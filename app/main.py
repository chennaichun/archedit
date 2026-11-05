"""HTTP server: web UI, line-art preview, Claude prompt writing and FLUX.2
generation through Cloudflare Workers AI.

Run:  uvicorn app.main:app --port 8765
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import random
import re
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from .align import align
from .agent import AgentStop, Brain, ThreadStore, build_graph, make_brain, run_input
from .catalog import MODES, SIZES, public_catalog
from .video import FILL_PROMPT, DepthEstimator, OnnxDepth, VideoError, keyframe_budget, make_video, public_motions
from .cloudflare import CloudflareImageClient, ImageAPIError, ImageRequest, get_model
from .config import Settings, get_settings
from .guards import DailyLimiter, ResultCache, delete_old_outputs
from .llm import ClaudePromptWriter, LLMError, PromptWriter
from .preprocess import (
    PreprocessError,
    canny,
    composite_local,
    fit_ratio,
    limit_size,
    plan_local_edit,
    reference_png,
)

MAX_UPLOAD = 20 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 40_000_000  # reject decompression bombs (~6300 x 6300)
VISITOR_COOKIE = "cs_visitor"
CLEANUP_EVERY = 600  # seconds between output-folder sweeps
MAX_VARIATIONS = 4
STYLE_HINT = " Match the style, colors and mood of image 1."
# The server pastes back only the brushed area, so the prompt just has to name the
# change; it should not guess where the area is ("at the center" misled the model).
LOCAL_HINT = " Keep everything else in image 0 the same, with the same framing, camera angle, perspective and lighting."


def parse_size(size: str, source: Optional[Image.Image] = None) -> tuple[int, int]:
    if size not in SIZES:
        raise HTTPException(400, f"Unsupported size {size}.")
    if size == "auto":
        return auto_size(source)
    w, h = size.split("x")
    return int(w), int(h)


def auto_size(source: Optional[Image.Image], megapixels: float = 1.0) -> tuple[int, int]:
    """About 1 MP in the source's aspect ratio (clamped to 1:3..3:1), multiples of 16."""
    if source is None:
        return 1024, 1024
    ratio = min(3.0, max(1 / 3, source.width / source.height))
    w = (megapixels * 1_000_000 * ratio) ** 0.5
    h = w / ratio
    return (int(min(1920, max(256, round(w / 16) * 16))), int(min(1920, max(256, round(h / 16) * 16))))


async def read_image(upload: Optional[UploadFile]) -> Optional[Image.Image]:
    if upload is None:
        return None
    data = await upload.read()
    if not data:
        return None
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Image is larger than 20 MB.")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(400, "Could not read this image. Upload a JPG, PNG or WebP under 40 megapixels.")
    return image.convert("RGB")


def png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def guide_image(source: Image.Image, mode: str, width: int, height: int, low: int, high: int) -> Image.Image:
    """The image sent to FLUX.2 as image 0 for this mode."""
    if mode == "local":
        return limit_size(source)
    fitted = fit_ratio(source, width, height)
    if mode == "edges":
        return canny(fitted, low, high)
    return fitted


def visitor_id(request: Request) -> str:
    """Opaque per-browser id from the cookie set by the middleware."""
    return request.state.visitor


def owner_tag(visitor: str) -> str:
    """What gets written to disk: a hash, so the cookie value itself is never stored."""
    return hashlib.sha256(visitor.encode()).hexdigest()[:16]


def create_app(
    settings: Optional[Settings] = None,
    image_client: Optional[CloudflareImageClient] = None,
    prompt_writer: Optional[PromptWriter] = None,
    agent_brain: Optional[Brain] = None,
    depth_estimator: Optional[DepthEstimator] = None,
) -> FastAPI:
    settings = settings or get_settings()
    image_client = image_client or CloudflareImageClient(settings)
    prompt_writer = prompt_writer or ClaudePromptWriter(settings)
    if agent_brain is None and settings.agent_brain != "off":
        agent_brain = make_brain(settings, image_client)
    depth_estimator = depth_estimator or OnnxDepth(settings.model_dir)
    video_slots = asyncio.Semaphore(1)  # videos run on this server's CPU: one at a time
    out_dir = settings.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    limiter = DailyLimiter(settings.daily_limit_per_visitor, settings.daily_limit_total)
    cache = ResultCache()
    last_cleanup = [0.0]

    def maybe_cleanup() -> None:
        now = time.time()
        if now - last_cleanup[0] >= CLEANUP_EVERY:
            last_cleanup[0] = now
            delete_old_outputs(out_dir, settings.output_ttl_hours, now)

    async def fix_drift(reference: Image.Image, output: Image.Image) -> tuple[Image.Image, Optional[dict]]:
        """Drift guard for whole-image rounds: warp the result back onto the input's framing."""
        if not settings.align_rounds:
            return output, None
        ref = reference.convert("RGB").resize(output.size, Image.LANCZOS)
        fixed, drift = await asyncio.to_thread(align, ref, output)
        return fixed, drift.public()

    def save_output(visitor: str, output: Image.Image, guide_png: Optional[bytes], meta: dict) -> dict:
        """Write the image, its guide and a JSON record (with the owner hash) to out_dir."""
        job_id = uuid.uuid4().hex
        output.convert("RGB").save(out_dir / f"{job_id}.png")
        if guide_png:
            (out_dir / f"{job_id}-guide.png").write_bytes(guide_png)
        meta = {
            "id": job_id,
            **meta,
            "image_url": f"/outputs/{job_id}.png",
            "guide_url": f"/outputs/{job_id}-guide.png" if guide_png else None,
        }
        record = {**meta, "owner": owner_tag(visitor)}
        (out_dir / f"{job_id}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
        return meta

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        maybe_cleanup()
        yield
        await image_client.aclose()

    app = FastAPI(title="ArchEdit", lifespan=lifespan)
    app.state.limiter = limiter
    app.state.cache = cache

    @app.middleware("http")
    async def _visitor_and_cache_headers(request: Request, call_next):
        visitor = request.cookies.get(VISITOR_COOKIE)
        new_visitor = not visitor or len(visitor) > 64
        if new_visitor:
            visitor = secrets.token_urlsafe(24)
        request.state.visitor = visitor
        response = await call_next(request)
        if new_visitor:
            response.set_cookie(
                VISITOR_COOKIE, visitor, max_age=365 * 24 * 3600, httponly=True,
                samesite="lax", secure=settings.secure_cookies,
            )
        # Make browsers revalidate the UI files so updates show up on reload.
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.exception_handler(LLMError)
    async def _llm_error(_req, exc: LLMError):
        return JSONResponse({"detail": exc.message}, status_code=exc.status)

    @app.exception_handler(VideoError)
    async def _video_error(_req, exc: VideoError):
        return JSONResponse({"detail": exc.message}, status_code=exc.status)

    @app.exception_handler(ImageAPIError)
    async def _image_error(_req, exc: ImageAPIError):
        return JSONResponse({"detail": exc.message}, status_code=exc.status)

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    @app.get("/api/status")
    async def status(request: Request):
        return {
            "cloudflare_ready": settings.cloudflare_ready,
            "claude_ready": settings.claude_ready,
            "default_model": settings.default_image_model,
            "claude_model": settings.claude_model,
            "agent_brain": agent_brain.name if agent_brain else None,
            "remaining_today": limiter.remaining(visitor_id(request)),
        }

    @app.get("/api/catalog")
    async def catalog():
        return {**public_catalog(), "motions": public_motions()}

    @app.post("/api/preview")
    async def preview(
        image: UploadFile = File(...),
        mode: str = Form("edit"),
        size: str = Form("auto"),
        low: int = Form(100),
        high: int = Form(200),
    ):
        """Show what will be sent to FLUX.2 as image 0 (cropped photo or line art)."""
        if mode not in MODES:
            raise HTTPException(400, f"Unknown mode: {mode}")
        src = await read_image(image)
        if src is None:
            raise HTTPException(400, "Upload an image.")
        width, height = parse_size(size, src)
        try:
            guide = await asyncio.to_thread(guide_image, src, mode, width, height, low, high)
        except PreprocessError as e:
            raise HTTPException(400, str(e))
        return Response(png_bytes(guide), media_type="image/png")

    @app.post("/api/prompt")
    async def write_prompt(
        idea: str = Form(...),
        mode: Optional[str] = Form(None),
        image: Optional[UploadFile] = File(None),
    ):
        idea = idea.strip()
        if not idea or len(idea) > 2000:
            raise HTTPException(400, "Describe the change in 1 to 2000 characters.")
        if not settings.claude_ready:
            raise HTTPException(503, "Claude is not set up on this server. Write the prompt yourself.")
        src = await read_image(image)
        image_png = None
        if src is not None:
            src.thumbnail((1024, 1024))
            image_png = png_bytes(src)
        suggestion = await prompt_writer.suggest(idea, image_png, mode)
        return suggestion.model_dump()

    @app.post("/api/generate")
    async def generate(
        request: Request,
        prompt: str = Form(...),
        mode: str = Form("edit"),
        model: str = Form(""),
        size: str = Form("auto"),
        seed: int = Form(-1),
        guidance: Optional[float] = Form(None),
        steps: Optional[int] = Form(None),
        low: int = Form(100),
        high: int = Form(200),
        image: Optional[UploadFile] = File(None),
        style_image: Optional[UploadFile] = File(None),
        mask: Optional[UploadFile] = File(None),
        variations: int = Form(1),
    ):
        visitor = visitor_id(request)
        if not 1 <= variations <= MAX_VARIATIONS:
            raise HTTPException(400, f"Variations must be between 1 and {MAX_VARIATIONS}.")
        if mode not in MODES:
            raise HTTPException(400, f"Unknown mode: {mode}")
        prompt = prompt.strip()
        if not prompt:
            raise HTTPException(400, "Enter a prompt.")
        if len(prompt) > 2000:
            raise HTTPException(400, "Keep the prompt under 2000 characters.")
        if guidance is not None and not 0 < guidance <= 20:
            raise HTTPException(400, "Guidance must be between 0 and 20.")
        if steps is not None and not 1 <= steps <= 50:
            raise HTTPException(400, "Steps must be between 1 and 50.")
        image_model = get_model(model or settings.default_image_model)
        if size not in SIZES:
            raise HTTPException(400, f"Unsupported size {size}.")

        references: list[bytes] = []
        guide_png: Optional[bytes] = None
        plan = None
        align_ref: Optional[Image.Image] = None  # whole-image edits are registered back onto this
        if mode == "local":
            src = await read_image(image)
            mask_img = await read_image(mask)
            if src is None:
                raise HTTPException(400, "Upload an image first.")
            if mask_img is None:
                raise HTTPException(400, "Brush the area you want to change first.")
            try:
                plan = await asyncio.to_thread(plan_local_edit, src, mask_img)
            except PreprocessError as e:
                raise HTTPException(400, str(e))
            width, height = plan.gen_width, plan.gen_height
            guide_png = png_bytes(plan.crop)
            references.append(reference_png(plan.crop))
        elif mode != "text":
            src = await read_image(image)
            if src is None:
                raise HTTPException(400, "This mode needs an image. Upload one first.")
            width, height = parse_size(size, src)
            try:
                guide = await asyncio.to_thread(guide_image, src, mode, width, height, low, high)
            except PreprocessError as e:
                raise HTTPException(400, str(e))
            guide_png = png_bytes(guide)
            references.append(reference_png(guide))
            if mode == "edit":
                align_ref = guide
        else:
            width, height = parse_size(size)
        style = await read_image(style_image)
        if style is not None:
            if not references:  # keep "image 1" meaning the style reference
                raise HTTPException(400, "A style image works with Restyle, Line art or Local mode.")
            references.append(reference_png(style))
            prompt_sent = prompt + STYLE_HINT
        else:
            prompt_sent = prompt
        if plan is not None:
            prompt_sent += LOCAL_HINT

        fixed_seed = seed >= 0
        base_seed = seed if fixed_seed else random.randint(0, 2**31 - 1 - MAX_VARIATIONS)
        effective_steps = steps if image_model.adjustable_steps else None
        out_size = size

        def cache_key_for(s: int) -> str:
            return ResultCache.key(
                image_model.model_id, prompt_sent, width, height, s, guidance, effective_steps,
                images=tuple(references),
            )

        # Variations are the same request with consecutive seeds. Reserve the daily
        # allowance for all of them up front, so a batch never stops halfway.
        seeds = [base_seed + i for i in range(variations)]
        cached_results = {s: cache.get(cache_key_for(s)) if fixed_seed else None for s in seeds}
        to_generate = [s for s in seeds if cached_results[s] is None]
        refusal = limiter.check(visitor, len(to_generate)) if to_generate else None
        if refusal:
            raise HTTPException(429, refusal)

        async def run_one(s: int) -> dict:
            started = time.time()
            result = cached_results[s]
            cached = result is not None
            if not cached:
                result = await image_client.generate(
                    ImageRequest(
                        prompt=prompt_sent, model=image_model, width=width, height=height,
                        references=references, seed=s, guidance=guidance, steps=effective_steps,
                    )
                )
                limiter.record(visitor)
                cache.put(cache_key_for(s), result)
            elapsed = round(time.time() - started, 1)
            try:
                output = Image.open(io.BytesIO(result))
                output.load()
            except (UnidentifiedImageError, OSError):
                raise HTTPException(502, "Cloudflare returned something that is not an image.")
            final_size = out_size
            drift = None
            if plan is not None:
                output = await asyncio.to_thread(composite_local, plan, output)
                final_size = f"{output.width}x{output.height}"
            elif align_ref is not None:
                output, drift = await fix_drift(align_ref, output)

            return save_output(visitor, output, guide_png, {
                "prompt": prompt,
                "mode": mode,
                "model": image_model.key,
                "size": final_size,
                "seed": s,
                "guidance": guidance,
                "steps": effective_steps,
                "style_reference": style is not None,
                "elapsed": elapsed,
                "cached": cached,
                "created": started,
                "drift": drift,
            })

        # Run the variations in parallel: total time is about one generation, not three.
        results = await asyncio.gather(*(run_one(s) for s in seeds), return_exceptions=True)
        variants = [r for r in results if isinstance(r, dict)]
        if not variants:
            raise next(r for r in results if isinstance(r, BaseException))
        maybe_cleanup()
        failed = len(results) - len(variants)
        return {**variants[0], "variants": variants, "failed": failed, "remaining_today": limiter.remaining(visitor)}

    # ---------------- chat editing agent (LangGraph) ----------------
    async def agent_edit(image_file: str, prompt: str, visitor: str, model_key: str, mask_file: Optional[str]) -> dict:
        """One round for the agent: the same path as /api/generate in edit mode, or in
        local mode (crop, edit, paste back) when the user brushed an area."""
        image_model = get_model(model_key or settings.default_image_model)
        refusal = limiter.check(visitor, 1)
        if refusal:
            raise AgentStop(refusal)
        src = Image.open(out_dir / image_file)
        src.load()
        src = src.convert("RGB")
        plan = None
        if mask_file:
            mask_img = Image.open(out_dir / mask_file)
            mask_img.load()
            try:
                plan = await asyncio.to_thread(plan_local_edit, src, mask_img)
            except PreprocessError as e:
                raise AgentStop(str(e))
            width, height, guide = plan.gen_width, plan.gen_height, plan.crop
            prompt_sent = prompt + LOCAL_HINT
        else:
            width, height = auto_size(src)
            guide = await asyncio.to_thread(guide_image, src, "edit", width, height, 100, 200)
            prompt_sent = prompt
        seed = random.randint(0, 2**31 - 1)
        started = time.time()
        result = await image_client.generate(
            ImageRequest(prompt=prompt_sent, model=image_model, width=width, height=height,
                         references=[reference_png(guide)], seed=seed)
        )
        limiter.record(visitor)
        try:
            output = Image.open(io.BytesIO(result))
            output.load()
        except (UnidentifiedImageError, OSError):
            raise ImageAPIError("Cloudflare returned something that is not an image.", 502)
        drift = None
        if plan is not None:
            output = await asyncio.to_thread(composite_local, plan, output)
        else:
            output, drift = await fix_drift(guide, output)
        return save_output(visitor, output, png_bytes(guide) if plan else None, {
            "prompt": prompt, "mode": "local" if plan else "edit", "model": image_model.key, "size": f"{output.width}x{output.height}",
            "seed": seed, "guidance": None, "steps": None, "style_reference": False,
            "elapsed": round(time.time() - started, 1), "cached": False, "created": started, "agent": True,
            "drift": drift,
        })

    threads = ThreadStore()
    graph = (
        build_graph(agent_brain, agent_edit, out_dir, max_steps=max(1, settings.agent_max_steps),
                    review=settings.agent_review, checkpointer=threads.saver)
        if agent_brain else None
    )
    app.state.threads = threads

    def sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    @app.post("/api/agent")
    async def agent_chat(
        request: Request,
        message: str = Form(...),
        thread: str = Form(...),
        image: UploadFile = File(...),
        mask: Optional[UploadFile] = File(None),
        model: str = Form(""),
    ):
        """One chat message: plan, edit round by round, check each round, reply.
        With a mask, every step changes only the brushed area.
        Streams Server-Sent Events: plan, attempt, check, round, done (or error)."""
        if graph is None:
            raise HTTPException(503, "The editing agent needs Cloudflare or a Claude API key on this server.")
        visitor = visitor_id(request)
        message = message.strip()
        if not message or len(message) > 2000:
            raise HTTPException(400, "Write a message of 1 to 2000 characters.")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", thread):
            raise HTTPException(400, "Invalid chat id.")
        get_model(model or settings.default_image_model)  # 400 on an unknown model
        refusal = limiter.check(visitor, 1)
        if refusal:
            raise HTTPException(429, refusal)
        src = await read_image(image)
        if src is None:
            raise HTTPException(400, "Upload an image first.")
        start_id = uuid.uuid4().hex
        start_file = f"{start_id}-start.png"
        await asyncio.to_thread(limit_size(src).save, out_dir / start_file)
        mask_file = None
        mask_img = await read_image(mask)
        if mask_img is not None:
            if mask_img.convert("L").point(lambda v: 255 if v > 127 else 0).getbbox() is None:
                raise HTTPException(400, "Brush the area you want to change first.")
            mask_file = f"{start_id}-mask.png"
            await asyncio.to_thread(mask_img.convert("L").save, out_dir / mask_file)

        thread_id = f"{owner_tag(visitor)}-{thread}"  # visitors can't reach each other's chats
        threads.touch(thread_id)
        config = {"configurable": {"thread_id": thread_id, "visitor": visitor, "model": model, "mask": mask_file},
                  "recursion_limit": 60}

        async def events():
            try:
                async for update in graph.astream(run_input(message, start_file), config, stream_mode="updates"):
                    for node, out in update.items():
                        out = out or {}
                        if node == "plan":
                            yield sse("plan", {"reply": out.get("plan_reply", ""),
                                               "steps": [s["title"] for s in out.get("steps", [])]})
                        elif node == "edit" and out.get("candidate"):
                            yield sse("attempt", {"image_url": out["candidate"]["image_url"]})
                        elif node == "check":
                            v = out.get("verdict") or {}
                            yield sse("check", {"ok": v.get("ok", True), "reason": v.get("reason", ""),
                                                "retry": bool(v.get("retry"))})
                        elif node == "accept":
                            yield sse("round", out["rounds"][-1])
                        elif node == "finish":
                            yield sse("done", {"reply": out.get("reply", ""), "notes": out.get("notes", []),
                                               "remaining_today": limiter.remaining(visitor)})
            except Exception as e:  # report instead of cutting the stream
                yield sse("error", {"detail": getattr(e, "message", None) or "The agent failed. Try again."})
            maybe_cleanup()

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ---------------- camera-move video (2.5D novel view synthesis) ----------------
    @app.post("/api/video")
    async def video(
        request: Request,
        image: UploadFile = File(...),
        motion: str = Form("push"),
        seconds: float = Form(4.0),
        fill: bool = Form(False),
        model: str = Form(""),
    ):
        """Estimate depth, move a virtual camera through it and return an MP4.
        Depth and rendering run on this server's CPU. With fill, the move is bigger and
        the areas it reveals are repainted by FLUX.2 (1-2 edits from the allowance)."""
        visitor = visitor_id(request)
        if not 2 <= seconds <= 8:
            raise HTTPException(400, "Length must be between 2 and 8 seconds.")
        src = await read_image(image)
        if src is None:
            raise HTTPException(400, "Upload an image first.")
        inpaint = None
        if fill:
            if not settings.cloudflare_ready:
                raise HTTPException(503, "Filling revealed areas needs Cloudflare. Turn it off or add the keys to .env.")
            image_model = get_model(model or settings.default_image_model)
            refusal = limiter.check(visitor, keyframe_budget(motion))
            if refusal:
                raise HTTPException(429, refusal)

            async def inpaint(view: Image.Image, hole_mask: Image.Image) -> Image.Image:
                plan = await asyncio.to_thread(plan_local_edit, view, hole_mask)
                result = await image_client.generate(ImageRequest(
                    prompt=FILL_PROMPT + LOCAL_HINT, model=image_model, width=plan.gen_width,
                    height=plan.gen_height, references=[reference_png(plan.crop)],
                    seed=random.randint(0, 2**31 - 1),
                ))
                limiter.record(visitor)
                try:
                    patch = Image.open(io.BytesIO(result))
                    patch.load()
                except (UnidentifiedImageError, OSError):
                    raise ImageAPIError("Cloudflare returned something that is not an image.", 502)
                return await asyncio.to_thread(composite_local, plan, patch)

        job_id = uuid.uuid4().hex
        video_path, depth_path = out_dir / f"{job_id}.mp4", out_dir / f"{job_id}-depth.png"
        started = time.time()
        async with video_slots:
            info = await make_video(src, depth_estimator, motion, seconds, video_path, depth_path, inpaint)
        maybe_cleanup()
        return {
            "id": job_id, "motion": motion, "seconds": seconds, "frames": info["frames"],
            "fills": info["fills"], "remaining_today": limiter.remaining(visitor),
            "elapsed": round(time.time() - started, 1),
            "video_url": f"/outputs/{job_id}.mp4", "depth_url": f"/outputs/{job_id}-depth.png",
        }

    @app.get("/api/history")
    async def history(request: Request):
        """This visitor's own recent edits only."""
        me = owner_tag(visitor_id(request))
        items = []
        metas = sorted(out_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for path in metas:
            try:
                record = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if record.pop("owner", None) != me:
                continue
            items.append(record)
            if len(items) >= 24:
                break
        return {"items": items}

    app.mount("/outputs", StaticFiles(directory=out_dir), name="outputs")
    app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")
    examples_dir = settings.static_dir.parent / "examples"
    if examples_dir.is_dir():
        app.mount("/examples", StaticFiles(directory=examples_dir), name="examples")

    @app.get("/")
    async def index():
        return FileResponse(settings.static_dir / "index.html")

    return app


app = create_app()
