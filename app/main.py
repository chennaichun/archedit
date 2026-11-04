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
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from .catalog import MODES, SIZES, public_catalog
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
STYLE_HINT = " Match the style, colors and mood of image 1."
# The server pastes back only the brushed area, so the prompt just has to name the
# change; it should not guess where the area is ("at the center" misled the model).
LOCAL_HINT = " Keep everything else in image 0 the same, with the same framing, camera angle, perspective and lighting."


def parse_size(size: str) -> tuple[int, int]:
    if size not in SIZES:
        raise HTTPException(400, f"Unsupported size {size}.")
    w, h = size.split("x")
    return int(w), int(h)


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
) -> FastAPI:
    settings = settings or get_settings()
    image_client = image_client or CloudflareImageClient(settings)
    prompt_writer = prompt_writer or ClaudePromptWriter(settings)
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
            "remaining_today": limiter.remaining(visitor_id(request)),
        }

    @app.get("/api/catalog")
    async def catalog():
        return public_catalog()

    @app.post("/api/preview")
    async def preview(
        image: UploadFile = File(...),
        mode: str = Form("edit"),
        size: str = Form("1024x1024"),
        low: int = Form(100),
        high: int = Form(200),
    ):
        """Show what will be sent to FLUX.2 as image 0 (cropped photo or line art)."""
        if mode not in MODES:
            raise HTTPException(400, f"Unknown mode: {mode}")
        width, height = parse_size(size)
        src = await read_image(image)
        if src is None:
            raise HTTPException(400, "Upload an image.")
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
        size: str = Form("1024x1024"),
        seed: int = Form(-1),
        guidance: Optional[float] = Form(None),
        steps: Optional[int] = Form(None),
        low: int = Form(100),
        high: int = Form(200),
        image: Optional[UploadFile] = File(None),
        style_image: Optional[UploadFile] = File(None),
        mask: Optional[UploadFile] = File(None),
    ):
        visitor = visitor_id(request)
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
        width, height = parse_size(size)

        references: list[bytes] = []
        guide_png: Optional[bytes] = None
        plan = None
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
            try:
                guide = await asyncio.to_thread(guide_image, src, mode, width, height, low, high)
            except PreprocessError as e:
                raise HTTPException(400, str(e))
            guide_png = png_bytes(guide)
            references.append(reference_png(guide))
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
        if not fixed_seed:
            seed = random.randint(0, 2**31 - 1)
        effective_steps = steps if image_model.adjustable_steps else None
        cache_key = ResultCache.key(
            image_model.model_id, prompt_sent, width, height, seed, guidance, effective_steps,
            images=tuple(references),
        )

        started = time.time()
        result = cache.get(cache_key) if fixed_seed else None
        cached = result is not None
        if not cached:
            refusal = limiter.check(visitor)
            if refusal:
                raise HTTPException(429, refusal)
            result = await image_client.generate(
                ImageRequest(
                    prompt=prompt_sent,
                    model=image_model,
                    width=width,
                    height=height,
                    references=references,
                    seed=seed,
                    guidance=guidance,
                    steps=effective_steps,
                )
            )
            limiter.record(visitor)
            cache.put(cache_key, result)
        elapsed = round(time.time() - started, 1)

        try:
            output = Image.open(io.BytesIO(result))
            output.load()
        except (UnidentifiedImageError, OSError):
            raise HTTPException(502, "Cloudflare returned something that is not an image.")
        if plan is not None:
            output = await asyncio.to_thread(composite_local, plan, output)
            size = f"{output.width}x{output.height}"

        job_id = uuid.uuid4().hex
        output.convert("RGB").save(out_dir / f"{job_id}.png")
        if guide_png:
            (out_dir / f"{job_id}-guide.png").write_bytes(guide_png)
        meta = {
            "id": job_id,
            "prompt": prompt,
            "mode": mode,
            "model": image_model.key,
            "size": size,
            "seed": seed,
            "guidance": guidance,
            "steps": effective_steps,
            "style_reference": style is not None,
            "elapsed": elapsed,
            "cached": cached,
            "created": started,
            "image_url": f"/outputs/{job_id}.png",
            "guide_url": f"/outputs/{job_id}-guide.png" if guide_png else None,
        }
        record = {**meta, "owner": owner_tag(visitor)}
        (out_dir / f"{job_id}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
        maybe_cleanup()
        return {**meta, "remaining_today": limiter.remaining(visitor)}

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
