"""HTTP server: web UI, line-art preview, Claude prompt writing and FLUX.2
generation through Cloudflare Workers AI.

Run:  uvicorn app.main:app --port 8000
"""

from __future__ import annotations

import io
import json
import random
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from .catalog import MODES, SIZES, public_catalog
from .cloudflare import CloudflareImageClient, ImageAPIError, ImageRequest, get_model
from .config import Settings, get_settings
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
STYLE_HINT = " Match the style, colors and mood of image 1."
LOCAL_HINT = (
    " Apply this change to the main subject at the center of image 0. Keep the same framing,"
    " camera angle, perspective, lighting and everything around it unchanged."
)


class PromptRequest(BaseModel):
    idea: str = Field(min_length=1, max_length=2000)
    mode: Optional[str] = None
    use_image: bool = False


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
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Could not read this image. Upload a JPG, PNG or WebP.")
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

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await image_client.aclose()

    app = FastAPI(title="ControlNet Studio", lifespan=lifespan)
    # The last uploaded photo, so Claude can look at it without a second upload.
    app.state.last_source = None

    @app.middleware("http")
    async def _no_stale_ui(request, call_next):
        # Make browsers revalidate the UI files so updates show up on reload.
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.exception_handler(LLMError)
    async def _llm_error(_req, exc: LLMError):
        return JSONResponse({"detail": exc.message}, status_code=exc.status)

    @app.exception_handler(ImageAPIError)
    async def _image_error(_req, exc: ImageAPIError):
        return JSONResponse({"detail": exc.message}, status_code=exc.status)

    @app.get("/api/status")
    async def status():
        return {
            "cloudflare_ready": settings.cloudflare_ready,
            "default_model": settings.default_image_model,
            "claude_model": settings.claude_model,
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
        app.state.last_source = src
        try:
            guide = guide_image(src, mode, width, height, low, high)
        except PreprocessError as e:
            raise HTTPException(400, str(e))
        return Response(png_bytes(guide), media_type="image/png")

    @app.post("/api/prompt")
    async def write_prompt(req: PromptRequest):
        image_png = None
        if req.use_image and app.state.last_source is not None:
            thumb = app.state.last_source.copy()
            thumb.thumbnail((1024, 1024))
            image_png = png_bytes(thumb)
        suggestion = await prompt_writer.suggest(req.idea, image_png, req.mode)
        return suggestion.model_dump()

    @app.post("/api/generate")
    async def generate(
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
        if mode not in MODES:
            raise HTTPException(400, f"Unknown mode: {mode}")
        prompt = prompt.strip()
        if not prompt:
            raise HTTPException(400, "Enter a prompt.")
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
            src = await read_image(image) or app.state.last_source
            mask_img = await read_image(mask)
            if src is None:
                raise HTTPException(400, "Upload an image first.")
            if mask_img is None:
                raise HTTPException(400, "Brush the area you want to change first.")
            try:
                plan = plan_local_edit(src, mask_img)
            except PreprocessError as e:
                raise HTTPException(400, str(e))
            width, height = plan.gen_width, plan.gen_height
            guide_png = png_bytes(plan.crop)
            references.append(reference_png(plan.crop))
        elif mode != "text":
            src = await read_image(image) or app.state.last_source
            if src is None:
                raise HTTPException(400, "This mode needs an image. Upload one first.")
            try:
                guide = guide_image(src, mode, width, height, low, high)
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

        if seed < 0:
            seed = random.randint(0, 2**31 - 1)
        started = time.time()
        result = await image_client.generate(
            ImageRequest(
                prompt=prompt_sent,
                model=image_model,
                width=width,
                height=height,
                references=references,
                seed=seed,
                guidance=guidance,
                steps=steps if image_model.adjustable_steps else None,
            )
        )
        elapsed = round(time.time() - started, 1)

        job_id = uuid.uuid4().hex[:12]
        try:
            output = Image.open(io.BytesIO(result))
            output.load()
        except (UnidentifiedImageError, OSError):
            raise HTTPException(502, "Cloudflare returned something that is not an image.")
        if plan is not None:
            output = composite_local(plan, output)
            size = f"{output.width}x{output.height}"
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
            "steps": steps if image_model.adjustable_steps else None,
            "style_reference": style is not None,
            "elapsed": elapsed,
            "created": started,
            "image_url": f"/outputs/{job_id}.png",
            "guide_url": f"/outputs/{job_id}-guide.png" if guide_png else None,
        }
        (out_dir / f"{job_id}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        return meta

    @app.get("/api/history")
    async def history():
        items = []
        metas = sorted(out_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:24]
        for path in metas:
            try:
                items.append(json.loads(path.read_text()))
            except (OSError, json.JSONDecodeError):
                continue
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
