"""Image helpers: crop to the output ratio, Canny line art, and reference
images sized for Cloudflare (each input must be under 512x512)."""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

MAX_REFERENCE_SIDE = 504  # Cloudflare: "All input images must be smaller than 512x512"
MAX_SOURCE_SIDE = 2048  # local edits keep the photo's own size, up to this


class PreprocessError(ValueError):
    pass


def fit_ratio(image: Image.Image, width: int, height: int) -> Image.Image:
    """Center-crop `image` to width:height, then resize to width x height."""
    image = image.convert("RGB")
    src_ratio = image.width / image.height
    dst_ratio = width / height
    if src_ratio > dst_ratio:
        new_w = round(image.height * dst_ratio)
        left = (image.width - new_w) // 2
        image = image.crop((left, 0, left + new_w, image.height))
    elif src_ratio < dst_ratio:
        new_h = round(image.width / dst_ratio)
        top = (image.height - new_h) // 2
        image = image.crop((0, top, image.width, top + new_h))
    return image.resize((width, height), Image.LANCZOS)


def canny(image: Image.Image, low: int = 100, high: int = 200, invert: bool = True) -> Image.Image:
    """Line art from Canny edges. Dark lines on white read best as a drawing for FLUX.2."""
    import cv2

    if not (0 <= low <= 255 and 0 <= high <= 255):
        raise PreprocessError("Canny thresholds must be between 0 and 255.")
    if low > high:
        low, high = high, low
    edges = cv2.Canny(np.array(image.convert("RGB")), low, high)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
    if invert:
        edges = 255 - edges
    return Image.fromarray(np.stack([edges] * 3, axis=-1))


def reference_png(image: Image.Image) -> bytes:
    """Shrink to fit under Cloudflare's 512px input limit and encode as PNG."""
    ref = image.convert("RGB").copy()
    ref.thumbnail((MAX_REFERENCE_SIDE, MAX_REFERENCE_SIDE), Image.LANCZOS)
    buf = io.BytesIO()
    ref.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Local edits: crop around the marked region, edit the crop, paste it back
# through a feathered mask so nothing outside the region changes.
# Benchmarks (ScribbleEdit, 2026) show editing models rarely understand marks
# drawn on the image, so the model only ever sees the plain crop.
# ---------------------------------------------------------------------------

@dataclass
class LocalPlan:
    source: Image.Image  # RGB, at most MAX_SOURCE_SIDE
    mask: Image.Image  # L, same size as source, 255 = edit
    box: tuple[int, int, int, int]  # crop box in source pixels
    gen_width: int  # size to request from the model
    gen_height: int

    @property
    def crop(self) -> Image.Image:
        return self.source.crop(self.box)


def limit_size(image: Image.Image, max_side: int = MAX_SOURCE_SIDE) -> Image.Image:
    image = image.convert("RGB")
    if max(image.size) > max_side:
        image = image.copy()
        image.thumbnail((max_side, max_side), Image.LANCZOS)
    return image


def _gen_size(w: int, h: int, long_side: int = 1024) -> tuple[int, int]:
    scale = long_side / max(w, h)

    def snap(v: float) -> int:
        return int(min(1920, max(256, round(v * scale / 16) * 16)))

    return snap(w), snap(h)


def plan_local_edit(source: Image.Image, mask: Image.Image, pad: float = 0.6, min_side: int = 256) -> LocalPlan:
    source = limit_size(source)
    W, H = source.size
    mask = mask.convert("L").resize((W, H), Image.NEAREST).point(lambda v: 255 if v > 127 else 0)
    bbox = mask.getbbox()
    if bbox is None:
        raise PreprocessError("Brush the area you want to change first.")
    x0, y0, x1, y1 = bbox
    # Crop that follows the region's own shape plus some context on every side.
    # (A square crop sized by the longest side made tall, thin regions such as a wall
    # use the whole room; the model then edited whatever sat in the middle instead.)
    cw = max((x1 - x0) * (1 + 2 * pad), min_side)
    ch = max((y1 - y0) * (1 + 2 * pad), min_side)
    # Keep the aspect ratio within 3:1 so the model gets a usable frame.
    cw, ch = max(cw, ch / 3), max(ch, cw / 3)
    cw, ch = min(round(cw), W), min(round(ch), H)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = int(min(max(cx - cw / 2, 0), W - cw))
    top = int(min(max(cy - ch / 2, 0), H - ch))
    box = (left, top, left + cw, top + ch)
    gw, gh = _gen_size(cw, ch)
    return LocalPlan(source=source, mask=mask, box=box, gen_width=gw, gen_height=gh)


def composite_local(plan: LocalPlan, edited: Image.Image) -> Image.Image:
    left, top, right, bottom = plan.box
    cw, ch = right - left, bottom - top
    patch = edited.convert("RGB").resize((cw, ch), Image.LANCZOS)
    feather = max(4, round(min(cw, ch) * 0.025))
    m = plan.mask.crop(plan.box)
    m = m.filter(ImageFilter.MaxFilter(2 * (feather // 2) + 1))  # grow a little so seams sit outside the object
    m = m.filter(ImageFilter.GaussianBlur(feather))
    out = plan.source.copy()
    out.paste(patch, (left, top), m)
    return out
