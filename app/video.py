"""Camera-move videos from one image (2.5D novel view synthesis).

1. Depth Anything V2 Small (ONNX, Apache-2.0) estimates relative depth on the CPU.
2. Every pixel becomes a 3D point. For each frame a virtual camera moves (push
   in, pan, orbit) and the points are splatted forward with a z-buffer, so near
   objects correctly cover far ones (no ghosting at object edges).
3. Optional generative fill (prior-based view synthesis): moving the camera
   reveals areas that were hidden behind objects. At one or two keyframes
   (the poses that reveal the most) those holes are repainted by FLUX.2 and
   the new pixels are added back to the point cloud behind the objects, so
   every later frame shows them consistently. That allows bigger moves.
4. The frames are encoded as an H.264 MP4 that browsers play.
"""

from __future__ import annotations

import asyncio
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Iterator, Optional, Protocol

import cv2
import httpx
import numpy as np
from PIL import Image

MODEL_URL = "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx"
MODEL_FILE = "depth-anything-v2-small.onnx"
MODEL_MIN_BYTES = 50_000_000  # the file is about 99 MB; anything much smaller is a broken download
DEPTH_SIDE = 518  # the model's training resolution (a multiple of its 14 px patches)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

MAX_VIDEO_SIDE = 1024
FPS = 24


class VideoError(Exception):
    def __init__(self, message: str, status: int = 500):
        super().__init__(message)
        self.message = message
        self.status = status


class DepthEstimator(Protocol):
    def estimate(self, image: Image.Image) -> np.ndarray:
        """Disparity in 0..1 (1 = nearest), float32, same height and width as the image."""
        ...


class OnnxDepth:
    """Depth Anything V2 Small with onnxruntime. The model is downloaded on first use."""

    def __init__(self, model_dir: Path, url: str = MODEL_URL):
        self.path = model_dir / MODEL_FILE
        self.url = url
        self._session = None
        self._lock = threading.Lock()

    def _download(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".part")
        try:
            with httpx.stream("GET", self.url, follow_redirects=True, timeout=httpx.Timeout(120, connect=15)) as r:
                r.raise_for_status()
                with tmp.open("wb") as f:
                    for chunk in r.iter_bytes(1 << 20):
                        f.write(chunk)
        except httpx.HTTPError:
            tmp.unlink(missing_ok=True)
            raise VideoError("Could not download the depth model. Check the internet connection and try again.", 503)
        if tmp.stat().st_size < MODEL_MIN_BYTES:
            tmp.unlink(missing_ok=True)
            raise VideoError("The depth model download was incomplete. Try again.", 503)
        tmp.replace(self.path)

    def session(self):
        with self._lock:
            if self._session is None:
                import onnxruntime as ort

                if not self.path.exists():
                    self._download()
                opts = ort.SessionOptions()
                opts.log_severity_level = 3
                self._session = ort.InferenceSession(str(self.path), opts, providers=["CPUExecutionProvider"])
            return self._session

    def estimate(self, image: Image.Image) -> np.ndarray:
        rgb = image.convert("RGB")
        W, H = rgb.size
        scale = DEPTH_SIDE / max(W, H)
        w = max(14, round(W * scale / 14) * 14)
        h = max(14, round(H * scale / 14) * 14)
        x = np.asarray(rgb.resize((w, h), Image.BICUBIC), dtype=np.float32) / 255.0
        x = ((x - MEAN) / STD).transpose(2, 0, 1)[None]
        sess = self.session()
        out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
        disp = np.squeeze(out).astype(np.float32)
        disp = cv2.resize(disp, (W, H), interpolation=cv2.INTER_CUBIC)
        return normalize(disp)


def normalize(disp: np.ndarray) -> np.ndarray:
    """Robust 0..1 scaling, so a few extreme pixels don't flatten the rest."""
    lo, hi = np.percentile(disp, [1, 99])
    return np.clip((disp - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


# --------------------------------------------------------------------------
# Camera moves
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Motion:
    key: str
    label: str
    description: str
    path: Callable[[float], tuple[float, float, float]]  # t in 0..1 -> camera (tx, ty, tz)


def _ease(t: float) -> float:
    return t * t * (3 - 2 * t)


MOTIONS: dict[str, Motion] = {
    m.key: m
    for m in [
        Motion("push", "Push in", "The camera moves slowly into the room.",
               lambda t: (0.0, 0.0, 0.18 * _ease(t))),
        Motion("pan", "Pan", "The camera slides left and right; near objects pass faster than far ones.",
               lambda t: (0.03 * math.sin(2 * math.pi * t), 0.0, 0.03)),
        Motion("orbit", "Orbit", "The camera circles gently, as if looking around.",
               lambda t: (0.028 * math.sin(2 * math.pi * t), 0.014 * math.cos(2 * math.pi * t) - 0.014,
                          0.06 + 0.04 * math.sin(math.pi * t))),
    ]
}


def video_size(w: int, h: int, max_side: int = MAX_VIDEO_SIDE) -> tuple[int, int]:
    """Fit inside max_side, both sides multiples of 16 (what H.264 encoders want)."""
    scale = min(1.0, max_side / max(w, h))
    return max(16, round(w * scale / 16) * 16), max(16, round(h * scale / 16) * 16)


def prepare_depth(disp: np.ndarray, edge: float = 0.05) -> np.ndarray:
    """Smooth the disparity but keep edges, then snap pixels on a depth edge to the
    near or the far side. The depth model blurs edges, and in-between depths become
    "flying pixels": a pale halo that floats between a chair and the floor."""
    d = cv2.bilateralFilter(disp.astype(np.float32), 9, 0.1, 5)
    k = np.ones((5, 5), np.uint8)
    # Repeated snapping (a shock filter) turns even a wide blurred ramp into a clean step.
    for _ in range(6):
        lo, hi = cv2.erode(d, k), cv2.dilate(d, k)
        on_edge = (hi - lo) > edge
        d = np.where(on_edge, np.where(d - lo < hi - d, lo, hi), d)
    return d


Pose = tuple[float, float, float]  # camera offset (tx, ty, tz) in units of the nearest depth


@dataclass
class View:
    image: np.ndarray  # h x w x 3 uint8
    depth: np.ndarray  # h x w float32 world depth, 0 where nothing landed
    holes: np.ndarray  # h x w bool


class PointCloud:
    """The image as coloured 3D points (one per pixel, plus any generated fill).

    A point is stored by its position on the original image plane (px, py) and its
    depth Z (1 = nearest, near_far = farthest). For a camera at pose t it lands at
    q = c + overscan * ((p - c) * Z - f * t) / (Z - tz).
    """

    near_far = 5.0
    overscan = 1.04  # zoom in a little so small moves never show empty borders

    def __init__(self, image: np.ndarray, disp: np.ndarray):
        h, w = disp.shape
        self.h, self.w = h, w
        self.f = float(max(w, h))  # ~53 degree field of view
        self.cx, self.cy = (w - 1) / 2, (h - 1) / 2
        dmin = 1.0 / self.near_far
        Z = 1.0 / (dmin + (1 - dmin) * prepare_depth(disp))
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        self._parts = [(xs.ravel(), ys.ravel(), Z.astype(np.float32).ravel(), image.reshape(-1, 3))]
        self._pack()

    def _pack(self) -> None:
        px, py, Z, rgb = (np.concatenate(c) for c in zip(*self._parts))
        order = np.argsort(Z, kind="stable")  # nearest first: rank 0 wins the z-buffer
        self.px, self.py, self.Z, self.rgb = px[order], py[order], Z[order], rgb[order]

    @property
    def size(self) -> int:
        return len(self.Z)

    def add(self, px: np.ndarray, py: np.ndarray, Z: np.ndarray, rgb: np.ndarray) -> None:
        if len(Z):
            self._parts.append((px.astype(np.float32), py.astype(np.float32), Z.astype(np.float32), rgb))
            self._pack()

    def project(self, pose: Pose) -> tuple[np.ndarray, np.ndarray]:
        tx, ty, tz = pose
        den = self.Z - tz
        qx = self.cx + self.overscan * ((self.px - self.cx) * self.Z - self.f * tx) / den
        qy = self.cy + self.overscan * ((self.py - self.cy) * self.Z - self.f * ty) / den
        return qx, qy

    def render(self, pose: Pose, fill_cracks: bool = True) -> View:
        """Forward splat with a z-buffer. Each point also covers the pixel to its
        right, below and diagonally, which closes the 1 px cracks that open when the camera
        moves closer. Depth decides first (key = depth rank * 4 + footprint), so a
        nearer point's extra footprint still beats a farther point: objects grow by
        at most a pixel but never let the background show through."""
        qx, qy = self.project(pose)
        n, w, h = self.size, self.w, self.h
        empty = 4 * n
        best = np.full(w * h, empty, dtype=np.int64)
        key = np.arange(n, dtype=np.int64) * 4
        xi, yi = np.floor(qx + 0.5).astype(np.int64), np.floor(qy + 0.5).astype(np.int64)
        for k, (dx, dy) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
            x, y = xi + dx, yi + dy
            ok = (x >= 0) & (x < w) & (y >= 0) & (y < h) & (self.Z - pose[2] > 0.05)
            np.minimum.at(best, y[ok] * w + x[ok], key[ok] + k)
        hit = best < empty
        winner = best[hit] // 4
        image = np.zeros((h * w, 3), np.uint8)
        depth = np.zeros(h * w, np.float32)
        image[hit] = self.rgb[winner]
        depth[hit] = self.Z[winner]
        holes = ~hit.reshape(h, w)
        image = image.reshape(h, w, 3)
        if fill_cracks and holes.any():
            image = cv2.inpaint(image, holes.astype(np.uint8), 3, cv2.INPAINT_TELEA)
        return View(image=image, depth=depth.reshape(h, w), holes=holes)

    def unproject(self, pose: Pose, xs: np.ndarray, ys: np.ndarray, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Inverse of project: output pixel (xs, ys) at depth Z seen from pose -> image-plane position."""
        tx, ty, tz = pose
        qx = self.cx + (xs - self.cx) / self.overscan
        qy = self.cy + (ys - self.cy) / self.overscan
        px = self.cx + ((qx - self.cx) * (Z - tz) + self.f * tx) / Z
        py = self.cy + ((qy - self.cy) * (Z - tz) + self.f * ty) / Z
        return px, py

    def add_view(self, pose: Pose, view: View, filled: np.ndarray) -> int:
        """Add the generated pixels of a filled keyframe as new points. A revealed area
        lies behind whatever covered it: it continues the surface right next to it on
        the far side (the floor behind a chair, not the distant window), so depth is
        grown into the hole from its edges, always taking the farther neighbour.
        Returns how many points were added."""
        holes = view.holes
        if not holes.any():
            return 0
        Z, _ = background_fill(view, float(self.Z.max()))
        ys, xs = np.nonzero(holes)
        Z = Z[ys, xs]
        px, py = self.unproject(pose, xs.astype(np.float32), ys.astype(np.float32), Z)
        self.add(px, py, Z, filled[ys, xs])
        return len(Z)


NEIGHBOURS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]


def background_fill(view: View, fallback: float) -> tuple[np.ndarray, np.ndarray]:
    """Grow depth and colour into the holes one pixel at a time from their edges,
    each new pixel copying its farthest known neighbour. A revealed area continues
    the surface behind the object (the floor behind a chair), so this is a far
    better first guess than blending in the object's own colour."""
    Z = np.where(view.holes, 0, view.depth).astype(np.float32)
    rgb = view.image.copy()
    todo = view.holes.copy()
    h, w = Z.shape
    # The far surface near each hole. Only pixels at about that depth may grow into the
    # hole, so the near object's edge never spreads into the gap it uncovered.
    far_ref = cv2.dilate(Z, np.ones((31, 31), np.uint8))
    strict = True
    for _ in range(2 * max(h, w)):
        if not todo.any():
            break
        best_z = np.zeros_like(Z)
        best_rgb = np.zeros_like(rgb)
        for dy, dx in NEIGHBOURS:
            nz = np.zeros_like(Z)
            nz[max(dy, 0): h + min(dy, 0), max(dx, 0): w + min(dx, 0)] = Z[max(-dy, 0): h + min(-dy, 0), max(-dx, 0): w + min(-dx, 0)]
            farther = nz > best_z
            if farther.any():
                best_z[farther] = nz[farther]
                shifted = np.zeros_like(rgb)
                shifted[max(dy, 0): h + min(dy, 0), max(dx, 0): w + min(dx, 0)] = rgb[max(-dy, 0): h + min(-dy, 0), max(-dx, 0): w + min(-dx, 0)]
                best_rgb[farther] = shifted[farther]
        step = todo & (best_z > 0)
        if strict:
            step &= best_z >= 0.8 * far_ref
        if not step.any():
            if strict:
                strict = False  # whatever is left has no far neighbour: take any side
                continue
            break
        Z[step] = best_z[step]
        rgb[step] = best_rgb[step]
        todo &= ~step
    Z[todo] = fallback  # nothing known anywhere (an empty frame)
    return Z, rgb


def hole_mask(view: View, min_area: float = 0.003) -> Optional[np.ndarray]:
    """Revealed areas worth repainting (cracks removed, grown a little for blending),
    or None when they are too small to bother."""
    big = cv2.morphologyEx(view.holes.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if big.mean() < min_area:
        return None
    return cv2.dilate(big, np.ones((9, 9), np.uint8)) > 0


def pick_keyframes(cloud: PointCloud, path: Callable[[float], Pose], max_keys: int = 2) -> list[float]:
    """The poses along the path that reveal the most hidden area: the largest one,
    then the largest one at least 0.3 away from it if it is still substantial."""
    ts = np.linspace(0, 1, 13)
    area = [float(cloud.render(path(float(t)), fill_cracks=False).holes.mean()) for t in ts]
    first = int(np.argmax(area))
    if area[first] < 0.003:
        return []
    keys = [float(ts[first])]
    far_ones = [i for i in range(len(ts)) if abs(ts[i] - ts[first]) >= 0.3]
    if max_keys > 1 and far_ones:
        second = max(far_ones, key=lambda i: area[i])
        if area[second] >= 0.3 * area[first]:
            keys.append(float(ts[second]))
    return keys


def render_frames(cloud: PointCloud, path: Callable[[float], Pose], seconds: float) -> Iterator[np.ndarray]:
    n = max(2, round(seconds * FPS))
    for i in range(n):
        yield cloud.render(path(i / (n - 1))).image


def encode_mp4(frames: Iterator[np.ndarray], path: Path, fps: int = FPS) -> int:
    """H.264 / yuv420p with faststart, which every browser plays. Returns the frame count."""
    import imageio_ffmpeg

    count = 0
    writer = None
    try:
        for frame in frames:
            if writer is None:
                h, w = frame.shape[:2]
                writer = imageio_ffmpeg.write_frames(
                    str(path), (w, h), fps=fps, codec="libx264", pix_fmt_out="yuv420p", quality=8,
                    macro_block_size=16, output_params=["-movflags", "+faststart"], ffmpeg_log_level="error",
                )
                writer.send(None)  # start the encoder
            writer.send(np.ascontiguousarray(frame))
            count += 1
    finally:
        if writer is not None:
            writer.close()
    return count


def depth_preview(disp: np.ndarray) -> Image.Image:
    colored = cv2.applyColorMap((disp * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return Image.fromarray(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))


FILL_PROMPT = (
    "Image 0 has a few blurry, smeared patches next to objects, where hidden parts of the room come into view. "
    "Repaint them as sharp, realistic surfaces that continue the surrounding walls, floor, furniture and objects, "
    "matching perspective, materials, texture and lighting."
)
LARGE_MOVE = 2.0  # how much bigger the moves get when revealed areas are filled

# inpaint(image, mask) -> image with the masked area repainted (FLUX.2 through the brushed-area path)
Inpaint = Callable[[Image.Image, Image.Image], Awaitable[Image.Image]]


async def make_video(
    image: Image.Image, depth: DepthEstimator, motion_key: str, seconds: float, out_path: Path,
    depth_path: Optional[Path] = None, inpaint: Optional[Inpaint] = None,
) -> dict:
    """Render the video. With `inpaint`, the move is LARGE_MOVE times bigger and the areas
    it reveals are repainted at up to two keyframes. Returns frames and fills used."""
    if motion_key not in MOTIONS:
        raise VideoError(f"Unknown camera move: {motion_key}", 400)
    motion = MOTIONS[motion_key]
    work = image.convert("RGB")
    work = work.resize(video_size(*work.size), Image.LANCZOS)
    disp = await asyncio.to_thread(depth.estimate, work)
    if depth_path is not None:
        await asyncio.to_thread(lambda: depth_preview(disp).save(depth_path))
    cloud = await asyncio.to_thread(PointCloud, np.asarray(work), disp)

    scale = LARGE_MOVE if inpaint else 1.0
    path = lambda t: tuple(v * scale for v in motion.path(t))  # noqa: E731
    fills = 0
    if inpaint:
        for t in await asyncio.to_thread(pick_keyframes, cloud, path):
            pose = path(t)
            view = await asyncio.to_thread(cloud.render, pose, False)
            mask = hole_mask(view)
            if mask is None:
                continue
            # First guess for FLUX.2: the background stretched into the holes, softened a little.
            _, guess = await asyncio.to_thread(background_fill, view, float(cloud.Z.max()))
            soft = cv2.GaussianBlur(guess, (0, 0), 2)
            guess = np.where(view.holes[..., None], soft, guess)
            filled = await inpaint(Image.fromarray(guess), Image.fromarray(mask.astype(np.uint8) * 255))
            filled_arr = np.asarray(filled.convert("RGB").resize((cloud.w, cloud.h)))
            await asyncio.to_thread(cloud.add_view, pose, view, filled_arr)
            fills += 1
    frames = await asyncio.to_thread(encode_mp4, render_frames(cloud, path, seconds), out_path)
    return {"frames": frames, "fills": fills, "points": cloud.size}


def keyframe_budget(motion_key: str) -> int:
    """Upper bound of FLUX.2 fills one video can use (reserved up front)."""
    return 1 if motion_key == "push" else 2


def public_motions() -> list[dict]:
    return [{"key": m.key, "label": m.label, "description": m.description} for m in MOTIONS.values()]
