"""Drift guard: keep every whole-image round registered to the image it edited.

FLUX.2 redraws the whole scene from a reference capped at 504 px, and in
testing each round came back about 1% smaller and a few pixels off, sometimes
3-9% for a restyle. Over several rounds that adds up to a visibly drifting
camera. After each round we estimate the shift / scale / rotation between the
input and the result and, when it is reliable, warp the result back.

1. SIFT features + RANSAC give a similarity transform (robust to big changes,
   but lighting edits leave few matches).
2. ECC (enhanced correlation coefficient) refines it on gradient images.
   ECC is invariant to brightness and contrast, so it still locks on after
   relighting, and its correlation doubles as a "same geometry" score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image

WORK_SIDE = 768  # alignment runs at this size; the transform is scaled back up
MIN_INLIERS = 20
MAX_SCALE_DRIFT = 0.12  # beyond this it is a different framing, not drift
MAX_SHIFT = 0.08  # of the image width
FIX_THRESHOLD_PX = 1.0  # don't resample for sub-pixel drift


@dataclass
class Drift:
    scale: float  # result content size relative to the input (0.988 = 1.2% smaller)
    shift_x: float  # how far the picture centre moved, in full-size pixels
    shift_y: float
    rotation: float  # degrees
    max_offset: float  # largest pixel displacement anywhere in the frame
    geometry: float  # 0..1 correlation of edges after alignment (1 = same lines)
    geometry_before: float  # the same before alignment
    inliers: int
    reliable: bool
    fixed: bool = False

    def public(self) -> dict:
        d = asdict(self)
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


def _gray(img: Image.Image, size: tuple[int, int]) -> np.ndarray:
    g = cv2.cvtColor(np.asarray(img.convert("RGB").resize(size, Image.LANCZOS)), cv2.COLOR_RGB2GRAY)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)  # evens out lighting changes


def _gradient(g: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 1.2)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _sift_similarity(ref: np.ndarray, res: np.ndarray) -> tuple[Optional[np.ndarray], int]:
    """2x3 matrix mapping result pixels onto reference pixels, and the inlier count."""
    sift = cv2.SIFT_create(6000)
    kr, dr = sift.detectAndCompute(ref, None)
    ks, ds = sift.detectAndCompute(res, None)
    if dr is None or ds is None or len(kr) < 8 or len(ks) < 8:
        return None, 0
    pairs = cv2.BFMatcher().knnMatch(ds, dr, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < 8:
        return None, 0
    src = np.float32([ks[m.queryIdx].pt for m in good])
    dst = np.float32([kr[m.trainIdx].pt for m in good])
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None:
        return None, 0
    return M.astype(np.float32), int(inl.sum())


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else 0.0


def _offset(M: np.ndarray, w: int, h: int) -> float:
    corners = np.float32([[0, 0], [w, 0], [0, h], [w, h], [w / 2, h / 2]])
    moved = corners @ M[:, :2].T + M[:, 2]
    return float(np.abs(moved - corners).max())


def measure(reference: Image.Image, result: Image.Image) -> tuple[Drift, Optional[np.ndarray]]:
    """How the result is displaced from the reference. Returns the drift and, when
    reliable, a 2x3 matrix (full result resolution) that maps the result onto the reference."""
    W, H = result.size
    s = WORK_SIDE / max(W, H)
    size = (max(16, round(W * s)), max(16, round(H * s)))
    ref, res = _gray(reference, size), _gray(result, size)
    gref, gres = _gradient(ref), _gradient(res)
    before = _corr(gref, gres)

    M, inliers = _sift_similarity(ref, res)
    start = M if (M is not None and inliers >= MIN_INLIERS) else np.eye(2, 3, dtype=np.float32)
    # ECC wants the matrix that maps template (reference) coordinates into the input (result).
    warp = cv2.invertAffineTransform(start).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
    try:
        _, warp = cv2.findTransformECC(gref, gres, warp, cv2.MOTION_AFFINE, criteria, None, 5)
        ecc_ok = True
    except cv2.error:
        ecc_ok = False
    M = cv2.invertAffineTransform(warp)  # result -> reference, at work size

    aligned = cv2.warpAffine(gres, M, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    after = _corr(gref, aligned)

    scale = float(np.sqrt(abs(np.linalg.det(M[:, :2]))))
    rotation = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    full = M.copy()
    full[:, 2] /= s  # translation back to full-resolution pixels
    offset = _offset(full, W, H)
    centre = np.float32([W / 2, H / 2])
    # Where the reference centre ended up in the result (inverse map), minus where it should be.
    moved = cv2.invertAffineTransform(full)[:, :2] @ centre + cv2.invertAffineTransform(full)[:, 2] - centre
    reliable = (
        ecc_ok
        and after > before + 0.005
        and abs(scale - 1) < MAX_SCALE_DRIFT
        and abs(full[0, 2]) < MAX_SHIFT * W
        and abs(full[1, 2]) < MAX_SHIFT * H
        and abs(rotation) < 3
    )
    drift = Drift(
        # M maps result -> reference, so content in the result is 1/scale of the reference.
        scale=1 / scale, shift_x=float(moved[0]), shift_y=float(moved[1]), rotation=-rotation,
        max_offset=offset, geometry=max(0.0, after if reliable else before), geometry_before=max(0.0, before),
        inliers=inliers, reliable=reliable,
    )
    return drift, (full if reliable else None)


def align(reference: Image.Image, result: Image.Image) -> tuple[Image.Image, Drift]:
    """Warp the result back onto the reference framing when the drift is reliable and
    larger than a pixel. Returns the (possibly unchanged) result and the measurement."""
    result = result.convert("RGB")
    drift, M = measure(reference, result)
    if M is None or drift.max_offset < FIX_THRESHOLD_PX:
        return result, drift
    arr = np.asarray(result)
    # Edges uncovered by the correction are a few pixels wide; mirror the image into them.
    fixed = cv2.warpAffine(arr, M, result.size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    drift.fixed = True
    return Image.fromarray(fixed), drift
