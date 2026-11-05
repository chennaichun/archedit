"""Camera-move videos. The depth model is faked except in one test that uses
the real model when it has been downloaded (models/)."""

from __future__ import annotations

import io
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import ROOT, Settings
from app.main import create_app
from app.video import MODEL_FILE, MOTIONS, OnnxDepth, PointCloud, hole_mask, normalize, pick_keyframes, video_size

from .test_api import real_png


class FakeDepth:
    """Bottom of the picture near, top far, like a floor running into the room."""

    def __init__(self):
        self.calls = 0

    def estimate(self, image):
        self.calls += 1
        w, h = image.size
        return np.repeat(np.linspace(0, 1, h, dtype=np.float32)[:, None], w, axis=1)


@pytest.fixture
def client(tmp_path):
    settings = replace(Settings(), output_dir=tmp_path / "out", cf_account_id="a", cf_api_token="t")
    app = create_app(settings=settings, depth_estimator=FakeDepth(), agent_brain=None)
    with TestClient(app) as c:
        yield c


def test_video_endpoint_returns_a_browser_playable_mp4(client, tmp_path):
    res = client.post("/api/video", data={"motion": "pan", "seconds": "2"},
                      files={"image": ("in.png", real_png(), "image/png")})
    assert res.status_code == 200, res.text
    r = res.json()
    assert r["frames"] == 48
    mp4 = client.get(r["video_url"])
    assert mp4.headers["content-type"] == "video/mp4"
    assert mp4.content[4:8] == b"ftyp"
    assert b"avc1" in mp4.content[:4096]  # H.264, not the mp4v codec browsers can't play
    depth = Image.open(io.BytesIO(client.get(r["depth_url"]).content))
    assert depth.size == Image.open(io.BytesIO(real_png())).size


def test_video_validation(client):
    img = {"image": ("in.png", real_png(), "image/png")}
    assert client.post("/api/video", data={"motion": "spin"}, files=img).status_code == 400
    assert client.post("/api/video", data={"seconds": "30"}, files=img).status_code == 400
    assert client.post("/api/video", data={"seconds": "1"}, files=img).status_code == 400


def test_catalog_lists_the_camera_moves(client):
    keys = [m["key"] for m in client.get("/api/catalog").json()["motions"]]
    assert keys == ["push", "pan", "orbit"]


def floor_cloud(w=320, h=240):
    """Bottom rows near, top rows far."""
    xs = np.linspace(0, 255, w, dtype=np.uint8)
    img = np.dstack([np.tile(xs, (h, 1))] * 3)
    disp = np.repeat(np.linspace(0, 1, h, dtype=np.float32)[:, None], w, axis=1)
    return PointCloud(img, disp)


def test_still_camera_only_applies_the_small_overscan_zoom():
    c = floor_cloud()
    qx, _ = c.project((0, 0, 0))
    expect = c.cx + c.overscan * (c.px - c.cx)
    assert np.allclose(qx, expect, atol=1e-3)


def test_pan_moves_near_points_more_than_far_ones():
    c = floor_cloud()
    still, _ = c.project((0, 0, 0))
    moved, _ = c.project((0.03, 0, 0))
    shift = np.abs(moved - still)
    near = shift[c.Z == c.Z.min()].mean()
    far = shift[c.Z == c.Z.max()].mean()
    assert near > 3 * far > 0


def test_zbuffer_keeps_the_near_object_in_front():
    """A near white square in front of a far black wall must stay solid white when the
    camera pans, and the wall it uncovers shows up as holes (not smeared white)."""
    w, h = 200, 150
    img = np.zeros((h, w, 3), np.uint8)
    disp = np.zeros((h, w), np.float32)
    img[50:100, 80:120] = 255
    disp[50:100, 80:120] = 1.0
    c = PointCloud(img, disp)
    view = c.render((0.04, 0, 0), fill_cracks=False)
    # The square moved left as one piece and is still white inside.
    xs = np.nonzero(view.image[75, :, 0] > 200)[0]
    assert len(xs) >= 38 and xs.mean() < 100 - 3
    # Uncovered wall right of the square: a gap of holes, never smeared white.
    gap = view.holes[75, int(xs.max()) + 1: 126]
    assert gap.sum() >= 3
    assert (view.image[75, int(xs.max()) + 1: 126, 0] == 0).all()


def test_filled_keyframe_closes_the_holes_for_every_later_frame():
    w, h = 200, 150
    img = np.full((h, w, 3), 40, np.uint8)
    disp = np.zeros((h, w), np.float32)
    img[50:100, 80:120] = 255
    disp[50:100, 80:120] = 1.0
    c = PointCloud(img, disp)
    pose = (0.06, 0, 0)
    view = c.render(pose, fill_cracks=False)
    assert hole_mask(view) is not None
    filled = view.image.copy()
    filled[view.holes] = (0, 200, 0)  # "generated" content
    added = c.add_view(pose, view, filled)
    assert added == int(view.holes.sum())
    again = c.render(pose, fill_cracks=False)
    assert not again.holes[50:100].any()
    # The new points sit behind the square: it still covers them when the camera is still.
    still = c.render((0, 0, 0), fill_cracks=False)
    assert (still.image[60:90, 90:110, 0] > 200).all()


def test_keyframes_are_the_most_revealing_poses():
    w, h = 200, 150
    img = np.zeros((h, w, 3), np.uint8)
    disp = np.zeros((h, w), np.float32)
    disp[50:100, 80:120] = 1.0
    c = PointCloud(img, disp)
    keys = pick_keyframes(c, lambda t: (0.06 * np.sin(2 * np.pi * t), 0, 0))
    assert sorted(keys) == [0.25, 0.75]  # the two pan extremes
    assert pick_keyframes(c, lambda t: (0, 0, 0)) == []


def test_every_motion_starts_close_to_the_still_frame():
    for m in MOTIONS.values():
        tx, ty, tz = m.path(0.0)
        assert abs(tx) < 0.01 and abs(ty) < 0.01 and 0 <= tz < 0.1


def test_video_size_is_h264_friendly():
    assert video_size(768, 512) == (768, 512)
    w, h = video_size(4032, 3024)
    assert max(w, h) <= 1024 and w % 16 == 0 and h % 16 == 0


def test_normalize_ignores_outliers():
    d = np.zeros((200, 200), np.float32)
    d[100:] = 1.0
    d[0, 0] = 1000.0  # one wild pixel must not squash everything else towards 0
    n = normalize(d)
    assert n[150, 150] == 1.0 and n[50, 50] == 0.0


@pytest.mark.skipif(not (ROOT / "models" / MODEL_FILE).exists(), reason="depth model not downloaded")
def test_real_depth_model_sees_the_desk_nearer_than_the_window():
    disp = OnnxDepth(ROOT / "models").estimate(Image.open(io.BytesIO(real_png())))
    h, w = disp.shape
    desk = disp[int(h * 0.85):, : int(w * 0.15)].mean()
    window = disp[int(h * 0.3): int(h * 0.5), int(w * 0.2): int(w * 0.35)].mean()
    assert desk > window + 0.4


class FakeFlux:
    """Cloudflare stub returning a flat green image, to see the fill land in the video."""

    def __init__(self):
        self.calls = 0

    def handler(self, request):
        import base64
        import httpx
        self.calls += 1
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), (0, 200, 0)).save(buf, "PNG")
        return httpx.Response(200, json={"success": True, "result": {"image": base64.b64encode(buf.getvalue()).decode()}})


class StepDepth:
    """A near box in the middle of a far wall: panning reveals the wall behind it."""

    def estimate(self, image):
        w, h = image.size
        d = np.zeros((h, w), np.float32)
        d[h // 3: 2 * h // 3, w // 3: 2 * w // 3] = 1.0
        return d


def make_fill_client(tmp_path, **overrides):
    import httpx
    from app.cloudflare import CloudflareImageClient
    flux = FakeFlux()
    settings = replace(Settings(), output_dir=tmp_path / "out", cf_account_id="a", cf_api_token="t",
                       default_image_model="klein-4b", **overrides)
    client = CloudflareImageClient(settings, httpx.AsyncClient(transport=httpx.MockTransport(flux.handler)))
    app = create_app(settings=settings, image_client=client, depth_estimator=StepDepth(), agent_brain=None)
    return TestClient(app), flux


def test_fill_repaints_revealed_areas_with_flux(tmp_path):
    client, flux = make_fill_client(tmp_path)
    with client:
        r = client.post("/api/video", data={"motion": "pan", "seconds": "2", "fill": "true"},
                        files={"image": ("in.png", real_png(), "image/png")})
        assert r.status_code == 200, r.text
        body = r.json()
    assert body["fills"] == flux.calls == 2  # one per pan extreme
    assert client.get(body["video_url"]).content[4:8] == b"ftyp"


def test_fill_reserves_the_allowance_up_front(tmp_path):
    client, flux = make_fill_client(tmp_path, daily_limit_per_visitor=1)
    with client:
        r = client.post("/api/video", data={"motion": "orbit", "seconds": "2", "fill": "true"},
                        files={"image": ("in.png", real_png(), "image/png")})
        assert r.status_code == 429
        # Without fill it is free and still works.
        ok = client.post("/api/video", data={"motion": "orbit", "seconds": "2"},
                         files={"image": ("in.png", real_png(), "image/png")})
        assert ok.status_code == 200 and ok.json()["fills"] == 0
    assert flux.calls == 0


def test_depth_edges_snap_to_near_or_far():
    """Blurred edges from the depth model must not leave in-between "flying" depths."""
    from app.video import prepare_depth
    d = np.zeros((60, 60), np.float32)
    d[:, 30:] = 1.0
    d = cv2_blur(d)
    out = prepare_depth(d)
    edge = out[30, 24:36]
    assert ((edge < 0.1) | (edge > 0.9)).all()


def cv2_blur(d):
    import cv2
    return cv2.GaussianBlur(d, (0, 0), 3)


def test_background_fill_continues_the_far_surface():
    """A hole between a near white object (left) and a far brown floor (right) is
    filled with floor, not with the object's colour."""
    from app.video import View, background_fill
    h, w = 20, 30
    image = np.zeros((h, w, 3), np.uint8)
    depth = np.zeros((h, w), np.float32)
    image[:, :10] = 255
    depth[:, :10] = 1.0  # near object
    image[:, 20:] = (120, 80, 40)
    depth[:, 20:] = 4.0  # far floor
    holes = np.zeros((h, w), bool)
    holes[:, 10:20] = True
    Z, rgb = background_fill(View(image=image, depth=depth, holes=holes), 5.0)
    assert (Z[:, 10:20] == 4.0).all()
    assert (rgb[:, 10:20] == (120, 80, 40)).all()
