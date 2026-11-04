"""API tests. Cloudflare is faked with httpx.MockTransport and Claude with a stub,
so they run offline without any keys."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace

import anthropic
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.cloudflare import CloudflareImageClient, ImageAPIError, ImageRequest
from app.catalog import MODELS
from app.config import Settings
from app.llm import ClaudePromptWriter, LLMError, PromptSuggestion
from app.main import create_app
from app.preprocess import fit_ratio


def sample_png(w=640, h=480, color="white") -> bytes:
    img = Image.new("RGB", (w, h), color)
    d = ImageDraw.Draw(img)
    d.rectangle((100, 100, 400, 300), outline="black", width=6)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def real_png(name="sample_input.png") -> bytes:
    """Photos from the original StableDiffusionWithGradio repo."""
    return (EXAMPLES / name).read_bytes()


def parse_multipart(request: httpx.Request) -> dict[str, bytes]:
    """Minimal multipart parser for assertions."""
    ctype = request.headers["content-type"]
    assert ctype.startswith("multipart/form-data")
    boundary = ctype.split("boundary=")[1].encode()
    parts = {}
    for chunk in request.content.split(b"--" + boundary):
        if b"\r\n\r\n" not in chunk:
            continue
        head, body = chunk.split(b"\r\n\r\n", 1)
        name = head.split(b'name="')[1].split(b'"')[0].decode()
        parts[name] = body.rstrip(b"\r\n")
    return parts


class FakeCloudflare:
    def __init__(self, status=200, payload=None):
        self.requests: list[httpx.Request] = []
        self.status = status
        out = base64.b64encode(sample_png(64, 64, "gray")).decode()
        self.payload = payload if payload is not None else {"success": True, "result": {"image": out}, "errors": []}

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(self.status, json=self.payload)


class FakeWriter:
    def __init__(self):
        self.calls = []

    async def suggest(self, idea, image_png=None, current_mode=None):
        self.calls.append((idea, image_png is not None, current_mode))
        return PromptSuggestion(prompt="a cozy room", mode="edges", notes="use line art")


def make_ctx(tmp_path, cf: FakeCloudflare, ready=True, **overrides):
    settings = replace(
        Settings(),
        output_dir=tmp_path / "out",
        cf_account_id="acc123" if ready else "",
        cf_api_token="tok" if ready else "",
        default_image_model="klein-4b",
        **overrides,
    )
    http = httpx.AsyncClient(transport=httpx.MockTransport(cf.handler))
    writer = FakeWriter()

    async def no_sleep(_seconds):
        return None

    client = CloudflareImageClient(settings, http, sleep=no_sleep)
    app = create_app(settings=settings, image_client=client, prompt_writer=writer)
    return SimpleNamespace(client=TestClient(app), cf=cf, writer=writer)


@pytest.fixture
def ctx(tmp_path):
    c = make_ctx(tmp_path, FakeCloudflare())
    with c.client:
        yield c


def test_index_status_catalog(ctx):
    assert ctx.client.get("/").status_code == 200
    assert ctx.client.get("/api/status").json()["cloudflare_ready"] is True
    cat = ctx.client.get("/api/catalog").json()
    assert {m["key"] for m in cat["models"]} == {"klein-4b", "klein-9b", "dev"}
    assert [m["key"] for m in cat["modes"]] == ["edit", "sketch", "edges", "text", "local"]
    assert [t["key"] for t in cat["tools"]] == ["edit", "sketch", "materials", "lighting", "style"]
    modes = {m["key"] for m in cat["modes"]}
    for tool in cat["tools"]:
        assert tool["mode"] in modes and tool["presets"]
        assert (EXAMPLES / tool["example"]).exists()


def test_preview_edges_is_line_art(ctx):
    r = ctx.client.post(
        "/api/preview",
        files={"image": ("a.png", real_png(), "image/png")},
        data={"mode": "edges", "size": "1216x832"},
    )
    assert r.status_code == 200, r.text
    img = np.array(Image.open(io.BytesIO(r.content)).convert("L"))
    assert img.shape == (832, 1216)
    assert img.min() == 0 and img.max() == 255  # dark lines on white


def test_generate_edit_sends_multipart_reference(ctx):
    r = ctx.client.post(
        "/api/generate",
        files={"image": ("a.png", real_png(), "image/png")},
        data={"prompt": "make it japanese", "mode": "edit", "model": "klein-9b", "size": "1024x1024", "seed": "7"},
    )
    assert r.status_code == 200, r.text
    meta = r.json()
    assert meta["seed"] == 7 and meta["model"] == "klein-9b"
    assert ctx.client.get(meta["image_url"]).status_code == 200
    assert ctx.client.get(meta["guide_url"]).status_code == 200

    req = ctx.cf.requests[-1]
    assert str(req.url).endswith("/accounts/acc123/ai/run/@cf/black-forest-labs/flux-2-klein-9b")
    assert req.headers["authorization"] == "Bearer tok"
    parts = parse_multipart(req)
    assert parts["prompt"] == b"make it japanese"
    assert parts["width"] == b"1024" and parts["seed"] == b"7"
    assert "steps" not in parts  # klein has fixed steps
    ref = Image.open(io.BytesIO(parts["input_image_0"]))
    assert max(ref.size) < 512  # Cloudflare input limit

    hist = ctx.client.get("/api/history").json()["items"]
    assert hist[0]["prompt"] == "make it japanese"


def test_generate_text_mode_is_multipart_without_images(ctx):
    r = ctx.client.post("/api/generate", data={"prompt": "a cat", "mode": "text", "model": "dev", "steps": "12"})
    assert r.status_code == 200, r.text
    parts = parse_multipart(ctx.cf.requests[-1])
    assert parts["prompt"] == b"a cat" and parts["steps"] == b"12"
    assert not any(k.startswith("input_image") for k in parts)


def test_style_reference_becomes_image_1(ctx):
    r = ctx.client.post(
        "/api/generate",
        files={"image": ("a.png", real_png(), "image/png"), "style_image": ("s.png", real_png("line.png"), "image/png")},
        data={"prompt": "restyle", "mode": "edit"},
    )
    assert r.status_code == 200, r.text
    parts = parse_multipart(ctx.cf.requests[-1])
    assert "input_image_1" in parts
    assert parts["prompt"].endswith(b"image 1.")


def test_generate_validation(ctx):
    assert ctx.client.post("/api/generate", data={"prompt": " ", "mode": "text"}).status_code == 400
    assert ctx.client.post("/api/generate", data={"prompt": "a", "mode": "x"}).status_code == 400
    assert ctx.client.post("/api/generate", data={"prompt": "a", "mode": "text", "model": "nope"}).status_code == 400
    # edit mode with no image uploaded yet
    assert ctx.client.post("/api/generate", data={"prompt": "a", "mode": "edit"}).status_code == 400
    assert ctx.client.post("/api/generate", data={"prompt": "a", "mode": "text", "size": "1x1"}).status_code == 400


@pytest.mark.parametrize("status,expected", [(401, 401), (429, 429), (400, 400), (500, 502)])
def test_cloudflare_errors_are_mapped(tmp_path, status, expected):
    cf = FakeCloudflare(status=status, payload={"success": False, "errors": [{"code": 1, "message": "boom"}]})
    c = make_ctx(tmp_path, cf)
    with c.client:
        r = c.client.post("/api/generate", data={"prompt": "a", "mode": "text"})
    assert r.status_code == expected
    assert r.json()["detail"]


def test_not_configured(tmp_path):
    c = make_ctx(tmp_path, FakeCloudflare(), ready=False)
    with c.client:
        r = c.client.post("/api/generate", data={"prompt": "a", "mode": "text"})
    assert r.status_code == 503
    assert not c.cf.requests


def test_prompt_sends_the_uploaded_image(ctx, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    r = ctx.client.post(
        "/api/prompt",
        files={"image": ("a.png", real_png(), "image/png")},
        data={"idea": "japanese bedroom", "mode": "edit"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["mode"] == "edges"
    assert ctx.writer.calls[-1] == ("japanese bedroom", True, "edit")


def test_prompt_without_claude_key(ctx, monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(k, raising=False)
    r = ctx.client.post("/api/prompt", data={"idea": "x"})
    assert r.status_code == 503
    assert not ctx.writer.calls


# ---- Claude prompt writer ------------------------------------------------------------

class FakeMessages:
    def __init__(self, response=None, exc=None):
        self.response, self.exc, self.kwargs = response, exc, None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        if self.exc:
            raise self.exc
        return self.response


def fake_client(messages):
    return SimpleNamespace(beta=SimpleNamespace(messages=messages))


def text_response(text, stop="end_turn"):
    return SimpleNamespace(stop_reason=stop, content=[SimpleNamespace(type="text", text=text)])


@pytest.mark.anyio
async def test_writer_builds_request_and_parses():
    msgs = FakeMessages(text_response('{"prompt":"p","mode":"text","notes":"x"}'))
    writer = ClaudePromptWriter(Settings(), client=fake_client(msgs))
    s = await writer.suggest("make the bed green", image_png=real_png(), current_mode="edit")
    assert s.mode == "text"
    kw = msgs.kwargs
    assert kw["model"] == "claude-opus-5-5"
    assert kw["fallbacks"] == "default" and kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert kw["output_config"]["format"]["type"] == "json_schema"
    content = kw["messages"][0]["content"]
    assert content[0]["type"] == "image" and content[-1]["type"] == "text"


@pytest.mark.anyio
async def test_writer_handles_refusal_bad_json_and_network():
    w = ClaudePromptWriter(Settings(), client=fake_client(FakeMessages(text_response("", stop="refusal"))))
    with pytest.raises(LLMError) as e:
        await w.suggest("x")
    assert e.value.status == 422
    w = ClaudePromptWriter(Settings(), client=fake_client(FakeMessages(text_response("not json"))))
    with pytest.raises(LLMError):
        await w.suggest("x")
    exc = anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com"))
    w = ClaudePromptWriter(Settings(), client=fake_client(FakeMessages(exc=exc)))
    with pytest.raises(LLMError) as e:
        await w.suggest("x")
    assert e.value.status == 502


@pytest.mark.anyio
async def test_client_rejects_too_many_references():
    settings = replace(Settings(), cf_account_id="a", cf_api_token="t")
    client = CloudflareImageClient(settings, httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200))))
    with pytest.raises(ImageAPIError):
        await client.generate(ImageRequest(prompt="x", model=MODELS["klein-4b"], references=[b"x"] * 5))


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ---- local (click-to-edit) mode ---------------------------------------------------------

def mask_png(w, h, box=None) -> bytes:
    m = Image.new("L", (w, h), 0)
    if box:
        ImageDraw.Draw(m).rectangle(box, fill=255)
    buf = io.BytesIO()
    m.save(buf, format="PNG")
    return buf.getvalue()


def test_local_edit_changes_only_masked_area(ctx):
    src_bytes = real_png()  # bedroom photo, 768x512
    r = ctx.client.post(
        "/api/generate",
        files={
            "image": ("a.png", src_bytes, "image/png"),
            "mask": ("m.png", mask_png(768, 512, (220, 300, 470, 420)), "image/png"),  # the bed
        },
        data={"prompt": "dark green velvet bedspread", "mode": "local", "model": "klein-9b"},
    )
    assert r.status_code == 200, r.text
    meta = r.json()
    assert meta["size"] == "768x512"  # keeps the photo's own size

    parts = parse_multipart(ctx.cf.requests[-1])
    assert parts["prompt"].startswith(b"dark green velvet bedspread")
    w, h = int(parts["width"]), int(parts["height"])
    assert w % 16 == 0 and h % 16 == 0 and max(w, h) == 1024
    ref = Image.open(io.BytesIO(parts["input_image_0"]))
    assert max(ref.size) < 512

    out = np.array(Image.open(io.BytesIO(ctx.client.get(meta["image_url"]).content)).convert("RGB")).astype(int)
    src = np.array(Image.open(io.BytesIO(src_bytes)).convert("RGB")).astype(int)
    assert np.abs(out[360, 345] - [128, 128, 128]).max() < 10  # middle of the bed -> gray from fake model
    assert (out[:250] == src[:250]).all()  # window, wall, cabinets: untouched
    assert (out[:, 560:] == src[:, 560:]).all()  # kitchen side: untouched


def test_local_edit_needs_a_mask(ctx):
    files = {"image": ("a.png", real_png(), "image/png")}
    assert ctx.client.post("/api/generate", files=files, data={"prompt": "x", "mode": "local"}).status_code == 400
    files["mask"] = ("m.png", mask_png(768, 512), "image/png")
    r = ctx.client.post("/api/generate", files=files, data={"prompt": "x", "mode": "local"})
    assert r.status_code == 400 and "Brush the area" in r.json()["detail"]


def test_plan_crop_stays_inside_image():
    from app.preprocess import plan_local_edit

    img = Image.new("RGB", (900, 600), "white")
    plan = plan_local_edit(img, Image.open(io.BytesIO(mask_png(900, 600, (850, 550, 899, 599)))))
    l, t, r_, b = plan.box
    assert 0 <= l < r_ <= 900 and 0 <= t < b <= 600
    assert r_ - l >= 256 and b - t >= 256


# ---- public deployment: visitors, limits, cache, retries ---------------------------------

def text_edit(client, **data):
    return client.post("/api/generate", data={"prompt": "a cat", "mode": "text", **data})


def test_history_is_per_visitor(tmp_path):
    c = make_ctx(tmp_path, FakeCloudflare())
    with c.client:
        assert text_edit(c.client, prompt="mine").status_code == 200
        assert [i["prompt"] for i in c.client.get("/api/history").json()["items"]] == ["mine"]
        assert "owner" not in c.client.get("/api/history").json()["items"][0]
        other = TestClient(c.client.app)  # a second browser: no cookie yet
        with other:
            assert other.get("/api/history").json()["items"] == []


def test_visitor_cookie_is_http_only(ctx):
    r = ctx.client.get("/api/status")
    cookie = r.headers["set-cookie"]
    assert "cs_visitor=" in cookie and "HttpOnly" in cookie
    assert "set-cookie" not in ctx.client.get("/api/status").headers  # reused afterwards


def test_daily_limit_per_visitor(tmp_path):
    c = make_ctx(tmp_path, FakeCloudflare(), daily_limit_per_visitor=2)
    with c.client:
        assert text_edit(c.client).json()["remaining_today"] == 1
        assert text_edit(c.client).json()["remaining_today"] == 0
        r = text_edit(c.client)
        assert r.status_code == 429 and "2 edits" in r.json()["detail"]
        assert len(c.cf.requests) == 2  # the refused request never reached Cloudflare


def test_daily_limit_total_across_visitors(tmp_path):
    c = make_ctx(tmp_path, FakeCloudflare(), daily_limit_total=1)
    with c.client:
        assert text_edit(c.client).status_code == 200
        with TestClient(c.client.app) as other:
            r = text_edit(other)
            assert r.status_code == 429 and "00:00 UTC" in r.json()["detail"]


def test_fixed_seed_requests_are_cached(ctx):
    first = text_edit(ctx.client, seed="7").json()
    second = text_edit(ctx.client, seed="7").json()
    assert len(ctx.cf.requests) == 1
    assert first["cached"] is False and second["cached"] is True
    text_edit(ctx.client, seed="8")
    text_edit(ctx.client)  # random seed: never served from cache
    assert len(ctx.cf.requests) == 3


class FlakyCloudflare(FakeCloudflare):
    def __init__(self, failures: int, fail_status: int = 503):
        super().__init__()
        self.failures, self.fail_status = failures, fail_status

    def handler(self, request):
        self.requests.append(request)
        if self.failures > 0:
            self.failures -= 1
            return httpx.Response(self.fail_status, json={"success": False, "errors": [{"message": "busy"}]})
        return httpx.Response(200, json=self.payload)


def test_retries_transient_server_errors(tmp_path):
    c = make_ctx(tmp_path, FlakyCloudflare(failures=2))
    with c.client:
        assert text_edit(c.client).status_code == 200
    assert len(c.cf.requests) == 3


def test_gives_up_after_max_retries(tmp_path):
    c = make_ctx(tmp_path, FlakyCloudflare(failures=10))
    with c.client:
        assert text_edit(c.client).status_code == 502
    assert len(c.cf.requests) == 3  # 1 try + 2 retries


def test_does_not_retry_client_errors(tmp_path):
    c = make_ctx(tmp_path, FlakyCloudflare(failures=10, fail_status=429))
    with c.client:
        assert text_edit(c.client).status_code == 429
    assert len(c.cf.requests) == 1


def test_healthz(ctx):
    assert ctx.client.get("/healthz").json() == {"ok": True}


def test_old_outputs_are_deleted(tmp_path):
    import os
    import time as _time

    from app.guards import delete_old_outputs

    old, new = tmp_path / "old.png", tmp_path / "new.png"
    old.write_bytes(b"x"); new.write_bytes(b"x")
    two_days_ago = _time.time() - 48 * 3600
    os.utime(old, (two_days_ago, two_days_ago))
    assert delete_old_outputs(tmp_path, ttl_hours=24) == 1
    assert not old.exists() and new.exists()


def test_daily_limiter_resets_at_utc_midnight():
    from app.guards import DailyLimiter

    now = [1_800_000_000.0]
    lim = DailyLimiter(per_visitor=1, total=0, clock=lambda: now[0])
    lim.record("a")
    assert lim.check("a")
    now[0] += 24 * 3600
    assert lim.check("a") is None


def test_sketch_mode_sends_the_drawing_as_is(ctx):
    """Sketch mode must not re-trace an existing line drawing (that would double every line)."""
    line = real_png("line.png")
    r = ctx.client.post(
        "/api/generate",
        files={"image": ("line.png", line, "image/png")},
        data={"prompt": "render it", "mode": "sketch", "size": "1216x832"},
    )
    assert r.status_code == 200, r.text
    guide = np.array(Image.open(io.BytesIO(ctx.client.get(r.json()["guide_url"]).content)).convert("L")).astype(int)
    expected = np.array(fit_ratio(Image.open(io.BytesIO(line)), 1216, 832).convert("L")).astype(int)
    assert np.abs(guide - expected).mean() < 1  # the drawing itself, not Canny edges of it
    assert "input_image_0" in parse_multipart(ctx.cf.requests[-1])


def test_presets_keep_the_geometry():
    from app.catalog import TOOLS

    for key in ("lighting", "style"):
        for preset in TOOLS[key].presets:
            assert "image 0" in preset.prompt and "camera angle" in preset.prompt


def test_material_presets_name_the_target():
    from app.catalog import TOOLS

    tool = TOOLS["materials"]
    assert "wall" in tool.targets
    for preset in tool.presets:
        assert "{target}" in preset.prompt


def test_image_edit_tool():
    from app.catalog import TOOLS

    tool = TOOLS["edit"]
    assert tool.brushable and tool.mode == "edit"
    by_label = {p.label: p for p in tool.presets}
    assert by_label["Remove object"].area and by_label["Repaint"].area
    assert "{target}" in by_label["Remove object"].prompt and "plant" in tool.targets
    assert not by_label["Declutter"].area


def test_remove_object_on_brushed_area(ctx):
    from app.catalog import TOOLS

    prompt = next(p.prompt for p in TOOLS["edit"].presets if p.label == "Remove object").replace("{target}", "plant")
    src = real_png()
    r = ctx.client.post(
        "/api/generate",
        files={"image": ("a.png", src, "image/png"), "mask": ("m.png", mask_png(768, 512, (20, 250, 150, 340)), "image/png")},
        data={"prompt": prompt, "mode": "local"},
    )
    assert r.status_code == 200, r.text
    out = np.array(Image.open(io.BytesIO(ctx.client.get(r.json()["image_url"]).content)).convert("RGB")).astype(int)
    orig = np.array(Image.open(io.BytesIO(src)).convert("RGB")).astype(int)
    assert (out[:, 300:] == orig[:, 300:]).all()  # right of the brushed plant: untouched
