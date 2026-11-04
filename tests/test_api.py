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


def make_ctx(tmp_path, cf: FakeCloudflare, ready=True):
    settings = replace(
        Settings(),
        output_dir=tmp_path / "out",
        cf_account_id="acc123" if ready else "",
        cf_api_token="tok" if ready else "",
        default_image_model="klein-4b",
    )
    http = httpx.AsyncClient(transport=httpx.MockTransport(cf.handler))
    writer = FakeWriter()
    app = create_app(settings=settings, image_client=CloudflareImageClient(settings, http), prompt_writer=writer)
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
    assert [m["key"] for m in cat["modes"]] == ["edit", "edges", "text", "local"]


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


def test_prompt_uses_last_source_image(ctx):
    ctx.client.post("/api/preview", files={"image": ("a.png", real_png(), "image/png")}, data={"mode": "edit"})
    r = ctx.client.post("/api/prompt", json={"idea": "japanese bedroom", "mode": "edit", "use_image": True})
    assert r.status_code == 200
    assert r.json()["mode"] == "edges"
    assert ctx.writer.calls[-1] == ("japanese bedroom", True, "edit")


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
