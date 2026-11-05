"""Chat editing agent (LangGraph). The brain and Cloudflare are faked, so these
run offline; the graph, memory and streaming are real."""

from __future__ import annotations

import io
import json
from dataclasses import replace

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.agent import CloudflareBrain, Plan, Review, Step, side_by_side
from app.cloudflare import CloudflareImageClient
from app.config import Settings
from app.main import create_app

from .test_api import FakeCloudflare, parse_multipart, real_png


class FakeBrain:
    name = "fake"

    def __init__(self, steps=("Brick wall", "Blue hour"), fail_first_review=False):
        self.steps = steps
        self.fail_first_review = fail_first_review
        self.plans: list[tuple[list[dict], str]] = []
        self.reviews: list[str] = []
        self.areas: list[bool] = []
        self.plan_images: list[bytes] = []
        self.review_pairs: list[bytes] = []

    async def plan(self, history, message, image_png, max_steps, area=False):
        Image.open(io.BytesIO(image_png)).verify()  # the current image reaches the brain
        self.plans.append((list(history), message))
        self.areas.append(area)
        self.plan_images.append(image_png)
        return Plan(reply="好的", steps=[Step(title=t, prompt=f"prompt for {t}") for t in self.steps])

    async def review(self, step, prompt, pair_jpeg, language_hint):
        self.reviews.append(prompt)
        self.review_pairs.append(pair_jpeg)
        if self.fail_first_review and len(self.reviews) == 1:
            return Review(ok=False, reason="牆沒有變", retry_prompt="Change the wall behind the bed to red brick.")
        return Review(ok=True, reason="看起來不錯", retry_prompt="")


def make(tmp_path, brain, **overrides):
    settings = replace(
        Settings(), output_dir=tmp_path / "out", cf_account_id="acc", cf_api_token="tok",
        default_image_model="klein-4b", **overrides,
    )
    cf = FakeCloudflare()
    http = httpx.AsyncClient(transport=httpx.MockTransport(cf.handler))
    client = CloudflareImageClient(settings, http)
    app = create_app(settings=settings, image_client=client, agent_brain=brain)
    return TestClient(app), cf


def chat(client, message, thread="t1"):
    res = client.post(
        "/api/agent",
        data={"message": message, "thread": thread},
        files={"image": ("in.png", real_png(), "image/png")},
    )
    assert res.status_code == 200, res.text
    events = []
    for block in res.text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_plans_and_edits_round_by_round(tmp_path):
    brain = FakeBrain()
    client, cf = make(tmp_path, brain)
    with client:
        events = chat(client, "牆改紅磚，然後變成藍調時刻")
    kinds = [k for k, _ in events]
    assert kinds == ["plan", "attempt", "check", "round", "attempt", "check", "round", "done"]
    assert events[0][1]["steps"] == ["Brick wall", "Blue hour"]
    rounds = [d for k, d in events if k == "round"]
    assert [r["title"] for r in rounds] == ["Brick wall", "Blue hour"]
    assert all(r["image_url"].startswith("/outputs/") for r in rounds)
    assert len(cf.requests) == 2  # one FLUX.2 call per step
    # Step 2 edits the result of step 1, not the original upload.
    # The fake FLUX.2 returns a plain gray image, so step 2's reference must be gray.
    refs = [Image.open(io.BytesIO(parse_multipart(r)["input_image_0"])).convert("L") for r in cf.requests]
    assert refs[0].getextrema()[1] - refs[0].getextrema()[0] > 100  # the real photo
    assert refs[1].getextrema()[1] - refs[1].getextrema()[0] < 10   # step 1's gray result
    assert events[-1][1]["reply"] == "好的"


def test_missed_edit_is_retried_once_with_the_better_prompt(tmp_path):
    brain = FakeBrain(steps=("Brick wall",), fail_first_review=True)
    client, cf = make(tmp_path, brain)
    with client:
        events = chat(client, "牆改紅磚")
    kinds = [k for k, _ in events]
    assert kinds == ["plan", "attempt", "check", "attempt", "check", "round", "done"]
    assert events[2][1] == {"ok": False, "reason": "牆沒有變", "retry": True}
    rnd = next(d for k, d in events if k == "round")
    assert rnd["attempts"] == 2
    assert rnd["prompt"] == "Change the wall behind the bed to red brick."
    assert brain.reviews == ["prompt for Brick wall", "Change the wall behind the bed to red brick."]
    assert len(cf.requests) == 2


def test_never_retries_more_than_once(tmp_path):
    class AlwaysNo(FakeBrain):
        async def review(self, step, prompt, pair_jpeg, language_hint):
            self.reviews.append(prompt)
            return Review(ok=False, reason="還是不對", retry_prompt="try again")

    brain = AlwaysNo(steps=("Brick wall",))
    client, cf = make(tmp_path, brain)
    with client:
        events = chat(client, "牆改紅磚")
    rnd = next(d for k, d in events if k == "round")
    assert rnd["ok"] is False and rnd["attempts"] == 2
    assert len(cf.requests) == 2
    assert "還是不對" in events[-1][1]["reply"]  # the user is told the check failed


def test_review_can_be_turned_off(tmp_path):
    brain = FakeBrain()
    client, _ = make(tmp_path, brain, agent_review=False)
    with client:
        kinds = [k for k, _ in chat(client, "two things")]
    assert "check" not in kinds and kinds.count("round") == 2
    assert brain.reviews == []


def test_chat_memory_carries_over_per_thread(tmp_path):
    brain = FakeBrain(steps=("Brick wall",))
    client, _ = make(tmp_path, brain)
    with client:
        chat(client, "牆改紅磚")
        chat(client, "再暖一點")
        chat(client, "新的對話", thread="t2")
    history_2nd = brain.plans[1][0]
    assert [m["role"] for m in history_2nd] == ["user", "assistant"]
    assert history_2nd[0]["content"] == "牆改紅磚"
    assert brain.plans[2][0] == []  # another thread starts empty


def test_threads_are_private_per_visitor(tmp_path):
    brain = FakeBrain(steps=())
    client, _ = make(tmp_path, brain)
    with client:
        chat(client, "secret plan", thread="same")
        other = TestClient(client.app)
        with other:
            chat(other, "hello", thread="same")
    assert brain.plans[1][0] == []


def test_no_steps_just_replies(tmp_path):
    brain = FakeBrain(steps=())
    client, cf = make(tmp_path, brain)
    with client:
        events = chat(client, "這是什麼風格？")
    assert [k for k, _ in events] == ["plan", "done"]
    assert cf.requests == []


def test_daily_limit_stops_between_rounds(tmp_path):
    brain = FakeBrain(steps=("A", "B", "C"))
    client, cf = make(tmp_path, brain, daily_limit_per_visitor=1, agent_review=False)
    with client:
        events = chat(client, "three things")
        assert [k for k, _ in events].count("round") == 1
        assert "used your 1 edits" in events[-1][1]["reply"]
        assert len(cf.requests) == 1
        # Nothing left: refused before planning.
        res = client.post("/api/agent", data={"message": "more", "thread": "t1"},
                          files={"image": ("in.png", real_png(), "image/png")})
        assert res.status_code == 429


def test_agent_validation_and_status(tmp_path):
    client, _ = make(tmp_path, FakeBrain())
    with client:
        assert client.get("/api/status").json()["agent_brain"] == "fake"
        bad = client.post("/api/agent", data={"message": "x", "thread": "../etc"},
                          files={"image": ("in.png", real_png(), "image/png")})
        assert bad.status_code == 400
        empty = client.post("/api/agent", data={"message": "  ", "thread": "t"},
                            files={"image": ("in.png", real_png(), "image/png")})
        assert empty.status_code == 400


def test_agent_off_without_a_brain(tmp_path):
    client, _ = make(tmp_path, None, agent_brain="off")
    with client:
        assert client.get("/api/status").json()["agent_brain"] is None
        res = client.post("/api/agent", data={"message": "x", "thread": "t"},
                          files={"image": ("in.png", real_png(), "image/png")})
        assert res.status_code == 503


def test_auto_brain_falls_back_to_cloudflare_without_claude(tmp_path, monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(k, raising=False)
    client, _ = make(tmp_path, None)
    with client:
        assert client.get("/api/status").json()["agent_brain"] == "cloudflare"


@pytest.mark.anyio
@pytest.mark.parametrize("result", [
    {"response": {"reply": "ok", "steps": [{"title": "A", "prompt": "p"}]}},
    {"response": '{"reply": "ok", "steps": [{"title": "A", "prompt": "p"}]}'},
    {"choices": [{"message": {"content": '{"reply": "ok", "steps": [{"title": "A", "prompt": "p"}]}'}}]},
])
async def test_cloudflare_brain_reads_every_response_shape(result):
    seen = []

    def handler(request: httpx.Request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"success": True, "result": result})

    settings = replace(Settings(), cf_account_id="acc", cf_api_token="tok")
    client = CloudflareImageClient(settings, httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    plan = await CloudflareBrain(settings, client).plan([], "make it brick", real_png(), 4)
    assert plan.steps[0].title == "A"
    body = seen[0]
    assert body["response_format"]["type"] == "json_schema"
    image_part = body["messages"][1]["content"][1]
    assert image_part["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_side_by_side_is_one_labelled_image():
    a = Image.new("RGB", (800, 600), "white")
    b = Image.new("RGB", (800, 600), "gray")
    pair = Image.open(io.BytesIO(side_by_side(a, b)))
    assert pair.height == 448 + 36 and pair.width > 2 * 597


@pytest.mark.parametrize("answers, ok", [
    ((True, True, True), True),
    ((False, True, True), False),
    ((True, False, True), False),  # e.g. the whole picture tinted blue
    ((True, True, False), False),
])
def test_review_checklist_needs_every_answer(answers, ok):
    from app.agent import ReviewChecklist
    applied, colors, scene = answers
    v = ReviewChecklist(change_applied=applied, natural_colors=colors, same_scene=scene,
                        reason="r", retry_prompt="better").verdict()
    assert v.ok is ok
    assert v.retry_prompt == ("" if ok else "better")


def area_mask(box=(60, 60, 200, 180)) -> bytes:
    src = Image.open(io.BytesIO(real_png()))
    m = Image.new("L", src.size, 0)
    from PIL import ImageDraw
    ImageDraw.Draw(m).rectangle(box, fill=255)
    buf = io.BytesIO()
    m.save(buf, "PNG")
    return buf.getvalue()


def chat_area(client, message, mask_png, thread="t1"):
    res = client.post(
        "/api/agent",
        data={"message": message, "thread": thread},
        files={"image": ("in.png", real_png(), "image/png"), "mask": ("mask.png", mask_png, "image/png")},
    )
    assert res.status_code == 200, res.text
    return [
        (dict(l.split(": ", 1) for l in b.splitlines())["event"],
         json.loads(dict(l.split(": ", 1) for l in b.splitlines())["data"]))
        for b in res.text.strip().split("\n\n")
    ]


def test_brushed_area_changes_only_that_area(tmp_path):
    brain = FakeBrain(steps=("紅磚", "再深一點"))
    client, cf = make(tmp_path, brain)
    with client:
        events = chat_area(client, "這面牆改紅磚", area_mask())
    rounds = [d for k, d in events if k == "round"]
    assert len(rounds) == 2 and all(r["mode"] == "local" for r in rounds)
    assert brain.areas == [True]
    # The planner sees the brushed area tinted red.
    seen = Image.open(io.BytesIO(brain.plan_images[0])).convert("RGB")
    r, g, b = seen.getpixel((130, 120))
    assert r > g + 40 and r > b + 40
    # FLUX.2 gets a crop around the area, not the whole photo, and the local hint.
    parts = parse_multipart(cf.requests[0])
    ref = Image.open(io.BytesIO(parts["input_image_0"]))
    full = Image.open(io.BytesIO(real_png()))
    assert ref.width * ref.height < 0.6 * min(504 * 504, full.width * full.height)
    assert b"Keep everything else" in parts["prompt"]
    # Outside the brushed area every pixel is identical to the upload, after both rounds.
    out = Image.open(tmp_path / "out" / f"{rounds[-1]['id']}.png").convert("RGB")
    orig = full.convert("RGB")
    assert out.size == orig.size
    for xy in [(5, 5), (full.width - 5, full.height - 5), (300, 300), (5, full.height - 5)]:
        assert out.getpixel(xy) == orig.getpixel(xy)
    assert out.getpixel((130, 120)) != orig.getpixel((130, 120))
    # The check looks at a close-up crop, not the whole picture.
    pair = Image.open(io.BytesIO(brain.review_pairs[0]))
    assert pair.height == 448 + 36


def test_empty_mask_is_rejected(tmp_path):
    client, cf = make(tmp_path, FakeBrain())
    with client:
        blank = io.BytesIO()
        Image.new("L", (64, 64), 0).save(blank, "PNG")
        res = client.post("/api/agent", data={"message": "x", "thread": "t"},
                          files={"image": ("in.png", real_png(), "image/png"),
                                 "mask": ("m.png", blank.getvalue(), "image/png")})
    assert res.status_code == 400
    assert cf.requests == []


def test_area_rules_only_in_area_mode():
    from app.agent import plan_system
    assert "BRUSHED AREA" in plan_system(4, area=True)
    assert "BRUSHED AREA" not in plan_system(4)
