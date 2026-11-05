"""Chat editing agent: a LangGraph graph that turns one message ("brick wall,
then blue hour, then declutter") into several rounds of image edits.

    plan ──► edit ──► review ──► accept ──► edit (next step) … ──► finish
                ▲        │
                └────────┘ retry once with a better prompt

Each step is one whole-image FLUX.2 edit of the latest image. After every
edit a vision model compares before and after; a missed edit is retried once
with a rewritten prompt, then accepted anyway so a batch never loops.

The "brain" (planner + reviewer) is Claude when an Anthropic key is set and
otherwise a free vision model on Cloudflare Workers AI. Conversation memory
lives in a LangGraph checkpointer per chat thread, so a follow-up message
("now make it warmer") continues from the latest image and knows what came
before.
"""

from __future__ import annotations

import base64
import io
import operator
from collections import OrderedDict
from pathlib import Path
from typing import Annotated, Awaitable, Callable, Optional, Protocol, TypedDict

import anthropic
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field, ValidationError

from .catalog import TOOLS
from .cloudflare import CloudflareImageClient, ImageAPIError
from .config import Settings
from .llm import LLMError, claude_json, image_block
from .preprocess import limit_size, plan_local_edit

HISTORY_TURNS = 12  # chat messages passed back to the planner


# --------------------------------------------------------------------------
# What the brain returns
# --------------------------------------------------------------------------

class Step(BaseModel):
    title: str = Field(max_length=120)
    prompt: str = Field(max_length=2000)


class Plan(BaseModel):
    reply: str
    steps: list[Step]


class Review(BaseModel):
    ok: bool
    reason: str
    retry_prompt: str = ""


class ReviewChecklist(BaseModel):
    """What the reviewer answers. Three plain yes/no questions are judged more
    reliably by small vision models than one overall "is it ok"."""

    change_applied: bool
    natural_colors: bool
    same_scene: bool
    reason: str
    retry_prompt: str = ""

    def verdict(self) -> Review:
        ok = self.change_applied and self.natural_colors and self.same_scene
        return Review(ok=ok, reason=self.reason, retry_prompt="" if ok else self.retry_prompt)


class AgentStop(Exception):
    """Ends the run early with a message for the user (daily limit reached)."""


PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"title": {"type": "string"}, "prompt": {"type": "string"}},
                "required": ["title", "prompt"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["reply", "steps"],
    "additionalProperties": False,
}

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "change_applied": {"type": "boolean"},
        "natural_colors": {"type": "boolean"},
        "same_scene": {"type": "boolean"},
        "reason": {"type": "string"},
        "retry_prompt": {"type": "string"},
    },
    "required": ["change_applied", "natural_colors", "same_scene", "reason", "retry_prompt"],
    "additionalProperties": False,
}


def _example_prompts() -> str:
    """Presets that worked in testing, as models of a good step prompt. Short prompts
    like "Change the lighting to blue hour" made FLUX.2 tint the whole image blue."""
    picks = [
        ("materials", "Exposed brick", "wall behind the bed"),
        ("lighting", "Blue hour", ""),
        ("edit", "Declutter", ""),
        ("style", "Japandi", ""),
    ]
    lines = []
    for tool, label, target in picks:
        preset = next(p for p in TOOLS[tool].presets if p.label == label)
        lines.append(f"- {label}: {preset.prompt.replace('{target}', target)}")
    return "\n".join(lines)


AREA_RULES = """
BRUSHED AREA: the user brushed one area of the image; in the attached image it is tinted red with a red outline. Every step changes ONLY that area: FLUX.2 receives a plain crop around it (without the red tint) and the result is pasted back into the brushed shape, so everything outside stays pixel-identical.
- First identify what the brushed area is (for example the wall behind the bed, the floor, a plant, the sofa) and name it in every prompt.
- prompt: English, 15-40 words describing only what that thing should become, matching the photo's lighting and perspective. Do not mention the red tint, the crop, or the rest of the room.
- Whole-scene changes (lighting, time of day, overall style) cannot be done inside a brushed area: return no step for them and say in reply that the user should switch the chat to Whole image for that.
"""


def plan_system(max_steps: int, area: bool = False) -> str:
    return f"""You are the planning agent of ArchEdit, an image editing tool for architecture and interiors.
The user chats with you in any language about the attached image (the latest version of their photo or render) and asks for changes.
Split the request into at most {max_steps} edit steps, in a sensible order. Every step is ONE edit by FLUX.2, an instruction-following image model that receives the current image as image 0 and returns the edited image; the next step edits that result.
{AREA_RULES if area else ""}
Rules for each step:
- title: 2-6 words in the user's language, e.g. "牆面改成紅磚" or "Blue hour lighting".
- prompt: English, 25-70 words, one change only. Name the target explicitly ("the wall behind the bed", "the wooden floor"), never a vague "the surface" or "the object". Say what to keep: "Keep the architecture, furniture, layout and camera angle of image 0 exactly the same." When removing something, describe what should be visible instead. FLUX.2 has no negative prompt, so phrase exclusions positively.
- Use precise architectural vocabulary for materials (board-formed concrete, lime plaster, white oak, terrazzo), light (golden hour, warm 2700K downlights) and style.
- Put whole-scene changes (lighting, time of day, style) before small object changes.
- Lighting and time of day: describe the light outside the windows and the interior lights separately and keep natural colors, e.g. "blue hour dusk outside, interior lamps on with a warm glow". Never just "make it blue hour", which tints the whole picture.

Good step prompts (follow their length and level of detail):
{_example_prompts()}

reply: one or two short sentences in the user's language saying what you will do.
If the message needs no edit (a question, thanks, or something you cannot do with whole-image edits), return no steps and answer in reply. You cannot undo; tell the user to use the Undo button in Rounds instead.
Keep the content suitable for a general audience."""


REVIEW_SYSTEM = """You check the result of one image edit in an architecture image editor.
The attached picture shows BEFORE on the left and AFTER on the right.
Answer each question about AFTER on its own:
- change_applied: is the requested change clearly visible on the right thing? Be lenient about details and style.
- natural_colors: do walls, furniture and fabrics keep believable, natural colors? false when the whole picture looks tinted or washed in one color (everything blue, orange or green), even if the instruction mentions that color for the light or the sky.
- same_scene: is it still the same room or building, with the same layout and camera?
reason: one short sentence in the language given below; when something is false, say what is wrong.
retry_prompt: when any answer is false, an improved English instruction for the same change that fixes the problem (name the target explicitly, describe the desired result concretely); otherwise an empty string."""


class Brain(Protocol):
    name: str

    async def plan(self, history: list[dict], message: str, image_png: bytes, max_steps: int, area: bool = False) -> Plan: ...

    async def review(self, step: Step, prompt: str, pair_jpeg: bytes, language_hint: str) -> Review: ...


def _history_text(history: list[dict]) -> str:
    if not history:
        return ""
    lines = [f"{m['role']}: {m['content']}" for m in history[-HISTORY_TURNS:]]
    return "Conversation so far:\n" + "\n".join(lines) + "\n\n"


def _plan_text(history: list[dict], message: str) -> str:
    return _history_text(history) + f"New message from the user: {message}\nThe attached image is the current version."


def _review_text(step: Step, prompt: str, language_hint: str) -> str:
    return (
        f"Step: {step.title}\nInstruction sent to the image model: {prompt}\n"
        "(When only a brushed area was edited, the picture is a close-up around that area.)\n"
        f"Write reason in the same language as this text: {language_hint}"
    )


def _parse(model: type[BaseModel], data: dict, who: str):
    try:
        return model.model_validate(data)
    except ValidationError:
        raise LLMError(f"{who} returned an unexpected format. Try again.", 502)


class ClaudeBrain:
    name = "claude"

    def __init__(self, settings: Settings, client: Optional[anthropic.AsyncAnthropic] = None):
        self.settings = settings
        self._client = client

    @property
    def client(self) -> anthropic.AsyncAnthropic:
        if self._client is None:
            self._client = anthropic.AsyncAnthropic()
        return self._client

    async def plan(self, history, message, image_png, max_steps, area=False):
        content = [image_block(image_png), {"type": "text", "text": _plan_text(history, message)}]
        data = await claude_json(self.client, self.settings, plan_system(max_steps, area), content, PLAN_SCHEMA)
        return _parse(Plan, data, "Claude")

    async def review(self, step, prompt, pair_jpeg, language_hint):
        content = [image_block(pair_jpeg, "image/jpeg"), {"type": "text", "text": _review_text(step, prompt, language_hint)}]
        data = await claude_json(self.client, self.settings, REVIEW_SYSTEM, content, REVIEW_SCHEMA)
        return _parse(ReviewChecklist, data, "Claude").verdict()


class CloudflareBrain:
    """Free: Llama 4 Scout (vision + JSON mode) on Workers AI, about 20 Neurons a call."""

    name = "cloudflare"

    def __init__(self, settings: Settings, client: CloudflareImageClient):
        self.settings = settings
        self.client = client

    @staticmethod
    def _image_part(jpeg: bytes) -> dict:
        return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}}

    async def _ask(self, system: str, text: str, jpeg: bytes, schema: dict) -> dict:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [{"type": "text", "text": text}, self._image_part(jpeg)]},
        ]
        return await self.client.chat_json(self.settings.agent_llm_model, messages, schema)

    async def plan(self, history, message, image_png, max_steps, area=False):
        data = await self._ask(plan_system(max_steps, area), _plan_text(history, message), to_jpeg(image_png), PLAN_SCHEMA)
        return _parse(Plan, data, "The Cloudflare language model")

    async def review(self, step, prompt, pair_jpeg, language_hint):
        data = await self._ask(REVIEW_SYSTEM, _review_text(step, prompt, language_hint), pair_jpeg, REVIEW_SCHEMA)
        return _parse(ReviewChecklist, data, "The Cloudflare language model").verdict()


def make_brain(settings: Settings, image_client: CloudflareImageClient) -> Optional[Brain]:
    """Claude when configured (or forced with AGENT_BRAIN=claude), else the free Cloudflare model."""
    choice = settings.agent_brain
    if choice == "claude" or (choice == "auto" and settings.claude_ready):
        return ClaudeBrain(settings)
    if choice in ("auto", "cloudflare") and settings.cloudflare_ready:
        return CloudflareBrain(settings, image_client)
    return None


# --------------------------------------------------------------------------
# Images for the brain
# --------------------------------------------------------------------------

def to_jpeg(png: bytes, max_side: int = 768) -> bytes:
    img = Image.open(io.BytesIO(png)).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def area_overlay(image: Image.Image, mask: Image.Image) -> bytes:
    """The image with the brushed area tinted and outlined in red, for the planner."""
    image = limit_size(image)
    m = mask.convert("L").resize(image.size, Image.NEAREST).point(lambda v: 255 if v > 127 else 0)
    red = Image.new("RGB", image.size, (255, 40, 30))
    out = Image.composite(Image.blend(image, red, 0.45), image, m)
    bbox = m.getbbox()
    if bbox:
        ImageDraw.Draw(out).rectangle(bbox, outline=(255, 40, 30), width=max(3, image.width // 200))
    buf = io.BytesIO()
    out.save(buf, "PNG")
    return buf.getvalue()


def area_crops(before: Image.Image, after: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Before and after cut to the crop the image model saw, so the check looks at the edit up close."""
    box = plan_local_edit(before, mask).box
    return limit_size(before).crop(box), limit_size(after).resize(limit_size(before).size).crop(box)


def side_by_side(before: Image.Image, after: Image.Image, height: int = 448) -> bytes:
    """BEFORE | AFTER in one labelled JPEG, so one image input is enough for any model."""
    tiles = []
    for img in (before, after):
        img = img.convert("RGB")
        w = max(1, round(img.width * height / img.height))
        tiles.append(img.resize((min(w, height * 2), height)))
    gap = 12
    canvas = Image.new("RGB", (tiles[0].width + gap + tiles[1].width, height + 36), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=24)
    x = 0
    for label, tile in zip(("BEFORE", "AFTER"), tiles):
        draw.text((x + 6, 4), label, fill="black", font=font)
        canvas.paste(tile, (x, 36))
        x += tile.width + gap
    buf = io.BytesIO()
    canvas.save(buf, "JPEG", quality=85)
    return buf.getvalue()


# --------------------------------------------------------------------------
# The graph
# --------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    messages: Annotated[list[dict], operator.add]  # chat memory, kept across runs
    # Per run (reset by every new message):
    request: str
    image: str            # file name in out_dir of the image the next step edits
    steps: list[dict]
    index: int            # current step
    attempt: int          # 0 = first try, 1 = retry
    prompt: str           # prompt for the current try
    candidate: Optional[dict]  # last generated image (output meta)
    verdict: Optional[dict]
    rounds: list[dict]    # accepted rounds of this run
    plan_reply: str
    stopped: str
    notes: list[str]
    reply: str


# edit_fn(image_file, prompt, visitor, model_key, mask_file or None) -> output meta with "id" and "image_url"
EditFn = Callable[[str, str, str, str, Optional[str]], Awaitable[dict]]


def run_input(message: str, image_file: str) -> AgentState:
    """Input for one chat message: append it to memory and reset the per-run fields."""
    return {
        "messages": [{"role": "user", "content": message}],
        "request": message, "image": image_file, "steps": [], "index": 0, "attempt": 0,
        "prompt": "", "candidate": None, "verdict": None, "rounds": [], "plan_reply": "",
        "stopped": "", "notes": [], "reply": "",
    }


def build_graph(
    brain: Brain,
    edit_fn: EditFn,
    out_dir: Path,
    max_steps: int = 4,
    max_retries: int = 1,
    review: bool = True,
    checkpointer=None,
):
    def opts(config: RunnableConfig) -> dict:
        return config.get("configurable", {})

    def load(name: str) -> Image.Image:
        img = Image.open(out_dir / name)
        img.load()
        return img

    async def plan(state: AgentState, config: RunnableConfig) -> dict:
        mask = opts(config).get("mask")
        if mask:
            png = area_overlay(load(state["image"]), load(mask))
        else:
            png = (out_dir / state["image"]).read_bytes()
        history = state.get("messages", [])[:-1]  # the last one is this request
        try:
            result = await brain.plan(history, state["request"], png, max_steps, area=bool(mask))
        except (LLMError, ImageAPIError) as e:
            return {"steps": [], "stopped": e.message}
        steps = [s.model_dump() for s in result.steps if s.prompt.strip()][:max_steps]
        return {
            "plan_reply": result.reply.strip(),
            "steps": steps,
            "index": 0,
            "attempt": 0,
            "prompt": steps[0]["prompt"] if steps else "",
        }

    async def edit(state: AgentState, config: RunnableConfig) -> dict:
        o = opts(config)
        try:
            meta = await edit_fn(state["image"], state["prompt"], o.get("visitor", ""), o.get("model", ""), o.get("mask"))
        except (AgentStop, ImageAPIError, LLMError) as e:
            return {"candidate": None, "stopped": getattr(e, "message", None) or str(e)}
        return {"candidate": meta, "verdict": None}

    async def check(state: AgentState, config: RunnableConfig) -> dict:
        step = Step(**state["steps"][state["index"]])
        cand = state["candidate"]
        try:
            before, after = load(state["image"]), load(cand["id"] + ".png")
            mask = opts(config).get("mask")
            if mask:
                before, after = area_crops(before, after, load(mask))
            pair = side_by_side(before, after)
            verdict = await brain.review(step, state["prompt"], pair, state["request"])
        except (LLMError, ImageAPIError, OSError):
            # A failed check should not throw away a finished edit.
            return {"verdict": {"ok": True, "reason": "", "retry_prompt": "", "skipped": True}}
        v = verdict.model_dump()
        if not v["ok"] and state["attempt"] < max_retries:
            v["retry"] = True
            return {"verdict": v, "attempt": state["attempt"] + 1, "prompt": v["retry_prompt"].strip() or state["prompt"]}
        return {"verdict": v}

    async def accept(state: AgentState) -> dict:
        step = state["steps"][state["index"]]
        cand = state["candidate"]
        verdict = state.get("verdict") or {}
        rnd = {**cand, "title": step["title"], "prompt": state["prompt"], "step": state["index"],
               "attempts": state["attempt"] + 1, "check": verdict.get("reason", ""), "ok": verdict.get("ok", True)}
        nxt = state["index"] + 1
        return {
            "rounds": state.get("rounds", []) + [rnd],
            "image": cand["id"] + ".png",
            "index": nxt,
            "attempt": 0,
            "prompt": state["steps"][nxt]["prompt"] if nxt < len(state["steps"]) else "",
            "candidate": None,
            "verdict": None,
        }

    async def finish(state: AgentState) -> dict:
        # The rounds themselves show up in the UI; the reply is the plan in the
        # user's language plus anything that went wrong.
        notes = [f"⚠ {r['title']}: {r['check']}" for r in state.get("rounds", []) if not r["ok"] and r["check"]]
        if state.get("stopped"):
            notes.append(state["stopped"])
        reply = "\n".join(p for p in [state.get("plan_reply", ""), *notes] if p).strip() or "OK."
        return {"reply": reply, "notes": notes, "messages": [{"role": "assistant", "content": reply}]}

    def after_plan(state: AgentState) -> str:
        return "edit" if state.get("steps") and not state.get("stopped") else "finish"

    def after_edit(state: AgentState) -> str:
        if state.get("stopped") or not state.get("candidate"):
            return "finish"
        return "check" if review else "accept"

    def after_check(state: AgentState) -> str:
        return "edit" if (state.get("verdict") or {}).get("retry") else "accept"

    def after_accept(state: AgentState) -> str:
        return "edit" if state["index"] < len(state.get("steps", [])) else "finish"

    g = StateGraph(AgentState)
    g.add_node("plan", plan)
    g.add_node("edit", edit)
    g.add_node("check", check)
    g.add_node("accept", accept)
    g.add_node("finish", finish)
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", after_plan, ["edit", "finish"])
    g.add_conditional_edges("edit", after_edit, ["check", "accept", "finish"])
    g.add_conditional_edges("check", after_check, ["edit", "accept"])
    g.add_conditional_edges("accept", after_accept, ["edit", "finish"])
    g.add_edge("finish", END)
    return g.compile(checkpointer=checkpointer)


class ThreadStore:
    """In-memory chat memory with a cap: the oldest threads are forgotten first."""

    def __init__(self, max_threads: int = 200):
        self.saver = InMemorySaver()
        self.max_threads = max_threads
        self._seen: OrderedDict[str, None] = OrderedDict()

    def touch(self, thread_id: str) -> None:
        self._seen[thread_id] = None
        self._seen.move_to_end(thread_id)
        while len(self._seen) > self.max_threads:
            old, _ = self._seen.popitem(last=False)
            self.saver.delete_thread(old)
