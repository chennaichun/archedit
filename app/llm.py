"""Claude turns a rough idea (any language, optional reference image) into a
FLUX.2 prompt and recommends a generation mode. Optional: the app works
without it, the user just writes the prompt by hand."""

from __future__ import annotations

import base64
import json
from typing import Optional, Protocol

import anthropic
from pydantic import BaseModel, ValidationError

from .catalog import MODES
from .config import Settings

MODE_KEYS = list(MODES)

SYSTEM_PROMPT = f"""You write prompts for FLUX.2, an image model that follows natural-language instructions and can take reference images, inside an architecture and interior design editing tool.
The user is an architect or designer. They describe what they want, in any language, and may attach the photo, render or sketch they are working from.
Use precise architectural vocabulary: name materials and finishes (board-formed concrete, lime plaster, white oak, terrazzo, travertine), lighting (golden hour, diffuse overcast, warm 2700K downlights) and camera (eye-level, 24mm, two-point perspective). Always keep the building's geometry, openings and camera unless the user asks to change them.

Modes:
- edit: the photo is sent to FLUX.2 as image 0. Write an edit instruction that keeps the architecture, layout and camera of image 0 and states exactly what changes (style, materials, colors, lighting, furniture added or removed).
- sketch: the user's own line drawing or sketch is sent as image 0. Write a full description of the finished photorealistic render and say it follows every line, wall and opening of image 0.
- edges: a black-on-white line drawing traced from the photo is sent as image 0. Write a full description of the finished scene and say it follows the lines and layout of image 0.
- text: no image is sent. Write a complete scene description.
- local: the user marked one region of the photo. FLUX.2 receives only a crop centered on that region, and the result is pasted back into the marked area. Write a short instruction (10-30 words) describing only what the object or area at the center should become, matching the photo's lighting and perspective. Do not describe the rest of the scene. Choose local only when the current mode is local, and keep it local in that case.

Write:
- prompt: English, 30-90 words of flowing natural language, not a tag list. Cover subject, setting, materials, lighting, camera or art style. FLUX.2 has no negative prompt, so phrase exclusions positively (for example "clean uncluttered surfaces").
- mode: one of {MODE_KEYS}. Pick edit when the user wants to restyle, relight or furnish a photo while keeping it recognizable, sketch when the attached image is a drawing to render, edges when they want a very different rendering on the same layout, text when there is no image.
- notes: one or two sentences in English explaining the choice and any tip.

Keep the content suitable for a general audience."""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "prompt": {"type": "string"},
        "mode": {"type": "string", "enum": MODE_KEYS},
        "notes": {"type": "string"},
    },
    "required": ["prompt", "mode", "notes"],
    "additionalProperties": False,
}


class PromptSuggestion(BaseModel):
    prompt: str
    mode: str
    notes: str


class LLMError(Exception):
    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.message = message
        self.status = status


class PromptWriter(Protocol):
    async def suggest(
        self, idea: str, image_png: Optional[bytes] = None, current_mode: Optional[str] = None
    ) -> PromptSuggestion: ...


class ClaudePromptWriter:
    def __init__(self, settings: Settings, client: Optional[anthropic.AsyncAnthropic] = None):
        self.settings = settings
        self._client = client

    @property
    def client(self) -> anthropic.AsyncAnthropic:
        if self._client is None:
            # Resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN or an `ant auth login` profile.
            self._client = anthropic.AsyncAnthropic()
        return self._client

    async def suggest(
        self, idea: str, image_png: Optional[bytes] = None, current_mode: Optional[str] = None
    ) -> PromptSuggestion:
        content: list[dict] = []
        if image_png:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.standard_b64encode(image_png).decode("ascii"),
                    },
                }
            )
        text = f"What I want: {idea.strip()}"
        if current_mode:
            text += f"\nCurrent mode: {current_mode} (change it if another fits better)"
        if image_png:
            text += "\nThe attached image is my original photo."
        content.append({"type": "text", "text": text})

        try:
            response = await self.client.beta.messages.create(
                model=self.settings.claude_model,
                max_tokens=16000,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                system=SYSTEM_PROMPT,
                output_config={
                    "effort": self.settings.claude_effort,
                    "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA},
                },
                messages=[{"role": "user", "content": content}],
            )
        except anthropic.AuthenticationError:
            raise LLMError("The Claude API key is missing or invalid. Set ANTHROPIC_API_KEY, or write the prompt yourself.", 401)
        except anthropic.PermissionDeniedError:
            raise LLMError("This API key cannot use this model.", 403)
        except anthropic.RateLimitError:
            raise LLMError("Too many Claude API requests. Wait a minute and try again.", 429)
        except anthropic.BadRequestError as e:
            raise LLMError(f"Claude API rejected the request: {e.message}", 400)
        except anthropic.APIStatusError as e:
            raise LLMError(f"Claude API error ({e.status_code}). Try again later.", 502)
        except anthropic.APIConnectionError:
            raise LLMError("Could not reach the Claude API. Check your internet connection.", 502)

        if response.stop_reason == "refusal":
            raise LLMError("Claude could not write a prompt for this description. Try rewording it.", 422)
        if response.stop_reason == "max_tokens":
            raise LLMError("Claude's reply was cut off. Try again.", 502)

        raw = "".join(block.text for block in response.content if block.type == "text")
        try:
            suggestion = PromptSuggestion.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValidationError):
            raise LLMError("Claude returned an unexpected format. Try again.", 502)
        if suggestion.mode not in MODE_KEYS:
            suggestion.mode = current_mode or "edit"
        return suggestion
