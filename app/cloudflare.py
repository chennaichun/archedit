"""FLUX.2 image generation through the Cloudflare Workers AI REST API.

Docs: https://developers.cloudflare.com/workers-ai/models/flux-2-klein-4b/
The models take multipart form data; reference images go in
input_image_0 .. input_image_3. The response is JSON with a base64 image.
"""

from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass, field
from typing import Optional

import httpx

from .catalog import MODELS, ImageModel
from .config import Settings

API_BASE = "https://api.cloudflare.com/client/v4/accounts"


@dataclass
class ImageRequest:
    prompt: str
    model: ImageModel
    width: int = 1024
    height: int = 1024
    references: list[bytes] = field(default_factory=list)  # PNG bytes, each under 512px
    seed: Optional[int] = None
    guidance: Optional[float] = None
    steps: Optional[int] = None  # FLUX.2 [dev] only


class ImageAPIError(Exception):
    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.message = message
        self.status = status


class CloudflareImageClient:
    max_retries = 2
    backoff_base = 1.0  # seconds; waits 1s, then 2s

    def __init__(self, settings: Settings, http: Optional[httpx.AsyncClient] = None, sleep=asyncio.sleep):
        self.settings = settings
        self._http = http
        self._sleep = sleep

    @property
    def http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=15.0))
        return self._http

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()

    def _require_ready(self) -> None:
        if not self.settings.cloudflare_ready:
            raise ImageAPIError(
                "Cloudflare is not set up. Add CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN to .env.", 503
            )

    async def _run(self, model_id: str, **body) -> dict:
        """POST to a Workers AI model (body as `files=` or `json=`) and return the JSON payload."""
        url = f"{API_BASE}/{self.settings.cf_account_id}/ai/run/{model_id}"
        headers = {"Authorization": f"Bearer {self.settings.cf_api_token}"}

        # Retry transient failures (network errors, timeouts, 5xx) with exponential
        # backoff. 4xx errors (bad token, quota used up, bad input) are not retried:
        # repeating them would fail the same way.
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            last = attempt == attempts - 1
            try:
                resp = await self.http.post(url, headers=headers, **body)
            except httpx.TimeoutException:
                if last:
                    raise ImageAPIError("Cloudflare timed out. Try again.", 504)
            except httpx.HTTPError:
                if last:
                    raise ImageAPIError("Could not reach Cloudflare. Check your internet connection.", 502)
            else:
                if resp.status_code < 500 or last:
                    break
            await self._sleep(self.backoff_base * 2**attempt)

        payload = _json(resp)
        if resp.status_code >= 400 or payload.get("success") is False:
            raise _api_error(resp.status_code, payload)
        return payload

    async def chat_json(self, model_id: str, messages: list[dict], schema: dict, max_tokens: int = 1200) -> dict:
        """Run a Workers AI chat model in JSON mode and return the parsed object."""
        self._require_ready()
        payload = await self._run(
            model_id,
            json={
                "messages": messages,
                "response_format": {"type": "json_schema", "json_schema": schema},
                "max_tokens": max_tokens,
            },
        )
        result = payload.get("result") or {}
        answer = result.get("response") if isinstance(result, dict) else None
        if answer is None:  # OpenAI-style shape
            try:
                answer = result["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                answer = None
        if isinstance(answer, str):
            try:
                answer = json.loads(answer)
            except ValueError:
                answer = None
        if not isinstance(answer, dict):
            raise ImageAPIError("The Cloudflare language model returned an unexpected format. Try again.", 502)
        return answer

    async def generate(self, req: ImageRequest) -> bytes:
        self._require_ready()
        if len(req.references) > 4:
            raise ImageAPIError("Use at most 4 reference images.", 400)

        data: dict[str, str] = {"prompt": req.prompt, "width": str(req.width), "height": str(req.height)}
        if req.seed is not None and req.seed >= 0:
            data["seed"] = str(req.seed)
        if req.guidance is not None:
            data["guidance"] = str(req.guidance)
        if req.steps is not None and req.model.adjustable_steps:
            data["steps"] = str(req.steps)
        # Every field goes in `files` so the body is multipart even for prompt-only
        # requests: (None, value) is a plain form field, (filename, bytes, type) a file.
        parts: list[tuple[str, tuple]] = [(k, (None, v)) for k, v in data.items()]
        parts += [
            (f"input_image_{i}", (f"image_{i}.png", png, "image/png")) for i, png in enumerate(req.references)
        ]

        payload = await self._run(req.model.model_id, files=parts)

        result = payload.get("result") or {}
        image_b64 = result.get("image") if isinstance(result, dict) else None
        if not image_b64:
            raise ImageAPIError("Cloudflare returned no image.", 502)
        try:
            return base64.b64decode(image_b64)
        except ValueError:
            raise ImageAPIError("Could not decode the image Cloudflare returned.", 502)


def _json(resp: httpx.Response) -> dict:
    try:
        body = resp.json()
        return body if isinstance(body, dict) else {}
    except ValueError:
        return {}


def _api_error(status: int, payload: dict) -> ImageAPIError:
    errors = payload.get("errors") or []
    detail = "; ".join(str(e.get("message", "")) for e in errors if isinstance(e, dict)).strip()
    if status in (401, 403):
        return ImageAPIError("The Cloudflare API token is invalid or lacks Workers AI permission.", 401)
    if status == 429:
        return ImageAPIError("Today's free allowance is used up, or requests are too frequent. It resets daily at 00:00 UTC.", 429)
    if status == 404:
        return ImageAPIError("Model or account not found. Check CLOUDFLARE_ACCOUNT_ID.", 404)
    if status == 400:
        return ImageAPIError(f"Cloudflare rejected the request: {detail or 'invalid parameters'}", 400)
    return ImageAPIError(f"Cloudflare error ({status}): {detail or 'try again later'}", 502)


def get_model(key: str) -> ImageModel:
    if key not in MODELS:
        raise ImageAPIError(f"Unknown model: {key}", 400)
    return MODELS[key]
