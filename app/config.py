"""Settings read from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv is optional
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name, "").strip()
    return value or default


@dataclass(frozen=True)
class Settings:
    # Cloudflare Workers AI (image generation; free daily allocation)
    cf_account_id: str = field(default_factory=lambda: _env("CLOUDFLARE_ACCOUNT_ID"))
    cf_api_token: str = field(default_factory=lambda: _env("CLOUDFLARE_API_TOKEN"))
    default_image_model: str = field(default_factory=lambda: _env("IMAGE_MODEL", "klein-4b"))

    # Claude (optional prompt writer)
    claude_model: str = field(default_factory=lambda: _env("CLAUDE_MODEL", "claude-opus-5-5"))
    claude_effort: str = field(default_factory=lambda: _env("CLAUDE_EFFORT", "medium"))

    # Server
    output_dir: Path = field(default_factory=lambda: Path(_env("OUTPUT_DIR", str(ROOT / "outputs"))))
    static_dir: Path = ROOT / "static"

    @property
    def cloudflare_ready(self) -> bool:
        return bool(self.cf_account_id and self.cf_api_token)


def get_settings() -> Settings:
    return Settings()
