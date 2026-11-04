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
    # Public deployment guards. The Cloudflare free allowance is shared by every
    # visitor, so cap both each visitor and the whole server per UTC day (0 = no cap).
    daily_limit_per_visitor: int = field(default_factory=lambda: int(_env("DAILY_LIMIT_PER_VISITOR", "0")))
    daily_limit_total: int = field(default_factory=lambda: int(_env("DAILY_LIMIT_TOTAL", "0")))
    output_ttl_hours: float = field(default_factory=lambda: float(_env("OUTPUT_TTL_HOURS", "0")))
    secure_cookies: bool = field(default_factory=lambda: _env("SECURE_COOKIES", "0") == "1")

    @property
    def cloudflare_ready(self) -> bool:
        return bool(self.cf_account_id and self.cf_api_token)

    @property
    def claude_ready(self) -> bool:
        return any(os.environ.get(k) for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"))


def get_settings() -> Settings:
    return Settings()
