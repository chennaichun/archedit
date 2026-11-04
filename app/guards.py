"""Small in-process helpers for running the app publicly on a single server:
daily usage caps, a result cache, and cleanup of old output files.

All state lives in memory, so it resets on restart and is per process.
That fits one small instance; several instances would need Redis instead.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional


def utc_day(now: Optional[float] = None) -> str:
    return datetime.fromtimestamp(now if now is not None else time.time(), tz=timezone.utc).strftime("%Y-%m-%d")


class DailyLimiter:
    """Counts generations per visitor and in total, resetting at 00:00 UTC
    (when Cloudflare's free allowance resets). A limit of 0 means no cap."""

    def __init__(self, per_visitor: int, total: int, clock: Callable[[], float] = time.time):
        self.per_visitor = per_visitor
        self.total = total
        self._clock = clock
        self._lock = threading.Lock()
        self._day = ""
        self._counts: dict[str, int] = {}
        self._total = 0

    def _roll(self) -> None:
        today = utc_day(self._clock())
        if today != self._day:
            self._day, self._counts, self._total = today, {}, 0

    def check(self, visitor: str, needed: int = 1) -> Optional[str]:
        """Return an error message if this visitor may not run `needed` generations now, else None."""
        with self._lock:
            self._roll()
            if self.total and self._total + needed > self.total:
                if self._total >= self.total:
                    return "The demo has used today's free image allowance. It resets at 00:00 UTC."
                return f"Only {self.total - self._total} edits are left on the demo today. Try fewer variations."
            used = self._counts.get(visitor, 0)
            if self.per_visitor and used + needed > self.per_visitor:
                if used >= self.per_visitor:
                    return f"You've used your {self.per_visitor} edits for today. More are available after 00:00 UTC."
                return f"You have {self.per_visitor - used} edits left today. Try fewer variations."
        return None

    def record(self, visitor: str) -> None:
        with self._lock:
            self._roll()
            self._counts[visitor] = self._counts.get(visitor, 0) + 1
            self._total += 1

    def remaining(self, visitor: str) -> Optional[int]:
        with self._lock:
            self._roll()
            left = []
            if self.per_visitor:
                left.append(self.per_visitor - self._counts.get(visitor, 0))
            if self.total:
                left.append(self.total - self._total)
            return max(0, min(left)) if left else None


class ResultCache:
    """LRU cache of generated images keyed by everything that determines the
    output. Only used when the seed is fixed: the same request with the same
    seed returns the same image, so there is no reason to pay for it twice."""

    def __init__(self, max_items: int = 64):
        self.max_items = max_items
        self._items: OrderedDict[str, bytes] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0

    @staticmethod
    def key(*parts: object, images: tuple[bytes, ...] = ()) -> str:
        h = hashlib.sha256()
        for p in parts:
            h.update(repr(p).encode())
            h.update(b"\0")
        for img in images:
            h.update(hashlib.sha256(img).digest())
        return h.hexdigest()

    def get(self, key: str) -> Optional[bytes]:
        with self._lock:
            value = self._items.get(key)
            if value is not None:
                self._items.move_to_end(key)
                self.hits += 1
            return value

    def put(self, key: str, value: bytes) -> None:
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)


def delete_old_outputs(out_dir: Path, ttl_hours: float, now: Optional[float] = None) -> int:
    """Delete output files older than ttl_hours. Returns how many were removed."""
    if ttl_hours <= 0:
        return 0
    cutoff = (now if now is not None else time.time()) - ttl_hours * 3600
    removed = 0
    for path in out_dir.glob("*"):
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:
            continue
    return removed
