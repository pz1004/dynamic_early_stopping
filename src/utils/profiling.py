"""
Lightweight profiling utilities for DES-kNN.
"""

import os
import time
from typing import Dict


def _env_flag(name: str) -> bool:
    value = os.getenv(name, "0").strip().lower()
    return value in {"1", "true", "yes", "y", "t"}


class _NullTimer:
    __slots__ = ()

    def __enter__(self) -> "_NullTimer":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


_NULL_TIMER = _NullTimer()


class _Timer:
    __slots__ = ("_profiler", "_key", "_start")

    def __init__(self, profiler: "Profiler", key: str) -> None:
        self._profiler = profiler
        self._key = key
        self._start = 0.0

    def __enter__(self) -> "_Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        elapsed = time.perf_counter() - self._start
        self._profiler._add_time(self._key, elapsed)
        return False


class Profiler:
    """
    Aggregates timing stats by category.

    Enable via DESKNN_PROFILE=1.
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._timings: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

    @classmethod
    def from_env(cls) -> "Profiler":
        return cls(_env_flag("DESKNN_PROFILE"))

    def time(self, key: str):
        if not self.enabled:
            return _NULL_TIMER
        return _Timer(self, key)

    def count(self, key: str, value: int = 1) -> None:
        if not self.enabled:
            return
        self._counts[key] = self._counts.get(key, 0) + value

    def _add_time(self, key: str, elapsed: float) -> None:
        self._timings[key] = self._timings.get(key, 0.0) + elapsed
        self._counts[key] = self._counts.get(key, 0) + 1

    def summary(self) -> Dict[str, Dict[str, float]]:
        return {
            key: {
                "count": self._counts[key],
                "total_s": self._timings[key],
            }
            for key in sorted(self._timings)
        }

    def format_summary(self) -> str:
        lines = ["DES-kNN profile summary:"]
        for key in sorted(self._timings):
            total = self._timings[key]
            count = self._counts.get(key, 0)
            lines.append(f"{key}: count={count} total_s={total:.6f}")
        return "\n".join(lines)
