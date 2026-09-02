"""Content-free compatibility counters for semantic trace rollout."""

from __future__ import annotations

from threading import Lock
from types import MappingProxyType
from typing import Literal, Mapping, TypeAlias

TraceCompatibilityPath: TypeAlias = Literal[
    "normalized_write",
    "normalized_read",
    "legacy_read",
    "fallback_read",
    "incomplete",
]
_PATHS: tuple[TraceCompatibilityPath, ...] = (
    "normalized_write",
    "normalized_read",
    "legacy_read",
    "fallback_read",
    "incomplete",
)


class TraceCompatibilityMetrics:
    """Thread-safe counts that never accept trace identities or content."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._counts = {path: 0 for path in _PATHS}

    def record(self, path: TraceCompatibilityPath, count: int = 1) -> None:
        """Increment one closed compatibility path by a positive count.

        Args:
            path: Registered content-free compatibility path.
            count: Positive number of observations to add.

        Raises:
            ValueError: If the path is unknown or count is not positive.
        """

        if path not in _PATHS:
            raise ValueError("trace_compatibility_path")
        if type(count) is not int or count <= 0:
            raise ValueError("trace_compatibility_count")
        with self._lock:
            self._counts[path] += count

    def snapshot(self) -> Mapping[TraceCompatibilityPath, int]:
        """Return an immutable content-free counter snapshot.

        Returns:
            Current count for every registered compatibility path.
        """

        with self._lock:
            return MappingProxyType(dict(self._counts))


__all__ = ["TraceCompatibilityMetrics", "TraceCompatibilityPath"]
