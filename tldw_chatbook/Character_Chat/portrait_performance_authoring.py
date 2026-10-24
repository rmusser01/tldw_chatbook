"""Disposable, bounded six-pose draft without persistence or provider authority."""

from __future__ import annotations

import hashlib

from .portrait_performance import (
    FRAME_ROLES,
    MAX_FRAME_BYTES,
    MAX_FRAME_PIXELS,
    MAX_SET_BYTES,
    FrameRole,
    PerformanceFrame,
    PerformanceSource,
)


class PortraitPerformanceDraft:
    """Stage normalized stills until an explicitly admitted Apply or discard."""

    def __init__(
        self, source: PerformanceSource, source_bytes: bytes, *, expected_revision: int
    ) -> None:
        self.source = source
        self.source_bytes = source_bytes
        self.expected_revision = expected_revision
        self._frames: dict[FrameRole, PerformanceFrame] = {}
        self._cancelled = False

    @property
    def staged_roles(self) -> tuple[FrameRole, ...]:
        """Return populated roles in the stable preview order."""
        return tuple(role for role in FRAME_ROLES if role in self._frames)

    def stage(self, frame: PerformanceFrame) -> None:
        """Retain an already normalized frame, preserving the old slot on error."""
        if self._cancelled or frame.role not in FRAME_ROLES:
            raise ValueError("portrait_performance_draft_unavailable")
        if (
            not frame.png
            or len(frame.png) > MAX_FRAME_BYTES
            or min(frame.width, frame.height) < 1
            or frame.width * frame.height > MAX_FRAME_PIXELS
            or hashlib.sha256(frame.png).hexdigest() != frame.sha256
        ):
            raise ValueError("portrait_performance_frame_invalid")
        others = [item for role, item in self._frames.items() if role != frame.role]
        if any(
            (item.width, item.height) != (frame.width, frame.height) for item in others
        ):
            raise ValueError("portrait_performance_geometry_mismatch")
        if sum(len(item.png) for item in others) + len(frame.png) > MAX_SET_BYTES:
            raise ValueError("portrait_performance_set_bytes_exceeded")
        self._frames[frame.role] = frame

    def frame(self, role: FrameRole) -> PerformanceFrame | None:
        """Return one staged pose for incomplete-set review."""
        return self._frames.get(role)

    def remove(self, role: FrameRole) -> None:
        """Clear one staged slot without changing persisted artwork."""
        self._frames.pop(role, None)

    def frames(self) -> tuple[PerformanceFrame, ...]:
        """Return the complete ordered set, or refuse publication."""
        if self._cancelled or self.staged_roles != FRAME_ROLES:
            raise ValueError("portrait_performance_requires_six_roles")
        return tuple(self._frames[role] for role in FRAME_ROLES)

    def cancel(self) -> None:
        """Release source and pose bytes; this draft cannot be reused."""
        self._cancelled = True
        self._frames.clear()
        self.source_bytes = b""
