"""Disposable, bounded six-pose draft without persistence or provider authority."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from .portrait_performance import (
    FRAME_ROLES,
    MAX_FRAME_BYTES,
    MAX_FRAME_PIXELS,
    MAX_SET_BYTES,
    FrameRole,
    PerformanceFrame,
    PerformanceSource,
)

if TYPE_CHECKING:
    from ..Image_Generation.capabilities import ResolvedReferenceImage


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


def performance_reference(data: bytes, source_sha256: str) -> ResolvedReferenceImage:
    """Extract one bounded metadata-free frame zero without changing source identity."""
    from io import BytesIO
    from math import isqrt

    from ..Chat.character_expression_playback import prepare_expression
    from ..Image_Generation.capabilities import ResolvedReferenceImage

    if hashlib.sha256(data).hexdigest() != source_sha256:
        raise ValueError("portrait_performance_source_changed")
    prepared = prepare_expression(data, (isqrt(MAX_FRAME_PIXELS),) * 2, animate=False)
    try:
        image = prepared.frames[0]
        image.info.clear()
        with BytesIO() as output:
            image.save(output, format="PNG")
            content = output.getvalue()
        if len(content) > MAX_FRAME_BYTES:
            raise ValueError("portrait_performance_reference_bytes_exceeded")
        return ResolvedReferenceImage(
            file_id=source_sha256,
            filename=None,
            mime_type="image/png",
            width=image.width,
            height=image.height,
            bytes_len=len(content),
            content=content,
            temp_path=None,
        )
    finally:
        prepared.close()
