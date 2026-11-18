"""Disposable, bounded six-pose draft without persistence or provider authority."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .portrait_compositing import (
    COMPONENT_ROLES,
    ComponentRole,
    PortraitCompositionRecipe,
    validate_recipe_geometry,
)
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


@dataclass(frozen=True, slots=True)
class PortraitCompositionCandidate:
    """One disposable immutable source-bound snapshot; never logs image bytes."""

    recipe: PortraitCompositionRecipe
    regions_revision: int
    components: tuple[tuple[ComponentRole, bytes], ...] = field(repr=False)
    frames: tuple[PerformanceFrame, ...] = field(repr=False)


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
        self._composition: PortraitCompositionCandidate | None = None

    @property
    def staged_roles(self) -> tuple[FrameRole, ...]:
        """Return populated roles in the stable preview order."""
        return tuple(role for role in FRAME_ROLES if role in self._frames)

    @property
    def composition(self) -> PortraitCompositionCandidate | None:
        """Return the currently admitted immutable candidate, if composed."""
        return self._composition

    @property
    def can_apply(self) -> bool:
        """Distinguish manual six-role readiness from three-component readiness."""
        return (
            not self._cancelled
            and self.staged_roles == FRAME_ROLES
            and (
                self._composition is None
                or tuple(r for r, _ in self._composition.components) == COMPONENT_ROLES
            )
        )

    @staticmethod
    def _validated(
        frames: tuple[PerformanceFrame, ...],
    ) -> dict[FrameRole, PerformanceFrame]:
        result = {}
        for frame in frames:
            if (
                type(frame) is not PerformanceFrame
                or frame.role not in FRAME_ROLES
                or frame.role in result
                or type(frame.png) is not bytes
                or not frame.png
                or len(frame.png) > MAX_FRAME_BYTES
                or type(frame.width) is not int
                or type(frame.height) is not int
                or min(frame.width, frame.height) < 1
                or frame.width * frame.height > MAX_FRAME_PIXELS
                or hashlib.sha256(frame.png).hexdigest() != frame.sha256
            ):
                raise ValueError("portrait_performance_frame_invalid")
            result[frame.role] = frame
        if len({(f.width, f.height) for f in frames}) > 1:
            raise ValueError("portrait_performance_geometry_mismatch")
        if sum(len(f.png) for f in frames) > MAX_SET_BYTES:
            raise ValueError("portrait_performance_set_bytes_exceeded")
        return result

    def stage(self, frame: PerformanceFrame) -> None:
        """Stage a manual import; failure preserves frames and component lineage."""
        self.stage_group((frame,))

    def stage_group(self, frames: tuple[PerformanceFrame, ...]) -> None:
        """Atomically merge manual poses, dropping composition only on success."""
        if self._cancelled or type(frames) is not tuple or not frames:
            raise ValueError("portrait_performance_draft_unavailable")
        incoming = self._validated(frames)
        updated = self._validated(tuple((self._frames | incoming).values()))
        self._frames = updated
        self._composition = None

    def replace_composition(self, candidate: PortraitCompositionCandidate) -> None:
        """Replace every derived role and component together; never merge old slots."""
        if self._cancelled or type(candidate) is not PortraitCompositionCandidate:
            raise ValueError("portrait_performance_draft_unavailable")
        validate_recipe_geometry(candidate.recipe)
        if (
            type(candidate.regions_revision) is not int
            or candidate.regions_revision < 1
            or type(candidate.components) is not tuple
        ):
            raise ValueError("portrait_composition_candidate_invalid")
        roles = []
        for item in candidate.components:
            if (
                type(item) is not tuple
                or len(item) != 2
                or item[0] not in COMPONENT_ROLES
                or type(item[1]) is not bytes
                or not item[1]
                or len(item[1]) > MAX_FRAME_BYTES
            ):
                raise ValueError("portrait_composition_candidate_invalid")
            roles.append(item[0])
        if tuple(roles) != tuple(r for r in COMPONENT_ROLES if r in roles):
            raise ValueError("portrait_composition_candidate_invalid")
        if type(candidate.frames) is not tuple:
            raise ValueError("portrait_composition_candidate_invalid")
        updated = self._validated(candidate.frames)
        expected = {"rest"} | (set(roles) & {"small", "open"})
        if "eyelids" in roles:
            expected |= {"blink_rest"} | {"blink_" + r for r in roles if r != "eyelids"}
        if (
            set(updated) != expected
            or tuple(f.role for f in candidate.frames)
            != tuple(r for r in FRAME_ROLES if r in expected)
            or any(
                (f.width, f.height) != (candidate.recipe.width, candidate.recipe.height)
                for f in candidate.frames
            )
        ):
            raise ValueError("portrait_composition_candidate_invalid")
        self._frames = updated
        self._composition = candidate

    def frame(self, role: FrameRole) -> PerformanceFrame | None:
        """Return one staged pose for incomplete-set review."""
        return self._frames.get(role)

    def remove(self, role: FrameRole) -> None:
        """Clear one staged slot without changing persisted artwork."""
        if self._cancelled or role not in FRAME_ROLES:
            raise ValueError("portrait_performance_draft_unavailable")
        self._frames.pop(role, None)
        self._composition = None

    def frames(self) -> tuple[PerformanceFrame, ...]:
        """Return the complete ordered set, or refuse publication."""
        if not self.can_apply:
            raise ValueError("portrait_performance_requires_six_roles")
        return tuple(self._frames[role] for role in FRAME_ROLES)

    def cancel(self) -> None:
        """Release source and pose bytes; this draft cannot be reused."""
        self._cancelled = True
        self._frames.clear()
        self._composition = None
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
