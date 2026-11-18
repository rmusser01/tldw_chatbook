"""Pure validated memory-banner presentation for the native Console."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .console_chat_models import ConsoleChatMessage
    from .console_context_compaction import EffectiveMemoryResult


#: SP2 /rewind: render-derived (never a tree node) one-line banner shown above
#: the boundary message when "summarize up to here" is in effect.
CONSOLE_SUMMARY_BANNER_COPY = (
    "⤵ Earlier turns summarized for context — full history above"
)


@dataclass(frozen=True, slots=True)
class ConsoleMemoryBannerPresentation:
    """One content-free banner derived from validated effective memory."""

    kind: Literal["prefix", "range"]
    render_anchor_message_id: str
    start_message_id: str | None
    end_message_id: str
    copy: str

    def __post_init__(self) -> None:
        if self.kind not in {"prefix", "range"}:
            raise ValueError("memory banner kind must be prefix or range")
        for name in ("render_anchor_message_id", "end_message_id", "copy"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"memory banner {name} must be non-empty text")
        if self.kind == "range":
            if not isinstance(self.start_message_id, str) or not self.start_message_id:
                raise ValueError("range memory banner requires a start identity")
        elif self.start_message_id is not None:
            raise ValueError("prefix memory banner cannot carry a start identity")


def derive_console_memory_banner_presentation(
    effective: EffectiveMemoryResult,
    active_messages: Iterable[ConsoleChatMessage],
) -> ConsoleMemoryBannerPresentation | None:
    """Derive one banner from the same typed memory result used by dispatch.

    Every persisted identity lookup is exact. Missing or duplicate anchors,
    malformed effective state, and prefix memories without a real placement
    row return ``None`` rather than guessing from content or proximity.

    Args:
        effective: Validated effective-memory result used by provider dispatch.
        active_messages: Ordered native messages on the visible active branch.

    Returns:
        A content-free banner presentation, or ``None`` when exact placement
        cannot be proven.

    Raises:
        TypeError: If ``effective`` is not an ``EffectiveMemoryResult``.
    """

    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Chat.console_context_compaction import (
        EffectiveMemoryKind,
        EffectiveMemoryResult,
    )
    from tldw_chatbook.Chat.console_context_repository import (
        MemoryCoverageKind,
        MemoryOriginKind,
    )

    if not isinstance(effective, EffectiveMemoryResult):
        raise TypeError("effective must be an EffectiveMemoryResult")
    rows = tuple(active_messages)
    positions_by_persisted_id: dict[str, list[int]] = {}
    for index, message in enumerate(rows):
        persisted_id = message.persisted_message_id
        if isinstance(persisted_id, str) and persisted_id:
            positions_by_persisted_id.setdefault(persisted_id, []).append(index)

    def exact_index(persisted_id: str | None) -> int | None:
        if not isinstance(persisted_id, str) or not persisted_id:
            return None
        matches = positions_by_persisted_id.get(persisted_id, ())
        return matches[0] if len(matches) == 1 else None

    def prefix_presentation(
        *, render_index: int, end_message_id: str
    ) -> ConsoleMemoryBannerPresentation:
        return ConsoleMemoryBannerPresentation(
            kind="prefix",
            render_anchor_message_id=rows[render_index].id,
            start_message_id=None,
            end_message_id=end_message_id,
            copy=CONSOLE_SUMMARY_BANNER_COPY,
        )

    if effective.kind is EffectiveMemoryKind.RAW:
        return None
    if effective.kind is EffectiveMemoryKind.LEGACY_PREFIX:
        legacy = effective.legacy
        if legacy is None:
            return None
        boundary_index = exact_index(legacy.boundary_message_id)
        if boundary_index is None:
            return None
        return prefix_presentation(
            render_index=boundary_index,
            end_message_id=legacy.boundary_message_id,
        )

    memory = effective.memory
    scope = effective.scope
    if (
        memory is None
        or scope is None
        or not memory.active
        or memory.source_kind != "generated"
        or memory.memory_id != scope.memory_id
        or memory.conversation_id != scope.conversation_id
    ):
        return None
    boundary_index = exact_index(memory.boundary_message_id)
    if boundary_index is None:
        return None

    if effective.kind is EffectiveMemoryKind.GENERATED_RANGE:
        if (
            scope.coverage_kind is not MemoryCoverageKind.RANGE
            or scope.origin_kind is not MemoryOriginKind.MANUAL_REWIND
        ):
            return None
        start_index = exact_index(scope.selection_anchor_message_id)
        if (
            start_index is None
            or start_index >= boundary_index
            or rows[start_index].role is not ConsoleMessageRole.USER
        ):
            return None
        user_ordinals: dict[int, int] = {}
        ordinal = 0
        for index, message in enumerate(rows):
            if message.role is ConsoleMessageRole.USER:
                ordinal += 1
                user_ordinals[index] = ordinal
        start_ordinal = user_ordinals.get(start_index)
        end_ordinal = next(
            (
                user_ordinals[index]
                for index in range(boundary_index, start_index - 1, -1)
                if index in user_ordinals
            ),
            None,
        )
        if start_ordinal is None or end_ordinal is None:
            return None
        return ConsoleMemoryBannerPresentation(
            kind="range",
            render_anchor_message_id=rows[start_index].id,
            start_message_id=scope.selection_anchor_message_id,
            end_message_id=memory.boundary_message_id,
            copy=(
                "Context uses a summary of turns "
                f"#{start_ordinal}-#{end_ordinal} - full transcript remains visible."
            ),
        )

    if (
        effective.kind is not EffectiveMemoryKind.GENERATED_PREFIX
        or scope.coverage_kind is not MemoryCoverageKind.PREFIX
    ):
        return None
    if scope.origin_kind is MemoryOriginKind.MANUAL_REWIND:
        anchor_index = exact_index(scope.selection_anchor_message_id)
        if (
            anchor_index is None
            or boundary_index >= anchor_index
            or rows[anchor_index].role is not ConsoleMessageRole.USER
        ):
            return None
        return prefix_presentation(
            render_index=anchor_index,
            end_message_id=memory.boundary_message_id,
        )
    if (
        scope.origin_kind is not MemoryOriginKind.AUTOMATIC
        or scope.selection_anchor_message_id is not None
    ):
        return None
    next_user_index = next(
        (
            index
            for index in range(boundary_index + 1, len(rows))
            if rows[index].role is ConsoleMessageRole.USER
        ),
        None,
    )
    if next_user_index is None:
        return None
    return prefix_presentation(
        render_index=next_user_index,
        end_message_id=memory.boundary_message_id,
    )
