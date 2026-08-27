"""Provider-specific replay of optional displayable Console thinking history."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from tldw_chatbook.Chat.thinking_blocks import (
    THINKING_ENVELOPE_VERSION,
    DisplayableThinkingBlock,
    ThinkingEnvelope,
    ThinkingHistoryPolicy,
    normalize_thinking_history_policy,
)


EffectiveThinkingHistoryPolicy = Literal["auto", "include", "exclude", "required"]
ThinkingReplayDisposition = Literal["displayable", "proprietary", "ignored"]
_LOCAL_CHAT_TARGETS = frozenset({"llama_cpp", "local_llamacpp", "vllm", "local_vllm"})
_START_ANCHORED_FORMAT = "start_anchored_think"
_SERIALIZATION_ERROR = "Thinking history could not be serialized safely."


class ThinkingHistorySerializationError(ValueError):
    """Reject strict replay without exposing stored thinking text."""


@dataclass(frozen=True, slots=True)
class ProviderThinkingSidecar:
    """One supported thinking envelope attached to its exact assistant owner."""

    owner_message_id: str
    envelope: ThinkingEnvelope = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.owner_message_id) is not str or not self.owner_message_id.strip():
            raise ValueError("Thinking owner ID must be nonblank.")
        if not isinstance(self.envelope, ThinkingEnvelope):
            raise TypeError("Thinking envelope must be canonical.")


@dataclass(frozen=True, slots=True)
class ResolvedThinkingBlock:
    """One complete block approved for this exact target serializer."""

    owner_message_id: str
    source_format: str
    text: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class ThinkingOwnerGroup:
    """Optional replay blocks retained or evicted with their visible owner."""

    owner_message_id: str
    blocks: tuple[ResolvedThinkingBlock, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        if type(self.owner_message_id) is not str or not self.owner_message_id.strip():
            raise ValueError("Thinking owner ID must be nonblank.")
        blocks = tuple(self.blocks)
        if not blocks or any(
            not isinstance(block, ResolvedThinkingBlock)
            or block.owner_message_id != self.owner_message_id
            for block in blocks
        ):
            raise ValueError("Thinking blocks must match one nonempty owner group.")
        object.__setattr__(self, "blocks", blocks)


@dataclass(frozen=True, slots=True)
class ThinkingReplayTarget:
    """Frozen adapter facts that decide optional replay compatibility."""

    provider: str
    model: str
    protocol: str
    disposition: ThinkingReplayDisposition
    round_trip_version: int | None


@dataclass(frozen=True, slots=True)
class ResolvedThinkingHistory:
    """Saved preference, effective UI state, and provider-ready owner groups."""

    saved_policy: ThinkingHistoryPolicy
    effective_policy: EffectiveThinkingHistoryPolicy
    groups: tuple[ThinkingOwnerGroup, ...] = field(default=(), repr=False)


def effective_thinking_history_policy(
    policy: object,
    *,
    continuation_required: bool = False,
) -> EffectiveThinkingHistoryPolicy:
    """Resolve the effective replay policy without changing the saved value."""

    saved = normalize_thinking_history_policy(policy)
    return "required" if continuation_required else saved


def _claims_local_compatibility(
    block: DisplayableThinkingBlock,
    target: ThinkingReplayTarget,
) -> bool:
    return (
        target.provider in _LOCAL_CHAT_TARGETS
        and block.provider in _LOCAL_CHAT_TARGETS
        and target.protocol == block.protocol
        and target.disposition == "displayable"
        and target.round_trip_version == THINKING_ENVELOPE_VERSION
    )


def _safe_start_anchored_text(text: str) -> bool:
    lowered = text.lower()
    return "<think" not in lowered and "</think" not in lowered


def resolve_thinking_history(
    *,
    target: ThinkingReplayTarget,
    policy: object,
    sidecars: tuple[ProviderThinkingSidecar, ...] = (),
    continuation_required: bool = False,
) -> ResolvedThinkingHistory:
    """Resolve optional replay without changing its saved conversation policy."""

    if not isinstance(target, ThinkingReplayTarget):
        raise TypeError("target must be a ThinkingReplayTarget.")
    saved = normalize_thinking_history_policy(policy)
    effective = effective_thinking_history_policy(
        saved,
        continuation_required=continuation_required,
    )
    if saved == "exclude":
        return ResolvedThinkingHistory(saved, effective)

    groups: list[ThinkingOwnerGroup] = []
    seen_owners: set[str] = set()
    for sidecar in tuple(sidecars):
        if not isinstance(sidecar, ProviderThinkingSidecar):
            raise TypeError("thinking sidecars must be canonical.")
        if sidecar.owner_message_id in seen_owners:
            raise ValueError("Thinking owner IDs must be unique.")
        seen_owners.add(sidecar.owner_message_id)
        resolved: list[ResolvedThinkingBlock] = []
        for block in sidecar.envelope.blocks:
            if not isinstance(block, DisplayableThinkingBlock):
                continue
            if block.status != "complete" or not _claims_local_compatibility(
                block, target
            ):
                continue
            safely_serializable = (
                block.source_format == _START_ANCHORED_FORMAT
                and _safe_start_anchored_text(block.text)
            )
            if not safely_serializable:
                if saved == "include":
                    raise ThinkingHistorySerializationError(_SERIALIZATION_ERROR)
                continue
            resolved.append(
                ResolvedThinkingBlock(
                    owner_message_id=sidecar.owner_message_id,
                    source_format=block.source_format,
                    text=block.text,
                )
            )
        if resolved:
            groups.append(ThinkingOwnerGroup(sidecar.owner_message_id, tuple(resolved)))
    return ResolvedThinkingHistory(saved, effective, tuple(groups))


def serialize_start_anchored_thinking(
    visible_answer: object,
    group: ThinkingOwnerGroup,
) -> str:
    """Return exact local chat-template source encoding for one owner."""

    if type(visible_answer) is not str:
        raise ThinkingHistorySerializationError(_SERIALIZATION_ERROR)
    parts: list[str] = []
    for block in group.blocks:
        if (
            block.source_format != _START_ANCHORED_FORMAT
            or not _safe_start_anchored_text(block.text)
        ):
            raise ThinkingHistorySerializationError(_SERIALIZATION_ERROR)
        parts.append(f"<think>{block.text}</think>")
    parts.append(visible_answer)
    return "\n".join(parts)
