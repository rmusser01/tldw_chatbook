"""Private Buddy presentation preferences and exact local conversation bindings.

A binding never consults Console selection. It can reconnect a durable conversation
in a new runtime, but cannot follow a repurposed live slot or a remote identifier.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")


def _valid_id(value: object) -> bool:
    return type(value) is str and _ID.fullmatch(value) is not None


@dataclass(frozen=True, slots=True)
class BuddyBinding:
    """An explicit local target, independent of the selected Console session."""

    kind: Literal["conversation", "workspace"]
    target_id: str
    conversation_id: str | None = None
    binding_revision: int = 0
    ephemeral: bool = False

    def __post_init__(self) -> None:
        if self.kind not in {"conversation", "workspace"} or not _valid_id(
            self.target_id
        ):
            raise ValueError("Choose a valid conversation or workspace for the Buddy.")
        if self.conversation_id is not None and not _valid_id(self.conversation_id):
            raise ValueError("Invalid Buddy conversation reference.")
        if type(self.binding_revision) is not int or self.binding_revision < 0:
            raise ValueError("Invalid Buddy conversation binding revision.")
        if type(self.ephemeral) is not bool:
            raise ValueError("Invalid temporary conversation binding.")
        if self.kind == "workspace" and (
            self.conversation_id is not None or self.ephemeral or self.binding_revision
        ):
            raise ValueError("A workspace binding cannot carry conversation identity.")
        if self.ephemeral and self.conversation_id is not None:
            raise ValueError("A temporary conversation cannot carry durable identity.")

    @classmethod
    def for_session(cls, session: Any) -> BuddyBinding:
        """Capture an existing local session without changing or persisting it."""
        if session.runtime_backend != "local":
            raise ValueError(
                "Buddy interaction currently supports local conversations."
            )
        return cls(
            kind="conversation",
            target_id=session.id,
            conversation_id=session.persisted_conversation_id,
            binding_revision=session.conversation_binding_revision,
            ephemeral=session.ephemeral,
        )

    def resolve_session(self, sessions: Iterable[Any]) -> Any | None:
        """Find exactly the bound live session or its uniquely restored local record."""
        if self.kind != "conversation":
            return None
        candidates = list(sessions)
        # A present but repurposed live slot invalidates this binding. Never recover
        # by falling through to another copy of a durable conversation in that case.
        live = [row for row in candidates if row.id == self.target_id]
        if live:
            if len(live) != 1:
                return None
            row = live[0]
            return (
                row
                if (
                    row.runtime_backend == "local"
                    and row.conversation_binding_revision == self.binding_revision
                    and row.ephemeral == self.ephemeral
                    and (
                        self.conversation_id is None
                        or row.persisted_conversation_id == self.conversation_id
                    )
                )
                else None
            )
        if self.conversation_id is None or self.ephemeral:
            return None
        restored = [
            row
            for row in candidates
            if row.runtime_backend == "local"
            and not row.ephemeral
            and row.persisted_conversation_id == self.conversation_id
        ]
        return restored[0] if len(restored) == 1 else None

    def includes(self, session: Any) -> bool:
        """Test scope for a trusted runtime event without reading any transcript."""
        if self.kind == "workspace":
            return (
                session.runtime_backend == "local"
                and session.workspace_id == self.target_id
            )
        return self.resolve_session((session,)) is session


@dataclass(frozen=True, slots=True)
class BuddyInteractionPreferences:
    """Presentation-only settings; artwork and run authority have other owners."""

    binding: BuddyBinding | None = None
    animated: bool = True
    speak_responses: bool = False

    def __post_init__(self) -> None:
        if self.binding is not None and not isinstance(self.binding, BuddyBinding):
            raise ValueError("Invalid Buddy target.")
        if type(self.animated) is not bool or type(self.speak_responses) is not bool:
            raise ValueError("Invalid Buddy animation or speech preference.")


def serialize_preferences(
    preferences: BuddyInteractionPreferences,
) -> dict[str, object]:
    """Encode profile-safe values, excluding temporary conversation references."""
    binding = preferences.binding
    values: dict[str, object] = {
        "animated": preferences.animated,
        "speak_responses": preferences.speak_responses,
        # Explicit empty keys clear a previous target with the merge-based writer.
        "kind": "",
        "target_id": "",
        "conversation_id": "",
        "binding_revision": 0,
    }
    if binding is not None and not binding.ephemeral:
        values.update(
            kind=binding.kind,
            target_id=binding.target_id,
            conversation_id=binding.conversation_id or "",
            binding_revision=binding.binding_revision,
        )
    return values


def parse_preferences(
    raw: Mapping[str, object] | object,
) -> BuddyInteractionPreferences:
    """Parse untrusted profile settings with independent safe field fallbacks."""
    if not isinstance(raw, Mapping):
        return BuddyInteractionPreferences()
    binding = None
    try:
        binding = BuddyBinding(
            kind=raw.get("kind", ""),  # type: ignore[arg-type]
            target_id=raw.get("target_id", ""),  # type: ignore[arg-type]
            conversation_id=raw.get("conversation_id") or None,  # type: ignore[arg-type]
            binding_revision=raw.get("binding_revision", 0),  # type: ignore[arg-type]
        )
    except (TypeError, ValueError):
        pass
    return BuddyInteractionPreferences(
        binding=binding,
        animated=raw.get("animated") if type(raw.get("animated")) is bool else True,
        speak_responses=(
            raw.get("speak_responses")
            if type(raw.get("speak_responses")) is bool
            else False
        ),
    )
