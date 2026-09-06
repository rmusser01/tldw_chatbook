"""Trusted payloads for character-conversation cross-screen navigation."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from tldw_chatbook.Character_Chat.character_conversation_navigation import (
    ResolvedLocalCharacterKey,
    UnresolvedConversationKey,
    deserialize_character_conversation_key,
    serialize_character_conversation_key,
)

_PAYLOAD_VERSION = 1
_ID_MAX_BYTES = 256
_QUERY_MAX_CHARS = 4096
_SNAPSHOT_MAX_CHARS = 1024
_FOCUS_ID = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,127}\Z")
_RETURN_TARGETS = frozenset(
    {
        ("chat", "console-context-character"),
        ("personas", "personas-conversations-list"),
        ("personas", "personas-filter"),
    }
)


def _canonical_text(value: object, name: str, *, max_bytes: int = _ID_MAX_BYTES) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be text")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be nonblank canonical text")
    if len(value.encode("utf-8")) > max_bytes:
        raise ValueError(f"{name} is too long")
    return value


def _exact_keys(payload: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(payload) != expected:
        raise ValueError(f"invalid {name} fields")


@dataclass(frozen=True)
class RoleplayReturnTarget:
    """A stable screen and widget anchor for reverse navigation."""

    screen_id: str
    focus_id: str

    def __post_init__(self) -> None:
        _canonical_text(self.screen_id, "screen_id", max_bytes=128)
        if not isinstance(self.focus_id, str) or not _FOCUS_ID.fullmatch(self.focus_id):
            raise ValueError("focus_id is invalid")
        if (self.screen_id, self.focus_id) not in _RETURN_TARGETS:
            raise ValueError("return target is not allowed for character navigation")

    @classmethod
    def console_context_character(cls) -> RoleplayReturnTarget:
        """Return the sole Console origin supported by this flow."""

        return cls("chat", "console-context-character")

    @classmethod
    def personas_conversations(cls) -> RoleplayReturnTarget:
        """Return to Roleplay's stable conversations list anchor."""

        return cls("personas", "personas-conversations-list")

    @classmethod
    def personas_filter(cls) -> RoleplayReturnTarget:
        """Return to Roleplay's stable filter anchor."""

        return cls("personas", "personas-filter")


@dataclass(frozen=True)
class RoleplayCharacterConversationLink:
    """A local, exact-character Roleplay conversations deep link."""

    character: ResolvedLocalCharacterKey
    conversation_id: str | None = None
    query: str = ""
    data_revision: int | None = None
    return_target: RoleplayReturnTarget | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.character, ResolvedLocalCharacterKey):
            raise TypeError("character must be a ResolvedLocalCharacterKey")
        if self.conversation_id is not None:
            _canonical_text(self.conversation_id, "conversation_id")
            if (
                isinstance(self.data_revision, bool)
                or not isinstance(self.data_revision, int)
                or self.data_revision < 0
            ):
                raise ValueError(
                    "data_revision is required for an exact conversation link"
                )
        elif self.data_revision is not None:
            raise ValueError("data_revision requires an exact conversation_id")
        if not isinstance(self.query, str):
            raise TypeError("query must be text")
        if len(self.query) > _QUERY_MAX_CHARS:
            raise ValueError("query is too long")
        if self.return_target is not None and not isinstance(
            self.return_target, RoleplayReturnTarget
        ):
            raise TypeError("return_target must be a RoleplayReturnTarget")


@dataclass(frozen=True)
class LibraryCharacterRepairContext:
    """Library-only repair request for one unresolved local conversation."""

    unresolved: UnresolvedConversationKey
    expected_conversation_version: int
    historical_display_snapshot: str
    return_target: RoleplayReturnTarget

    def __post_init__(self) -> None:
        if not isinstance(self.unresolved, UnresolvedConversationKey):
            raise TypeError("unresolved must be an UnresolvedConversationKey")
        if isinstance(self.expected_conversation_version, bool) or not isinstance(
            self.expected_conversation_version, int
        ):
            raise TypeError("expected_conversation_version must be an integer")
        if self.expected_conversation_version < 1:
            raise ValueError("expected_conversation_version must be positive")
        _canonical_text(
            self.historical_display_snapshot,
            "historical_display_snapshot",
            max_bytes=_SNAPSHOT_MAX_CHARS * 4,
        )
        if len(self.historical_display_snapshot) > _SNAPSHOT_MAX_CHARS:
            raise ValueError("historical_display_snapshot is too long")
        if not isinstance(self.return_target, RoleplayReturnTarget):
            raise TypeError("return_target must be a RoleplayReturnTarget")


@dataclass(frozen=True)
class RoleplayDraftSnapshot:
    """Aggregate dirty and in-flight state owned by the Roleplay surface."""

    form_dirty: bool
    character_visual_dirty: bool
    persona_visual_dirty: bool
    attachments_dirty: bool
    inflight_save_domains: tuple[str, ...]

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, bool)
            for value in (
                self.form_dirty,
                self.character_visual_dirty,
                self.persona_visual_dirty,
                self.attachments_dirty,
            )
        ):
            raise TypeError("draft flags must be booleans")
        for domain in self.inflight_save_domains:
            _canonical_text(domain, "inflight save domain", max_bytes=128)

    @property
    def dirty_domains(self) -> tuple[str, ...]:
        """Return dirty owners in stable user-facing order."""

        return tuple(
            name
            for dirty, name in (
                (self.form_dirty, "character form"),
                (self.character_visual_dirty, "character visuals"),
                (self.persona_visual_dirty, "Persona visuals"),
                (self.attachments_dirty, "attachments"),
            )
            if dirty
        )

    @property
    def is_clean(self) -> bool:
        return not self.dirty_domains and not self.inflight_save_domains


class RoleplayDraftNavigationDialog(ModalScreen[str | None]):
    """Exactly-three-choice aggregate draft veto owned by app navigation."""

    BINDINGS: ClassVar[list[Binding]] = [Binding("escape", "stay", "Stay", show=False)]
    DEFAULT_CSS = """
    RoleplayDraftNavigationDialog { align: center middle; }
    RoleplayDraftNavigationDialog > Container {
        width: 72; max-width: 96%; height: auto; border: thick $accent;
        background: $surface; padding: 1 2;
    }
    RoleplayDraftNavigationDialog Vertical { height: auto; }
    RoleplayDraftNavigationDialog Button { width: 100%; margin-top: 1; }
    """

    def __init__(self, domains: tuple[str, ...]) -> None:
        super().__init__()
        self.domains = domains

    def compose(self) -> ComposeResult:
        with Container(id="roleplay-draft-navigation-dialog"):
            yield Static("Finish Roleplay drafts before leaving")
            yield Static(
                "Affected: " + ", ".join(self.domains),
                id="roleplay-draft-navigation-domains",
            )
            with Vertical():
                yield Button("Save and continue", id="roleplay-draft-save-continue")
                yield Button(
                    "Discard and continue", id="roleplay-draft-discard-continue"
                )
                yield Button("Stay", id="roleplay-draft-stay")

    @on(Button.Pressed)
    def _choice(self, event: Button.Pressed) -> None:
        event.stop()
        choice = {
            "roleplay-draft-save-continue": "save",
            "roleplay-draft-discard-continue": "discard",
            "roleplay-draft-stay": None,
        }.get(event.button.id)
        self.dismiss(choice)

    def action_stay(self) -> None:
        self.dismiss(None)


class RoleplayDraftRecoveryDialog(ModalScreen[str | None]):
    """Recover a partial aggregate save without losing the pending navigation."""

    BINDINGS: ClassVar[list[Binding]] = [Binding("escape", "stay", "Stay", show=False)]
    DEFAULT_CSS = RoleplayDraftNavigationDialog.DEFAULT_CSS

    def __init__(self, failed_domains: tuple[str, ...]) -> None:
        super().__init__()
        self.failed_domains = failed_domains

    def compose(self) -> ComposeResult:
        with Container(id="roleplay-draft-recovery-dialog"):
            yield Static("Some Roleplay drafts could not be saved")
            yield Static(
                "Failed: " + ", ".join(self.failed_domains),
                id="roleplay-draft-recovery-domains",
            )
            with Vertical():
                yield Button("Retry", id="roleplay-draft-retry")
                yield Button("Stay", id="roleplay-draft-recovery-stay")

    @on(Button.Pressed)
    def _choice(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss("retry" if event.button.id == "roleplay-draft-retry" else None)

    def action_stay(self) -> None:
        self.dismiss(None)


def _serialize_return_target(target: RoleplayReturnTarget) -> dict[str, str]:
    return {"screen_id": target.screen_id, "focus_id": target.focus_id}


def _deserialize_return_target(payload: object) -> RoleplayReturnTarget:
    if not isinstance(payload, Mapping):
        raise TypeError("return_target must be a mapping")
    _exact_keys(payload, {"screen_id", "focus_id"}, "return target")
    return RoleplayReturnTarget(
        screen_id=payload.get("screen_id"),  # type: ignore[arg-type]
        focus_id=payload.get("focus_id"),  # type: ignore[arg-type]
    )


def serialize_roleplay_character_conversation_link(
    link: RoleplayCharacterConversationLink,
) -> dict[str, object]:
    """Serialize one exact local Roleplay deep link."""

    if not isinstance(link, RoleplayCharacterConversationLink):
        raise TypeError("link must be a RoleplayCharacterConversationLink")
    return {
        "version": _PAYLOAD_VERSION,
        "source": "local",
        "character": serialize_character_conversation_key(link.character),
        "conversation_id": link.conversation_id,
        "query": link.query,
        "data_revision": link.data_revision,
        "return_target": (
            _serialize_return_target(link.return_target)
            if link.return_target is not None
            else None
        ),
    }


def deserialize_roleplay_character_conversation_link(
    payload: Mapping[str, Any],
) -> RoleplayCharacterConversationLink:
    """Validate and deserialize one supported local Roleplay deep link."""

    _exact_keys(
        payload,
        {
            "version",
            "source",
            "character",
            "conversation_id",
            "query",
            "data_revision",
            "return_target",
        },
        "Roleplay link",
    )
    if payload.get("version") != _PAYLOAD_VERSION:
        raise ValueError("unsupported Roleplay link version")
    if payload.get("source") != "local":
        raise ValueError("Roleplay link source must be local")
    character_payload = payload.get("character")
    if not isinstance(character_payload, Mapping):
        raise TypeError("character must be a mapping")
    character = deserialize_character_conversation_key(character_payload)
    if not isinstance(character, ResolvedLocalCharacterKey):
        raise TypeError("Roleplay link requires a resolved local character")
    conversation_id = payload.get("conversation_id")
    if conversation_id is not None:
        conversation_id = _canonical_text(conversation_id, "conversation_id")
    query = payload.get("query")
    if not isinstance(query, str):
        raise TypeError("query must be text")
    return_payload = payload.get("return_target")
    return RoleplayCharacterConversationLink(
        character=character,
        conversation_id=conversation_id,
        query=query,
        data_revision=payload.get("data_revision"),  # type: ignore[arg-type]
        return_target=(
            _deserialize_return_target(return_payload)
            if return_payload is not None
            else None
        ),
    )


def serialize_library_character_repair_context(
    context: LibraryCharacterRepairContext,
) -> dict[str, object]:
    """Serialize one same-authority Library repair context."""

    if not isinstance(context, LibraryCharacterRepairContext):
        raise TypeError("context must be a LibraryCharacterRepairContext")
    return {
        "version": _PAYLOAD_VERSION,
        "source": "local",
        "data_authority_id": context.unresolved.data_authority_id,
        "unresolved": serialize_character_conversation_key(context.unresolved),
        "expected_conversation_version": context.expected_conversation_version,
        "historical_display_snapshot": context.historical_display_snapshot,
        "return_target": _serialize_return_target(context.return_target),
    }


def deserialize_library_character_repair_context(
    payload: Mapping[str, Any],
) -> LibraryCharacterRepairContext:
    """Validate and deserialize one Library-owned repair context."""

    _exact_keys(
        payload,
        {
            "version",
            "source",
            "data_authority_id",
            "unresolved",
            "expected_conversation_version",
            "historical_display_snapshot",
            "return_target",
        },
        "Library repair context",
    )
    if payload.get("version") != _PAYLOAD_VERSION:
        raise ValueError("unsupported Library repair context version")
    if payload.get("source") != "local":
        raise ValueError("Library repair source must be local")
    unresolved_payload = payload.get("unresolved")
    if not isinstance(unresolved_payload, Mapping):
        raise TypeError("unresolved must be a mapping")
    unresolved = deserialize_character_conversation_key(unresolved_payload)
    if not isinstance(unresolved, UnresolvedConversationKey):
        raise TypeError("Library repair requires an unresolved conversation")
    authority = _canonical_text(payload.get("data_authority_id"), "data_authority_id")
    if authority != unresolved.data_authority_id:
        raise ValueError("repair authority components do not match")
    return LibraryCharacterRepairContext(
        unresolved=unresolved,
        expected_conversation_version=payload.get("expected_conversation_version"),  # type: ignore[arg-type]
        historical_display_snapshot=payload.get("historical_display_snapshot"),  # type: ignore[arg-type]
        return_target=_deserialize_return_target(payload.get("return_target")),
    )
