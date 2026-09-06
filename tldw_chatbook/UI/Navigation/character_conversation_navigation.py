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
    serialize_character_conversation_key,
)
from tldw_chatbook.Constants import TAB_CHAT, TAB_PERSONAS

_PAYLOAD_VERSION = 1
_ID_MAX_BYTES = 256
_QUERY_MAX_CHARS = 4096
_SNAPSHOT_MAX_CHARS = 1024
_FOCUS_ID = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,127}\Z")
_RETURN_TARGETS = frozenset(
    {
        (TAB_CHAT, "console-context-character"),
        (TAB_PERSONAS, "personas-conversations-list"),
        (TAB_PERSONAS, "personas-filter"),
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
        """Return the sole Console origin supported by this flow.

        Returns:
            The Console Context character anchor.
        """

        return cls(TAB_CHAT, "console-context-character")

    @classmethod
    def personas_conversations(cls) -> RoleplayReturnTarget:
        """Return to Roleplay's stable conversations list anchor.

        Returns:
            The Roleplay conversations list anchor.
        """

        return cls(TAB_PERSONAS, "personas-conversations-list")

    @classmethod
    def personas_filter(cls) -> RoleplayReturnTarget:
        """Return to Roleplay's stable filter anchor.

        Returns:
            The Roleplay character filter anchor.
        """

        return cls(TAB_PERSONAS, "personas-filter")


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
class LibraryUnavailableConversationInspection:
    """Library detail link for one exact unresolved local conversation."""

    unresolved: UnresolvedConversationKey
    return_target: RoleplayReturnTarget

    def __post_init__(self) -> None:
        if not isinstance(self.unresolved, UnresolvedConversationKey):
            raise TypeError("unresolved must be an UnresolvedConversationKey")
        if not isinstance(self.return_target, RoleplayReturnTarget):
            raise TypeError("return_target must be a RoleplayReturnTarget")
        if self.return_target != RoleplayReturnTarget.console_context_character():
            raise ValueError("unavailable inspection must return to Console Character")


@dataclass(frozen=True)
class LibraryUnavailableConversationsBrowse:
    """Library archive link for all unavailable chats with a selected anchor."""

    selected: UnresolvedConversationKey
    return_target: RoleplayReturnTarget

    def __post_init__(self) -> None:
        if not isinstance(self.selected, UnresolvedConversationKey):
            raise TypeError("selected must be an UnresolvedConversationKey")
        if not isinstance(self.return_target, RoleplayReturnTarget):
            raise TypeError("return_target must be a RoleplayReturnTarget")
        if self.return_target != RoleplayReturnTarget.console_context_character():
            raise ValueError("unavailable browse must return to Console Character")


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

    from ._character_conversation_wire import _RoleplayLinkWire

    wire = _RoleplayLinkWire.model_validate(payload)
    return RoleplayCharacterConversationLink(
        character=ResolvedLocalCharacterKey(
            wire.character.data_authority_id, wire.character.character_id
        ),
        conversation_id=wire.conversation_id,
        query=wire.query,
        data_revision=wire.data_revision,
        return_target=(
            RoleplayReturnTarget(
                wire.return_target.screen_id, wire.return_target.focus_id
            )
            if wire.return_target is not None
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

    from ._character_conversation_wire import _LibraryRepairWire

    wire = _LibraryRepairWire.model_validate(payload)
    return LibraryCharacterRepairContext(
        unresolved=UnresolvedConversationKey(
            wire.unresolved.data_authority_id, wire.unresolved.conversation_id
        ),
        expected_conversation_version=wire.expected_conversation_version,
        historical_display_snapshot=wire.historical_display_snapshot,
        return_target=RoleplayReturnTarget(
            wire.return_target.screen_id, wire.return_target.focus_id
        ),
    )


def _serialize_unresolved_library_link(
    unresolved: UnresolvedConversationKey,
    return_target: RoleplayReturnTarget,
    *,
    identity_field: str,
) -> dict[str, object]:
    return {
        "version": _PAYLOAD_VERSION,
        "source": "local",
        "data_authority_id": unresolved.data_authority_id,
        identity_field: serialize_character_conversation_key(unresolved),
        "return_target": _serialize_return_target(return_target),
    }


def _deserialize_unresolved_library_link(
    payload: Mapping[str, Any],
    *,
    identity_field: str,
    name: str,
) -> tuple[UnresolvedConversationKey, RoleplayReturnTarget]:
    from ._character_conversation_wire import (
        _LibraryUnavailableBrowseWire,
        _LibraryUnavailableInspectionWire,
    )

    model = (
        _LibraryUnavailableInspectionWire
        if identity_field == "unresolved"
        else _LibraryUnavailableBrowseWire
    )
    wire = model.model_validate(payload)
    identity = getattr(wire, identity_field)
    return (
        UnresolvedConversationKey(identity.data_authority_id, identity.conversation_id),
        RoleplayReturnTarget(wire.return_target.screen_id, wire.return_target.focus_id),
    )


def serialize_library_unavailable_inspection(
    link: LibraryUnavailableConversationInspection,
) -> dict[str, object]:
    """Serialize an exact non-mutating Library conversation inspection."""

    if not isinstance(link, LibraryUnavailableConversationInspection):
        raise TypeError("link must be a LibraryUnavailableConversationInspection")
    return _serialize_unresolved_library_link(
        link.unresolved, link.return_target, identity_field="unresolved"
    )


def deserialize_library_unavailable_inspection(
    payload: Mapping[str, Any],
) -> LibraryUnavailableConversationInspection:
    """Validate one exact non-mutating Library conversation inspection."""

    unresolved, return_target = _deserialize_unresolved_library_link(
        payload,
        identity_field="unresolved",
        name="Library unavailable inspection",
    )
    return LibraryUnavailableConversationInspection(unresolved, return_target)


def serialize_library_unavailable_browse(
    link: LibraryUnavailableConversationsBrowse,
) -> dict[str, object]:
    """Serialize a complete Library unavailable-list browse link."""

    if not isinstance(link, LibraryUnavailableConversationsBrowse):
        raise TypeError("link must be a LibraryUnavailableConversationsBrowse")
    return _serialize_unresolved_library_link(
        link.selected, link.return_target, identity_field="selected"
    )


def deserialize_library_unavailable_browse(
    payload: Mapping[str, Any],
) -> LibraryUnavailableConversationsBrowse:
    """Validate one complete Library unavailable-list browse link."""

    selected, return_target = _deserialize_unresolved_library_link(
        payload,
        identity_field="selected",
        name="Library unavailable browse",
    )
    return LibraryUnavailableConversationsBrowse(selected, return_target)
