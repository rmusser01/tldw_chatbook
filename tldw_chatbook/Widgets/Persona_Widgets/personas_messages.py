"""Screen-independent message contracts for Personas workbench widgets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from textual.message import Message


PersonaWorkbenchMode = Literal[
    "characters",
    "personas",
    "prompts",
    "dictionaries",
    "lore",
    "import_export",
]
PersonaEntityKind = Literal[
    "character",
    "persona",
    "prompt",
    "dictionary",
    "lore",
]
PersonaAction = Literal[
    "create",
    "create_actor_pack",
    "import",
    "export",
    "duplicate",
    "toggle_enabled",
    "attach_to_console",
    "start_chat",
    "save",
    "cancel",
    "refresh",
]
PersonaBuddyAction = Literal["use", "show", "close", "disable"]
PersonaBuddySource = Literal["local", "server"]
ActorPackExportActorKind = Literal["character", "persona"]


class ActorPackImportRequested(Message):
    """Request the dedicated Actor Pack picker without carrying path/content."""


@dataclass(frozen=True, slots=True)
class _PersonaBuddyActionPayload:
    action: PersonaBuddyAction
    source: PersonaBuddySource
    persona_id: str
    revision: int


class PersonaBuddyActionRequested(Message):
    """Request one explicit Buddy ownership or visibility change."""

    __slots__ = ("_payload",)

    def __init__(
        self,
        *,
        action: PersonaBuddyAction,
        source: PersonaBuddySource,
        persona_id: str,
        revision: int,
    ) -> None:
        super().__init__()
        if (
            action not in {"use", "show", "close", "disable"}
            or source not in {"local", "server"}
            or not persona_id
            or type(revision) is not int
            or revision < 1
        ):
            raise ValueError("invalid Persona Buddy action")
        self._payload = _PersonaBuddyActionPayload(
            action=action,
            source=source,
            persona_id=persona_id,
            revision=revision,
        )

    @property
    def action(self) -> PersonaBuddyAction:
        return self._payload.action

    @property
    def source(self) -> PersonaBuddySource:
        return self._payload.source

    @property
    def persona_id(self) -> str:
        return self._payload.persona_id

    @property
    def revision(self) -> int:
        return self._payload.revision


@dataclass(frozen=True, slots=True)
class _ActorPackExportPayload:
    actor_kind: ActorPackExportActorKind
    source: PersonaBuddySource
    local_actor_id: str
    actor_revision: int


class ActorPackExportRequested(Message):
    """Request export of one exact selected local actor revision."""

    __slots__ = ("_payload",)

    def __init__(
        self,
        *,
        actor_kind: ActorPackExportActorKind,
        source: PersonaBuddySource,
        local_actor_id: str,
        actor_revision: int,
    ) -> None:
        super().__init__()
        if (
            actor_kind not in {"character", "persona"}
            or source not in {"local", "server"}
            or type(local_actor_id) is not str
            or not local_actor_id
            or type(actor_revision) is not int
            or actor_revision < 1
        ):
            raise ValueError("invalid Actor Pack export request")
        self._payload = _ActorPackExportPayload(
            actor_kind=actor_kind,
            source=source,
            local_actor_id=local_actor_id,
            actor_revision=actor_revision,
        )

    @property
    def actor_kind(self) -> ActorPackExportActorKind:
        return self._payload.actor_kind

    @property
    def source(self) -> PersonaBuddySource:
        return self._payload.source

    @property
    def local_actor_id(self) -> str:
        return self._payload.local_actor_id

    @property
    def actor_revision(self) -> int:
        return self._payload.actor_revision


class PersonaModeChanged(Message):
    """Request a Personas workbench mode change."""

    def __init__(self, mode: PersonaWorkbenchMode) -> None:
        super().__init__()
        self.mode = mode


class PersonaEntitySelected(Message):
    """Notify the workbench that a character, persona, or related asset was selected."""

    def __init__(
        self,
        *,
        entity_kind: PersonaEntityKind,
        entity_id: str,
        entity_name: str,
        runtime_target: str | None = None,
    ) -> None:
        super().__init__()
        self.entity_kind = entity_kind
        self.entity_id = entity_id
        self.entity_name = entity_name
        self.runtime_target = runtime_target


class PersonaSearchChanged(Message):
    """Notify the workbench that list search or filter input changed."""

    def __init__(self, *, query: str = "", filter_text: str = "") -> None:
        super().__init__()
        self.query = query
        self.filter_text = filter_text


class PersonaActionRequested(Message):
    """Request a Personas action without coupling child widgets to a screen class."""

    def __init__(
        self,
        *,
        action: PersonaAction,
        entity_kind: PersonaEntityKind | None = None,
        entity_id: str | None = None,
    ) -> None:
        super().__init__()
        self.action = action
        self.entity_kind = entity_kind
        self.entity_id = entity_id


class PersonaSortCycleRequested(Message):
    """The user asked to advance the library sort (screen decides the next key)."""


class PersonaTagFilterRequested(Message):
    """The user asked to open the tag filter (characters only)."""


class PersonaPageChanged(Message):
    """The user asked to move the library page window."""

    def __init__(self, delta: int) -> None:
        super().__init__()
        self.delta = delta


class PersonaMarksChanged(Message):
    """The library pane's marked (multi-selected) row set changed (F-040).

    ``marks`` is a tuple of ``(kind, item_id, name)`` triples in row order;
    empty when the last mark was cleared.
    """

    def __init__(self, marks: tuple[tuple[str, str, str], ...]) -> None:
        super().__init__()
        self.marks = marks


__all__ = [
    "ActorPackExportActorKind",
    "ActorPackExportRequested",
    "PersonaAction",
    "PersonaActionRequested",
    "PersonaBuddyAction",
    "PersonaBuddyActionRequested",
    "PersonaBuddySource",
    "PersonaEntityKind",
    "PersonaEntitySelected",
    "PersonaMarksChanged",
    "PersonaModeChanged",
    "PersonaPageChanged",
    "PersonaSearchChanged",
    "PersonaSortCycleRequested",
    "PersonaTagFilterRequested",
    "PersonaWorkbenchMode",
]
