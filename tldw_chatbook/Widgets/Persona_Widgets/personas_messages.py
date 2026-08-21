"""Screen-independent message contracts for Personas workbench widgets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Self

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


@dataclass(frozen=True, slots=True)
class PersonaBuddyActionRequested(Message):
    """Request one explicit Buddy ownership or visibility change."""

    action: PersonaBuddyAction
    source: PersonaBuddySource
    persona_id: str
    revision: int

    def __post_init__(self) -> None:
        if (
            self.action not in {"use", "show", "close", "disable"}
            or self.source not in {"local", "server"}
            or not self.persona_id
            or type(self.revision) is not int
            or self.revision < 1
        ):
            raise ValueError("invalid Persona Buddy action")
        # Textual mutates Message's private delivery slots. Keep those slots
        # operational while the public action payload remains frozen.
        initialized = Message()
        for attribute in Message.__slots__:
            object.__setattr__(self, attribute, getattr(initialized, attribute))

    def set_sender(self, sender: Any) -> Self:
        object.__setattr__(self, "_sender", sender)
        return self

    def _set_forwarded(self) -> None:
        object.__setattr__(self, "_forwarded", True)

    def prevent_default(self, prevent: bool = True) -> Self:
        object.__setattr__(self, "_no_default_action", prevent)
        return self

    def stop(self, stop: bool = True) -> Self:
        object.__setattr__(self, "_stop_propagation", stop)
        return self

    def _bubble_to(self, widget: Any) -> None:
        object.__setattr__(self, "_no_default_action", False)
        widget.post_message(self)


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
