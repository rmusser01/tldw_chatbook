"""Screen-independent message contracts for Personas workbench widgets."""

from __future__ import annotations

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
    "persona_profile",
    "prompt",
    "dictionary",
    "lore",
]
PersonaAction = Literal[
    "create",
    "import",
    "export",
    "attach_to_console",
    "start_chat",
    "save",
    "cancel",
    "refresh",
]


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


__all__ = [
    "PersonaAction",
    "PersonaActionRequested",
    "PersonaEntityKind",
    "PersonaEntitySelected",
    "PersonaModeChanged",
    "PersonaSearchChanged",
    "PersonaWorkbenchMode",
]
