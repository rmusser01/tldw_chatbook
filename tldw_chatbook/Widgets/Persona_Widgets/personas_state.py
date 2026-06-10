"""Shared state model for the destination-native Personas workbench."""

from __future__ import annotations

from dataclasses import dataclass

from .personas_messages import PersonaEntityKind, PersonaWorkbenchMode


VALID_PERSONA_MODES: tuple[PersonaWorkbenchMode, ...] = (
    "characters",
    "personas",
    "prompts",
    "dictionaries",
    "lore",
    "import_export",
)
VALID_PERSONA_ENTITY_KINDS: tuple[PersonaEntityKind, ...] = (
    "character",
    "persona_profile",
    "prompt",
    "dictionary",
    "lore",
)
MODE_LABELS: dict[PersonaWorkbenchMode, str] = {
    "characters": "Characters",
    "personas": "Personas",
    "prompts": "Prompts",
    "dictionaries": "Dictionaries",
    "lore": "Lore",
    "import_export": "Import / Export",
}


@dataclass(slots=True)
class PersonasWorkbenchState:
    """Serializable state shared by future Personas workbench panes."""

    active_mode: PersonaWorkbenchMode = "characters"
    runtime_source: str = "local"
    search_query: str = ""
    filter_text: str = ""
    selected_entity_kind: PersonaEntityKind | None = None
    selected_entity_id: str | None = None
    selected_entity_name: str = ""
    selected_runtime_target: str | None = None
    is_loading: bool = False
    has_unsaved_changes: bool = False
    status_message: str = "Mode: Characters"

    def switch_mode(self, mode: PersonaWorkbenchMode) -> None:
        """Switch the active workbench mode and clear mode-scoped edit state."""
        if mode not in VALID_PERSONA_MODES:
            raise ValueError(f"Unsupported Personas mode: {mode}")
        self.active_mode = mode
        self.clear_selection()
        self.has_unsaved_changes = False
        self.is_loading = False
        self.status_message = f"Mode: {MODE_LABELS[mode]}"

    def select_entity(
        self,
        *,
        entity_kind: PersonaEntityKind,
        entity_id: str,
        entity_name: str,
        runtime_target: str | None = None,
    ) -> None:
        """Select an entity and derive its Console runtime target when omitted."""
        if entity_kind not in VALID_PERSONA_ENTITY_KINDS:
            raise ValueError(f"Unsupported Personas entity kind: {entity_kind}")
        self.selected_entity_kind = entity_kind
        self.selected_entity_id = str(entity_id)
        self.selected_entity_name = str(entity_name)
        self.selected_runtime_target = runtime_target or self._default_runtime_target(
            entity_kind,
            self.selected_entity_id,
        )
        self.status_message = f"Selected: {self.selected_entity_name}"

    def clear_selection(self) -> None:
        """Clear the selected entity without changing the active mode."""
        self.selected_entity_kind = None
        self.selected_entity_id = None
        self.selected_entity_name = ""
        self.selected_runtime_target = None

    def reset_for_runtime_source_change(self, runtime_source: str) -> None:
        """Reset source-scoped state after switching local/server runtime authority."""
        self.runtime_source = str(runtime_source or "local")
        self.search_query = ""
        self.filter_text = ""
        self.clear_selection()
        self.is_loading = False
        self.has_unsaved_changes = False
        self.status_message = f"Mode: {MODE_LABELS[self.active_mode]}"

    def selected_metadata(self) -> dict[str, str]:
        """Return stable Console handoff metadata for the selected entity."""
        if (
            self.selected_entity_kind is None
            or self.selected_entity_id is None
            or self.selected_runtime_target is None
        ):
            return {}
        return {
            "selected_kind": self.selected_entity_kind,
            "selected_record_id": self.selected_entity_id,
            "selected_name": self.selected_entity_name,
            "selected_target_id": self.selected_runtime_target,
        }

    def _default_runtime_target(self, entity_kind: PersonaEntityKind, entity_id: str) -> str:
        return f"{self.runtime_source}:{entity_kind}:{entity_id}"


__all__ = [
    "MODE_LABELS",
    "VALID_PERSONA_ENTITY_KINDS",
    "VALID_PERSONA_MODES",
    "PersonasWorkbenchState",
]
