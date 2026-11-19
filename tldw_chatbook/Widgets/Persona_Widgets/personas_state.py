"""Shared state model for the destination-native Personas workbench."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import threading
    import weakref
    from pathlib import Path

    from tldw_chatbook.Character_Chat.persona_visual_identity import (
        LocalPersonaVisualIdentityAuthority,
    )
    from tldw_chatbook.Character_Chat.visual_identity import VisualIdentityCandidate
    from tldw_chatbook.Persona_Visual.authoring import PersonaVisualAuthoringDraft
    from tldw_chatbook.Persona_Visual.authoring_workspace import (
        PersonaVisualAuthoringWorkspace,
    )
    from tldw_chatbook.Persona_Visual.importer import PersonaVisualImportReview
    from tldw_chatbook.TTS import (
        AssignedTTSProfileSnapshot,
        CharacterRef,
        LoadedTTSProfile,
        TTSProfileAvailability,
    )
    from tldw_chatbook.Widgets.Persona_Widgets.persona_profile_editor_widget import (
        PersonaProfileEditorWidget,
    )
    from tldw_chatbook.Widgets.Persona_Widgets.personas_character_editor_widget import (
        PersonasCharacterEditorWidget,
    )
    from tldw_chatbook.Widgets.Persona_Widgets.personas_pane_messages import (
        VisualIdentityAssetMetadata,
        VisualIdentityPackMetadata,
    )
    from tldw_chatbook.Widgets.Persona_Widgets.personas_persona_visual_pack_widget import (
        PersonasPersonaVisualPackWidget,
    )
    from tldw_chatbook.Widgets.Persona_Widgets.personas_visual_identity_pack_widget import (
        PersonasVisualIdentityPackWidget,
    )


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
    "persona",
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
    #: Active library sort key (see ``PersonasScreen._character_sort_cycle``).
    sort_key: str = "name_asc"
    #: Active characters-only tag filter (``None`` means "all tags").
    tag_filter: str | None = None
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
        # Each mode starts with a fresh list: default sort and no tag filter.
        # Search is cleared by the screen's _apply_mode.
        self.sort_key = "name_asc"
        self.tag_filter = None
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
        self.sort_key = "name_asc"
        self.tag_filter = None
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

    def _default_runtime_target(
        self, entity_kind: PersonaEntityKind, entity_id: str
    ) -> str:
        return f"{self.runtime_source}:{entity_kind}:{entity_id}"


__all__ = [
    "MODE_LABELS",
    "VALID_PERSONA_ENTITY_KINDS",
    "VALID_PERSONA_MODES",
    "PersonasWorkbenchState",
]


@dataclasses.dataclass(frozen=True, slots=True)
class _CharacterTTSControlSnapshot:
    """Exact service and authority tokens behind one rendered control state."""

    request_generation: int
    runtime_source: str
    character_id: str
    character_ref: CharacterRef
    repository_generation: int
    loaded_profiles: tuple[LoadedTTSProfile, ...]
    availability: tuple[TTSProfileAvailability, ...]
    current: AssignedTTSProfileSnapshot | None
    assignment_count: int | None
    configuration_revision: int
    catalog_revision: int | None
    expected_server_id: str | None
    server_context_capture: object | None = dataclasses.field(
        repr=False,
        compare=False,
    )

@dataclasses.dataclass(frozen=True, slots=True)
class _ActorPackExportUIAuthority:
    """Exact screen and selection authority for one app-owned export."""

    source: str
    actor_kind: str
    local_actor_id: str
    actor_revision: int
    session_generation: int
    screen: object = dataclasses.field(repr=False, compare=False)
    controller: object = dataclasses.field(repr=False, compare=False)
    service: object = dataclasses.field(repr=False, compare=False)

class _CharacterTTSAuthorityUnavailable(RuntimeError):
    """Stable character authority could not be proved."""

class _CharacterTTSMixedSnapshot(RuntimeError):
    """Repository or capability observations did not form one snapshot."""

@dataclasses.dataclass(frozen=True, slots=True)
class _CharacterVisualIdentityLoadSnapshot:
    """Authority captured before an editor pack-metadata worker starts."""

    editor_ref: weakref.ReferenceType[PersonasCharacterEditorWidget]
    db: object | None
    character_id: int | None
    screen_generation: int
    editor_session_token: int

@dataclasses.dataclass(frozen=True, slots=True)
class _VisualIdentityPreviewSnapshot:
    """Weak, path-free authority snapshot for one selected preview."""

    editor_ref: weakref.ReferenceType[PersonasCharacterEditorWidget]
    browser_ref: weakref.ReferenceType[PersonasVisualIdentityPackWidget]
    db: object
    character_id: int
    screen_generation: int
    editor_session_token: int
    binding_id: int
    pack_id: int
    pack_version_id: int
    source_kind: str
    asset: VisualIdentityAssetMetadata

@dataclasses.dataclass(frozen=True, slots=True)
class _VisualIdentityAuthorSnapshot:
    """Exact editor/browser authority for one unpublished candidate."""

    editor_ref: weakref.ReferenceType[PersonasCharacterEditorWidget]
    browser_ref: weakref.ReferenceType[PersonasVisualIdentityPackWidget]
    db: object
    character_id: int
    screen_generation: int
    editor_session_token: int
    binding_id: int
    pack_id: int
    pack_version_id: int

@dataclasses.dataclass(slots=True)
class _VisualIdentityAuthoringState:
    """One candidate plus cancellation shared by its provider calls."""

    snapshot: _VisualIdentityAuthorSnapshot
    candidate: VisualIdentityCandidate
    cancel_event: threading.Event
    authoritative_pack: VisualIdentityPackMetadata

@dataclasses.dataclass(frozen=True, slots=True)
class _PersonaVisualAuthorSnapshot:
    """Exact local Persona editor authority for one isolated visual draft."""

    editor_ref: weakref.ReferenceType[PersonaProfileEditorWidget]
    browser_ref: weakref.ReferenceType[PersonasPersonaVisualPackWidget]
    db: object
    local_service: object
    persona_id: str
    persona_revision: int
    screen_generation: int
    editor_session_token: int

@dataclasses.dataclass(frozen=True, slots=True)
class _PersonaSharedVisualIdentityLoadSnapshot:
    """Exact local Persona editor authority for Shared Identity metadata."""

    editor_ref: weakref.ReferenceType[PersonaProfileEditorWidget]
    db: object
    local_service: object
    persona_id: str
    persona_revision: int
    screen_generation: int
    editor_session_token: int

@dataclasses.dataclass(frozen=True, slots=True)
class _PersonaSharedVisualIdentityAuthorSnapshot:
    """Exact Persona/editor/pack authority for one unpublished reaction draft."""

    editor_ref: weakref.ReferenceType[PersonaProfileEditorWidget]
    browser_ref: weakref.ReferenceType[PersonasVisualIdentityPackWidget]
    db: object
    local_service: object
    persona_id: str
    persona_revision: int
    persona_authority: LocalPersonaVisualIdentityAuthority
    screen_generation: int
    editor_session_token: int
    binding_id: int
    pack_id: int
    pack_version_id: int

@dataclasses.dataclass(slots=True)
class _PersonaSharedVisualIdentityAuthoringState:
    """One Persona reaction candidate plus its cancellation authority."""

    snapshot: _PersonaSharedVisualIdentityAuthorSnapshot
    candidate: VisualIdentityCandidate
    cancel_event: threading.Event
    authoritative_pack: VisualIdentityPackMetadata

@dataclasses.dataclass(slots=True)
class _PersonaVisualAuthoringState:
    """One unpublished Persona Visual draft plus its private source lease."""

    snapshot: _PersonaVisualAuthorSnapshot
    draft: PersonaVisualAuthoringDraft
    source_root: Path = dataclasses.field(repr=False)
    workspace: PersonaVisualAuthoringWorkspace | None = dataclasses.field(
        default=None, repr=False
    )
    import_review: PersonaVisualImportReview | None = dataclasses.field(
        default=None, repr=False
    )
    dirty: bool = False

@dataclasses.dataclass(frozen=True, slots=True)
class _CharacterSaveAuthority:
    """Editor identity allowed to reconcile one completed character save."""

    editor_ref: weakref.ReferenceType[PersonasCharacterEditorWidget]
    selected_id: str | None
    edit_mode: str
    screen_generation: int
    editor_session_token: int

@dataclasses.dataclass(frozen=True, slots=True)
class _ActorPackPortraitChoice:
    """One immutable eligible local portrait option."""

    character_id: int
    name: str
    revision: int
    sha256: str

@dataclasses.dataclass(frozen=True, slots=True)
class _ActorPackCreateSession:
    """Workbench authority for one pack-ready actor draft."""

    generation: int
    actor_kind: str
    source: str
    editor_generation: int
    editor_session_token: int
    portrait_choices: tuple[_ActorPackPortraitChoice, ...] = ()
