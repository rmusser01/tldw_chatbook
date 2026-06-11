"""Destination-native Personas workbench (Characters mode wired)."""

from __future__ import annotations

from typing import Any

from loguru import logger
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, Static

from ...Widgets.confirmation_dialog import UnsavedChangesDialog
from ...Widgets.CCP_Widgets.ccp_character_card_widget import (
    CCPCharacterCardWidget,
    EditCharacterRequested,
)
from ...Widgets.CCP_Widgets.ccp_character_editor_widget import (
    CCPCharacterEditorWidget,
    CharacterEditorCancelled,
    CharacterSaveRequested,
)
from ...Widgets.destination_workbench import DestinationModeStrip
from ...Widgets.Persona_Widgets.personas_inspector_pane import PersonasInspectorPane
from ...Widgets.Persona_Widgets.personas_library_pane import LibraryRow, PersonasLibraryPane
from ...Widgets.Persona_Widgets.personas_messages import (
    PersonaActionRequested,
    PersonaEntitySelected,
)
from ...Widgets.Persona_Widgets.personas_state import MODE_LABELS, PersonasWorkbenchState
from ..CCP_Modules import ccp_character_handler
from ..CCP_Modules.ccp_character_handler import CCPCharacterHandler
from ..CCP_Modules.ccp_persona_handler import CCPPersonaHandler
from ..Navigation.base_app_screen import BaseAppScreen


logger = logger.bind(module="PersonasScreen")

#: Modes rendered as chips in the strip; "import_export" is intentionally
#: excluded until import/export is wired as an action rather than a mode.
MODE_CHIP_ORDER: tuple[str, ...] = ("characters", "personas", "prompts", "dictionaries", "lore")

PLACEHOLDER_COPY = "This mode is not available yet. Characters and Personas are the supported modes."

#: Center-area widgets toggled by ``_show_center``.
_CENTER_VIEW_IDS: tuple[str, ...] = (
    "#ccp-character-card-view",
    "#ccp-character-editor-view",
    "#personas-mode-placeholder",
)


class PersonasScreen(BaseAppScreen):
    """Characters, personas, prompts, dictionaries, and behavior profiles."""

    # Baseline workbench geometry so the screen renders correctly even without
    # the app stylesheet (e.g. harness tests). The agentic-terminal TCSS uses
    # equal-specificity selectors and takes precedence when loaded.
    DEFAULT_CSS = """
    PersonasScreen {
        background: $background;
    }

    #personas-mode-strip {
        height: 1;
        min-height: 1;
        padding: 0 1;
        overflow: hidden;
    }

    #personas-mode-label {
        width: 8;
        min-width: 8;
        height: 1;
        min-height: 1;
    }

    Button.personas-mode-chip {
        width: auto;
        min-width: 0;
        height: 1;
        min-height: 1;
        padding: 0 1;
        border: none;
    }

    .personas-mode-chip.is-active {
        border: none;
        background: $primary;
        color: $background;
        text-style: bold underline;
    }

    #personas-workbench {
        height: 1fr;
        min-height: 20;
        padding: 1;
        border: solid $surface-lighten-1;
    }

    #personas-workbench .destination-workbench-pane {
        min-width: 0;
        height: 100%;
        min-height: 18;
        padding: 0 1;
        border: solid $surface-lighten-1;
    }

    #personas-library-pane {
        width: 2fr;
    }

    #personas-work-area {
        width: 4fr;
    }

    #personas-inspector-pane {
        width: 2fr;
    }

    #personas-detail-stack {
        width: 100%;
        height: 1fr;
        min-height: 0;
    }

    #personas-library-rows {
        height: 1fr;
        min-height: 3;
    }

    #personas-library-pane Button.personas-library-row {
        width: 100%;
        min-width: 0;
        height: 1;
        min-height: 1;
        padding: 0 1;
        border: none;
    }

    .personas-library-row.is-active {
        background: $primary;
        color: $background;
        text-style: bold;
    }
    """

    def __init__(self, app_instance: Any, **kwargs: Any) -> None:
        super().__init__(app_instance, "personas", **kwargs)
        self.state = PersonasWorkbenchState()
        self._edit_mode: str = "view"
        self._characters: list[dict] = []
        self.character_handler = CCPCharacterHandler(self)
        self.persona_handler = CCPPersonaHandler(self)

    # ===== Compose =====

    def compose_content(self) -> ComposeResult:
        with Vertical(id="personas-shell"):
            yield Static(
                "Personas | Behavior profiles for chat and agents | Ready | Local",
                id="personas-title",
                classes="ds-destination-header",
            )
            yield Static(
                "Create and manage behavior profiles - characters, personas, prompts, "
                "dictionaries, and lore - and attach them to Console.",
                id="personas-purpose",
                classes="destination-purpose",
            )
            yield Static(
                self._status_row_text(),
                id="personas-status-row",
                classes="destination-status-row",
            )
            with DestinationModeStrip(id="personas-mode-strip", classes="destination-mode-strip"):
                yield Static("Modes:", id="personas-mode-label", classes="destination-section")
                for mode in MODE_CHIP_ORDER:
                    classes = "personas-mode-chip"
                    if mode == self.state.active_mode:
                        classes = f"{classes} is-active"
                    yield Button(
                        MODE_LABELS[mode],
                        id=f"personas-mode-{mode}",
                        classes=classes,
                        tooltip=f"Switch the workbench to {MODE_LABELS[mode]}.",
                    )
            with Horizontal(id="personas-workbench", classes="ds-panel destination-workbench"):
                yield PersonasLibraryPane(
                    id="personas-library-pane",
                    classes="destination-workbench-pane",
                )
                with Vertical(id="personas-work-area", classes="destination-workbench-pane"):
                    with Container(id="personas-detail-stack"):
                        yield CCPCharacterCardWidget(parent_screen=self)
                        yield CCPCharacterEditorWidget(parent_screen=self)
                        yield Static(PLACEHOLDER_COPY, id="personas-mode-placeholder")
                yield PersonasInspectorPane(
                    id="personas-inspector-pane",
                    classes="destination-workbench-pane ds-inspector",
                )

    async def on_mount(self) -> None:
        super().on_mount()
        self.query_one(PersonasLibraryPane).set_mode(self.state.active_mode)
        self._show_center(None)
        await self.character_handler.refresh_character_list()

    # ===== Library rendering =====

    async def refresh_character_library_list(self, characters: list[dict] | None) -> None:
        """Destination-native hook called by ``CCPCharacterHandler``."""
        self._characters = [dict(record) for record in (characters or [])]
        if self.state.active_mode != "characters":
            return
        try:
            await self._render_library_rows()
        except Exception:
            # Tolerate refreshes that race screen teardown.
            logger.warning("Could not render the character library rows.", exc_info=True)

    async def _render_library_rows(self) -> None:
        rows = tuple(
            LibraryRow(
                item_id=str(record.get("id")),
                kind="character",
                name=str(record.get("name") or "Unnamed"),
            )
            for record in self._characters
            if record.get("id") is not None
        )
        library = self.query_one(PersonasLibraryPane)
        await library.update_rows(rows, total=len(rows), noun="characters")
        if self.state.selected_entity_kind == "character" and self.state.selected_entity_id:
            library.mark_active_row("character", self.state.selected_entity_id)

    def _character_record(self, item_id: str | None) -> dict | None:
        if item_id is None:
            return None
        for record in self._characters:
            if str(record.get("id")) == str(item_id):
                return record
        return None

    # ===== Mode switching =====

    @on(Button.Pressed, ".personas-mode-chip")
    async def _handle_mode_chip(self, event: Button.Pressed) -> None:
        event.stop()
        mode = str(event.button.id or "").removeprefix("personas-mode-")
        if mode not in MODE_CHIP_ORDER or mode == self.state.active_mode:
            return
        if self.state.has_unsaved_changes:
            self.run_worker(self._switch_mode_after_guard(mode))
            return
        await self._apply_mode(mode)

    async def _switch_mode_after_guard(self, mode: str) -> None:
        if not await self._confirm_discard_unsaved():
            return
        await self._apply_mode(mode)

    async def _apply_mode(self, mode: str) -> None:
        self.state.switch_mode(mode)
        self._edit_mode = "view"
        for chip_mode in MODE_CHIP_ORDER:
            self.query_one(f"#personas-mode-{chip_mode}", Button).set_class(
                chip_mode == mode, "is-active"
            )
        self.query_one("#personas-status-row", Static).update(self._status_row_text())
        library = self.query_one(PersonasLibraryPane)
        library.set_mode(mode)
        await self.query_one(PersonasInspectorPane).clear_selection()
        if mode == "characters":
            await self._render_library_rows()
            self._show_center(None)
        elif mode == "personas":
            # Persona profile listing is wired in a follow-up task.
            await library.update_rows((), total=0, noun="persona profiles")
            self._show_center(None)
        else:
            await library.update_rows((), total=0, noun=MODE_LABELS[mode].lower())
            self._show_center("#personas-mode-placeholder")

    def _status_row_text(self) -> str:
        return f"Mode: {MODE_LABELS[self.state.active_mode]} | Source: Local | Attachments: Console"

    # ===== Selection =====

    @on(PersonaEntitySelected)
    async def _handle_entity_selected(self, message: PersonaEntitySelected) -> None:
        message.stop()
        if message.entity_kind != "character":
            # Persona profiles and other kinds are wired in follow-up tasks.
            return
        if self.state.has_unsaved_changes:
            self.run_worker(
                self._select_character_after_guard(message.entity_id, message.entity_name)
            )
            return
        await self._select_character(message.entity_id, message.entity_name)

    async def _select_character_after_guard(self, entity_id: str, entity_name: str) -> None:
        if not await self._confirm_discard_unsaved():
            return
        self.state.has_unsaved_changes = False
        await self._select_character(entity_id, entity_name)

    async def _select_character(self, entity_id: str, entity_name: str) -> None:
        self.state.select_entity(
            entity_kind="character",
            entity_id=entity_id,
            entity_name=entity_name,
        )
        self._edit_mode = "view"
        self.query_one(PersonasLibraryPane).mark_active_row("character", entity_id)
        await self.character_handler.load_character(entity_id)
        self._show_center("#ccp-character-card-view")
        inspector = self.query_one(PersonasInspectorPane)
        inspector.show_selection(name=entity_name, kind="character", authority="Local")
        inspector.set_unsaved(False)

    # ===== Create / edit =====

    @on(PersonaActionRequested)
    async def _handle_action_requested(self, message: PersonaActionRequested) -> None:
        message.stop()
        if message.action != "create":
            # Import/export/delete and the rest are wired in follow-up tasks.
            return
        if self.state.active_mode != "characters":
            # Persona profile creation is wired in a follow-up task.
            return
        if self.state.has_unsaved_changes:
            self.run_worker(self._begin_create_after_guard())
            return
        self._begin_create_character()

    async def _begin_create_after_guard(self) -> None:
        if not await self._confirm_discard_unsaved():
            return
        self._begin_create_character()

    def _begin_create_character(self) -> None:
        self._edit_mode = "create"
        self.state.clear_selection()
        self.state.has_unsaved_changes = True
        self.query_one(CCPCharacterEditorWidget).new_character()
        self._show_center("#ccp-character-editor-view")
        self.query_one(PersonasInspectorPane).set_unsaved(True)

    @on(EditCharacterRequested)
    def _handle_edit_requested(self, message: EditCharacterRequested) -> None:
        message.stop()
        record = self._full_character_record(str(message.character_id))
        if record is None:
            self._notify("Character data is not loaded yet.", severity="warning")
            return
        self._edit_mode = "edit"
        self.state.has_unsaved_changes = True
        self.query_one(CCPCharacterEditorWidget).load_character(record)
        self._show_center("#ccp-character-editor-view")
        self.query_one(PersonasInspectorPane).set_unsaved(True)

    def _full_character_record(self, character_id: str) -> dict | None:
        """Prefer the handler's fully-loaded card over the id/name list rows."""
        loaded = self.character_handler.current_character_data
        if loaded and str(self.character_handler.current_character_id) == character_id:
            return dict(loaded)
        return self._character_record(character_id)

    # ===== Save =====

    def _validate_character(self, data: dict) -> tuple[str, ...]:
        """Field-level validation; failures block Save and render in the inspector."""
        errors: list[str] = []
        if not str(data.get("name", "")).strip():
            errors.append("name: required")
        book = data.get("character_book")
        if book:
            from ...Character_Chat.Character_Chat_Lib import validate_character_book

            ok, book_errors = validate_character_book(book)
            if not ok:
                errors.extend(str(error) for error in book_errors)
        return tuple(errors)

    @on(CharacterSaveRequested)
    def _handle_save_requested(self, message: CharacterSaveRequested) -> None:
        message.stop()
        data = dict(message.character_data or {})
        errors = self._validate_character(data)
        self.query_one(PersonasInspectorPane).show_validation(errors)
        if errors:
            return
        self._save_character_worker(data, self.state.selected_entity_id)

    @work(thread=True, exclusive=True)
    def _save_character_worker(self, data: dict, selected_id: str | None) -> None:
        """Persist via the legacy module-level helpers off the UI thread."""
        try:
            if self._edit_mode == "create" or not selected_id:
                saved_id = ccp_character_handler.create_character(data)
                if not saved_id:
                    raise RuntimeError("Character creation returned no id.")
            else:
                if not ccp_character_handler.update_character(selected_id, data):
                    raise RuntimeError(f"Character update failed for id {selected_id}.")
                saved_id = selected_id
        except Exception as exc:
            logger.error(f"Error saving character: {exc}", exc_info=True)
            self.app.call_from_thread(
                self._notify, f"Failed to save character: {exc}", "error"
            )
            return
        self.app.call_from_thread(self._after_character_save, str(saved_id))

    async def _after_character_save(self, saved_id: str) -> None:
        self._edit_mode = "view"
        await self.character_handler.refresh_character_list()
        record = self._character_record(saved_id)
        name = str(record.get("name") or "Saved character") if record else "Saved character"
        self.state.select_entity(entity_kind="character", entity_id=saved_id, entity_name=name)
        self.state.has_unsaved_changes = False
        inspector = self.query_one(PersonasInspectorPane)
        inspector.show_selection(name=name, kind="character", authority="Local")
        inspector.set_unsaved(False)
        inspector.show_validation(())
        self.query_one(PersonasLibraryPane).mark_active_row("character", saved_id)
        if record is not None:
            await self.character_handler.load_character(saved_id)
        self._show_center("#ccp-character-card-view")
        self._notify("Character saved.", severity="information")

    # ===== Cancel =====

    @on(CharacterEditorCancelled)
    async def _handle_editor_cancelled(self, message: CharacterEditorCancelled) -> None:
        message.stop()
        if self.state.has_unsaved_changes:
            self.run_worker(self._cancel_edit_after_guard())
            return
        self._finish_cancel_edit()

    async def _cancel_edit_after_guard(self) -> None:
        if not await self._confirm_discard_unsaved():
            return
        self._finish_cancel_edit()

    def _finish_cancel_edit(self) -> None:
        self._edit_mode = "view"
        self.state.has_unsaved_changes = False
        self.query_one(PersonasInspectorPane).set_unsaved(False)
        if self.state.selected_entity_id:
            self._show_center("#ccp-character-card-view")
        else:
            self._show_center(None)

    # ===== Helpers =====

    def _show_center(self, visible_id: str | None) -> None:
        """Show one center-area view (or none); tolerate missing nodes."""
        for selector in _CENTER_VIEW_IDS:
            try:
                widget = self.query_one(selector)
            except Exception:
                continue
            visible = selector == visible_id
            # The CCP widgets carry a `.hidden` class with display:none !important.
            widget.set_class(not visible, "hidden")
            widget.display = visible

    async def _confirm_discard_unsaved(self) -> bool:
        """True when it is safe to discard the in-progress edit.

        The dialog path requires a worker context (``push_screen_wait``);
        message handlers route through ``self.run_worker`` before calling this
        when ``has_unsaved_changes`` is set.
        """
        if not self.state.has_unsaved_changes:
            return True
        dialog = UnsavedChangesDialog(
            tab_title=self.state.selected_entity_name or "New character",
        )
        try:
            return bool(await self.app.push_screen_wait(dialog))
        except Exception:
            logger.warning("Could not show unsaved-changes dialog; keeping edits.", exc_info=True)
            return False

    def _notify(self, message: str, severity: str = "warning") -> None:
        notify = getattr(self.app_instance, "notify", None)
        if callable(notify):
            notify(message, severity=severity)
