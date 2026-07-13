"""Destination-native Personas workbench (Characters mode wired)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Awaitable, Callable

from loguru import logger
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.css.query import QueryError
from textual.timer import Timer
from textual.widgets import Button, Input, ListView, Static, TextArea

from ...Character_Chat.Character_Chat_Lib import (
    export_character_card_to_json,
    export_character_card_to_png,
    validate_character_book,
)
from ...Chat.chat_handoff_models import ChatHandoffPayload
from ...DB.ChaChaNotes_DB import ConflictError
from ...tldw_api import PersonaProfileCreate, PersonaProfileUpdate
from ...Utils.path_validation import validate_path_simple
from ...Widgets.Console.console_rail_handle import ConsoleRailHandle
from ...Widgets.confirmation_dialog import ConfirmationDialog, UnsavedChangesDialog
from ...Widgets.destination_workbench import DestinationModeStrip
from ...Widgets.Persona_Widgets.persona_profile_card_widget import PersonaProfileCardWidget
from ...Widgets.Persona_Widgets.persona_profile_editor_widget import PersonaProfileEditorWidget
from ...Widgets.Persona_Widgets.personas_character_card_widget import (
    PersonasCharacterCardWidget,
)
from ...Widgets.Persona_Widgets.personas_character_editor_widget import (
    PersonasCharacterEditorWidget,
)
from ...Widgets.Persona_Widgets.personas_conversation_transcript_widget import (
    PersonasConversationTranscriptWidget,
)
from ...Widgets.Persona_Widgets.personas_inspector_pane import PersonasInspectorPane
from ...Widgets.Persona_Widgets.personas_library_pane import LibraryRow, PersonasLibraryPane
from ...Widgets.Persona_Widgets.personas_messages import (
    PersonaActionRequested,
    PersonaEntitySelected,
    PersonaSearchChanged,
)
from ...Widgets.Persona_Widgets.personas_pane_messages import (
    CharacterEditorCancelled,
    CharacterImageUploadRequested,
    CharacterSaveRequested,
    ConversationRowSelected,
    EditCharacterRequested,
    EditorContentChanged,
    EditPersonaRequested,
    PersonaProfileEditCancelled,
    PersonaProfileSaveRequested,
    PreviewOpenInConsoleRequested,
    PreviewReplyRequested,
    PreviewResetRequested,
)
from ...Widgets.Persona_Widgets.personas_dictionary_detail import (
    DictionaryEntriesReorderRequested,
    DictionaryEntryAddRequested,
    DictionaryEntryDeleteRequested,
    DictionaryEntryUpdateRequested,
    DictionarySettingsEdited,
    DictionarySettingsSaveRequested,
    PersonasDictionaryDetailWidget,
)
from ...Widgets.Persona_Widgets.personas_preview_pane import PersonasPreviewPane
from ...Widgets.Persona_Widgets.personas_state import MODE_LABELS, PersonasWorkbenchState
from ...Widgets.workbench_focus import WorkbenchPaneTarget, focus_relative_workbench_pane
from ..CCP_Modules import ccp_character_handler
from ..CCP_Modules.ccp_character_handler import CCPCharacterHandler
from ..CCP_Modules.ccp_enhanced_handlers import setup_ccp_enhancements
from ..CCP_Modules.ccp_messages import CharacterMessage
from ..CCP_Modules.ccp_persona_handler import CCPPersonaHandler
from .destination_recovery import DestinationRecoveryState
from ..Navigation.base_app_screen import BaseAppScreen
from ..Navigation.shortcut_context import ShortcutAction, ShortcutContext
from ..Persona_Modules.personas_conversations_controller import (
    _CONVERSATION_VIEW_ID,
    PersonasConversationsController,
)
from ..Persona_Modules.personas_preview_controller import PersonasPreviewController


logger = logger.bind(module="PersonasScreen")

#: Modes rendered as chips in the strip; "import_export" is intentionally
#: excluded until import/export is wired as an action rather than a mode.
MODE_CHIP_ORDER: tuple[str, ...] = ("characters", "personas", "prompts", "dictionaries", "lore")

#: One-line "what this mode is" copy, shown under the title and as chip tooltips.
_MODE_DESCRIPTORS: dict[str, str] = {
    "characters": "Characters — who the AI plays.",
    "personas": "Personas — who you are.",
    "prompts": "Prompts — moving to the Library.",
    "dictionaries": "Dictionaries — text find/replace rules.",
    "lore": "Lore — world facts injected on keywords.",
}

#: Modes genuinely coming to Roleplay — their chips carry the "· soon" marker.
#: Departing modes (prompts) are deliberately excluded: they are leaving, not arriving.
_COMING_SOON_MODES: frozenset[str] = frozenset({"lore"})

#: Placeholder body per not-yet-built (or departing) mode; generic fallback for others.
_MODE_PLACEHOLDER_BODY: dict[str, str] = {
    "lore": "Lore — build world facts that get injected when keywords appear. Coming soon.",
    "prompts": "Prompts are moving to the Library — you'll manage them there.",
}
_PLACEHOLDER_FALLBACK = "This mode is coming soon."
PERSONAS_SEARCH_DEBOUNCE_SECONDS = 0.2
PERSONAS_AVATAR_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif"})
PERSONAS_AVATAR_IMAGE_SUFFIX_COPY = "PNG, JPG, JPEG, WEBP, or GIF"
PERSONAS_AVATAR_MAX_BYTES = 5 * 1024 * 1024
PERSONAS_AVATAR_MAX_SIZE_COPY = "5 MB"

# 80-column terminals need a tighter three-pane split than the default
# 2:4:2 workbench minimums. Keep this screen-owned so a later rail-collapse
# task can replace it without changing pane widgets.
PERSONAS_COMPACT_WORKBENCH_MAX_WIDTH = 90
PERSONAS_LIBRARY_RAIL_HANDLE_WIDTH = 13
PERSONAS_INSPECTOR_RAIL_HANDLE_WIDTH = 11

#: Center-area widgets toggled by ``_show_center``.
_CENTER_VIEW_IDS: tuple[str, ...] = (
    "#personas-dictionary-detail",
    "#ccp-character-card-view",
    "#ccp-character-editor-view",
    "#ccp-persona-card-view",
    "#ccp-persona-editor-view",
    _CONVERSATION_VIEW_ID,
    "#personas-mode-placeholder",
)


class PersonasScreen(BaseAppScreen):
    """Characters, personas, prompts, dictionaries, and behavior profiles."""

    #: Page size above which the loaded list may be truncated and FTS is used
    #: instead of filtering the in-memory list. Must stay in sync with the
    #: ``fetch_character_names(limit=1000)`` default in
    #: ``Character_Chat/Character_Chat_Lib.py``, which caps the loaded list.
    LIBRARY_FTS_THRESHOLD: int = 1000

    # Escape/Ctrl+S deliberately do NOT use priority=True: on Textual 8.2.7
    # neither Input nor TextArea (with the default tab_behavior="focus")
    # consumes those keys, so they bubble from the editor fields to this
    # screen's bindings; the actions are strict no-ops outside their contexts.
    # Esc inside an editor TextArea is cancel-with-guard by design (the
    # unsaved-changes dialog prompts before anything is discarded).
    BINDINGS = [
        *BaseAppScreen.BINDINGS,
        Binding("f6", "focus_next_workbench_pane", "Next pane", show=False, priority=True),
        Binding(
            "shift+f6",
            "focus_previous_workbench_pane",
            "Previous pane",
            show=False,
            priority=True,
        ),
        Binding("ctrl+n", "personas_new", "New"),
        Binding("ctrl+f", "personas_search", "Search"),
        Binding("ctrl+enter", "personas_attach", "Attach"),
        Binding("ctrl+s", "personas_save", "Save", show=False),
        Binding("escape", "personas_escape", "Back", show=False),
        # Ctrl+1..5 mirror the mode strip order (MODE_CHIP_ORDER).
        *[
            Binding(
                f"ctrl+{index + 1}",
                f"personas_mode('{mode}')",
                MODE_LABELS.get(mode, mode),
                show=False,
            )
            for index, mode in enumerate(MODE_CHIP_ORDER)
        ],
        # [ / ] cycle the mode strip. They are printable keys, so text widgets
        # consume them as input first; they only act from list/button focus.
        Binding("left_square_bracket", "personas_mode_cycle(-1)", "Prev mode", show=False),
        Binding("right_square_bracket", "personas_mode_cycle(1)", "Next mode", show=False),
    ]
    _WORKBENCH_FOCUS_TARGETS = (
        WorkbenchPaneTarget(
            "personas-library-rail-handle",
            ("personas-library-rail-open",),
        ),
        WorkbenchPaneTarget(
            "personas-library-pane",
            (
                "personas-library-search",
                "personas-library-rail-collapse",
                "personas-library-new",
            ),
        ),
        WorkbenchPaneTarget(
            "personas-work-area",
            ("personas-preview-input", "personas-preview-toggle"),
        ),
        WorkbenchPaneTarget(
            "personas-inspector-pane",
            (
                "personas-conversations-list",
                "personas-inspector-rail-collapse",
                "personas-attach-to-console",
            ),
        ),
        WorkbenchPaneTarget(
            "personas-inspector-rail-handle",
            ("personas-inspector-rail-open",),
        ),
    )

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
        text-style: bold underline;
    }

    #personas-workbench {
        height: 1fr;
        min-height: 20;
        padding: 1;
    }

    #personas-workbench .destination-workbench-pane {
        min-width: 0;
        height: 100%;
        min-height: 18;
        padding: 0 1;
    }

    #personas-library-pane {
        width: 2fr;
    }

    #personas-work-area {
        width: 4fr;
        /* The center card is the workbench's primary surface; keep it from
           collapsing to an unusable sliver when the window narrows (~80 col).
           A full responsive pass (collapsing a side pane, etc.) is tracked
           separately. */
        min-width: 40;
    }

    #personas-inspector-pane {
        width: 2fr;
    }

    #personas-workbench.personas-workbench-compact {
        padding: 0;
    }

    #personas-library-pane.personas-workbench-compact-pane {
        width: 1fr;
        min-width: 16;
        padding: 0 1;
    }

    #personas-work-area.personas-workbench-compact-pane {
        width: 3fr;
        min-width: 34;
        padding: 0 1;
    }

    #personas-inspector-pane.personas-workbench-compact-pane {
        width: 1fr;
        min-width: 22;
        padding: 0 1;
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

    #personas-library-pane ListItem.personas-library-row {
        width: 100%;
        min-width: 0;
        height: 1;
        min-height: 1;
        padding: 0 1;
        border: none;
    }

    #personas-conversation-actions {
        height: 3;
        min-height: 3;
        width: 100%;
    }

    #personas-conversation-actions Button {
        width: auto;
        min-width: 0;
        height: 3;
        margin-right: 1;
    }
    """

    def __init__(self, app_instance: Any, **kwargs: Any) -> None:
        super().__init__(app_instance, "personas", **kwargs)
        self.state = PersonasWorkbenchState()
        self._edit_mode: str = "view"
        self._guard_active: bool = False
        # Refuse-reentry flag for the import/export file dialogs. Cancelling
        # an in-flight dialog worker (exclusive=True) would orphan a modal
        # pushed via push_screen_wait, whose dismissal then calls
        # set_result on a cancelled future (InvalidStateError on Textual
        # 8.2.7), so a second request is ignored instead.
        self._io_dialog_active: bool = False
        # Same refuse-reentry idiom for the delete confirmation dialog.
        self._delete_dialog_active: bool = False
        self._character_editor_generation: int = 0
        self._profile_save_inflight: bool = False
        self._characters: list[dict] = []
        self._profiles: list[dict] = []
        self._dictionaries_cache: list[dict] = []
        self._selected_dictionary_version: int | None = None
        self._profile_lookup_recovery_state: DestinationRecoveryState | None = None
        self._search_debounce_timer: Timer | None = None
        # Serializes library renders: the pane's update_rows has two
        # suspension points, so interleaved renders could double-mount rows.
        self._render_lock = asyncio.Lock()
        self._workbench_compact: bool | None = None
        self._library_rail_collapsed: bool = False
        self._inspector_rail_collapsed: bool = False
        self.character_handler = CCPCharacterHandler(self)
        self.persona_handler = CCPPersonaHandler(self)
        self.conversations = PersonasConversationsController(self)
        self.preview = PersonasPreviewController(self)
        setup_ccp_enhancements(self)

    # ===== Compose =====

    def compose_content(self) -> ComposeResult:
        """Compose the Personas destination shell and workbench rails.

        Returns:
            Textual compose result for the Personas content tree.
        """
        with Vertical(id="personas-shell"):
            yield Static(
                self._title_text(),
                id="personas-title",
                classes="ds-destination-header",
            )
            yield Static(
                self._mode_descriptor_text(self.state.active_mode),
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
                    label = MODE_LABELS.get(mode, mode)
                    if mode in _COMING_SOON_MODES:
                        label = f"{label} · soon"
                    yield Button(
                        label,
                        id=f"personas-mode-{mode}",
                        classes=classes,
                        tooltip=self._mode_descriptor_text(mode),
                    )
            with Horizontal(id="personas-workbench", classes="ds-panel destination-workbench"):
                library_handle = ConsoleRailHandle(
                    label="Library",
                    button_id="personas-library-rail-open",
                    badge_id="personas-library-rail-badge",
                    side="left",
                    id="personas-library-rail-handle",
                )
                library_handle.styles.width = PERSONAS_LIBRARY_RAIL_HANDLE_WIDTH
                library_handle.styles.min_width = PERSONAS_LIBRARY_RAIL_HANDLE_WIDTH
                library_handle.styles.max_width = PERSONAS_LIBRARY_RAIL_HANDLE_WIDTH
                if not self._library_rail_collapsed:
                    library_handle.display = False
                yield library_handle

                library_pane = PersonasLibraryPane(
                    id="personas-library-pane",
                    classes="destination-workbench-pane",
                )
                if self._library_rail_collapsed:
                    library_pane.display = False
                yield library_pane

                with Vertical(id="personas-work-area", classes="destination-workbench-pane"):
                    with Container(id="personas-detail-stack"):
                        yield PersonasCharacterCardWidget()
                        yield PersonasCharacterEditorWidget()
                        yield PersonaProfileCardWidget()
                        yield PersonaProfileEditorWidget()
                        with Horizontal(id="personas-conversation-actions"):
                            yield Button("Back to card", id="personas-conversation-back")
                            yield Button(
                                "Continue in Console",
                                id="personas-conversation-continue-console",
                            )
                            yield Button(
                                "Open in Library",
                                id="personas-conversation-open-library",
                            )
                        yield PersonasDictionaryDetailWidget(id="personas-dictionary-detail")
                        yield PersonasConversationTranscriptWidget()
                        yield Static(self._mode_placeholder_text("lore"), id="personas-mode-placeholder")
                    yield PersonasPreviewPane(id="personas-preview-pane")

                inspector_pane = PersonasInspectorPane(
                    id="personas-inspector-pane",
                    classes="destination-workbench-pane ds-inspector",
                )
                if self._inspector_rail_collapsed:
                    inspector_pane.display = False
                yield inspector_pane

                inspector_handle = ConsoleRailHandle(
                    label="Inspector",
                    button_id="personas-inspector-rail-open",
                    badge_id="personas-inspector-rail-badge",
                    side="right",
                    id="personas-inspector-rail-handle",
                )
                inspector_handle.styles.width = PERSONAS_INSPECTOR_RAIL_HANDLE_WIDTH
                inspector_handle.styles.min_width = PERSONAS_INSPECTOR_RAIL_HANDLE_WIDTH
                inspector_handle.styles.max_width = PERSONAS_INSPECTOR_RAIL_HANDLE_WIDTH
                if not self._inspector_rail_collapsed:
                    inspector_handle.display = False
                yield inspector_handle

    async def on_mount(self) -> None:
        super().on_mount()
        loading_manager = getattr(self, "loading_manager", None)
        setup_loading = getattr(loading_manager, "setup", None)
        if callable(setup_loading):
            await setup_loading()
        self._sync_responsive_workbench()
        self._sync_personas_rails()
        self._sync_personas_rail_tooltips()
        self.query_one(PersonasLibraryPane).set_mode(self.state.active_mode)
        self._show_center(None)
        await self.character_handler.refresh_character_list()
        self._register_footer_shortcuts()

    async def on_unmount(self) -> None:
        super().on_unmount()
        self._cancel_search_debounce()
        self._clear_footer_shortcuts()
        await self.preview.close_gateway()

    def on_resize(self, event: Any) -> None:
        """Refresh compact workbench classes after terminal size changes.

        Args:
            event: Textual resize event emitted when the screen size changes.
        """
        self._sync_responsive_workbench()

    def _sync_responsive_workbench(self) -> None:
        compact = self.size.width <= PERSONAS_COMPACT_WORKBENCH_MAX_WIDTH
        if self._workbench_compact == compact:
            return
        try:
            workbench = self.query_one("#personas-workbench")
        except QueryError:
            return
        self._workbench_compact = compact
        workbench.set_class(compact, "personas-workbench-compact")
        for pane_id in (
            "#personas-library-pane",
            "#personas-work-area",
            "#personas-inspector-pane",
        ):
            try:
                self.query_one(pane_id).set_class(compact, "personas-workbench-compact-pane")
            except QueryError:
                continue

    def _sync_personas_rails(self) -> None:
        """Mirror Console/Notes collapsible rails for Library and Inspector."""
        if not self.is_mounted:
            return
        try:
            library_open = not self._library_rail_collapsed
            inspector_open = not self._inspector_rail_collapsed
            self.query_one("#personas-library-pane").display = library_open
            self.query_one("#personas-library-rail-handle").display = not library_open
            self.query_one("#personas-inspector-pane").display = inspector_open
            self.query_one("#personas-inspector-rail-handle").display = not inspector_open
        except QueryError:
            return

    def _sync_personas_rail_tooltips(self) -> None:
        """Set Personas-specific collapsed rail tooltips on shared handles."""
        try:
            self.query_one("#personas-library-rail-open", Button).tooltip = (
                "Open Library rail"
            )
            self.query_one("#personas-inspector-rail-open", Button).tooltip = (
                "Open Inspector rail"
            )
        except QueryError:
            return

    # ===== Library rendering =====

    async def refresh_character_library_list(self, characters: list[dict] | None) -> None:
        """Destination-native hook called by ``CCPCharacterHandler``."""
        self._characters = [dict(record) for record in (characters or [])]
        self._update_status_row()
        if self.state.active_mode != "characters":
            return
        try:
            await self._render_library_rows()
        except Exception:
            # Tolerate refreshes that race screen teardown.
            logger.opt(exception=True).warning("Could not render the character library rows.")

    @staticmethod
    def _build_library_rows(records: list[dict], kind: str) -> tuple[LibraryRow, ...]:
        """Map id/name records onto library rows, skipping id-less records."""
        return tuple(
            LibraryRow(
                item_id=str(record.get("id")),
                kind=kind,
                name=str(record.get("name") or "Unnamed"),
            )
            for record in records
            if record.get("id") is not None
        )

    def _library_render_snapshot_is_current(
        self,
        *,
        expected_query: str | None = None,
        expected_mode: str | None = None,
    ) -> bool:
        """Return whether a delayed library render still matches live state."""

        has_snapshot = expected_query is not None or expected_mode is not None
        if has_snapshot and not self.is_mounted:
            return False
        if expected_mode is not None and expected_mode != self.state.active_mode:
            return False
        if expected_query is not None and expected_query != self.state.search_query:
            return False
        return True

    async def _render_library_rows(
        self,
        *,
        expected_query: str | None = None,
        expected_mode: str | None = None,
    ) -> None:
        if not self._library_render_snapshot_is_current(
            expected_query=expected_query,
            expected_mode=expected_mode,
        ):
            return

        query = expected_query if expected_query is not None else self.state.search_query
        total = len(self._characters)
        filtered_total_unbounded = False
        if query:
            if total >= self.LIBRARY_FTS_THRESHOLD:
                # Large library: use FTS so the full DB corpus is searched
                # even when the loaded list is a page-size truncation. The
                # query runs in a thread so the DB call never blocks the UI
                # loop (the render lock below is only taken afterwards, so
                # the await cannot deadlock it).
                matched = await asyncio.to_thread(
                    ccp_character_handler.search_characters_fts, query
                )
                filtered_total_unbounded = True
            else:
                # Small library: filter in-memory, case-insensitively on name.
                q_lower = query.lower()
                matched = [r for r in self._characters if q_lower in str(r.get("name") or "").lower()]
            filtered = True
        else:
            matched = self._characters
            filtered = False
        if not self._library_render_snapshot_is_current(
            expected_query=expected_query,
            expected_mode=expected_mode,
        ):
            return
        async with self._render_lock:
            if not self._library_render_snapshot_is_current(
                expected_query=expected_query,
                expected_mode=expected_mode,
            ):
                return
            rows = self._build_library_rows(matched, "character")
            library = self.query_one(PersonasLibraryPane)
            await library.update_rows(
                rows,
                total=total,
                noun="characters",
                filtered=filtered,
                filtered_total_unbounded=filtered_total_unbounded,
            )
            if self.state.selected_entity_kind == "character" and self.state.selected_entity_id:
                library.mark_active_row("character", self.state.selected_entity_id)

    def _character_record(self, item_id: str | None) -> dict | None:
        if item_id is None:
            return None
        for record in self._characters:
            if str(record.get("id")) == str(item_id):
                return record
        return None

    def _profile_list_recovery_state(self, exc: Exception) -> DestinationRecoveryState:
        """Build recovery copy when persona profile listing is unavailable."""

        reason = str(exc).strip() or "The current backend did not return persona profiles."
        disabled_tooltip = (
            f"{reason} Retry Personas or use Characters until persona profiles are available."
        )
        return DestinationRecoveryState(
            status_label="Persona profiles unavailable",
            unavailable_what="Browse persona profiles in Personas",
            why=reason,
            next_action=(
                "Check the current runtime backend or retry after persona profile support is available"
            ),
            recovery_action="Retry Personas or use Characters",
            authority_owner="persona scope service",
            stable_selector="personas-service-error",
            disabled_tooltip=disabled_tooltip,
        )

    @work(exclusive=True, group="personas-list-refresh")
    async def _refresh_profile_rows_worker(self) -> None:
        """Fetch persona profile rows and render them while still in Personas mode."""
        try:
            profiles = await self.persona_handler.refresh_persona_list(
                raise_on_unavailable=True
            )
        except Exception as exc:
            logger.opt(exception=True).warning("Could not refresh the persona profile list.")
            self._profile_lookup_recovery_state = self._profile_list_recovery_state(exc)
            profiles = []
        else:
            self._profile_lookup_recovery_state = None
        self._profiles = [dict(record) for record in (profiles or [])]
        self._update_status_row()
        if not self.is_mounted or self.state.active_mode != "personas":
            # A late result must not render persona rows into another mode.
            return
        try:
            await self._render_profile_rows()
        except Exception:
            # Tolerate refreshes that race screen teardown.
            logger.opt(exception=True).warning("Could not render the persona profile rows.")

    async def _render_profile_rows(
        self,
        *,
        expected_query: str | None = None,
        expected_mode: str | None = None,
    ) -> None:
        if not self._library_render_snapshot_is_current(
            expected_query=expected_query,
            expected_mode=expected_mode,
        ):
            return

        query = expected_query if expected_query is not None else self.state.search_query
        total = len(self._profiles)
        if query:
            q_lower = query.lower()
            matched = [r for r in self._profiles if q_lower in str(r.get("name") or "").lower()]
            filtered = True
        else:
            matched = self._profiles
            filtered = False
        async with self._render_lock:
            if not self._library_render_snapshot_is_current(
                expected_query=expected_query,
                expected_mode=expected_mode,
            ):
                return
            rows = self._build_library_rows(matched, "persona_profile")
            library = self.query_one(PersonasLibraryPane)
            recovery_state = self._profile_lookup_recovery_state
            await library.update_rows(
                rows,
                total=total,
                noun="persona profiles",
                filtered=filtered,
                recovery_copy=(
                    recovery_state.visible_copy if recovery_state is not None else None
                ),
                recovery_id=(
                    recovery_state.stable_selector
                    if recovery_state is not None
                    else "personas-library-recovery"
                ),
            )
            if self.state.selected_entity_kind == "persona_profile" and self.state.selected_entity_id:
                library.mark_active_row("persona_profile", self.state.selected_entity_id)

    @on(PersonaSearchChanged)
    def _handle_search_changed(self, message: PersonaSearchChanged) -> None:
        message.stop()
        # Search does not change selection or center pane — no unsaved guard needed.
        self.state.search_query = message.query.strip()
        self._cancel_search_debounce()
        query = self.state.search_query
        mode = self.state.active_mode
        self._search_debounce_timer = self.set_timer(
            PERSONAS_SEARCH_DEBOUNCE_SECONDS,
            lambda: self._start_debounced_search_render(query=query, mode=mode),
        )

    def _cancel_search_debounce(self) -> None:
        """Cancel a pending search render when newer state supersedes it."""

        if self._search_debounce_timer is not None:
            self._search_debounce_timer.stop()
            self._search_debounce_timer = None

    def _start_debounced_search_render(self, *, query: str, mode: str) -> None:
        """Start the debounced render after the active timer has fired."""

        self._search_debounce_timer = None
        self.run_worker(
            self._render_search_query(query=query, mode=mode),
            exclusive=True,
            group="personas-library-search",
        )

    async def _render_search_query(self, *, query: str, mode: str) -> None:
        """Render the latest debounced library search for the active mode."""

        if not self._library_render_snapshot_is_current(
            expected_query=query,
            expected_mode=mode,
        ):
            return
        if mode == "characters":
            try:
                await self._render_library_rows(
                    expected_query=query,
                    expected_mode=mode,
                )
            except Exception:
                logger.opt(exception=True).warning("Could not re-render character rows after search.")
        elif mode == "personas":
            try:
                await self._render_profile_rows(
                    expected_query=query,
                    expected_mode=mode,
                )
            except Exception:
                logger.opt(exception=True).warning("Could not re-render profile rows after search.")
        elif mode == "dictionaries":
            try:
                await self._render_dictionary_rows(query=query)
            except Exception:
                logger.opt(exception=True).warning("Could not re-render dictionary rows after search.")

    def _profile_record(self, item_id: str | None) -> dict | None:
        if item_id is None:
            return None
        for record in self._profiles:
            if str(record.get("id")) == str(item_id):
                return record
        return None

    async def _fetch_profile_record(self, persona_id: str) -> dict:
        """Full profile from the scope service; falls back to the cached list row."""
        record, _complete = await self._fetch_profile_record_checked(persona_id)
        return record

    async def _fetch_profile_record_checked(self, persona_id: str) -> tuple[dict, bool]:
        """Full profile plus whether it actually came from the scope service.

        Returns ``(record, complete)``. ``complete`` is ``True`` only for a
        service-backed record; the cached list-row / bare-id fallback is a
        summary (no ``system_prompt``), so callers that need the full card
        (e.g. Console handoffs) must check the flag, while display paths can
        keep showing the partial record.
        """
        fallback = dict(self._profile_record(persona_id) or {"id": persona_id})
        service = getattr(self.app_instance, "character_persona_scope_service", None)
        if service is None or not hasattr(service, "get_persona_profile"):
            return fallback, False
        try:
            record = await service.get_persona_profile(
                persona_id, mode=self.persona_handler.current_mode()
            )
        except Exception:
            logger.opt(exception=True).warning(
                f"Could not fetch persona profile {persona_id}; using the list row.",
            )
            return fallback, False
        if hasattr(record, "model_dump"):
            record = record.model_dump(mode="json")
        if not isinstance(record, dict):
            return fallback, False
        return dict(record), True

    def _dictionary_scope_service(self) -> Any:
        """The app-level dictionaries scope service, or None when absent."""
        return getattr(self.app_instance, "chat_dictionary_scope_service", None)

    @staticmethod
    def _dictionary_row(record: dict) -> LibraryRow:
        entries = record.get("entries") or []
        state = "on" if record.get("enabled", record.get("is_active", True)) else "off"
        return LibraryRow(
            item_id=str(record.get("id")),
            kind="dictionary",
            name=str(record.get("name") or "Unnamed"),
            meta=f"{len(entries)} entries · {state}",
        )

    async def _render_dictionary_rows(self, query: str = "") -> None:
        """Fetch and render dictionary rows; degrade to recovery copy on failure."""
        library = self.query_one(PersonasLibraryPane)
        service = self._dictionary_scope_service()
        if service is None:
            await library.update_rows(
                (), total=0, noun="dictionaries",
                recovery_copy="Dictionaries are unavailable: the service is not configured.",
            )
            return
        try:
            response = await service.list_dictionaries(mode="local", include_inactive=True)
            records = list(response.get("dictionaries") or [])
        except Exception:
            logger.opt(exception=True).warning("Could not list chat dictionaries.")
            await library.update_rows(
                (), total=0, noun="dictionaries",
                recovery_copy="Dictionaries could not be loaded.\nSwitch modes and back to retry.",
            )
            return
        self._dictionaries_cache = records
        needle = query.strip().lower()
        visible = [r for r in records if needle in str(r.get("name", "")).lower()] if needle else records
        rows = tuple(self._dictionary_row(r) for r in visible)
        await library.update_rows(
            rows, total=len(records), noun="dictionaries", filtered=bool(needle),
        )

    # ===== Mode switching =====

    @on(Button.Pressed, ".personas-mode-chip")
    async def _handle_mode_chip(self, event: Button.Pressed) -> None:
        event.stop()
        mode = str(event.button.id or "").removeprefix("personas-mode-")
        if mode not in MODE_CHIP_ORDER or mode == self.state.active_mode:
            return
        await self._run_guarded(lambda: self._apply_mode(mode))

    @on(Button.Pressed, "#personas-library-rail-collapse")
    def _handle_library_rail_collapse(self, event: Button.Pressed) -> None:
        event.stop()
        self._library_rail_collapsed = True
        self._sync_personas_rails()

    @on(Button.Pressed, "#personas-library-rail-open")
    def _handle_library_rail_open(self, event: Button.Pressed) -> None:
        event.stop()
        self._library_rail_collapsed = False
        self._sync_personas_rails()

    @on(Button.Pressed, "#personas-inspector-rail-collapse")
    def _handle_inspector_rail_collapse(self, event: Button.Pressed) -> None:
        event.stop()
        self._inspector_rail_collapsed = True
        self._sync_personas_rails()

    @on(Button.Pressed, "#personas-inspector-rail-open")
    def _handle_inspector_rail_open(self, event: Button.Pressed) -> None:
        event.stop()
        self._inspector_rail_collapsed = False
        self._sync_personas_rails()

    async def _apply_mode(self, mode: str) -> None:
        self._cancel_search_debounce()
        self.state.switch_mode(mode)
        # switch_mode does not reset search_query; clear it explicitly and
        # reset the Input widget so the library starts unfiltered in the new mode.
        self.state.search_query = ""
        try:
            self.query_one("#personas-library-search", Input).value = ""
        except Exception:
            pass
        self._edit_mode = "view"
        for chip_mode in MODE_CHIP_ORDER:
            self.query_one(f"#personas-mode-{chip_mode}", Button).set_class(
                chip_mode == mode, "is-active"
            )
        self.query_one("#personas-status-row", Static).update(self._status_row_text())
        self.query_one("#personas-purpose", Static).update(self._mode_descriptor_text(mode))
        library = self.query_one(PersonasLibraryPane)
        library.set_mode(mode)
        # clear_selection empties the conversations panel; drop the caches too.
        self.conversations.reset()
        await self.preview.reset("")
        await self.query_one(PersonasInspectorPane).clear_selection()
        if mode == "characters":
            await self._render_library_rows()
            self._show_center(None)
        elif mode == "personas":
            self._profile_lookup_recovery_state = None
            await library.update_rows((), total=0, noun="persona profiles")
            self._show_center(None)
            self._refresh_profile_rows_worker()
        elif mode == "dictionaries":
            await self._render_dictionary_rows()
            self._show_center(None)
        else:
            await library.update_rows((), total=0, noun=MODE_LABELS.get(mode, mode).lower())
            self.query_one("#personas-mode-placeholder", Static).update(self._mode_placeholder_text(mode))
            self._show_center("#personas-mode-placeholder")

    def _title_text(self) -> str:
        """Live header line: destination identity plus the editing state.

        "Local" deliberately stays out of the title - the status row directly
        below already says "Source: Local" (de-dup, P3-15).
        """
        base = "Roleplay | Author the pieces that shape a chat"
        suffix = " - unsaved" if self.state.has_unsaved_changes else ""
        if self._edit_mode == "create":
            noun = "persona" if self.state.active_mode == "personas" else "character"
            return f"{base} | New {noun}{suffix}"
        if self._edit_mode == "edit":
            name = self.state.selected_entity_name or "item"
            return f"{base} | Editing {name}{suffix}"
        if self.state.has_unsaved_changes and self.state.selected_entity_name:
            return f"{base} | {self.state.selected_entity_name}{suffix}"
        return f"{base} | Ready"

    def _mode_descriptor_text(self, mode: str) -> str:
        """The visible one-line meaning of a mode (falls back for un-described modes)."""
        return _MODE_DESCRIPTORS.get(mode, MODE_LABELS.get(mode, mode))

    def _mode_placeholder_text(self, mode: str) -> str:
        """The inviting placeholder body for a not-yet-built (or departing) mode."""
        return _MODE_PLACEHOLDER_BODY.get(mode, _PLACEHOLDER_FALLBACK)

    def _update_title(self) -> None:
        """Refresh the header line; tolerate updates racing teardown."""
        try:
            self.query_one("#personas-title", Static).update(self._title_text())
        except Exception:
            logger.opt(exception=True).debug("Could not update the personas title.")

    def _status_row_text(self) -> str:
        mode = self.state.active_mode
        if mode == "characters":
            return f"Characters: {len(self._characters)} | Source: Local | Attachments: Console"
        if mode == "personas":
            return f"Personas: {len(self._profiles)} | Source: Local | Attachments: Console"
        return f"Mode: {MODE_LABELS.get(mode, mode)} | Source: Local | Attachments: Console"

    def _update_status_row(self) -> None:
        """Refresh the status row text; tolerate refreshes racing teardown."""
        try:
            self.query_one("#personas-status-row", Static).update(self._status_row_text())
        except Exception:
            logger.opt(exception=True).debug("Could not update the personas status row.")

    # ===== Selection =====

    @on(PersonaEntitySelected)
    async def _handle_entity_selected(self, message: PersonaEntitySelected) -> None:
        message.stop()
        if message.entity_kind == "character":
            await self._run_guarded(
                lambda: self._select_character(message.entity_id, message.entity_name)
            )
        elif message.entity_kind == "persona_profile":
            await self._run_guarded(
                lambda: self._select_profile(message.entity_id, message.entity_name)
            )
        elif message.entity_kind == "dictionary":
            await self._run_guarded(
                lambda: self._select_dictionary(message.entity_id, message.entity_name)
            )
        # Prompts and lore are wired in follow-up tasks.

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
        inspector.show_validation(())
        self._sync_inspector_console_actions()
        # Drop any previous character's rows immediately and show a loading
        # placeholder; the worker fills the panel in once the listing returns
        # (or replaces the placeholder with the empty-state copy).
        self.conversations.reset()
        await inspector.show_conversations_loading()
        self.conversations.load_conversations(entity_id)
        # Seed the ephemeral preview with the character's greeting. The list
        # rows are id/name-only summaries and load_character only SCHEDULES a
        # thread worker, so the full record (with first_message) is usually
        # not available yet here. Instant path: when the handler already holds
        # this character's full card (re-selection), seed now; otherwise clear
        # the preview and let the CharacterMessage.Loaded handler seed it.
        record = self._full_character_record(entity_id)
        await self.preview.reset_for_character(
            character_id=entity_id,
            character_name=entity_name,
            record=record,
        )

    async def _select_profile(self, entity_id: str, entity_name: str) -> None:
        self.state.select_entity(
            entity_kind="persona_profile",
            entity_id=entity_id,
            entity_name=entity_name,
        )
        self._edit_mode = "view"
        self.query_one(PersonasLibraryPane).mark_active_row("persona_profile", entity_id)
        record = await self._fetch_profile_record(entity_id)
        self.query_one(PersonaProfileCardWidget).show_persona(record)
        self._show_center("#ccp-persona-card-view")
        inspector = self.query_one(PersonasInspectorPane)
        inspector.show_selection(name=entity_name, kind="persona_profile", authority="Local")
        inspector.set_unsaved(False)
        inspector.show_validation(())
        self._sync_inspector_console_actions()
        # Persona profiles have no conversation linkage in the local data.
        self.conversations.reset()
        await inspector.show_conversations(())
        # Profiles have no first_message concept; start the preview empty.
        await self.preview.reset("")

    async def _select_dictionary(self, entity_id: str, entity_name: str) -> None:
        """Load one dictionary into the center detail; inspector shows the selection."""
        service = self._dictionary_scope_service()
        if service is None:
            self._notify("Dictionaries service is not configured.", "error")
            return
        try:
            record = await service.get_dictionary(int(entity_id), mode="local")
        except Exception as exc:
            logger.opt(exception=True).warning(f"Could not load dictionary {entity_id}.")
            self._notify(f"Could not load dictionary: {exc}", "error")
            return
        self._edit_mode = "view"
        self.state.has_unsaved_changes = False
        raw_version = record.get("version")
        self._selected_dictionary_version = int(raw_version) if raw_version is not None else None
        self.state.select_entity(
            entity_kind="dictionary", entity_id=entity_id, entity_name=entity_name
        )
        detail = self.query_one(PersonasDictionaryDetailWidget)
        detail.load_dictionary(record)
        self._show_center("#personas-dictionary-detail")
        library = self.query_one(PersonasLibraryPane)
        library.mark_active_row("dictionary", entity_id)
        inspector = self.query_one(PersonasInspectorPane)
        inspector.show_selection(name=entity_name, kind="Dictionary", authority="Local")
        self._sync_inspector_console_actions()
        self._update_title()
        self._update_status_row()

    @on(DictionarySettingsEdited)
    def _handle_dictionary_settings_edited(self, message: DictionarySettingsEdited) -> None:
        message.stop()
        if self.state.selected_entity_kind != "dictionary":
            return
        if not self.state.has_unsaved_changes:
            self.state.has_unsaved_changes = True
            self._update_title()
            self._sync_inspector_console_actions()

    @on(DictionarySettingsSaveRequested)
    async def _handle_dictionary_settings_save(self, message: DictionarySettingsSaveRequested) -> None:
        message.stop()
        if self.state.selected_entity_kind != "dictionary" or not self.state.selected_entity_id:
            return
        detail = self.query_one(PersonasDictionaryDetailWidget)
        payload = dict(message.payload)
        if not payload.get("name"):
            detail.set_status("A name is required.")
            return
        service = self._dictionary_scope_service()
        if service is None:
            self._notify("Dictionaries service is not configured.", "error")
            return
        entity_id = self.state.selected_entity_id
        try:
            record = await service.update_dictionary(
                int(entity_id), payload, mode="local",
                expected_version=self._selected_dictionary_version,
            )
        except ConflictError:
            detail.set_status(
                "Save failed: the dictionary changed since it was loaded. Reselect and try again."
            )
            return
        except Exception as exc:
            logger.opt(exception=True).warning(f"Could not save dictionary {entity_id}.")
            detail.set_status(f"Save failed: {exc}")
            return
        raw_version = record.get("version")
        self._selected_dictionary_version = int(raw_version) if raw_version is not None else None
        self.state.has_unsaved_changes = False
        self.state.selected_entity_name = str(record.get("name") or "")
        detail.load_dictionary(record)
        detail.set_status("Saved.")
        self._update_title()
        await self._render_dictionary_rows(query=self.state.search_query)
        self.query_one(PersonasLibraryPane).mark_active_row("dictionary", entity_id)
        self._sync_inspector_console_actions()

    async def _reload_selected_dictionary_entries(self) -> None:
        """Re-fetch entries + version after a mutation (positional ids shift)."""
        entity_id = self.state.selected_entity_id
        service = self._dictionary_scope_service()
        if service is None or self.state.selected_entity_kind != "dictionary" or not entity_id:
            return
        detail = self.query_one(PersonasDictionaryDetailWidget)
        try:
            response = await service.list_entries(int(entity_id), mode="local")
            record = await service.get_dictionary(int(entity_id), mode="local")
        except Exception as exc:
            logger.opt(exception=True).warning(f"Could not reload dictionary {entity_id} entries.")
            detail.set_status(f"Reload failed: {exc}")
            return
        raw_version = record.get("version")
        self._selected_dictionary_version = int(raw_version) if raw_version is not None else None
        detail.update_entries(list(response.get("entries") or []))
        await self._render_dictionary_rows(query=self.state.search_query)
        self.query_one(PersonasLibraryPane).mark_active_row("dictionary", entity_id)

    async def _run_dictionary_entry_op(self, op: Callable[[Any], Awaitable[Any]], failure: str) -> None:
        """One guarded service mutation + the mandatory entries reload."""
        service = self._dictionary_scope_service()
        detail = self.query_one(PersonasDictionaryDetailWidget)
        if service is None or self.state.selected_entity_kind != "dictionary":
            return
        try:
            await op(service)
        except ConflictError:
            detail.set_status(
                "Change failed: the dictionary changed since it was loaded. Reselect and try again."
            )
            return
        except Exception as exc:
            logger.opt(exception=True).warning(failure)
            detail.set_status(f"{failure}: {exc}")
            return
        await self._reload_selected_dictionary_entries()
        detail.set_status("")

    @on(DictionaryEntryAddRequested)
    async def _handle_dictionary_entry_add(self, message: DictionaryEntryAddRequested) -> None:
        message.stop()
        entity_id = self.state.selected_entity_id
        if not entity_id:
            return
        await self._run_dictionary_entry_op(
            lambda service: service.add_entry(int(entity_id), message.payload, mode="local"),
            "Could not add the entry",
        )

    @on(DictionaryEntryUpdateRequested)
    async def _handle_dictionary_entry_update(self, message: DictionaryEntryUpdateRequested) -> None:
        message.stop()
        await self._run_dictionary_entry_op(
            lambda service: service.update_entry(message.entry_id, message.payload, mode="local"),
            "Could not update the entry",
        )

    @on(DictionaryEntryDeleteRequested)
    async def _handle_dictionary_entry_delete(self, message: DictionaryEntryDeleteRequested) -> None:
        message.stop()
        await self._run_dictionary_entry_op(
            lambda service: service.delete_entry(message.entry_id, mode="local"),
            "Could not delete the entry",
        )

    @on(DictionaryEntriesReorderRequested)
    async def _handle_dictionary_entries_reorder(self, message: DictionaryEntriesReorderRequested) -> None:
        message.stop()
        entity_id = self.state.selected_entity_id
        if not entity_id:
            return
        await self._run_dictionary_entry_op(
            lambda service: service.reorder_entries(
                int(entity_id), {"entry_ids": list(message.entry_ids)}, mode="local"
            ),
            "Could not reorder entries",
        )

    # ===== Saved conversations =====

    def _character_db(self) -> Any:
        """Local character DB, via the same resolver import/export uses."""
        return ccp_character_handler._default_character_db()

    @on(ConversationRowSelected)
    async def _handle_conversation_row_selected(self, message: ConversationRowSelected) -> None:
        message.stop()
        if (
            self.state.active_mode != "characters"
            or self.state.selected_entity_kind != "character"
        ):
            return
        conversation_id = str(message.conversation_id)
        await self._run_guarded(
            lambda: self.conversations.open_conversation(conversation_id)
        )

    @on(Button.Pressed, "#personas-conversation-back")
    def _handle_conversation_back(self, event: Button.Pressed) -> None:
        event.stop()
        self._show_center("#ccp-character-card-view")
        self._register_footer_shortcuts()
        self._focus_conversations_list()

    @on(Button.Pressed, "#personas-conversation-open-library")
    def _handle_conversation_open_library(self, event: Button.Pressed) -> None:
        event.stop()
        self.conversations.open_in_library()

    @on(Button.Pressed, "#personas-conversation-continue-console")
    def _handle_conversation_continue_console(self, event: Button.Pressed) -> None:
        event.stop()
        self.conversations.continue_in_console()

    # ===== Console handoff (attach / start chat) =====

    def _stage_handoff(
        self,
        *,
        item_type: str,
        title: str,
        body: str,
        body_truncated: bool = False,
        source_id: str | None = None,
        suggested_prompt: str = "",
        display_summary: str = "",
        extra_metadata: dict | None = None,
    ) -> bool:
        """Single seam for staging Personas context into Console.

        Builds a ``ChatHandoffPayload`` with the workbench's fixed
        source/runtime identity ("personas" / local) and the selection
        metadata from ``PersonasWorkbenchState.selected_metadata()``, then
        hands it to the app's ``open_chat_with_handoff``. Returns ``True``
        when a payload was staged.
        """
        open_handoff = getattr(self.app_instance, "open_chat_with_handoff", None)
        if not callable(open_handoff):
            self._notify("Console handoff is unavailable.", "warning")
            return False
        payload = ChatHandoffPayload.from_source_content(
            source="personas",
            item_type=item_type,
            title=title,
            body=body,
            body_truncated=body_truncated,
            source_id=source_id,
            suggested_prompt=suggested_prompt,
            display_summary=display_summary,
            runtime_backend="local",
            source_owner="local",
            source_selector_state="local",
            metadata={
                **self.state.selected_metadata(),
                "backend": "local",
                **(extra_metadata or {}),
            },
        )
        open_handoff(payload)
        return True

    def _console_action_allowed(self) -> bool:
        """True when a saved character/persona profile selection is attachable."""
        return bool(
            self.state.selected_entity_id
            and self.state.selected_entity_kind in ("character", "persona_profile")
            and not self.state.has_unsaved_changes
        )

    def _console_action_block_reason(self) -> str:
        """Return a readable reason for a blocked screen-owned Console action.

        Returns:
            Human-readable block reason suitable for inspector readiness copy.
        """
        if self.state.has_unsaved_changes:
            return "unsaved edits"
        if not self.state.selected_entity_id:
            return "select an item"
        if self.state.selected_entity_kind == "dictionary":
            return "attach arrives in a later update"
        if self.state.selected_entity_kind not in ("character", "persona_profile"):
            return "select a character or persona"
        return "unavailable"

    def _sync_inspector_console_actions(self) -> None:
        """Push the single screen-owned Console gate into the inspector pane."""
        try:
            inspector = self.query_one(PersonasInspectorPane)
        except QueryError:
            return
        allowed = self._console_action_allowed()
        inspector.set_console_actions_enabled(
            allowed,
            reason=None if allowed else self._console_action_block_reason(),
        )

    async def _selection_handoff_body(self) -> str | None:
        """Readable card summary for the selected item, or ``None`` when stale."""
        kind = self.state.selected_entity_kind
        entity_id = str(self.state.selected_entity_id or "")
        name = self.state.selected_entity_name or "Unnamed"
        if kind == "character":
            record = self._full_character_record(entity_id)
            if record is None:
                self._notify("Character data is not loaded yet.", "warning")
                return None
            field_specs = (
                ("Description", "description"),
                ("Personality", "personality"),
                ("Scenario", "scenario"),
                ("System prompt", "system_prompt"),
            )
        else:
            # List rows are summaries; the full record carries system_prompt.
            # A degraded (fallback) record must fail the handoff loudly, the
            # same way a not-yet-loaded character does, rather than staging a
            # body that silently lacks the system prompt.
            record, complete = await self._fetch_profile_record_checked(entity_id)
            if not complete:
                self._notify(
                    "Persona profile is not fully loaded; try reselecting it.",
                    "warning",
                )
                return None
            field_specs = (
                ("Description", "description"),
                ("System prompt", "system_prompt"),
            )
        lines = [f"Name: {name}"]
        for label, key in field_specs:
            value = str(record.get(key) or "").strip()
            if value:
                lines.append(f"{label}: {value}")
        return "\n".join(lines)

    async def _attach_selection_to_console(self, *, intent: str) -> None:
        """Stage the selected card in Console (intent: "attach" or "start_chat")."""
        if not self._console_action_allowed():
            # The inspector disables these buttons without a saved selection;
            # this is a defensive re-check (and the ctrl+enter guard).
            self._notify("Select a saved item before using Console actions.", "warning")
            return
        kind = str(self.state.selected_entity_kind)
        name = self.state.selected_entity_name or "Unnamed"
        body = await self._selection_handoff_body()
        if body is None:
            return
        if intent == "start_chat":
            suggested_prompt = f"Respond as {name}."
            extra_metadata: dict | None = {"intent": "start_chat"}
            display_summary = f"{name} staged to start a Console chat."
            success_copy = "Chat staged in Console."
        else:
            suggested_prompt = f"Use {name} to guide the next response."
            extra_metadata = None
            display_summary = f"{name} ({kind}) staged."
            success_copy = "Staged in Console."
        staged = self._stage_handoff(
            item_type=f"{kind}-card",
            title=f"{name} ({kind})",
            body=body,
            source_id=str(self.state.selected_entity_id or ""),
            suggested_prompt=suggested_prompt,
            display_summary=display_summary,
            extra_metadata=extra_metadata,
        )
        if staged:
            self._notify(success_copy, "information")

    @on(Button.Pressed, "#personas-attach-to-console")
    async def _handle_attach_to_console(self, event: Button.Pressed) -> None:
        event.stop()
        await self._attach_selection_to_console(intent="attach")

    @on(Button.Pressed, "#personas-start-chat")
    async def _handle_start_chat(self, event: Button.Pressed) -> None:
        # The legacy CCP route launched a blank main-chat tab directly via the
        # chat tab container, but that container is only queryable while the
        # chat screen is mounted - never true from a pushed destination
        # screen. The workbench therefore routes Start Chat through the
        # app-level open_chat_with_handoff API with an explicit intent marker.
        event.stop()
        await self._attach_selection_to_console(intent="start_chat")

    @on(CharacterMessage.Loaded)
    async def _handle_character_loaded(self, message: CharacterMessage.Loaded) -> None:
        message.stop()
        await self.preview.handle_character_loaded(
            character_id=str(message.character_id),
            card_data=message.card_data,
        )

    @on(PreviewReplyRequested)
    def _handle_preview_reply(self, message: PreviewReplyRequested) -> None:
        message.stop()
        self.preview.handle_reply_requested(message.user_message)

    @on(PreviewResetRequested)
    def _handle_preview_reset(self, message: PreviewResetRequested) -> None:
        message.stop()
        self.preview.handle_reset()

    @on(PreviewOpenInConsoleRequested)
    def _handle_preview_open_console(self, message: PreviewOpenInConsoleRequested) -> None:
        message.stop()
        self.preview.open_in_console()

    # ===== Create / edit =====

    @on(PersonaActionRequested)
    async def _handle_action_requested(self, message: PersonaActionRequested) -> None:
        message.stop()
        if message.action == "create":
            if self.state.active_mode == "characters":
                await self._run_guarded(self._begin_create_character)
            elif self.state.active_mode == "personas":
                await self._run_guarded(self._begin_create_profile)
            # Creation in the remaining modes is wired in follow-up tasks.
        elif message.action == "import":
            # Character-card import only; the library pane hides the Import
            # button outside Characters mode, so other modes are a no-op.
            if self.state.active_mode != "characters":
                return
            await self._run_guarded(self._open_import_dialog)
        # Delete and the rest are wired in follow-up tasks.

    async def _begin_create_character(self) -> None:
        self._character_editor_generation += 1
        self._edit_mode = "create"
        self.state.clear_selection()
        # Change-based dirty tracking: the session starts clean; the editor
        # posts EditorContentChanged on the first real modification.
        self.query_one(PersonasCharacterEditorWidget).new_character()
        self._show_center("#ccp-character-editor-view")
        inspector = self.query_one(PersonasInspectorPane)
        # Create mode: the previous selection's identity (and conversation
        # rows) must not linger in the inspector. clear_selection also resets
        # the unsaved gating, which is correct for a pristine session.
        await inspector.clear_selection()
        # While an editor is open the editor footer owns validation detail;
        # the inspector line must not claim "OK".
        inspector.show_validation_editing()
        self.call_after_refresh(self._focus_editor_name)

    async def _begin_create_profile(self) -> None:
        self._edit_mode = "create"
        self.state.clear_selection()
        # Change-based dirty tracking: the session starts clean (see
        # _begin_create_character).
        self.query_one(PersonaProfileEditorWidget).new_persona()
        self._show_center("#ccp-persona-editor-view")
        inspector = self.query_one(PersonasInspectorPane)
        # Same identity reset as _begin_create_character: no stale selection.
        await inspector.clear_selection()
        inspector.show_validation_editing()
        self.call_after_refresh(self._focus_editor_name)

    @on(EditPersonaRequested)
    async def _handle_persona_edit_requested(self, message: EditPersonaRequested) -> None:
        message.stop()
        if str(message.persona_id) != (self.state.selected_entity_id or ""):
            self._notify("Selection out of sync; reselect the persona profile.", "warning")
            return
        record = await self._fetch_profile_record(str(message.persona_id))
        self._edit_mode = "edit"
        # Change-based dirty tracking: the session starts clean; the editor
        # posts EditorContentChanged on the first real modification.
        self.query_one(PersonaProfileEditorWidget).load_persona(record)
        self._show_center("#ccp-persona-editor-view")
        inspector = self.query_one(PersonasInspectorPane)
        inspector.set_unsaved(False)
        inspector.show_validation_editing()
        self._register_footer_shortcuts()
        self.call_after_refresh(self._focus_editor_name)

    @on(EditCharacterRequested)
    def _handle_edit_requested(self, message: EditCharacterRequested) -> None:
        message.stop()
        if str(message.character_id) != (self.state.selected_entity_id or ""):
            self._notify("Selection out of sync; reselect the character.", "warning")
            return
        record = self._full_character_record(str(message.character_id))
        if record is None:
            self._notify("Character data is not loaded yet.", severity="warning")
            return
        self._character_editor_generation += 1
        self._edit_mode = "edit"
        # Change-based dirty tracking: the session starts clean; the editor
        # posts EditorContentChanged on the first real modification.
        self.query_one(PersonasCharacterEditorWidget).load_character(record)
        self._show_center("#ccp-character-editor-view")
        inspector = self.query_one(PersonasInspectorPane)
        inspector.set_unsaved(False)
        inspector.show_validation_editing()
        self._register_footer_shortcuts()
        self.call_after_refresh(self._focus_editor_name)

    @on(EditorContentChanged)
    def _handle_editor_content_changed(self, message: EditorContentChanged) -> None:
        """First real modification of an editing session: mark it unsaved.

        The editors post this once per load/new session; the screen owns the
        ``has_unsaved_changes`` flag, the inspector gating, the title state,
        the footer hints, and the library row badge.
        """
        message.stop()
        if self._edit_mode not in ("create", "edit"):
            # A stray Changed outside an editing session (e.g. racing a
            # save/cancel finisher) must not resurrect the dirty flag.
            return
        if self.state.has_unsaved_changes:
            return
        self.state.has_unsaved_changes = True
        try:
            self.query_one(PersonasInspectorPane).set_unsaved(True)
        except QueryError:
            pass
        self._set_active_row_unsaved(True)
        # Attach availability changed (unsaved edits block Console actions),
        # and the title gains the "- unsaved" segment.
        self._register_footer_shortcuts()

    def _set_active_row_unsaved(self, unsaved: bool) -> None:
        """Badge (or un-badge) the selected library row for the edit session."""
        try:
            pane = self.query_one(PersonasLibraryPane)
        except QueryError:
            return
        kind = self.state.selected_entity_kind
        entity_id = self.state.selected_entity_id
        if unsaved and kind and entity_id:
            pane.set_row_unsaved(kind, str(entity_id), True)
        else:
            pane.set_row_unsaved(None, None, False)

    def _full_character_record(self, character_id: str) -> dict | None:
        """Return the handler's fully-loaded card, or ``None`` when stale.

        The list rows in ``_characters`` are id/name-only; falling back to
        them would feed the editor (and a later save) empty fields, so a
        mismatch deliberately returns ``None``.
        """
        loaded = self.character_handler.current_character_data
        if loaded and str(self.character_handler.current_character_id) == character_id:
            return dict(loaded)
        return None

    def _character_editor_is_active(self) -> bool:
        """Return whether the visible edit session is the character editor."""
        if self._edit_mode not in ("create", "edit"):
            return False
        try:
            editor = self.query_one(PersonasCharacterEditorWidget)
        except QueryError:
            return False
        return editor.display is True

    def _character_editor_session_token(
        self,
    ) -> tuple[int, str, str, str | None, str | None] | None:
        """Return the current character edit session identity, if visible."""
        if not self._character_editor_is_active():
            return None
        return (
            self._character_editor_generation,
            self._edit_mode,
            self.state.active_mode,
            self.state.selected_entity_kind,
            self.state.selected_entity_id,
        )

    def _read_avatar_image_bytes(self, path: str) -> bytes:
        candidate = validate_path_simple(path, require_exists=True)
        if not candidate.is_file():
            raise ValueError("Choose an existing avatar image file.")
        if candidate.suffix.lower() not in PERSONAS_AVATAR_IMAGE_SUFFIXES:
            raise ValueError(
                f"Unsupported avatar image type. Use {PERSONAS_AVATAR_IMAGE_SUFFIX_COPY}."
            )
        if candidate.stat().st_size > PERSONAS_AVATAR_MAX_BYTES:
            raise ValueError(
                f"Avatar image must be {PERSONAS_AVATAR_MAX_SIZE_COPY} or smaller."
            )
        data = candidate.read_bytes()
        if not data:
            raise ValueError("Avatar image file is empty.")
        return data

    async def _stage_character_avatar_from_path(self, path: str) -> None:
        session_token = self._character_editor_session_token()
        if session_token is None:
            self._notify("Open a character editor before uploading an avatar.", "warning")
            return
        try:
            image_data = await asyncio.to_thread(self._read_avatar_image_bytes, path)
        except ValueError as exc:
            self._notify(str(exc), "error")
            return
        except OSError as exc:
            logger.opt(exception=True).error(f"Error reading avatar image from {path}: {exc}")
            self._notify(f"Avatar upload failed: {exc}", "error")
            return
        if self._character_editor_session_token() != session_token:
            logger.debug(
                "Avatar upload result ignored because the character editor session "
                f"changed. path={path!r}, original_session={session_token!r}, "
                f"current_session={self._character_editor_session_token()!r}"
            )
            return
        try:
            self.query_one(PersonasCharacterEditorWidget).set_avatar_image(image_data)
        except Exception as exc:
            logger.opt(exception=True).error(
                "Could not stage avatar image in editor. "
                f"path={path!r}, edit_mode={self._edit_mode!r}, "
                f"active_mode={self.state.active_mode!r}, "
                f"selected_kind={self.state.selected_entity_kind!r}, "
                f"selected_id={self.state.selected_entity_id!r}, "
                f"image_size_bytes={len(image_data)}: {exc}",
            )
            self._notify(f"Avatar upload failed: {exc}", "error")
            return
        self._notify("Avatar staged. Save the character to persist it.", "information")

    # ===== Import / export =====
    #
    # Dialog flows run in workers (push_screen_wait requires one); the
    # path-based methods below them are dialog-free so tests can call them
    # directly. Sync DB/file work runs via asyncio.to_thread instead of a
    # threaded Textual worker because import and export need their result
    # awaited inline for the follow-up selection/notification steps.

    @on(CharacterImageUploadRequested)
    def _handle_character_image_upload_requested(
        self, message: CharacterImageUploadRequested
    ) -> None:
        message.stop()
        if not self._character_editor_is_active():
            self._notify("Open a character editor before uploading an avatar.", "warning")
            return
        if self._io_dialog_active:
            logger.debug("Import/export dialog already active; ignoring avatar upload request.")
            return
        self._io_dialog_active = True
        self.run_worker(self._avatar_upload_dialog_worker(), group="personas-io")

    async def _avatar_upload_dialog_worker(self) -> None:
        from ...Widgets.enhanced_file_picker import EnhancedFileOpen, Filters

        try:
            picker = EnhancedFileOpen(
                title="Upload Character Avatar",
                filters=Filters(
                    (
                        "Image Files",
                        lambda p: p.suffix.lower() in PERSONAS_AVATAR_IMAGE_SUFFIXES,
                    ),
                    ("PNG Files", lambda p: p.suffix.lower() == ".png"),
                    ("JPEG Files", lambda p: p.suffix.lower() in (".jpg", ".jpeg")),
                    ("WEBP Files", lambda p: p.suffix.lower() == ".webp"),
                    ("GIF Files", lambda p: p.suffix.lower() == ".gif"),
                ),
                context="character_avatar_upload",
            )
            try:
                file_path = await self.app.push_screen_wait(picker)
            except Exception:
                logger.opt(exception=True).warning("Could not show the avatar upload file dialog.")
                return
            if file_path:
                await self._stage_character_avatar_from_path(str(file_path))
        finally:
            self._io_dialog_active = False

    async def _open_import_dialog(self) -> None:
        """Continuation for the guarded import action: launch the dialog worker."""
        if self._io_dialog_active:
            logger.debug("Import/export dialog already active; ignoring import request.")
            return
        self._io_dialog_active = True
        self.run_worker(self._import_dialog_worker(), group="personas-io")

    async def _import_dialog_worker(self) -> None:
        # Same dialog family as the legacy CCP import route
        # (ccp_character_handler.handle_import).
        from ...Widgets.enhanced_file_picker import EnhancedFileOpen, Filters

        try:
            picker = EnhancedFileOpen(
                title="Import Character Card",
                filters=Filters(
                    (
                        "Character Cards",
                        lambda p: p.suffix.lower()
                        in (".json", ".md", ".markdown", ".png"),
                    ),
                    ("JSON Files", lambda p: p.suffix.lower() == ".json"),
                    (
                        "Markdown Files",
                        lambda p: p.suffix.lower() in (".md", ".markdown"),
                    ),
                    ("PNG Files (with embedded data)", lambda p: p.suffix.lower() == ".png"),
                    ("All Files", lambda p: True),
                ),
                context="character_import",
            )
            try:
                file_path = await self.app.push_screen_wait(picker)
            except Exception:
                logger.opt(exception=True).warning("Could not show the import file dialog.")
                return
            if file_path:
                await self._import_character_from_path(str(file_path))
        finally:
            self._io_dialog_active = False

    async def _import_character_from_path(self, path: str) -> None:
        """Import a character card file, then refresh, select, and reveal it."""
        # On a name conflict the importer returns the EXISTING character's id;
        # snapshot the pre-import ids so the notification can say so.
        pre_import_ids = {str(c.get("id")) for c in self._characters}
        try:
            # Sync DB call; see the section comment for the threading choice.
            imported_id = await asyncio.to_thread(
                ccp_character_handler.import_character_card, path
            )
        except Exception as exc:
            logger.opt(exception=True).error(f"Error importing character card from {path}: {exc}")
            self._notify(f"Import failed: {exc}", "error")
            return
        if imported_id is None:
            self._notify(
                "Import failed: the file did not contain a valid character card.",
                "error",
            )
            return
        imported_id = str(imported_id)
        # Clear any active search (state + Input, as _apply_mode does) so the
        # imported character is visible in the refreshed list.
        self.state.search_query = ""
        try:
            self.query_one("#personas-library-search", Input).value = ""
        except Exception:
            pass
        await self.character_handler.refresh_character_list()
        if not self.is_mounted or self.state.active_mode != "characters":
            # The user left Characters mode while the import ran; the list is
            # refreshed but selection/center pane belong to the new mode.
            return
        record = self._character_record(imported_id)
        name = str((record or {}).get("name") or "Imported character")
        await self._select_character(imported_id, name)
        # The selection changed outside _run_guarded; refresh the footer hints
        # (attach is now available) and the header state.
        self._register_footer_shortcuts()
        if imported_id in pre_import_ids:
            self._notify("Character already existed; selected it.", "information")
        else:
            self._notify("Character imported.", "information")

    @on(Button.Pressed, "#personas-export-json")
    async def _handle_export_json_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._open_export_dialog("json")

    @on(Button.Pressed, "#personas-export-png")
    async def _handle_export_png_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._open_export_dialog("png")

    def _open_export_dialog(self, fmt: str) -> None:
        """Validate the selection and launch the save-dialog worker."""
        kind = self.state.selected_entity_kind
        if not self.state.selected_entity_id or kind not in ("character", "persona_profile"):
            # The inspector disables these buttons without a saved selection;
            # this is a defensive re-check.
            self._notify("Select a saved item before exporting.", "warning")
            return
        if fmt == "png" and kind != "character":
            self._notify("PNG export is only available for characters.", "warning")
            return
        if self._io_dialog_active:
            logger.debug("Import/export dialog already active; ignoring export request.")
            return
        self._io_dialog_active = True
        self.run_worker(self._export_dialog_worker(fmt), group="personas-io")

    async def _export_dialog_worker(self, fmt: str) -> None:
        # Same dialog family as import (enhanced_file_picker).
        from ...Widgets.enhanced_file_picker import EnhancedFileSave, Filters

        try:
            name = self.state.selected_entity_name or "export"
            # Filename sanitization mirrors the legacy CCP export route.
            safe_name = "".join(c for c in name if c.isalnum() or c in " -_").rstrip()
            default_filename = f"{safe_name or 'export'}.{fmt}"
            if fmt == "png":
                filters = Filters(
                    ("PNG Files", lambda p: p.suffix.lower() == ".png"),
                    ("All Files", lambda p: True),
                )
            else:
                filters = Filters(
                    ("JSON Files", lambda p: p.suffix.lower() == ".json"),
                    ("All Files", lambda p: True),
                )
            picker = EnhancedFileSave(
                title=f"Export as {fmt.upper()}",
                default_filename=default_filename,
                filters=filters,
                context="character_export",
            )
            try:
                target_path = await self.app.push_screen_wait(picker)
            except Exception:
                logger.opt(exception=True).warning("Could not show the export file dialog.")
                return
            if target_path:
                await self._export_selected_character(str(target_path), fmt=fmt)
        finally:
            self._io_dialog_active = False

    async def _export_selected_character(self, target_path: str, *, fmt: str) -> None:
        """Export the current selection to ``target_path`` as JSON or PNG."""
        kind = self.state.selected_entity_kind
        entity_id = self.state.selected_entity_id
        if not entity_id:
            self._notify("Select a saved item before exporting.", "warning")
            return
        try:
            if kind == "character":
                character_id = int(str(entity_id))
                if fmt == "png":
                    await asyncio.to_thread(
                        self._export_character_png_sync, character_id, target_path
                    )
                else:
                    await asyncio.to_thread(
                        self._export_character_json_sync, character_id, target_path
                    )
            elif kind == "persona_profile":
                if fmt != "json":
                    self._notify("PNG export is only available for characters.", "warning")
                    return
                record = await self._fetch_profile_record(str(entity_id))
                content = json.dumps(record, indent=2, ensure_ascii=False, default=str)
                await asyncio.to_thread(self._write_text_file, target_path, content)
            else:
                self._notify("Export is not available for this selection.", "warning")
                return
        except Exception as exc:
            logger.opt(exception=True).error(f"Error exporting to {target_path}: {exc}")
            self._notify(f"Export failed: {exc}", "error")
            return
        self._notify(f"Exported to {target_path}", "information")

    def _export_character_json_sync(self, character_id: int, target_path: str) -> None:
        """Sync JSON export; raises on failure (runs off the UI thread)."""
        db = ccp_character_handler._default_character_db()
        content = export_character_card_to_json(db, character_id, include_image=True)
        if content is None:
            raise RuntimeError("export returned no data")
        self._write_text_file(target_path, content)

    def _export_character_png_sync(self, character_id: int, target_path: str) -> None:
        """Sync PNG export; the library writes the file and validates the path.

        ``export_character_card_to_png`` validates ``output_path`` against a
        base directory (defaulting to its own exports folder), so the chosen
        path's parent is passed to keep user-selected destinations valid.
        """
        db = ccp_character_handler._default_character_db()
        target = Path(target_path).expanduser()
        ok = export_character_card_to_png(
            db, character_id, str(target), base_directory=str(target.parent)
        )
        if not ok:
            # The library returns False for several causes; surface them all.
            raise RuntimeError(
                "PNG export failed - the destination may be invalid (hidden/dot "
                "directories are rejected) or the character lacks image data; "
                "check the log for details."
            )

    @staticmethod
    def _write_text_file(target_path: str, content: str) -> None:
        """Write export text after validating the destination at the boundary.

        Mirrors the PNG export's check: the chosen path is validated against
        its own parent (``validate_path`` rejects hidden/dot components and
        resolution escapes) so legitimate user-chosen directories stay valid.
        """
        from ...Utils.path_validation import validate_path

        target = Path(target_path).expanduser()
        if not target.parent.exists():
            raise ValueError(f"destination directory does not exist: {target.parent}")
        validated = validate_path(target, base_directory=target.parent)
        validated.write_text(content, encoding="utf-8")

    # ===== Delete =====

    @on(Button.Pressed, "#personas-delete")
    async def _handle_delete_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        # The inspector enables Delete whenever a selection exists, even with
        # unsaved edits - deleting discards them by definition. The flow still
        # routes through the unsaved guard so a dirty session shows the
        # discard dialog FIRST, then the delete confirm. Two dialogs in
        # sequence is deliberate: the user explicitly approves both losses.
        await self._run_guarded(self._begin_delete_selection)

    async def _begin_delete_selection(self) -> None:
        """Validate the selection and launch the delete-confirm dialog worker."""
        kind = self.state.selected_entity_kind
        entity_id = str(self.state.selected_entity_id or "")
        if not entity_id or kind not in ("character", "persona_profile"):
            # The inspector disables Delete without a selection; defensive.
            self._notify("Select a saved item before deleting.", "warning")
            return
        name = self.state.selected_entity_name or "Unnamed"
        if kind == "character":
            # The sparse list rows do not carry `version` reliably; the
            # optimistic-lock delete needs the handler's fully loaded card.
            record = self._full_character_record(entity_id)
            if record is None:
                self._notify("Character data is not loaded yet.", "warning")
                return
            version: int | None = int(record.get("version") or 1)
        else:
            record = await self._fetch_profile_record(entity_id)
            raw_version = record.get("version")
            version = int(raw_version) if raw_version is not None else None
        if self._delete_dialog_active:
            logger.debug("Delete dialog already active; ignoring delete request.")
            return
        self._delete_dialog_active = True
        self.run_worker(
            self._delete_dialog_worker(kind, entity_id, name, version),
            group="personas-io",
        )

    async def _delete_dialog_worker(
        self, kind: str, entity_id: str, name: str, version: int | None
    ) -> None:
        try:
            if not await self._confirm_delete(name):
                return
            await self._delete_entity(kind, entity_id, version)
        finally:
            self._delete_dialog_active = False

    async def _confirm_delete(self, name: str) -> bool:
        """True when the user confirmed the delete (requires a worker context)."""
        dialog = ConfirmationDialog(
            title="Delete",
            message=f"Delete {name}? This cannot be undone here.",
            confirm_label="Delete",
            cancel_label="Cancel",
        )
        try:
            return bool(await self.app.push_screen_wait(dialog))
        except Exception:
            logger.opt(exception=True).warning(
                "Could not show the delete confirmation dialog; keeping the item.",
            )
            return False

    async def _delete_entity(self, kind: str, entity_id: str, version: int | None) -> None:
        """Perform the confirmed delete, then clean up selection-coupled state."""
        conflict_copy = (
            "Delete failed: the {noun} changed since it was loaded. "
            "Reselect and try again."
        )
        if kind == "character":
            try:
                # Sync DB call; same threading choice as import/export.
                ok = await asyncio.to_thread(
                    ccp_character_handler.delete_character, entity_id, int(version or 1)
                )
            except ConflictError:
                logger.warning(f"Optimistic-lock conflict deleting character {entity_id}.")
                self._notify(conflict_copy.format(noun="character"), "error")
                return
            except Exception as exc:
                logger.opt(exception=True).error(f"Error deleting character {entity_id}: {exc}")
                self._notify(f"Delete failed: {exc}", "error")
                return
            if not ok:
                # The DB API signals conflicts by raising; treat a False/None
                # return (e.g. stubbed/alternate backends) the same way.
                self._notify(conflict_copy.format(noun="character"), "error")
                return
        else:
            service = getattr(self.app_instance, "character_persona_scope_service", None)
            if service is None or not hasattr(service, "delete_persona_profile"):
                self._notify("Delete failed: persona profiles are unavailable.", "error")
                return
            try:
                await service.delete_persona_profile(
                    entity_id,
                    expected_version=version,
                    mode=self.persona_handler.current_mode(),
                )
            except Exception as exc:
                logger.opt(exception=True).error(
                    f"Error deleting persona profile {entity_id}: {exc}")
                # The local backend signals optimistic-lock loss with a
                # `..._version_conflict:` ValueError marker; map it onto the
                # same recovery copy the character path uses.
                if "version_conflict" in str(exc):
                    self._notify(conflict_copy.format(noun="persona profile"), "error")
                else:
                    self._notify(f"Delete failed: {exc}", "error")
                return
        await self._after_delete(kind)

    async def _after_delete(self, kind: str) -> None:
        if kind == "character":
            # The handler still holds the deleted card; drop it so
            # _full_character_record cannot serve stale data.
            self.character_handler.current_character_id = None
            self.character_handler.current_character_data = {}
        expected_mode = "characters" if kind == "character" else "personas"
        stale = not self.is_mounted or self.state.active_mode != expected_mode
        if not stale:
            self.state.clear_selection()
            self.state.has_unsaved_changes = False
            self._edit_mode = "view"
            # clear_selection on the inspector also empties the conversations
            # panel; drop the controller caches and the ephemeral preview too.
            self.conversations.reset()
            await self.preview.reset("")
            await self.query_one(PersonasInspectorPane).clear_selection()
            self._show_center(None)
            self._register_footer_shortcuts()
        # Refresh the cached rows even when the user already left the screen
        # or switched modes (the render paths are mode-guarded downstream).
        if kind == "character":
            try:
                await self.character_handler.refresh_character_list()
            except Exception:
                logger.opt(exception=True).warning("Could not refresh characters after a delete.")
        else:
            self._refresh_profile_rows_worker()
        if not stale:
            self._notify("Deleted.", "information")

    # ===== Save =====

    def _validate_character(self, data: dict) -> tuple[str, ...]:
        """Field-level validation; failures block Save and render in the inspector."""
        errors: list[str] = []
        if not str(data.get("name", "")).strip():
            errors.append("name: required")
        book = data.get("character_book")
        if book:
            ok, book_errors = validate_character_book(book)
            if not ok:
                errors.extend(str(error) for error in book_errors)
        return tuple(errors)

    @on(CharacterSaveRequested)
    def _handle_save_requested(self, message: CharacterSaveRequested) -> None:
        message.stop()
        data = dict(message.character_data or {})
        errors = self._validate_character(data)
        # The editor footer is the single in-editor validation surface: the
        # screen-side results (name, character_book) render there, in the
        # same format as the editor's own check.
        self.query_one(PersonasCharacterEditorWidget).show_validation(errors)
        if errors:
            # The editor stays open, so the inspector line stays in its
            # editing state instead of duplicating the error detail.
            self.query_one(PersonasInspectorPane).show_validation_editing()
            return
        # Snapshot UI-thread state here; the background persistence call must
        # not read mutable screen state.
        self._save_character_worker(data, self.state.selected_entity_id, self._edit_mode)

    @work(exclusive=True, group="personas-save")
    async def _save_character_worker(self, data: dict, selected_id: str | None, edit_mode: str) -> None:
        """Persist via the legacy module-level helpers off the UI thread."""
        try:
            def persist_character() -> str:
                if edit_mode == "create" or not selected_id:
                    created_id = ccp_character_handler.create_character(data)
                    if not created_id:
                        raise RuntimeError("Character creation returned no id.")
                    return str(created_id)
                if not ccp_character_handler.update_character(selected_id, data):
                    raise RuntimeError(f"Character update failed for id {selected_id}.")
                return str(selected_id)

            saved_id = await asyncio.to_thread(persist_character)
        except Exception as exc:
            logger.opt(exception=True).error(f"Error saving character: {exc}")
            self._notify(f"Save failed: {exc}", "error")
            return
        await self._after_character_save(saved_id, str(data.get("name") or ""))

    async def _after_character_save(self, saved_id: str, submitted_name: str = "") -> None:
        if not self.is_mounted or self.state.active_mode != "characters":
            # The save completed after the user left the screen or switched
            # modes; refresh the cached list but leave the selection,
            # inspector, and center pane alone.
            try:
                await self.character_handler.refresh_character_list()
            except Exception:
                logger.opt(exception=True).warning("Could not refresh characters after a late save.")
            return
        self._character_editor_generation += 1
        self._edit_mode = "view"
        self._set_active_row_unsaved(False)
        await self.character_handler.refresh_character_list()
        record = self._character_record(saved_id)
        name = str((record or {}).get("name") or submitted_name or "Saved character")
        self.state.select_entity(entity_kind="character", entity_id=saved_id, entity_name=name)
        self.state.has_unsaved_changes = False
        inspector = self.query_one(PersonasInspectorPane)
        inspector.show_selection(name=name, kind="character", authority="Local")
        inspector.set_unsaved(False)
        inspector.show_validation(())
        self._sync_inspector_console_actions()
        self.query_one(PersonasLibraryPane).mark_active_row("character", saved_id)
        if record is not None:
            await self.character_handler.load_character(saved_id)
        self._show_center("#ccp-character-card-view")
        self._register_footer_shortcuts()
        self.call_after_refresh(self._focus_library_list)
        self._notify("Character saved.", severity="information")

    @on(PersonaProfileSaveRequested)
    async def _handle_profile_save_requested(self, message: PersonaProfileSaveRequested) -> None:
        """Persist a persona profile through the async scope service.

        The editor validates before posting (its inline
        ``#personas-editor-validation`` panel is the personas-mode source of
        truth), so the screen only clears the inspector summary here.
        """
        message.stop()
        if self._profile_save_inflight:
            logger.debug("Persona save already in flight; ignoring duplicate request.")
            return
        if self._edit_mode not in ("create", "edit"):
            # Message dispatch is serial, so a double-posted Save arrives after
            # the first save already finished and returned to view mode; a save
            # without an open edit session is a stale duplicate.
            logger.debug("Persona save without an open edit session; ignoring.")
            return
        data = dict(message.data or {})
        # The editor is still open until the save lands; keep the inspector
        # line in its editing state (a failed save leaves the editor open).
        self.query_one(PersonasInspectorPane).show_validation_editing()
        service = getattr(self.app_instance, "character_persona_scope_service", None)
        if service is None:
            self._notify("Save failed: persona profiles are unavailable.", "error")
            return
        self._profile_save_inflight = True
        try:
            mode = self.persona_handler.current_mode()
            persona_id = str(data.get("id") or "")
            try:
                if self._edit_mode == "create" or not persona_id:
                    request = PersonaProfileCreate(
                        id=data.get("id") or None,
                        name=str(data.get("name") or ""),
                        description=data.get("description"),
                        mode=data.get("mode") or "session_scoped",
                        system_prompt=data.get("system_prompt"),
                    )
                    result = await service.create_persona_profile(request, mode=mode)
                else:
                    request = PersonaProfileUpdate(
                        name=str(data.get("name") or ""),
                        description=data.get("description"),
                        mode=data.get("mode"),
                        system_prompt=data.get("system_prompt"),
                    )
                    result = await service.update_persona_profile(
                        persona_id,
                        request,
                        expected_version=data.get("version"),
                        mode=mode,
                    )
            except Exception as exc:
                logger.opt(exception=True).error(f"Error saving persona profile: {exc}")
                self._notify(f"Save failed: {exc}", "error")
                return
            if hasattr(result, "model_dump"):
                result = result.model_dump(mode="json")
            if not isinstance(result, dict):
                # Tolerate backends that return ids/None: keep the submitted data.
                result = dict(data)
            saved = dict(result)
            saved.setdefault("id", persona_id)
            await self._after_profile_save(saved)
        finally:
            self._profile_save_inflight = False

    async def _after_profile_save(self, saved: dict) -> None:
        # Refresh the cached profile list tolerantly even when the user has
        try:
            profiles = await self.persona_handler.refresh_persona_list(
                raise_on_unavailable=True
            )
        except Exception as exc:
            logger.opt(exception=True).warning("Could not refresh persona profiles after a save.")
            self._profile_lookup_recovery_state = self._profile_list_recovery_state(exc)
            profiles = []
        else:
            self._profile_lookup_recovery_state = None
        self._profiles = [dict(record) for record in (profiles or [])]
        self._update_status_row()
        self._update_status_row()
        if not self.is_mounted or self.state.active_mode != "personas":
            # Leave the selection, inspector, and center pane alone.
            return
        self._edit_mode = "view"
        self.state.has_unsaved_changes = False
        self._set_active_row_unsaved(False)
        saved_id = str(saved.get("id") or "")
        name = str(saved.get("name") or "Saved persona")
        self.state.select_entity(
            entity_kind="persona_profile", entity_id=saved_id, entity_name=name
        )
        inspector = self.query_one(PersonasInspectorPane)
        inspector.show_selection(name=name, kind="persona_profile", authority="Local")
        inspector.set_unsaved(False)
        inspector.show_validation(())
        self._sync_inspector_console_actions()
        await self._render_profile_rows()
        self.query_one(PersonaProfileCardWidget).show_persona(saved)
        self._show_center("#ccp-persona-card-view")
        self._register_footer_shortcuts()
        self.call_after_refresh(self._focus_library_list)
        self._notify("Persona saved.", "information")

    # ===== Cancel =====

    @on(CharacterEditorCancelled)
    async def _handle_editor_cancelled(self, message: CharacterEditorCancelled) -> None:
        message.stop()

        async def _finish() -> None:
            self._finish_cancel_edit()

        await self._run_guarded(_finish)

    def _finish_cancel_edit(self) -> None:
        self._character_editor_generation += 1
        self._edit_mode = "view"
        self.state.has_unsaved_changes = False
        inspector = self.query_one(PersonasInspectorPane)
        inspector.set_unsaved(False)
        # The editor closed: the inspector validation line returns to normal.
        inspector.show_validation(())
        self._set_active_row_unsaved(False)
        if self.state.selected_entity_id:
            self._show_center("#ccp-character-card-view")
        else:
            self._show_center(None)
        self.call_after_refresh(self._focus_library_list)

    @on(PersonaProfileEditCancelled)
    async def _handle_profile_edit_cancelled(self, message: PersonaProfileEditCancelled) -> None:
        message.stop()

        async def _finish() -> None:
            self._finish_cancel_profile_edit()

        await self._run_guarded(_finish)

    def _finish_cancel_profile_edit(self) -> None:
        self._edit_mode = "view"
        self.state.has_unsaved_changes = False
        inspector = self.query_one(PersonasInspectorPane)
        inspector.set_unsaved(False)
        # The editor closed: the inspector validation line returns to normal.
        inspector.show_validation(())
        self._set_active_row_unsaved(False)
        if self.state.selected_entity_id:
            self._show_center("#ccp-persona-card-view")
        else:
            self._show_center(None)
        self.call_after_refresh(self._focus_library_list)

    # ===== Helpers =====

    def _show_center(self, visible_id: str | None) -> None:
        """Show one center-area view (or none); tolerate missing nodes."""
        for selector in _CENTER_VIEW_IDS:
            try:
                widget = self.query_one(selector)
            except Exception:
                continue
            # All center views are ds-native widgets without `.hidden`-class
            # styling; plain display toggling is the whole mechanism.
            widget.display = selector == visible_id
        # The conversation actions row is chrome shown alongside (not instead
        # of) the read-only conversation view.
        try:
            actions = self.query_one("#personas-conversation-actions")
        except Exception:
            return
        actions.display = visible_id == _CONVERSATION_VIEW_ID

    async def _run_guarded(self, continuation: Callable[[], Awaitable[None]]) -> None:
        """Run ``continuation``, confirming first when an edit would be discarded.

        The clean fast path runs the continuation inline (the calling message
        handler awaits it), matching the pre-helper behavior tests rely on.
        The dirty path needs a worker context for ``push_screen_wait`` and is
        protected by ``_guard_active`` so a queued second trigger cannot
        double-fire the confirm dialog or its continuation.
        """
        if not self.state.has_unsaved_changes:
            await continuation()
            # Guarded continuations are exactly the transitions that change
            # edit mode / selection, so the footer hints refresh here.
            self._register_footer_shortcuts()
            return
        if self._guard_active:
            return
        self._guard_active = True
        self.run_worker(self._confirm_then_run(continuation), group="personas-guard")

    async def _confirm_then_run(self, continuation: Callable[[], Awaitable[None]]) -> None:
        try:
            if not await self._confirm_discard_unsaved():
                return
            self.state.has_unsaved_changes = False
            # The discarded session's row badge must not survive the discard
            # (the continuation may move the selection without a row rebuild).
            self._set_active_row_unsaved(False)
            await continuation()
            self._register_footer_shortcuts()
        finally:
            self._guard_active = False

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
            logger.opt(exception=True).warning("Could not show unsaved-changes dialog; keeping edits.")
            return False

    def _notify(self, message: str, severity: str = "warning") -> None:
        notify = getattr(self.app_instance, "notify", None)
        if callable(notify):
            notify(message, severity=severity)

    # ===== Key bindings =====

    def action_personas_new(self) -> None:
        """Ctrl+N: same code path as the library New button."""
        self.post_message(PersonaActionRequested(action="create"))

    def action_personas_search(self) -> None:
        """Ctrl+F: focus the library search input."""
        try:
            self.query_one("#personas-library-search", Input).focus()
        except QueryError:
            pass

    async def action_personas_attach(self) -> None:
        """Ctrl+Enter: same path as the inspector Attach button.

        No-ops silently when the attach buttons would be disabled (no saved
        selection, or unsaved edits) so the shortcut cannot bypass the guard.
        """
        if not self._console_action_allowed():
            return
        await self._attach_selection_to_console(intent="attach")

    def action_personas_save(self) -> None:
        """Ctrl+S: trigger the visible editor's Save path; no-op otherwise.

        Pressing the editor's own Save button reuses its validation flow
        (name-required check, message posting) without a second code path.
        """
        if self._edit_mode not in ("create", "edit"):
            return
        for view_id, button_id in (
            ("#ccp-character-editor-view", "#personas-char-editor-save"),
            ("#ccp-persona-editor-view", "#personas-editor-save"),
        ):
            try:
                view = self.query_one(view_id)
            except QueryError:
                continue
            if view.display:
                try:
                    view.query_one(button_id, Button).press()
                except QueryError:
                    logger.warning(f"Save button {button_id} is not mounted.")
                return

    def action_personas_escape(self) -> None:
        """Escape, context-sensitive; a strict no-op outside its contexts.

        Editor open -> the SAME cancel path as the Cancel button (the posted
        cancel message routes through the unsaved guard; never bypassed).
        Transcript open -> back to the card, focus the conversations list.
        Search focused -> move focus to the library list.
        """
        if self._edit_mode in ("create", "edit"):
            for view_id, message in (
                ("#ccp-character-editor-view", CharacterEditorCancelled),
                ("#ccp-persona-editor-view", PersonaProfileEditCancelled),
            ):
                try:
                    view = self.query_one(view_id)
                except QueryError:
                    continue
                if view.display:
                    self.post_message(message())
                    return
            return
        try:
            transcript = self.query_one(_CONVERSATION_VIEW_ID)
        except QueryError:
            transcript = None
        if transcript is not None and transcript.display:
            # Same path as the "Back to card" button.
            self._show_center("#ccp-character-card-view")
            self._register_footer_shortcuts()
            self._focus_conversations_list()
            return
        focused = self.app.focused
        if focused is not None and focused.id == "personas-library-search":
            self._focus_library_list(force=True)

    async def action_personas_mode(self, mode: str) -> None:
        """Ctrl+1..5: same guarded path as the mode chips."""
        if mode not in MODE_CHIP_ORDER or mode == self.state.active_mode:
            return
        await self._run_guarded(lambda: self._apply_mode(mode))

    async def action_personas_mode_cycle(self, delta: int) -> None:
        """[ / ]: cycle the mode strip relative to the active mode."""
        try:
            index = MODE_CHIP_ORDER.index(self.state.active_mode)
        except ValueError:
            index = 0
        await self.action_personas_mode(
            MODE_CHIP_ORDER[(index + delta) % len(MODE_CHIP_ORDER)]
        )

    # ===== Focus management =====

    def _focus_steal_blocked(self) -> bool:
        """True when focus sits in an active text input outside the editors.

        Mirrors the chat screen's "don't steal focus from an active input"
        rule. Editor fields are exempt: focus moves are requested exactly when
        the editor is being dismissed (save/cancel), so they are not active.
        """
        focused = self.app.focused
        if not isinstance(focused, (Input, TextArea)):
            return False
        return not any(
            getattr(node, "id", None)
            in ("ccp-character-editor-view", "ccp-persona-editor-view")
            for node in focused.ancestors
        )

    def _focus_editor_name(self) -> None:
        """Focus the visible editor's Name input (after the center switch)."""
        for view_id, input_id in (
            ("#ccp-character-editor-view", "#personas-char-editor-name"),
            ("#ccp-persona-editor-view", "#personas-editor-name"),
        ):
            try:
                view = self.query_one(view_id)
            except QueryError:
                continue
            if view.display:
                try:
                    view.query_one(input_id, Input).focus()
                except QueryError:
                    pass
                return

    def _focus_library_list(self, *, force: bool = False) -> None:
        """Focus the library rows unless the user is typing elsewhere.

        ``force`` is for explicit user intent (Esc in the search input), where
        leaving the text input IS the request.
        """
        if not force and self._focus_steal_blocked():
            return
        try:
            self.query_one("#personas-library-rows", ListView).focus()
        except QueryError:
            pass

    def action_focus_next_workbench_pane(self) -> None:
        """F6: move focus to the next Personas workbench pane."""
        focus_relative_workbench_pane(
            self,
            self._WORKBENCH_FOCUS_TARGETS,
            direction=1,
        )

    def action_focus_previous_workbench_pane(self) -> None:
        """Shift+F6: move focus to the previous Personas workbench pane."""
        focus_relative_workbench_pane(
            self,
            self._WORKBENCH_FOCUS_TARGETS,
            direction=-1,
        )

    def _focus_conversations_list(self) -> None:
        """Focus the inspector's conversations list (transcript Back path)."""
        if self._focus_steal_blocked():
            return
        try:
            self.query_one("#personas-conversations-list", ListView).focus()
        except QueryError:
            pass

    def _focus_conversation_transcript(self) -> None:
        """Focus the transcript scroll so arrow keys scroll it."""
        if self._focus_steal_blocked():
            return
        try:
            self.query_one("#personas-transcript-scroll").focus()
        except QueryError:
            pass

    # ===== Footer shortcut context =====

    def _shortcut_context(self) -> ShortcutContext:
        """Truthful footer hints built from the live workbench state.

        Re-registered (replace-on-change) on every transition that changes an
        availability: entering/leaving an editor, selection changes, saves,
        deletes, and mode switches.
        """
        editing = self._edit_mode in ("create", "edit")
        try:
            transcript_open = bool(self.query_one(_CONVERSATION_VIEW_ID).display)
        except QueryError:
            transcript_open = False
        return ShortcutContext(
            source="personas",
            actions=(
                ShortcutAction("ctrl+n", "new"),
                ShortcutAction("ctrl+f", "search"),
                ShortcutAction("ctrl+s", "save", available=editing),
                ShortcutAction("esc", "back", available=editing or transcript_open),
                ShortcutAction(
                    "ctrl+enter", "attach", available=self._console_action_allowed()
                ),
                ShortcutAction("[ ]", "mode"),
            ),
        )

    def _register_footer_shortcuts(self) -> None:
        # The footer re-registers on exactly the transitions that change the
        # editing/selection state, which are also the transitions the live
        # header reflects; refresh the title here so the two stay in lockstep.
        self._update_title()
        self._sync_inspector_console_actions()
        try:
            footer = self.app.query_one("AppFooterStatus")
        except QueryError:
            return
        set_ctx = getattr(footer, "set_shortcut_context", None)
        if callable(set_ctx):
            set_ctx(self._shortcut_context())

    def _clear_footer_shortcuts(self) -> None:
        try:
            footer = self.app.query_one("AppFooterStatus")
        except QueryError:
            return
        clear_ctx = getattr(footer, "clear_shortcut_context", None)
        if callable(clear_ctx):
            clear_ctx(source="personas")
