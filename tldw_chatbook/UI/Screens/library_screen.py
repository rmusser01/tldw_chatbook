"""Library destination shell for source material and Search/RAG."""

from __future__ import annotations

import asyncio
import inspect
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger
from rich.markup import escape as escape_markup
from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches, QueryError
from textual.timer import Timer
from textual.widgets import Button, Input, Static, TextArea

from ...Chat.chat_handoff_models import ChatHandoffPayload
from ...config import get_cli_setting, save_setting_to_cli_config
from ...Constants import (
    LIBRARY_MODE_CONVERSATIONS,
    LIBRARY_NAV_CONTEXT_CONVERSATION_ID,
    LIBRARY_NAV_CONTEXT_MODE,
)
from ...DB.ChaChaNotes_DB import ConflictError
from ...Library.library_collections_service import LibraryCollectionsServiceError
from ...Library.library_collections_state import LibraryCollectionsPanelState
from ...Library.library_conversations_state import build_library_conversations_state
from ...Library.library_media_state import (
    LibraryMediaCanvasState,
    build_library_media_state,
)
from ...Library.library_media_viewer_state import (
    build_library_media_highlight_rows,
    build_library_media_viewer_state,
    find_content_matches,
)
from ...Library.library_notes_state import (
    LibraryNoteEditorState,
    build_library_note_editor_state,
    build_library_notes_list_state,
    build_note_export_content,
    next_notes_sort_mode,
    note_template_keywords,
    notes_autosave_status_text,
    resolve_note_template_placeholders,
    sort_notes_records,
)
from ...Library.library_notes_sync_state import (
    LibraryNotesSyncState,
    append_activity,
    next_sync_conflict,
    next_sync_direction,
    sync_status_line,
)
from ...Library.library_rag_service import (
    LibraryRagSearchOutcome,
    LibraryRagSearchRequest,
    run_library_rag_search,
)
from ...Library.library_rag_state import LibraryRagPanelState
from ...Library.library_rail_state import (
    LIBRARY_RAIL_SECTION_IDS,
    coerce_library_rail_preferences,
    serialize_library_rail_preferences,
)
from ...Library.library_shell_state import (
    LIBRARY_ROW_BROWSE_CONVERSATIONS,
    LIBRARY_ROW_BROWSE_MEDIA,
    LIBRARY_ROW_BROWSE_NOTES,
    LibraryShellInput,
    build_library_shell_state,
)
from ...runtime_policy.server_event_scope import event_principal_id_from_active_context
from ...runtime_policy.types import PolicyDeniedError, RuntimeSourceState
from ...Sync_Interop.sync_promotion_state import build_sync_promotion_state
from ...Sync_Interop.sync_readiness import DEFAULT_SYNC_ELIGIBILITY_REGISTRY, build_sync_readiness_report
from ...Third_Party.textual_fspicker import FileOpen, FileSave
from ...Utils.input_validation import sanitize_string, validate_text_input
from ...Utils.path_validation import validate_path_simple
from ...Workspaces import LibraryWorkspaceDepthState, build_library_workspace_depth_state
from ...Widgets.Console.console_rail_section import (
    CONSOLE_RAIL_SECTION_TOGGLE_PREFIX,
    ConsoleRailSectionHeader,
)
from ...Widgets.Library import (
    LibraryCollectionsPanel,
    LibraryConversationsCanvas,
    LibraryMediaCanvas,
    LibraryMediaViewer,
    LibraryNotesCanvas,
    LibraryRail,
    LibrarySearchRagInspectorPanel,
    LibrarySearchRagPanel,
    library_dim_label_text,
)
from ..Navigation.base_app_screen import BaseAppScreen
from ..Navigation.main_navigation import NavigateToScreen
from ..Views.RAGSearch.search_handoff import build_library_rag_console_live_work_payload
from .destination_recovery import DestinationRecoveryState, policy_denied_recovery_state
from .study_scope_models import (
    MATERIAL_SOURCE_LIBRARY,
    MATERIAL_TITLE_LIBRARY_SOURCES,
    StudyScopeContext,
    StudySourceItem,
)


logger = logger.bind(module="LibraryScreen")
LIBRARY_SOURCE_PAGE_SIZES = {"notes": 100, "media": 50, "conversations": 50}
LIBRARY_SERVICE_ERROR_COPY = "Library source services unavailable; retry Library later."
LIBRARY_SERVICE_UNAVAILABLE_COPY = "Library source services are unavailable in this runtime."
LIBRARY_EMPTY_COPY = "No local Library content yet."
LIBRARY_EMPTY_NEXT_ACTION_COPY = (
    "Import media, create notes, or open Library Search/RAG after indexing."
)
LIBRARY_INSPECTOR_EMPTY_COPY = "No source selected."
LIBRARY_INSPECTOR_EMPTY_NEXT_ACTION_COPY = (
    "Library remains a hub; Notes, Media, Search/RAG, and Study own deeper work."
)
LIBRARY_SOURCE_SNAPSHOT_TIMEOUT_SECONDS = 5.0
LIBRARY_NOTES_AUTOSAVE_SECONDS = 2.0
# Matches the standalone NotesSyncPane.AUTO_SYNC_INTERVAL_SECONDS -- same
# 5-minute cadence, same "while this screen instance lives" scope.
LIBRARY_NOTES_AUTO_SYNC_INTERVAL_SECONDS = 300
LIBRARY_NOTE_CONTENT_MAX_CHARS = 2_000_000
LIBRARY_COLLECTION_SYNC_CONFLICT_LIMIT = 200
LIBRARY_HANDOFF_LABEL_PREFIX = "Console/RAG handoff: "
LIBRARY_LOCAL_SNAPSHOT_MODES = frozenset({"sources", "conversations", "import-export"})
LIBRARY_WORKSPACE_SOURCE_COLUMN_WIDTH = 30
LIBRARY_WORKSPACE_SCOPE_COLUMN_WIDTH = 18
LIBRARY_WORKSPACE_VISIBLE_COLUMN_WIDTH = 7
LIBRARY_WORKSPACE_CONTEXT_COLUMN_WIDTH = 11
LIBRARY_HUB_RECENT_LABEL_WIDTH = 32
LIBRARY_HUB_INVENTORY_SOURCE_COLUMN_WIDTH = 14
LIBRARY_HUB_INVENTORY_READINESS_COLUMN_WIDTH = 16
LIBRARY_HUB_INVENTORY_OWNER_COLUMN_WIDTH = 22
LIBRARY_HUB_INVENTORY_ACTION_COLUMN_WIDTH = 18
LIBRARY_MEDIA_HANDOFF_EXCERPT_CHARS = 500
LIBRARY_RAG_RESULTS_STATIC_WIDGET_IDS = frozenset(
    {
        "library-rag-results-section-rule",
        "library-rag-results-heading",
        "library-rag-attribution-placeholder",
    }
)
LIBRARY_COLUMN_TITLES = {
    "sources": ("Source Map", "Active Workbench", "Inspector"),
    "conversations": ("Source Map", "Saved Conversations", "Conversation Inspector"),
    "search": ("Source Map", "Search/RAG Workbench", "Evidence Inspector"),
    "import-export": ("Source Map", "Import/Export Workbench", "Import/Export Inspector"),
    "workspaces": ("Source Map", "Workspace Context", "Handoff Rules"),
    "collections": ("Source Map", "Collections Reader", "Collection Inspector"),
    "study": ("Source Map", "Study Handoff", "Inspector"),
    "flashcards": ("Source Map", "Flashcards Handoff", "Inspector"),
    "quizzes": ("Source Map", "Quizzes Handoff", "Inspector"),
}
LIBRARY_MODES = {
    "sources": {
        "label": "Content Hub",
        "button_id": "library-mode-sources",
        "description": (
            "Content Hub mode: Library landing page for ingested content, notes, media, "
            "conversations, collections, and retrieval."
        ),
        "next_action": "Open the owning module for deep work; Console handoff is secondary.",
    },
    "conversations": {
        "label": "Conversations",
        "button_id": "library-mode-conversations",
        "description": (
            "Conversations mode: browse saved chats inside Library, inspect metadata, "
            "and prepare eligible conversation context."
        ),
        "next_action": "Select a saved conversation to inspect metadata and handoff eligibility.",
        "show_in_strip": False,
    },
    "search": {
        "label": "Search/RAG",
        "button_id": "library-mode-search",
        "description": (
            "Ask Library sources, inspect evidence, then send selected snippets to Console."
        ),
        "next_action": "Query first; scope, evidence, and Console handoff stay visible below.",
    },
    "import-export": {
        "label": "Import/Export",
        "button_id": "library-mode-import-export",
        "description": (
            "Import/Export mode: Library owns source acquisition framing; "
            "Ingest and Media own deeper file handling."
        ),
        "next_action": "Choose a handoff action below; imported material returns as Library inventory.",
    },
    "workspaces": {
        "label": "Workspaces",
        "button_id": "library-mode-workspaces",
        "description": "Workspaces mode: scope Library material to project or task contexts.",
        "next_action": "Workspace scoping is shown here before material is staged in Console.",
    },
    "collections": {
        "label": "Collections",
        "button_id": "library-mode-collections",
        "description": "Collections mode: read and review saved Library content.",
        "next_action": "Select or create a Collection record before item actions become available.",
    },
    "study": {
        "label": "Study",
        "button_id": "library-mode-study",
        "description": "Study mode: turn Library material into study sessions.",
        "next_action": "Open Study Dashboard to continue due cards, decks, and quizzes.",
    },
    "flashcards": {
        "label": "Flashcards",
        "button_id": "library-mode-flashcards",
        "description": "Flashcards mode: generate or review cards from Library sources.",
        "next_action": "Open Flashcards to work with the current source snapshot.",
    },
    "quizzes": {
        "label": "Quizzes",
        "button_id": "library-mode-quizzes",
        "description": "Quizzes mode: generate or resume quizzes from Library sources.",
        "next_action": "Open Quizzes to test recall against the current source snapshot.",
    },
}

LIBRARY_STUDY_HANDOFF_MODES = {
    "study": {
        "label": "Study",
        "action_label": "Study Dashboard",
    },
    "flashcards": {
        "label": "Flashcards",
        "action_label": "Flashcards",
    },
    "quizzes": {
        "label": "Quizzes",
        "action_label": "Quizzes",
    },
}


def _active_library_sync_scope(app_instance: Any) -> dict[str, str | None]:
    runtime_state = getattr(getattr(app_instance, "runtime_policy", None), "state", None)
    active_source = str(getattr(runtime_state, "active_source", "local") or "local").lower()
    server_profile_id = getattr(runtime_state, "active_server_id", None)
    source_authority = "server" if active_source == "server" and server_profile_id else "local"
    authenticated_principal_id = None
    if source_authority == "server":
        server_context_provider = getattr(app_instance, "server_context_provider", None)
        get_active_context = getattr(server_context_provider, "get_active_context", None)
        if callable(get_active_context):
            try:
                authenticated_principal_id = event_principal_id_from_active_context(get_active_context())
            except Exception:
                authenticated_principal_id = None
    workspace_scope = None
    workspace_service = getattr(app_instance, "workspace_registry_service", None)
    get_active_workspace = getattr(workspace_service, "get_active_workspace", None)
    if callable(get_active_workspace):
        try:
            active_workspace = get_active_workspace()
            workspace_scope = getattr(active_workspace, "workspace_id", None)
        except Exception:
            workspace_scope = None
    return {
        "source_authority": source_authority,
        "server_profile_id": str(server_profile_id) if server_profile_id else None,
        "authenticated_principal_id": authenticated_principal_id,
        "workspace_scope": workspace_scope,
    }
LIBRARY_MODE_BY_BUTTON_ID = {
    mode["button_id"]: mode_id for mode_id, mode in LIBRARY_MODES.items()
}

# Maps a Library mode id to the shell rail row that selects that canvas so
# navigation context and legacy mode switches land on the right canvas.
LIBRARY_MODE_TO_ROW_ID = {
    "conversations": LIBRARY_ROW_BROWSE_CONVERSATIONS,
    "collections": "browse-collections",
    "search": "browse-search",
    "study": "create-study",
    "flashcards": "create-flashcards",
    "quizzes": "create-quizzes",
    "import-export": "ingest-import-export",
}


def _record_value(record: Any, key: str, fallback: Any = "") -> Any:
    if isinstance(record, Mapping):
        return record.get(key, fallback)
    return getattr(record, key, fallback)


def _library_collection_record_data(record: Any) -> dict[str, Any]:
    return {
        "collection_id": _record_value(record, "collection_id"),
        "name": _record_value(record, "name"),
        "description": _record_value(record, "description"),
        "item_count": _record_value(record, "item_count", 0),
        "source_authority": _record_value(record, "source_authority", "local"),
        "sync_status": _record_value(record, "sync_status", "local-only"),
        "created_at": _record_value(record, "created_at"),
        "updated_at": _record_value(record, "updated_at"),
    }


def _collection_scoped_mirror_report(
    report: Mapping[str, Any] | None,
    collection_id: str,
) -> dict[str, Any] | None:
    if not report:
        return None
    actions = tuple(
        action
        for action in report.get("actions", ())
        if isinstance(action, Mapping)
        and isinstance(action.get("identity"), Mapping)
        and str(action["identity"].get("local_entity_id", "")) == collection_id
    )
    if not actions:
        return None
    scoped_report = dict(report)
    scoped_report["actions"] = actions
    scoped_report["mapped_count"] = len(actions)
    scoped_report["dry_run"] = bool(report.get("dry_run", True))
    scoped_report["write_enabled"] = bool(report.get("write_enabled", False))
    return scoped_report


def _collection_scoped_conflicts(
    conflict_reports: Sequence[Mapping[str, Any]],
    collection_id: str,
) -> tuple[Mapping[str, Any], ...]:
    scoped: list[Mapping[str, Any]] = []
    local_side_suffix = f":local:{collection_id}"
    remote_side_suffix = f":remote:{collection_id}"
    for conflict in conflict_reports:
        local_side_key = str(conflict.get("local_side_key") or "")
        remote_side_key = str(conflict.get("remote_side_key") or "")
        if local_side_key or remote_side_key:
            if (
                local_side_key.endswith(local_side_suffix)
                or remote_side_key.endswith(remote_side_suffix)
            ):
                scoped.append(conflict)
            continue
        details = conflict.get("details", {})
        if isinstance(details, Mapping):
            local_entity_id = details.get("local_entity_id")
            if local_entity_id is not None and str(local_entity_id) != collection_id:
                continue
        scoped.append(conflict)
    return tuple(scoped)


class LibraryScreen(BaseAppScreen):
    """Source material, imports/exports, conversations, and Search/RAG entry."""

    BINDINGS = [
        ("u", "library_rag_use_in_console", "Use Library context in Console"),
    ]

    # Baseline workbench geometry so the screen renders correctly even without
    # the app stylesheet (e.g. harness tests). The agentic-terminal TCSS uses
    # equal-specificity selectors and takes precedence when loaded.
    DEFAULT_CSS = """
    #library-mode-bar {
        height: 1;
        min-height: 1;
        padding: 0 1;
        overflow: hidden;
    }

    #library-mode-label {
        width: 8;
        min-width: 8;
        height: 1;
        min-height: 1;
    }

    Button.library-mode-chip {
        width: auto;
        min-width: 10;
        height: 1;
        min-height: 1;
        padding: 0 1;
        border: none;
    }

    .library-mode-chip.is-active {
        border: none;
        text-style: bold underline;
    }

    /* Standalone fallback chrome: the app bundle overrides these ID/class
       rules with $ds-grid-line tokens (css/tldw_cli_modular.tcss), but the
       screen must render its workbench borders when mounted outside TldwCli
       (e.g. test harnesses), where the bundle stylesheet is not loaded. */
    #library-contract-grid {
        height: 1fr;
        min-height: 20;
        padding: 1;
        border: solid $surface-lighten-1;
    }

    .library-region {
        min-width: 0;
        height: 100%;
        min-height: 20;
        padding: 1;
        border: solid $surface-lighten-1;
    }

    #library-source-browser {
        width: 31;
        min-width: 31;
        max-width: 31;
    }

    #library-source-detail {
        width: 5fr;
    }

    #library-source-inspector {
        width: 2fr;
    }

    #library-source-browser .library-source-action {
        height: 1;
        min-height: 1;
        width: auto;
        min-width: 0;
        padding: 0 1;
        border: none;
        background: transparent;
        content-align: left middle;
        text-style: none;
    }

    #library-source-browser .library-source-action-spacer {
        height: 0;
        min-height: 0;
    }

    .library-source-active-marker {
        height: 1;
        min-height: 1;
        padding: 0 1;
        background: $surface;
        color: $text;
        text-style: bold;
    }

    .library-source-group-rule {
        height: 1;
        min-height: 1;
        color: $text-muted;
    }

    .library-source-action-meta {
        color: $text-muted;
        height: 1;
        min-height: 1;
        margin-bottom: 1;
    }

    .library-hub-spacer {
        height: 1;
        min-height: 1;
    }

    #library-collection-form Input {
        height: 3;
        min-height: 3;
        padding: 0 1;
        border: tall $surface-lighten-1;
        background: $surface;
        color: $text;
    }

    #library-collection-actions Button {
        height: 1;
        min-height: 1;
        padding: 0 1;
        border: none;
        background: transparent;
        content-align: left middle;
    }
    """

    def __init__(self, app_instance: Any, **kwargs: Any) -> None:
        super().__init__(app_instance, "library", **kwargs)
        self._local_source_records: dict[str, tuple[Mapping[str, Any], ...]] = {
            "notes": (),
            "media": (),
            "conversations": (),
        }
        self._local_source_counts: dict[str, int] = {
            "notes": 0,
            "media": 0,
            "conversations": 0,
        }
        self._local_source_total_known: dict[str, bool] = {
            "notes": True,
            "media": True,
            "conversations": True,
        }
        self._library_lookup_error: str | None = None
        self._library_lookup_recovery_state: DestinationRecoveryState | None = None
        self._library_loaded = False
        self._active_mode = "sources"
        self._library_rag_query = ""
        self._library_rag_results = ()
        self._library_rag_retrieval_status = ""
        self._library_rag_recovery_state: DestinationRecoveryState | None = None
        self._library_rag_selected_result_id = ""
        self._library_collections_loaded = False
        self._library_collections_records = ()
        self._library_collections_selected_id = ""
        self._library_collections_error = ""
        self._library_sync_profile_summary: Mapping[str, Any] | None = None
        self._library_collection_name_input = ""
        self._library_collection_description_input = ""
        self._library_collection_pending_delete_id = ""
        self._library_workspace_depth_state_cache: LibraryWorkspaceDepthState | None = None
        self._selected_conversation_id = ""
        self._library_selected_row_id: str = ""
        self._library_conversation_query: str = ""
        self._library_media_type_filter: str = "All"
        self._selected_media_id: str = ""
        self._library_media_view: str = "list"
        self._library_media_detail: Mapping[str, Any] | None = None
        self._library_media_editing: bool = False
        self._library_media_confirming_delete: bool = False
        self._library_media_highlights: list[dict[str, Any]] = []
        self._library_media_editing_analysis: bool = False
        self._library_media_content_query: str = ""
        self._library_media_content_match_index: int = 0
        self._library_notes_view: str = "list"
        self._library_notes_sort: str = "newest"
        self._library_notes_filter: str = ""
        self._library_notes_filter_records: list | None = None
        self._library_note_detail: Mapping[str, Any] | None = None
        self._selected_note_id: str = ""
        self._library_note_version: int | None = None
        self._library_note_dirty: bool = False
        self._library_note_autosave_state: str = "idle"
        self._library_notes_autosave_timer: Timer | None = None
        self._library_note_conflict_snapshot: LibraryNoteEditorState | None = None
        self._library_note_confirming_delete: bool = False
        self._library_note_preview: bool = False
        # Live-capture snapshot seeded whenever the Preview toggle fires, so
        # neither entering nor leaving preview drops unsaved edits (the
        # Markdown widget shown in preview mode has no widget text of its
        # own to read back). See ``handle_library_note_preview_toggle``.
        self._library_note_preview_snapshot: LibraryNoteEditorState | None = None
        # Guards against the spurious ``Input.Changed`` that Textual fires
        # when an ``Input(value=...)`` widget mounts with a non-empty
        # initial value: without this, opening a note (or leaving a
        # conflict) would immediately mark the note dirty and arm an
        # autosave even though the user never typed anything. Re-armed via
        # ``call_after_refresh`` after every notes-editor (re)compose.
        self._library_note_editor_armed: bool = False
        # Notes sync panel state. Seeded from config lazily on first entry
        # into sync mode (``_ensure_library_notes_sync_config_loaded``), not
        # here in __init__, so tests/screens that never open the sync panel
        # never pay for a config read.
        self._library_notes_sync_config_loaded: bool = False
        self._library_notes_sync_direction: str = "bidirectional"
        self._library_notes_sync_conflict: str = "newer_wins"
        self._library_notes_sync_auto: bool = False
        self._library_notes_sync_status: str = "idle"
        self._library_notes_sync_activity: tuple[str, ...] = ()
        self._library_notes_sync_running: bool = False
        self._library_notes_auto_sync_timer: Timer | None = None

    def on_mount(self) -> None:
        super().on_mount()
        self.set_timer(
            LIBRARY_SOURCE_SNAPSHOT_TIMEOUT_SECONDS,
            self._apply_source_snapshot_timeout,
        )
        self._refresh_local_source_snapshot()
        if self._active_mode == "collections" and not self._library_collections_loaded:
            # Deep-links that preset mode=collections call apply_navigation_context
            # BEFORE the screen is mounted (see app.py handle_screen_navigation),
            # so the is_mounted-guarded load there never fires. Kick the same
            # snapshot load here once the canvas has actually been composed.
            self.run_worker(self._sync_collections_panel(refresh_snapshot=True))

    def apply_navigation_context(self, context: Mapping[str, Any]) -> None:
        """Apply route context supplied by shell navigation.

        Args:
            context: Navigation payload from ``NavigateToScreen``. A valid
                Library mode switches the active mode. A ``conversation_id``
                selects that conversation when the local source snapshot
                arrives, defaulting the mode to Conversations when no valid
                mode is supplied.
        """
        if not isinstance(context, Mapping):
            return
        requested_mode = self._safe_text(
            context.get(LIBRARY_NAV_CONTEXT_MODE),
            max_length=64,
        )
        conversation_id = self._safe_text(
            context.get(LIBRARY_NAV_CONTEXT_CONVERSATION_ID),
            max_length=200,
        )
        target_mode = requested_mode if requested_mode in LIBRARY_MODES else ""
        if conversation_id and not target_mode:
            target_mode = LIBRARY_MODE_CONVERSATIONS
        if target_mode:
            self._active_mode = target_mode
            selected_row_id = LIBRARY_MODE_TO_ROW_ID.get(target_mode)
            if selected_row_id:
                self._library_selected_row_id = selected_row_id
            self._invalidate_library_workspace_depth_state()
        if conversation_id:
            self._selected_conversation_id = conversation_id
            self._library_selected_row_id = LIBRARY_ROW_BROWSE_CONVERSATIONS
        if self.is_mounted:
            if self._active_mode == "collections" and not self._library_collections_loaded:
                # Deep-link into Collections must load the snapshot the retired
                # chip flow ran; the panel shows the records once loaded.
                self.run_worker(self._sync_collections_panel(refresh_snapshot=True))
            else:
                self.refresh(recompose=True)

    @work(exclusive=True)
    async def _refresh_local_source_snapshot(self) -> None:
        (
            records,
            counts,
            total_known,
            lookup_error,
            recovery_state,
        ) = await self._list_local_source_snapshot()
        self._apply_local_source_snapshot(records, counts, total_known, lookup_error, recovery_state)

    def _apply_local_source_snapshot(
        self,
        records: dict[str, tuple[Mapping[str, Any], ...]],
        counts: dict[str, int],
        total_known: dict[str, bool],
        lookup_error: str | None = None,
        recovery_state: DestinationRecoveryState | None = None,
    ) -> None:
        self._local_source_records = records
        self._local_source_counts = counts
        self._local_source_total_known = total_known
        self._library_lookup_error = lookup_error
        self._library_lookup_recovery_state = recovery_state
        self._library_loaded = True
        self._invalidate_library_workspace_depth_state()
        if self.is_mounted:
            self.refresh(recompose=True)

    def _apply_source_snapshot_timeout(self) -> None:
        """Avoid leaving Library in an indefinite loading state."""
        if self._library_loaded:
            return
        self._apply_local_source_snapshot(
            {"notes": (), "media": (), "conversations": ()},
            {"notes": 0, "media": 0, "conversations": 0},
            {"notes": True, "media": True, "conversations": True},
            LIBRARY_SERVICE_ERROR_COPY,
            None,
        )

    @staticmethod
    async def _run_library_service_call(
        callable_obj: Any,
        *args: Any,
        isolate_in_worker: bool = False,
        **kwargs: Any,
    ) -> Any:
        if inspect.iscoroutinefunction(callable_obj) and not isolate_in_worker:
            return await callable_obj(*args, **kwargs)

        def invoke_service_in_worker() -> Any:
            result = callable_obj(*args, **kwargs)
            if inspect.isawaitable(result):
                async def await_result() -> Any:
                    return await result

                # Runs via asyncio.to_thread: this thread has no event loop,
                # and the awaitable must complete here so blocking async
                # services stay off the UI loop.
                return asyncio.run(await_result())  # policy-exception: worker-thread loop
            return result

        if isolate_in_worker:
            return await asyncio.to_thread(invoke_service_in_worker)

        result = await asyncio.to_thread(lambda: callable_obj(*args, **kwargs))
        if inspect.isawaitable(result):
            return await result
        return result

    @staticmethod
    def _safe_text(value: Any, fallback: str = "", *, max_length: int = 500) -> str:
        text = sanitize_string(str(value or ""), max_length=max_length).strip()
        if not text:
            return fallback
        text = text.replace("<", "").replace(">", "")
        for pattern in ("javascript:", "onclick=", "onerror="):
            text = text.replace(pattern, "")
        if validate_text_input(text, max_length=max_length, allow_html=False):
            return text
        return fallback

    @staticmethod
    def _sanitize_media_field(value: Any, *, max_length: int) -> str:
        """Sanitize a user-entered media field at the UI boundary.

        Runs the shared ``input_validation`` helpers before the value is
        handed to a persistence service (``update_media_item`` /
        ``create_highlight`` / ``save_analysis_version``), so over-long
        input is truncated and control characters are stripped, and any
        HTML/script markup that trips ``validate_text_input`` is neutralized
        -- keeping unsafe or oversized content from reaching storage. Normal
        text (including a lone ``<``/``>`` in prose) is returned unchanged.

        Args:
            value: The raw widget value to sanitize.
            max_length: Maximum length to allow before truncation.

        Returns:
            The sanitized field value.
        """
        text = sanitize_string(str(value or ""), max_length=max_length)
        if not validate_text_input(text, max_length=max_length, allow_html=False):
            # A dangerous HTML/script pattern was detected: break it up
            # (drop the angle brackets and inline handlers) but keep the
            # rest of the field rather than discarding user content.
            text = text.replace("<", "").replace(">", "")
            for pattern in ("javascript:", "onclick=", "onerror="):
                text = text.replace(pattern, "")
        return text

    @staticmethod
    def _sanitize_note_content(value: Any, *, max_length: int) -> str:
        """Length-cap Library note body content without HTML neutering.

        ``_sanitize_media_field`` is right for short metadata fields (a
        title, a highlight quote), but note bodies routinely contain code
        blocks, HTML/JS snippets in prose, or long-form markdown -- running
        them through the same control-character stripping and
        angle-bracket/``onclick=``-pattern neutering would silently mangle
        legitimate content. The only boundary a note body needs is a hard
        length cap before it reaches the persistence seam.

        Args:
            value: The raw ``TextArea`` text to sanitize.
            max_length: Maximum length to allow before truncation.

        Returns:
            ``value`` unchanged, truncated to ``max_length`` if needed.
        """
        text = str(value or "")
        if len(text) > max_length:
            text = text[:max_length]
        return text

    @staticmethod
    def _note_word_count(content: str) -> int:
        """Count words in a Library note body for the meta line's status text.

        Counts whitespace-delimited runs via a regex scan instead of
        ``len(content.split())``, which would materialize a full list of
        every word just to throw it away -- wasteful for the very large
        note bodies this editor allows (see ``LIBRARY_NOTE_CONTENT_MAX_CHARS``).

        Args:
            content: The note body text to count words in.

        Returns:
            The number of whitespace-delimited words.
        """
        return sum(1 for _ in re.finditer(r"\S+", content))

    def _library_note_keywords_from_input(self, raw_value: str) -> list[str] | None:
        """Parse and sanitize the Library note editor's keywords Input.

        Args:
            raw_value: The raw ``#library-note-keywords`` Input value.

        Returns:
            A list of sanitized, non-empty keyword tokens (each capped at
            100 characters), or ``None`` when no keywords were entered --
            matching the ``save_note`` seam's "omit keywords" contract.
        """
        parsed = [item.strip() for item in (raw_value or "").split(",") if item.strip()]
        if not parsed:
            return None
        sanitized = [self._sanitize_media_field(item, max_length=100) for item in parsed]
        sanitized = [item for item in sanitized if item]
        return sanitized or None

    def _library_notes_user_id(self) -> str:
        """Return the notes-service user id, falling back to the shared default."""
        return getattr(self.app_instance, "notes_user_id", None) or "default_user"

    @staticmethod
    def _safe_sync_scope_text(value: Any, *, max_length: int = 200) -> str | None:
        """Return a validated Sync scope value or None when unsafe/empty."""

        text = sanitize_string(str(value or ""), max_length=max_length).strip()
        text = " ".join(text.split())
        if not text:
            return None
        if validate_text_input(text, max_length=max_length, allow_html=False):
            return text
        return None

    @classmethod
    def _source_title(cls, source_type: str, record: Mapping[str, Any]) -> str:
        title_keys_by_source = {
            "notes": ("title", "name", "note_title", "note_name"),
            "media": ("title", "name", "media_title", "file_name", "url"),
            "conversations": ("title", "name", "conversation_title", "label"),
        }
        for key in title_keys_by_source[source_type]:
            title = cls._safe_text(record.get(key))
            if title:
                return title
        return "Untitled source"

    @staticmethod
    def _response_records_and_count(result: Any) -> tuple[tuple[Mapping[str, Any], ...], int, bool]:
        total = None
        if isinstance(result, Mapping):
            raw_items = result.get("items")
            pagination = result.get("pagination")
            total = result.get("total")
            if isinstance(pagination, Mapping):
                total = pagination.get("total", pagination.get("total_items", total))
        elif isinstance(result, Sequence) and not isinstance(result, (str, bytes, bytearray)):
            raw_items = result
        else:
            raw_items = ()

        records = tuple(record for record in tuple(raw_items or ()) if isinstance(record, Mapping))
        total_known = total is not None
        try:
            count = int(total) if total is not None else len(records)
        except (TypeError, ValueError):
            count = len(records)
            total_known = False
        return records, max(count, 0), total_known

    async def _notes_true_count_or_none(self, count_notes: Any, **kwargs: Any) -> int | None:
        """Fetch the authoritative local notes total, degrading quietly on failure.

        Runs inside the same ``asyncio.gather`` as the paginated ``list_notes``
        fetch (see ``_list_local_source_snapshot``). ``list_notes`` on the
        real local backend returns a plain list with no total, so the rail
        badge would otherwise always show the "showing up to N" sample-cap
        suffix. When the seam is missing entirely, the caller never invokes
        this method (guarded by ``callable(count_notes)``); when it *is*
        present but raises here, the failure is swallowed and ``None`` is
        returned so the caller falls back to the paginated response's own
        record count instead of surfacing an error or failing the whole
        snapshot fetch.

        Args:
            count_notes: The bound ``count_notes`` callable to invoke.
            **kwargs: Forwarded to ``count_notes`` (``scope``, ``user_id``).

        Returns:
            The exact notes count, or ``None`` if the call failed or
            returned something other than an ``int``.
        """
        try:
            result = await self._run_library_service_call(count_notes, isolate_in_worker=True, **kwargs)
        except Exception:
            logger.warning("Failed to fetch exact local notes count; using sample count.", exc_info=True)
            return None
        return result if isinstance(result, int) else None

    async def _list_local_source_snapshot(
        self,
    ) -> tuple[
        dict[str, tuple[Mapping[str, Any], ...]],
        dict[str, int],
        dict[str, bool],
        str | None,
        DestinationRecoveryState | None,
    ]:
        notes_service = getattr(self.app_instance, "notes_scope_service", None)
        media_service = getattr(self.app_instance, "media_reading_scope_service", None)
        conversation_service = getattr(self.app_instance, "chat_conversation_scope_service", None)
        list_notes = getattr(notes_service, "list_notes", None)
        list_media = getattr(media_service, "list_media_items", None)
        list_conversations = getattr(conversation_service, "list_conversations", None)
        count_notes = getattr(notes_service, "count_notes", None)
        count_notes_available = callable(count_notes)
        notes_user_id = getattr(self.app_instance, "notes_user_id", None) or "default_user"

        empty_records: dict[str, tuple[Mapping[str, Any], ...]] = {
            "notes": (),
            "media": (),
            "conversations": (),
        }
        empty_counts = {"notes": 0, "media": 0, "conversations": 0}
        empty_total_known = {"notes": True, "media": True, "conversations": True}
        if not all(callable(call) for call in (list_notes, list_media, list_conversations)):
            return empty_records, empty_counts, empty_total_known, LIBRARY_SERVICE_UNAVAILABLE_COPY, None

        gathered_calls = [
            self._run_library_service_call(
                list_notes,
                scope="local_note",
                limit=LIBRARY_SOURCE_PAGE_SIZES["notes"],
                offset=0,
                user_id=notes_user_id,
                isolate_in_worker=True,
            ),
            self._run_library_service_call(
                list_media,
                mode="local",
                page=1,
                results_per_page=LIBRARY_SOURCE_PAGE_SIZES["media"],
                include_keywords=False,
                isolate_in_worker=True,
            ),
            self._run_library_service_call(
                list_conversations,
                mode="local",
                limit=LIBRARY_SOURCE_PAGE_SIZES["conversations"],
                offset=0,
                isolate_in_worker=True,
            ),
        ]
        if count_notes_available:
            gathered_calls.append(
                self._notes_true_count_or_none(count_notes, scope="local_note", user_id=notes_user_id)
            )

        try:
            gathered_results = await asyncio.wait_for(
                asyncio.gather(*gathered_calls),
                timeout=LIBRARY_SOURCE_SNAPSHOT_TIMEOUT_SECONDS,
            )
        except PolicyDeniedError as exc:
            policy_message = self._safe_text(exc.user_message, LIBRARY_SERVICE_ERROR_COPY)
            recovery_state = policy_denied_recovery_state(
                exc,
                unavailable_what="Use Library sources in Console",
                stable_selector="library-source-error",
                policy_message=policy_message,
            )
            return (
                empty_records,
                empty_counts,
                empty_total_known,
                recovery_state.visible_copy,
                recovery_state,
            )
        except Exception:
            logger.warning(
                "Failed to load local Library source snapshot.",
                exc_info=True,
            )
            return empty_records, empty_counts, empty_total_known, LIBRARY_SERVICE_ERROR_COPY, None

        if count_notes_available:
            notes_result, media_result, conversation_result, notes_true_count = gathered_results
        else:
            notes_result, media_result, conversation_result = gathered_results
            notes_true_count = None

        notes, notes_count, notes_total_known = self._response_records_and_count(notes_result)
        if notes_true_count is not None:
            notes_count = notes_true_count
            notes_total_known = True
        media, media_count, media_total_known = self._response_records_and_count(media_result)
        (
            conversations,
            conversations_count,
            conversations_total_known,
        ) = self._response_records_and_count(conversation_result)
        return (
            {
                "notes": notes,
                "media": media,
                "conversations": conversations,
            },
            {
                "notes": notes_count,
                "media": media_count,
                "conversations": conversations_count,
            },
            {
                "notes": notes_total_known,
                "media": media_total_known,
                "conversations": conversations_total_known,
            },
            None,
            None,
        )

    def _has_local_sources(self) -> bool:
        return any(count > 0 for count in self._local_source_counts.values())

    def _source_snapshot_body(self) -> str:
        lines = ["Local Library source snapshot staged for Console:", ""]
        for source_type, label in (
            ("notes", "Notes"),
            ("media", "Media"),
            ("conversations", "Conversations"),
        ):
            lines.append(self._source_count_label(source_type, label))
            for index, record in enumerate(self._local_source_records[source_type], start=1):
                lines.append(f"  {index}. {self._source_title(source_type, record)}")
            lines.append("")
        return "\n".join(lines).strip()

    def _source_count_label(self, source_type: str, label: str) -> str:
        count = self._local_source_counts[source_type]
        if self._local_source_total_known[source_type]:
            return f"{label}: {count}"
        return f"{label} (showing up to {LIBRARY_SOURCE_PAGE_SIZES[source_type]}): {count}"

    def _hub_source_count_label(self, source_type: str, label: str) -> str:
        count = self._local_source_counts[source_type]
        suffix = "" if self._local_source_total_known[source_type] else "+"
        return f"{label}: {count}{suffix}"

    def _source_sample_titles(self, source_type: str) -> list[str]:
        return [
            self._source_title(source_type, record)
            for record in self._local_source_records[source_type]
        ]

    def _conversation_records(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self._local_source_records.get("conversations", ()))

    def _conversation_record_id(self, record: Mapping[str, Any], index: int) -> str:
        return self._source_record_id(record) or f"conversation-{index + 1}"

    def _ensure_selected_conversation_id(self) -> str:
        records = self._conversation_records()
        record_ids = {
            self._conversation_record_id(record, index)
            for index, record in enumerate(records)
        }
        if self._selected_conversation_id in record_ids:
            return self._selected_conversation_id
        self._selected_conversation_id = (
            self._conversation_record_id(records[0], 0) if records else ""
        )
        return self._selected_conversation_id

    def _selected_conversation_record(self) -> tuple[int, Mapping[str, Any]] | None:
        selected_id = self._ensure_selected_conversation_id()
        if not selected_id:
            return None
        for index, record in enumerate(self._conversation_records()):
            if self._conversation_record_id(record, index) == selected_id:
                return index, record
        return None

    @classmethod
    def _conversation_message_count_label(cls, record: Mapping[str, Any]) -> str:
        for key in ("message_count", "messages_count", "messageCount", "message_total", "messages_total"):
            value = record.get(key)
            if isinstance(value, int):
                return f"Messages: {value}"
            if isinstance(value, str) and value.strip().isdigit():
                return f"Messages: {value.strip()}"
        messages = record.get("messages")
        if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes, bytearray)):
            return f"Messages: {len(messages)}"
        return "Messages: unknown"

    @classmethod
    def _conversation_workspace_label(cls, record: Mapping[str, Any]) -> str:
        for key in ("workspace_name", "workspace_id", "workspace", "scope_id"):
            value = cls._safe_text(record.get(key), max_length=64)
            if value:
                return f"Workspace: {value}"
        return "Workspace: unassigned"

    @classmethod
    def _conversation_updated_label(cls, record: Mapping[str, Any]) -> str:
        for key in ("updated_at", "last_modified", "last_updated", "modified_at", "created_at"):
            value = cls._safe_text(record.get(key), max_length=64)
            if value:
                return f"Updated: {value}"
        return "Updated: unknown"

    def _conversation_handoff_enabled(
        self,
        workspace_depth_state: LibraryWorkspaceDepthState,
    ) -> bool:
        return bool(
            self._selected_conversation_record()
            and workspace_depth_state.context_handoff_enabled
            and not self._library_lookup_error
        )

    def _conversation_handoff_label(
        self,
        workspace_depth_state: LibraryWorkspaceDepthState,
    ) -> str:
        if not self._selected_conversation_record():
            return "Handoff eligibility: select a conversation first."
        if workspace_depth_state.context_handoff_enabled:
            return "Handoff eligibility: ready for Console context."
        return f"Handoff eligibility: blocked. {workspace_depth_state.context_handoff_tooltip}"

    def _conversation_browser_rows(
        self,
        workspace_depth_state: LibraryWorkspaceDepthState,
    ) -> tuple[Static | Button, ...]:
        records = self._conversation_records()
        self._ensure_selected_conversation_id()
        rows: list[Static | Button] = [
            Static(
                "Saved Conversations",
                id="library-conversations-browser-title",
                classes="destination-section",
            ),
            Static(
                "Library-owned browser for saved chats; select one to inspect metadata before handoff.",
                id="library-conversations-browser-purpose",
            ),
        ]
        if not records:
            rows.extend(
                (
                    Static(
                        "No saved conversations available in Library.",
                        id="library-conversations-empty",
                        classes="ds-recovery-callout is-blocked",
                    ),
                    Static(
                        "Create or save a Console chat, then return here to browse it.",
                        id="library-conversations-empty-recovery",
                    ),
                    Button(
                        "Open Console",
                        id="library-conversations-open-console-empty",
                        classes="library-source-action",
                        tooltip="Open Console to create or save a conversation.",
                    ),
                    Static(
                        "What appears here:",
                        id="library-conversations-empty-contents-title",
                        classes="destination-section",
                    ),
                    Static(
                        "Saved chats with title, message count, workspace, and updated time.",
                        id="library-conversations-empty-contents-metadata",
                    ),
                    Static(
                        "Select a row to enable Console handoff actions.",
                        id="library-conversations-empty-contents-handoff",
                    ),
                )
            )
            return tuple(rows)
        for index, record in enumerate(records):
            conversation_id = self._conversation_record_id(record, index)
            title = self._source_title("conversations", record)
            selected_prefix = "> " if conversation_id == self._selected_conversation_id else "  "
            rows.append(
                Button(
                    f"{selected_prefix}{title}",
                    id=f"library-conversation-select-{index}",
                    classes="library-source-action library-conversation-select",
                    tooltip="Select this conversation for Library inspection.",
                )
            )
            rows.append(
                Static(
                    " | ".join(
                        (
                            self._conversation_message_count_label(record),
                            self._conversation_workspace_label(record),
                        )
                    ),
                    id=f"library-conversation-row-{index}",
                )
            )
        selected = self._selected_conversation_record()
        if selected is None:
            return tuple(rows)
        _, selected_record = selected
        rows.extend(
            (
                Static(
                    "Selected conversation",
                    classes="destination-section",
                ),
                Static(
                    self._source_title("conversations", selected_record),
                    id="library-selected-conversation-title",
                ),
                Static(
                    self._conversation_message_count_label(selected_record),
                    id="library-selected-conversation-message-count",
                ),
                Static(
                    self._conversation_workspace_label(selected_record),
                    id="library-selected-conversation-workspace",
                ),
                Static(
                    self._conversation_updated_label(selected_record),
                    id="library-selected-conversation-updated",
                ),
                Static(
                    "Source authority: local",
                    id="library-selected-conversation-authority",
                ),
                Static(
                    self._conversation_handoff_label(workspace_depth_state),
                    id="library-selected-conversation-handoff",
                ),
            )
        )
        return tuple(rows)

    def _selected_conversation_handoff_payload(self) -> ChatHandoffPayload | None:
        selected = self._selected_conversation_record()
        if selected is None:
            return None
        index, record = selected
        conversation_id = self._conversation_record_id(record, index)
        title = self._source_title("conversations", record)
        message_count = self._conversation_message_count_label(record)
        workspace_label = self._conversation_workspace_label(record)
        updated_label = self._conversation_updated_label(record)
        body = "\n".join(
            (
                f"Conversation: {title}",
                f"Conversation ID: {conversation_id}",
                message_count,
                workspace_label,
                updated_label,
                "Source authority: local",
            )
        )
        return ChatHandoffPayload(
            source="library",
            item_type="conversation",
            title=title,
            body=body,
            source_id=conversation_id,
            display_summary=f"Conversation staged: {title}",
            suggested_prompt="Use this conversation as source context for my next question.",
            runtime_backend="local",
            source_owner="local",
            source_selector_state="local",
            discovery_owner="conversation",
            discovery_entity_id=conversation_id,
            metadata={
                "conversation_id": conversation_id,
                "conversation_title": title,
                "message_count_label": message_count,
                "workspace_label": workspace_label,
                "updated_label": updated_label,
                "source_authority": "local",
            },
        )

    def _selected_media_handoff_payload(self) -> ChatHandoffPayload | None:
        """Build the Console handoff payload for the open Library media item.

        Mirrors ``_selected_conversation_handoff_payload``, but reads the
        currently loaded media detail (``_library_media_detail``) instead of
        a selected browser row -- the media viewer's "Use in Chat" action
        stages whatever item is open in the in-canvas viewer, not a row
        selection from the list.

        Returns:
            A ``ChatHandoffPayload`` staging the open media item as Console
            context, or None when no media item is currently loaded.
        """
        detail = self._library_media_detail
        if not isinstance(detail, Mapping):
            return None
        viewer = build_library_media_viewer_state(detail)
        media_id = viewer.media_id or self._safe_text(self._selected_media_id)
        if not media_id:
            return None
        title = viewer.title or "Untitled source"
        media_type = (
            self._safe_text(detail.get("type"))
            or self._safe_text(detail.get("media_type"))
            or "unknown"
        )
        content = viewer.content
        excerpt = content[:LIBRARY_MEDIA_HANDOFF_EXCERPT_CHARS]
        body_truncated = len(content) > LIBRARY_MEDIA_HANDOFF_EXCERPT_CHARS
        body_lines = [f"Media: {title}", f"Media ID: {media_id}"]
        body_lines.extend(
            line
            for line in viewer.metadata_lines
            if line.startswith(("Type:", "Author:", "Keywords:"))
        )
        body_lines.append("Source authority: local")
        body_lines.append("")
        body_lines.append("Content excerpt:")
        body_lines.append(excerpt if excerpt else "No stored content.")
        body = "\n".join(body_lines)
        return ChatHandoffPayload(
            source="library",
            item_type="media",
            title=title,
            body=body,
            body_truncated=body_truncated,
            source_id=media_id,
            display_summary=f"Media staged: {title}",
            suggested_prompt="Use this media as source context for my next question.",
            runtime_backend="local",
            source_owner="local",
            source_selector_state="local",
            discovery_owner="media",
            discovery_entity_id=media_id,
            metadata={
                "media_id": media_id,
                "media_title": title,
                "media_type": media_type,
            },
        )

    def _source_recent_label(self, source_type: str) -> str:
        recent = self._source_recent_value(source_type)
        return f"Recent: {recent}"

    def _hub_table_cell(self, value: str, width: int = LIBRARY_HUB_RECENT_LABEL_WIDTH) -> str:
        """Keep hub table cells readable in terminal-width layouts."""
        clean_value = " ".join(value.split())
        if len(clean_value) <= width:
            return clean_value
        suffix = "..."
        limit = max(1, width - len(suffix))
        shortened = clean_value[:limit].rsplit(" ", 1)[0].strip()
        if not shortened:
            shortened = clean_value[:limit].strip()
        return f"{shortened}{suffix}"

    def _hub_section_rule(self, label: str, widget_id: str) -> Static:
        rule_width = 74
        suffix_width = max(3, rule_width - len(label) - 4)
        return Static(
            f"-- {label} {'-' * suffix_width}",
            id=widget_id,
            classes="destination-section",
        )

    def _hub_source_count_value(self, source_type: str) -> str:
        count = self._local_source_counts.get(source_type, 0)
        suffix = "" if self._local_source_total_known.get(source_type, True) else "+"
        return f"{count}{suffix}"

    def _hub_console_status(self, source_type: str) -> str:
        if self._local_source_counts.get(source_type, 0) <= 0:
            return "blocked: no source"
        workspace_depth_state = self._library_workspace_depth_state()
        if workspace_depth_state.context_handoff_enabled:
            return "ready"
        return "blocked: workspace gate"

    def _hub_recent_sources_label(self) -> str:
        return "; ".join(
            (
                f"Notes: {self._source_recent_value('notes')}",
                f"Media: {self._source_recent_value('media')}",
                f"Conversations: {self._source_recent_value('conversations')}",
            )
        )

    def _hub_readiness_counts(self) -> tuple[int, int, int]:
        active_modules = sum(
            1
            for source_type in ("notes", "media", "conversations")
            if self._local_source_counts.get(source_type, 0) > 0
        )
        workspace_depth_state = self._library_workspace_depth_state()
        eligible_modules = active_modules if workspace_depth_state.context_handoff_enabled else 0
        blocked_modules = max(0, active_modules - eligible_modules)
        return active_modules, eligible_modules, blocked_modules

    def _hub_state_summary(self) -> str:
        _, eligible_modules, blocked_modules = self._hub_readiness_counts()
        workspace_depth_state = self._library_workspace_depth_state()
        console_state = "ready" if workspace_depth_state.context_handoff_enabled else "blocked"
        return "\n".join(
            (
                f"State: Local workspace | Browse all workspaces | Console staging {console_state}",
                (
                    f"Inventory: Notes {self._hub_source_count_value('notes')} | "
                    f"Media {self._hub_source_count_value('media')} | "
                    f"Conversations {self._hub_source_count_value('conversations')} | "
                    f"Console eligible {eligible_modules} | Blocked {blocked_modules}"
                ),
            )
        )

    def _hub_readiness_summary(self) -> str:
        _, eligible_modules, blocked_modules = self._hub_readiness_counts()
        blocked_suffix = "module" if blocked_modules == 1 else "modules"
        eligible_suffix = "module" if eligible_modules == 1 else "modules"
        return "\n".join(
            (
                self._hub_key_value_row("Eligible", f"{eligible_modules} {eligible_suffix}"),
                self._hub_key_value_row(
                    "Blocked",
                    f"{blocked_modules} workspace-gated {blocked_suffix}",
                ),
                self._hub_key_value_row("Recent", self._hub_recent_sources_label()),
                self._hub_key_value_row(
                    "Next",
                    "Link sources to the active workspace or open an owner screen.",
                ),
            )
        )

    def _hub_key_value_row(self, label: str, value: str, *, label_width: int = 14) -> str:
        return f"{label:<{label_width}} {value}"

    def _source_recent_value(self, source_type: str) -> str:
        titles = self._source_sample_titles(source_type)
        if not titles:
            return "none"
        return self._hub_table_cell(titles[0])

    def _hub_inventory_readiness_label(self, source_type: str, unit: str) -> str:
        count_label = self._hub_source_count_value(source_type)
        try:
            count = int(count_label.rstrip("+"))
        except ValueError:
            count = 0
        if count == 1:
            return f"{count_label} {unit}"
        return f"{count_label} {unit}s"

    def _hub_inventory_console_label(self, source_type: str) -> str:
        return f"Console {self._hub_console_status(source_type)}"

    def _hub_inventory_row(
        self,
        *,
        source: str,
        readiness: str,
        owner: str,
        action: str,
        console: str,
        widget_id: str,
    ) -> Static:
        source_cell = self._hub_table_cell(
            source,
            LIBRARY_HUB_INVENTORY_SOURCE_COLUMN_WIDTH,
        )
        readiness_cell = self._hub_table_cell(
            readiness,
            LIBRARY_HUB_INVENTORY_READINESS_COLUMN_WIDTH,
        )
        owner_cell = self._hub_table_cell(
            owner,
            LIBRARY_HUB_INVENTORY_OWNER_COLUMN_WIDTH,
        )
        action_cell = self._hub_table_cell(
            action,
            LIBRARY_HUB_INVENTORY_ACTION_COLUMN_WIDTH,
        )
        return Static(
            "\n".join(
                (
                    (
                        f"{source_cell:<{LIBRARY_HUB_INVENTORY_SOURCE_COLUMN_WIDTH}} "
                        f"{readiness_cell:<{LIBRARY_HUB_INVENTORY_READINESS_COLUMN_WIDTH}} "
                        f"{owner_cell:<{LIBRARY_HUB_INVENTORY_OWNER_COLUMN_WIDTH}} "
                        f"{action_cell:<{LIBRARY_HUB_INVENTORY_ACTION_COLUMN_WIDTH}}"
                    ),
                    f"  {console}",
                )
            ),
            markup=False,
            id=widget_id,
            classes="library-hub-card",
        )

    def _hub_spacer(self, widget_id: str) -> Static:
        return Static("", id=widget_id, classes="library-hub-spacer")

    def _import_export_workflow_rows(self) -> tuple[Static, ...]:
        return (
            Static(
                "Library Import/Export Workflow",
                id="library-import-export-workflow-title",
                classes="destination-section",
            ),
            Static(
                "Library owns source acquisition framing; Ingest and Media own deeper file handling.",
                id="library-import-export-owner-boundary",
            ),
            Static(
                "Import source material",
                id="library-import-export-import-title",
                classes="destination-section",
            ),
            Static(
                "Open Ingest to add files, URLs, transcripts, source packages, or external material.",
                id="library-import-export-ingest-copy",
            ),
            Static(
                "Imported material returns here as notes, media, conversations, or indexed sources.",
                id="library-import-export-return-copy",
            ),
            Static(
                "Media review",
                id="library-import-export-media-title",
                classes="destination-section",
            ),
            Static(
                "Full Media ingestion and review stays in Media.",
                id="library-import-export-media-boundary",
            ),
            Static(
                "Ownership boundaries",
                id="library-import-export-boundaries-title",
                classes="destination-section",
            ),
            Static(
                "Artifact export stays in Artifacts.",
                id="library-import-export-artifact-boundary",
            ),
            Static(
                "Generic file management stays outside Library.",
                id="library-import-export-file-boundary",
            ),
            Static(
                "Export is not wired here yet.",
                id="library-import-export-export-blocked",
                classes="ds-recovery-callout is-blocked",
            ),
            Static(
                "Return path: come back to Library after import to see new hub inventory.",
                id="library-import-export-return-path",
            ),
        )

    def _import_export_inspector_rows(self) -> tuple[Static, ...]:
        return (
            Static(
                "Import/Export inspector",
                id="library-inspector-title",
                classes="destination-section",
            ),
            Static(
                "Current scope: source-level Library acquisition.",
                id="library-import-export-inspector-scope",
            ),
            Static(
                "Handoff target: Ingest for new source material; Media for media review.",
                id="library-import-export-inspector-targets",
            ),
            Static(
                "Prerequisite: choose the owner workflow before leaving Library.",
                id="library-import-export-inspector-prerequisite",
            ),
            Static(
                "Blocked: Library source export is planned but not implemented here.",
                id="library-import-export-inspector-blocked",
                classes="ds-recovery-callout is-blocked",
            ),
            Static(
                "Recovery: use owner screens for existing export paths until Library export is wired.",
                id="library-import-export-inspector-recovery",
            ),
        )

    def _source_action_meta(self, widget_id: str) -> str:
        if widget_id == "library-open-notes":
            return (
                f"Notes: {self._hub_source_count_value('notes')} | "
                "global browse | stage gated"
            )
        if widget_id == "library-open-media":
            return (
                f"Media: {self._hub_source_count_value('media')} | "
                "global browse | stage gated"
            )
        if widget_id == "library-open-conversations":
            return (
                f"Conversations: {self._hub_source_count_value('conversations')} | "
                "global browse | stage gated"
            )
        if widget_id == "library-open-search":
            return "Retrieval | query first | stage evidence"
        if widget_id == "library-open-collections":
            return "Collections | read/review | items WIP"
        return ""

    def _source_module_action_widgets(self) -> tuple[Button | Static, ...]:
        action_groups: tuple[tuple[str, tuple[tuple[str, str, str], ...]], ...] = (
            (
                "Sources",
                (
                    ("Open Notes", "library-open-notes", "Open saved notes and workspaces."),
                    (
                        "Open Media",
                        "library-open-media",
                        "Browse the media library, transcripts, analysis, and read-it-later.",
                    ),
                    (
                        "Open Conversations",
                        "library-open-conversations",
                        "Open saved conversation browsing inside Library.",
                    ),
                ),
            ),
            (
                "Retrieval",
                (
                    ("Search/RAG", "library-open-search", "Search or ask over indexed sources."),
                ),
            ),
            (
                "Movement",
                (
                    (
                        "Import/Export Sources",
                        "library-open-import-export",
                        "Open source import and export tools.",
                    ),
                    (
                        "Collections",
                        "library-open-collections",
                        "Read, review, and reuse saved Library content.",
                    ),
                ),
            ),
            (
                "Learning",
                (
                    ("Study Dashboard", "library-open-study", "Open Study globally or with Library sources."),
                    ("Flashcards", "library-open-flashcards", "Open Flashcards globally or with Library sources."),
                    ("Quizzes", "library-open-quizzes", "Open Quizzes globally or with Library sources."),
                ),
            ),
        )

        widgets: list[Button | Static] = []
        active_action_id = self._active_source_action_id()
        hide_search_action = self._active_mode == "search"
        hide_learning_actions = self._active_mode == "collections" or self._active_mode in LIBRARY_STUDY_HANDOFF_MODES
        for group_index, (group_label, actions) in enumerate(action_groups):
            if group_label == "Retrieval" and hide_search_action:
                continue
            if group_label == "Learning" and hide_learning_actions:
                continue
            if group_index > 0:
                widgets.append(
                    Static(
                        "----",
                        id=f"library-source-group-rule-{group_label.lower().replace(' ', '-')}",
                        classes="library-source-group-rule",
                    )
                )
            group_heading = Static(group_label, classes="destination-section library-source-group")
            widgets.append(group_heading)
            for index, (label, widget_id, tooltip) in enumerate(actions):
                classes = "library-source-action"
                if widget_id == active_action_id:
                    classes = f"{classes} is-active"
                    widgets.append(
                        Static(
                            f"> Active: {label}",
                            id="library-source-active-marker",
                            classes="library-source-active-marker",
                        )
                    )
                button = Button(
                    label,
                    id=widget_id,
                    classes=classes,
                    tooltip=tooltip,
                )
                widgets.append(button)
                meta = self._source_action_meta(widget_id)
                if meta:
                    widgets.append(
                        Static(
                            meta,
                            id=f"{widget_id}-meta",
                            classes="library-source-action-meta",
                        )
                    )
                if index < len(actions) - 1:
                    spacer = Static("", classes="library-source-action-spacer")
                    widgets.append(spacer)
        return tuple(widgets)

    def _hub_inspector_rows(
        self,
        workspace_depth_state: LibraryWorkspaceDepthState,
    ) -> tuple[Static, ...]:
        handoff_copy = (
            "Console handoff is secondary and uses eligible Library content only."
            if self._has_local_sources()
            else "Console handoff is secondary and unavailable until Library content exists."
        )
        return (
            Static(
                "Hub inspector",
                id="library-inspector-title",
                classes="destination-section",
            ),
            Static("Selected", id="library-hub-inspector-selected-title", classes="destination-section"),
            Static(LIBRARY_INSPECTOR_EMPTY_COPY, id="library-inspector-empty"),
            Static("Available now", id="library-hub-inspector-available-title", classes="destination-section"),
            Static(
                "Browse Notes, Media, Conversations, Collections, and Search/RAG owner screens.",
                id="library-hub-inspector-available-now",
            ),
            Static("Blocked", id="library-hub-inspector-blocked-title", classes="destination-section"),
            Static(
                handoff_copy,
                id="library-hub-inspector-console-boundary",
                classes="ds-recovery-callout is-blocked",
            ),
            Static("Next action", id="library-hub-inspector-next-title", classes="destination-section"),
            Static(
                LIBRARY_INSPECTOR_EMPTY_NEXT_ACTION_COPY,
                id="library-inspector-empty-next-action",
            ),
            Static("Details", id="library-hub-inspector-details-title", classes="destination-section"),
            Static(
                "Notes owner: Notes screen handles editing, sync, templates, export, and delete.",
                id="library-hub-inspector-notes-owner",
            ),
            Static(
                "Media owner: Media screen handles browse, ingest review, analysis, and read-it-later.",
                id="library-hub-inspector-media-owner",
            ),
            Static(
                "Search/RAG owner: Library Search/RAG handles retrieval, evidence, and saved searches.",
                id="library-hub-inspector-rag-owner",
            ),
            Static(
                "Workspace boundary: browse/search remains global; active workspace gates staging and manipulation.",
                id="library-hub-inspector-workspace-boundary",
            ),
            Static(
                workspace_depth_state.handoff_label,
                id="library-hub-inspector-handoff-state",
            ),
        )

    @classmethod
    def _source_record_id(cls, record: Mapping[str, Any]) -> str | None:
        for key in (
            "id",
            "uuid",
            "record_id",
            "backing_id",
            "source_id",
            "item_id",
            "media_id",
            "note_id",
            "conversation_id",
            "backing_media_id",
        ):
            value = cls._safe_text(record.get(key), max_length=128)
            if value:
                return value
        return None

    def _study_source_items(self) -> tuple[StudySourceItem, ...]:
        source_items: list[StudySourceItem] = []
        source_type_map = {
            "notes": "note",
            "media": "media",
        }
        for source_type, study_source_type in source_type_map.items():
            for record in self._local_source_records[source_type]:
                source_id = self._source_record_id(record)
                if not source_id:
                    continue
                source_items.append(
                    StudySourceItem(
                        source_type=study_source_type,
                        source_id=source_id,
                        label=self._source_title(source_type, record),
                    )
                )
        return tuple(source_items)

    def _source_count_metadata(self) -> dict[str, int | None]:
        metadata: dict[str, int | None] = {}
        for source_type in ("notes", "media", "conversations"):
            count = self._local_source_counts[source_type]
            metadata[f"{source_type}_sample_count"] = len(self._local_source_records[source_type])
            metadata[f"{source_type}_total_count"] = (
                count if self._local_source_total_known[source_type] else None
            )
            if self._local_source_total_known[source_type]:
                metadata[f"{source_type}_count"] = count
        return metadata

    def _source_snapshot_metadata(self) -> dict[str, Any]:
        return {
            **self._source_count_metadata(),
            "note_titles": self._source_sample_titles("notes"),
            "media_titles": self._source_sample_titles("media"),
            "conversation_titles": self._source_sample_titles("conversations"),
        }

    def _workspace_source_records(self) -> Mapping[str, tuple[Mapping[str, Any], ...]]:
        return {
            source_type: tuple(self._local_source_records[source_type])
            for source_type in ("notes", "media", "conversations")
        }

    def _invalidate_library_workspace_depth_state(self) -> None:
        self._library_workspace_depth_state_cache = None

    def _next_local_workspace_identity(self) -> tuple[str, str]:
        """Return a collision-free local workspace id and display name."""
        registry_service = getattr(self.app_instance, "workspace_registry_service", None)
        existing_workspaces = (
            tuple(registry_service.list_workspaces(include_archived=True))
            if registry_service is not None
            else ()
        )
        existing_ids = {workspace.workspace_id for workspace in existing_workspaces}
        existing_names = {workspace.name for workspace in existing_workspaces}
        index = 1
        while True:
            workspace_id = f"workspace-local-{index}"
            workspace_name = f"Workspace {index}"
            if workspace_id not in existing_ids and workspace_name not in existing_names:
                return workspace_id, workspace_name
            index += 1

    def _library_workspace_depth_state(
        self,
        *,
        refresh: bool = False,
    ) -> LibraryWorkspaceDepthState:
        if refresh or self._library_workspace_depth_state_cache is None:
            self._library_workspace_depth_state_cache = build_library_workspace_depth_state(
                registry_service=getattr(self.app_instance, "workspace_registry_service", None),
                source_records=self._workspace_source_records(),
            )
        return self._library_workspace_depth_state_cache

    def _library_workspace_scope_label(
        self,
        workspace_depth_state: LibraryWorkspaceDepthState,
    ) -> Text:
        """Return the left-rail workspace scope copy for the active Library mode."""
        return Text.from_markup(
            "Active workspace: "
            f"{escape_markup(workspace_depth_state.workspace_name)}"
        )

    def _source_study_context(self) -> StudyScopeContext | None:
        if not self._has_source_study_context():
            return None
        material_titles: list[str] = []
        for source_type in ("notes", "media", "conversations"):
            material_titles.extend(self._source_sample_titles(source_type))
        return StudyScopeContext(
            material_source=MATERIAL_SOURCE_LIBRARY,
            material_title=MATERIAL_TITLE_LIBRARY_SOURCES,
            material_summary=self._source_snapshot_body(),
            material_titles=tuple(material_titles),
            source_items=self._study_source_items(),
            return_hint=MATERIAL_SOURCE_LIBRARY,
        )

    def _source_study_handoff_titles(self) -> tuple[str, ...]:
        material_titles: list[str] = []
        for source_type in ("notes", "media", "conversations"):
            material_titles.extend(self._source_sample_titles(source_type))
        return tuple(material_titles)

    def _has_source_study_context(self) -> bool:
        return self._has_local_sources()

    def _study_handoff_copy(self) -> dict[str, str]:
        mode = LIBRARY_STUDY_HANDOFF_MODES.get(
            self._active_mode,
            LIBRARY_STUDY_HANDOFF_MODES["study"],
        )
        titles = self._source_study_handoff_titles()
        has_context = self._has_source_study_context()
        action_label = mode["action_label"]
        if has_context and titles:
            context_copy = f"Carries forward: {', '.join(titles)}"
        elif has_context:
            context_copy = "Carries forward: Library source snapshot (titles unavailable)"
        else:
            context_copy = "No Library source snapshot will be carried forward."
        return {
            "label": mode["label"],
            "action_label": action_label,
            "context": context_copy,
            "owner": (
                "Library prepares source context only; Study owns sessions, "
                "generation, review, and attempts."
            ),
            "wip": (
                "WIP: provider-backed generation and collection-scoped study "
                "remain owned by later Study slices."
            ),
            "recovery": (
                "Source snapshot is ready; open "
                f"{action_label} to continue with this Library context."
                if has_context
                else (
                    "Import sources or create notes first, or open "
                    f"{action_label} globally without Library context."
                )
            ),
        }

    def _status_label(self) -> str:
        if not self._library_loaded:
            return "Loading"
        if self._library_lookup_recovery_state is not None:
            return self._library_lookup_recovery_state.status_label
        if self._library_lookup_error is None:
            return "Ready" if self._has_local_sources() else "Empty"
        if "unavailable" in self._library_lookup_error.lower():
            return "Unavailable"
        return "Blocked"

    def _search_rag_status_label(self) -> str:
        panel_state = self._library_rag_panel_state()
        if panel_state.scope.status == "blocked":
            if not panel_state.scope.has_available_sources:
                return "Blocked: no Library sources"
            return "Blocked: select Library source scope"
        if panel_state.query_state.status == "blocked":
            return f"Blocked: {panel_state.query_state.run_action.disabled_reason}"
        if panel_state.retrieval_status == "searching":
            return "Searching"
        if panel_state.retrieval_status in {"blocked", "failed"}:
            return "Blocked: retrieval unavailable"
        if panel_state.retrieval_status == "empty":
            return "No results"
        if panel_state.selected_result is not None:
            return "Evidence selected"
        if panel_state.results:
            return "Results ready"
        return "Ready"

    def _status_row_copy(self) -> str:
        if self._active_mode == "conversations":
            return f"Library | Conversations | {self._status_label()} | Local"
        if self._active_mode == "search":
            return f"Library | Search/RAG | {self._search_rag_status_label()} | Local"
        if self._active_mode == "import-export":
            return f"Library | Import/Export | {self._status_label()} | Local"
        if self._active_mode == "collections":
            collections_status = "Empty"
            if self._library_collections_error:
                collections_status = "Unavailable"
            elif self._library_collections_records:
                collections_status = "Ready"
            return f"Library | Collections | {collections_status} | Local"
        return (
            "Library | Content hub, imports, Search/RAG, Workspaces, "
            f"Collections, Study | {self._status_label()} | Local"
        )

    def _active_mode_contract(self) -> Mapping[str, str]:
        return LIBRARY_MODES.get(self._active_mode, LIBRARY_MODES["sources"])

    def _active_column_titles(self) -> tuple[str, str, str]:
        return LIBRARY_COLUMN_TITLES.get(self._active_mode, LIBRARY_COLUMN_TITLES["sources"])

    def _active_source_action_id(self) -> str:
        return {
            "conversations": "library-open-conversations",
            "search": "library-open-search",
            "collections": "library-open-collections",
            "import-export": "library-open-import-export",
        }.get(self._active_mode, "")

    def _should_show_local_snapshot_region(self) -> bool:
        return self._active_mode in LIBRARY_LOCAL_SNAPSHOT_MODES

    def _library_rag_panel_state(self) -> LibraryRagPanelState:
        return LibraryRagPanelState.from_values(
            source_counts={
                "notes": self._local_source_counts.get("notes", 0),
                "media": self._local_source_counts.get("media", 0),
                "conversations": self._local_source_counts.get("conversations", 0),
                "workspaces": 0,
                "collections": 0,
            },
            query=self._library_rag_query,
            mode="rag",
            results=self._library_rag_results,
            selected_result_id=self._library_rag_selected_result_id,
            retrieval_status=self._library_rag_retrieval_status,
            recovery_copy=(
                self._library_rag_recovery_state.visible_copy
                if self._library_rag_recovery_state is not None
                else ""
            ),
            recovery_selector=(
                self._library_rag_recovery_state.stable_selector
                if self._library_rag_recovery_state is not None
                else ""
            ),
            dependencies_ready=True,
            index_ready=True,
            provider_ready=True,
        )

    def _library_collections_panel_state(self) -> LibraryCollectionsPanelState:
        status = "loading"
        if self._library_collections_error:
            status = "error"
        elif self._library_collections_loaded:
            status = "ready"
        return LibraryCollectionsPanelState.from_values(
            collections=self._library_collections_records,
            selected_collection_id=self._library_collections_selected_id,
            status=status,
            error_message=self._library_collections_error,
            create_name=self._library_collection_name_input,
            rename_name=self._library_collection_name_input,
            sync_profile_summary=self._library_sync_profile_summary,
        )

    def _collections_inspector_rows(
        self,
        panel_state: LibraryCollectionsPanelState,
    ) -> tuple[Static, ...]:
        selected = panel_state.selected_collection
        if selected is None:
            return (
                Static("Collections Inspector", id="library-inspector-title", classes="destination-section"),
                Static("Selected: none", id="library-collection-inspector-empty"),
                Static(
                    (
                        "Collections are for reading and reviewing saved content; "
                        "the local item reader is not wired in this slice."
                    ),
                    id="library-collection-inspector-empty-next-action",
                ),
                Static(
                    (
                        "Global browsing/search remains available; active staging and manipulation "
                        "stay workspace-gated."
                    ),
                    id="library-collection-inspector-global-rule",
                ),
                Static("Action status", classes="destination-section"),
                Static(
                    "Available now: create, rename, delete records",
                    id="library-collection-inspector-empty-local-actions",
                ),
                Static(
                    "Blocked later: item reader, Search/RAG, Study, Console handoff, server sync",
                    id="library-collection-inspector-empty-later-actions",
                    classes="ds-recovery-callout is-blocked",
                ),
                Static(
                    "Next: select or create a Collection record to inspect local item-reader readiness.",
                    id="library-collection-inspector-empty-recovery",
                ),
            )
        return (
            Static("Selected Collection Record", id="library-inspector-title", classes="destination-section"),
            Static(f"Selected: {selected.name}", id="library-collection-inspector-selected"),
            Static(selected.name, id="library-collection-inspector-name"),
            Static(
                f"Stored item count: {selected.item_count_label}",
                id="library-collection-inspector-item-count",
            ),
            Static(
                "Collection item reader: not wired locally yet.",
                id="library-collection-inspector-reader-state",
            ),
            Static(
                "Workspace rule: Library browsing/search stays global; Console/RAG staging follows active workspace.",
                id="library-collection-inspector-workspace-rule",
            ),
            Static("Action status", classes="destination-section"),
            Static(
                "Available now: create, rename, delete records",
                id="library-collection-inspector-local-actions",
            ),
            Static(
                "Blocked later: item reader, Search/RAG, Study, Console handoff, server sync",
                id="library-collection-inspector-later-actions",
                classes="ds-recovery-callout is-blocked",
            ),
            Static(
                "Next: collection item adapters are required before item-level actions unlock.",
                id="library-collection-inspector-next",
            ),
            Static(
                "Disabled: collection item Search/RAG is not wired yet.",
                id="library-collection-inspector-rag-blocked",
                classes="ds-recovery-callout is-blocked",
            ),
            Static(
                "Disabled: collection item Console handoff is not wired yet.",
                id="library-collection-inspector-console-blocked",
                classes="ds-recovery-callout is-blocked",
            ),
            Static(
                (
                    "Recovery: use existing Library Search/RAG or individual eligible sources "
                    "until collection item adapters are available."
                ),
                id="library-collection-inspector-recovery",
            ),
            Static("What this means", classes="destination-section"),
            Static(
                "This is a read-only sync dry run. No server writes can run from this screen.",
                id="library-collection-inspector-sync-meaning",
            ),
            Static(selected.sync_status_label, id="library-collection-inspector-sync-status"),
            Static(selected.sync_status_detail, id="library-collection-inspector-sync-detail"),
        )

    def _workspace_handoff_summary_label(self, state: LibraryWorkspaceDepthState) -> str:
        """Shorten ``state.handoff_label`` for the Workspace group's Handoff row.

        Drops the redundant "Console/RAG handoff: " prefix (the "Handoff"
        row label already says as much) and prefixes a "●" glyph directly
        before a nonzero blocked count, so this single line carries the
        signal the retired blocked-state callouts used to repeat three
        times.
        """
        label = state.handoff_label
        if label.startswith(LIBRARY_HANDOFF_LABEL_PREFIX):
            label = label[len(LIBRARY_HANDOFF_LABEL_PREFIX) :]
        match = re.search(r"(\d+) blocked", label)
        if match and int(match.group(1)) > 0:
            label = f"{label[: match.start()]}● {match.group(0)}{label[match.end() :]}"
        return label

    def _workspaces_detail_rows(
        self,
        state: LibraryWorkspaceDepthState,
    ) -> tuple[Static, ...]:
        """Build the Workspace group's Details rail rows.

        Only the two facts that change staging eligibility survive here: the
        active workspace and the Console/RAG handoff eligible/blocked
        counts. The former policy-prose lines (global browse/search rule,
        staging-scope rule, per-source visibility label, collections and
        import/export capability notes) restated the same "browse stays
        global, staging is workspace-scoped" rule several different ways;
        the Handoff row below now carries that state on its own.
        """
        return (
            Static(
                "Workspace",
                id="library-details-group-workspace",
                classes="library-details-group",
            ),
            Static(
                library_dim_label_text("Active", state.workspace_name),
                id="library-workspaces-active-workspace",
                classes="library-details-row",
            ),
            Static(
                library_dim_label_text(
                    "Handoff", self._workspace_handoff_summary_label(state)
                ),
                id="library-workspaces-handoff",
                classes="library-details-row",
            ),
        )

    def _workspaces_inspector_rows(
        self,
        state: LibraryWorkspaceDepthState,
    ) -> tuple[Static, ...]:
        return (
            Static("Handoff status", id="library-inspector-title", classes="destination-section"),
            Static(state.handoff_label, id="library-workspaces-inspector-handoff"),
            Static(
                self._workspace_handoff_blocked_label(state),
                id="library-workspaces-inspector-blocked",
            ),
            Static(
                self._workspace_handoff_why_label(state),
                id="library-workspaces-inspector-why",
            ),
            Static(
                self._workspace_handoff_fix_label(state),
                id="library-workspaces-inspector-fix",
            ),
            Static(
                self._workspace_handoff_action_label(state),
                id="library-workspaces-inspector-action",
            ),
            Static(
                "Rule: browse/search stay global; staging is workspace-scoped.",
                id="library-workspaces-inspector-rule",
            ),
        )

    def _workspace_handoff_recovery_label(self, state: LibraryWorkspaceDepthState) -> str:
        if not state.source_rows:
            workspace_name = state.workspace_name.strip()
            if workspace_name and workspace_name not in {"unavailable"}:
                return f"import or assign sources to {workspace_name}"
            return "import or assign sources to a workspace"
        workspace_name = state.workspace_name.strip()
        if workspace_name and workspace_name not in {"Local Default", "unavailable"}:
            return f"Copy/link blocked sources to {workspace_name}"
        return "Assign blocked sources to the active workspace"

    def _workspace_handoff_blocked_label(self, state: LibraryWorkspaceDepthState) -> str:
        if state.context_handoff_enabled:
            return "Ready: all visible sources can be staged"
        if not state.source_rows:
            return "Blocked: no workspace sources"
        workspace_name = state.workspace_name.strip()
        if workspace_name and workspace_name not in {"Local Default", "unavailable"}:
            return f"Blocked: some sources are outside {workspace_name}"
        return "Blocked: sources need active workspace assignment"

    def _workspace_handoff_fix_label(self, state: LibraryWorkspaceDepthState) -> str:
        if state.context_handoff_enabled:
            return "Fix: no action needed"
        if not state.source_rows:
            return f"Fix: {self._workspace_handoff_recovery_label(state)}"
        return f"Fix: {self._workspace_handoff_recovery_label(state)}"

    def _workspace_handoff_why_label(self, state: LibraryWorkspaceDepthState) -> str:
        if state.context_handoff_enabled:
            return "Why: all visible sources belong to the active workspace."
        if not state.source_rows:
            return "Why: no sources are assigned to this workspace yet."
        return "Why: at least one visible source belongs to another workspace."

    def _workspace_handoff_action_label(self, state: LibraryWorkspaceDepthState) -> str:
        if state.context_handoff_enabled:
            return "Action: Use in Console is available."
        if not state.source_rows:
            return "Action: Import sources or assign sources before staging."
        return "Action: Copy/link blocked sources before staging."

    def _study_handoff_detail_widget(self) -> Vertical:
        copy = self._study_handoff_copy()
        recovery_classes = (
            "ds-recovery-callout"
            if self._has_local_sources()
            else "ds-recovery-callout is-blocked"
        )
        action_button_id = {
            "study": "library-open-study",
            "flashcards": "library-open-flashcards",
            "quizzes": "library-open-quizzes",
        }.get(self._active_mode, "library-open-study")
        handoff_toolbar = Horizontal(
            Button(
                copy["action_label"],
                id=action_button_id,
                classes="library-canvas-action",
                compact=True,
                tooltip=(
                    f"Open {copy['action_label']} with the current Library "
                    "source snapshot, or globally when none is available."
                ),
            ),
            id="library-study-handoff-actions",
            classes="ds-toolbar",
        )
        handoff_toolbar.styles.height = "auto"
        return Vertical(
            Static(
                f"{copy['label']} handoff",
                id="library-study-handoff-purpose",
                classes="destination-section",
            ),
            Static(
                f"Primary action: {copy['action_label']}",
                id="library-study-handoff-primary-action",
            ),
            Static(
                copy["context"],
                id="library-study-handoff-context",
            ),
            Static(
                copy["owner"],
                id="library-study-handoff-owner",
            ),
            Static(
                copy["wip"],
                id="library-study-handoff-wip",
                classes="ds-recovery-callout",
            ),
            Static(
                copy["recovery"],
                id="library-study-handoff-recovery",
                classes=recovery_classes,
            ),
            handoff_toolbar,
            id="library-study-handoff-detail",
            classes="library-rag-region",
        )

    def _workspace_handoff_action_state(
        self,
        workspace_depth_state: LibraryWorkspaceDepthState,
    ) -> tuple[bool, str]:
        """Return the Workspaces handoff button disabled flag and tooltip."""
        handoff_disabled = True
        handoff_tooltip = "Stage Library source context after Library finishes loading."
        if self._library_lookup_error:
            recovery_state = self._library_lookup_recovery_state
            handoff_tooltip = (
                recovery_state.disabled_tooltip
                if recovery_state is not None
                else "Library source services are unavailable; retry Library later."
            )
        elif not self._has_local_sources():
            handoff_tooltip = "Stage Library source context after adding notes, media, or conversations."
        else:
            handoff_disabled = not workspace_depth_state.context_handoff_enabled
            handoff_tooltip = (
                workspace_depth_state.context_handoff_tooltip
                if handoff_disabled
                else "Stage Library source context in Console."
            )
        return handoff_disabled, handoff_tooltip

    def _workspace_action_widgets(
        self,
        workspace_depth_state: LibraryWorkspaceDepthState,
        *,
        handoff_disabled: bool,
        handoff_tooltip: str,
    ) -> tuple[Any, ...]:
        """Build the Actions group's Details rail rows.

        Only the two action buttons plus one dim WIP note survive here; the
        Workspace group's Handoff row now carries the eligible/blocked
        status, so the retired ready/blocked/next-step callouts that used to
        repeat it are gone.
        """
        widgets: list[Any] = [
            Static(
                "Actions",
                id="library-details-group-actions",
                classes="library-details-group",
            ),
            Button(
                "Create local workspace",
                id="library-create-local-workspace",
                classes="library-source-action",
                tooltip=(
                    "Create a local-only workspace and make it active. "
                    "Server sync and ACP handoff remain WIP."
                ),
            ),
        ]
        if not workspace_depth_state.source_rows:
            widgets.append(
                Button(
                    "Import sources",
                    id="library-workspace-import-sources",
                    classes="library-source-action",
                    tooltip="Open Library Import/Export to add workspace-eligible sources.",
                )
            )
        widgets.append(
            Button(
                "Use in Console",
                id="library-use-in-console",
                classes="library-source-action",
                disabled=handoff_disabled,
                tooltip=handoff_tooltip,
            )
        )
        widgets.append(
            Static(
                "Server sync WIP · local only",
                id="library-workspace-create-local-copy",
                classes="library-rail-empty-copy",
            )
        )
        return tuple(widgets)

    def _compose_workspaces_rail_body(self) -> list[Any]:
        """Build the Workspaces body for the rail Details section.

        Returns the workspace-context depth panel followed by the workspace
        action controls, mirroring the retired Workspaces mode body. Preserves
        every legacy widget id so the flows remain reachable from the rail.

        Returns:
            Freshly constructed widgets for the Details disclosure.
        """
        state = self._library_workspace_depth_state()
        handoff_disabled, handoff_tooltip = self._workspace_handoff_action_state(state)
        depth_panel = Vertical(
            *self._workspaces_detail_rows(state),
            id="library-workspaces-depth-panel",
        )
        # Vertical defaults to `height: 1fr`, which inside the Details body's
        # auto-height flow starves this panel of space for its own rows. The
        # panel then clips mid-row and the next widget (the "Actions" group
        # header) renders immediately after the truncated content, reading
        # as a visual collision. Hug the panel's real content height instead
        # so nothing below it is squeezed or overlapped.
        depth_panel.styles.height = "auto"
        widgets: list[Any] = [depth_panel]
        widgets.extend(
            self._workspace_action_widgets(
                state,
                handoff_disabled=handoff_disabled,
                handoff_tooltip=handoff_tooltip,
            )
        )
        return widgets

    def _library_action_widgets(
        self,
        *,
        workspace_depth_state: LibraryWorkspaceDepthState,
        collection_scoped_actions_deferred: bool,
        handoff_disabled: bool,
        handoff_tooltip: str,
        collections_panel_state: LibraryCollectionsPanelState | None = None,
    ) -> tuple[Any, ...]:
        if self._active_mode == "workspaces":
            return self._workspace_action_widgets(
                workspace_depth_state,
                handoff_disabled=handoff_disabled,
                handoff_tooltip=handoff_tooltip,
            )
        if self._active_mode == "conversations":
            handoff_ready = self._conversation_handoff_enabled(workspace_depth_state)
            recovery_copy = self._conversation_handoff_label(workspace_depth_state)
            return (
                Static("Conversation actions", classes="destination-section"),
                Button(
                    "Open in Console",
                    id="library-conversation-open-console",
                    classes="library-source-action",
                    disabled=not handoff_ready,
                    tooltip=(
                        "Open this conversation as Console context."
                        if handoff_ready
                        else recovery_copy
                    ),
                ),
                Static(
                    (
                        "Selected conversation can be handed off when workspace policy allows it."
                        if self._selected_conversation_record()
                        else "Select a conversation first to enable these actions."
                    ),
                    id="library-conversation-action-disabled-reason",
                ),
                Button(
                    "Use as source",
                    id="library-conversation-use-source",
                    classes="library-source-action",
                    disabled=not handoff_ready,
                    tooltip=(
                        "Use this conversation as a source for Console/RAG context."
                        if handoff_ready
                        else recovery_copy
                    ),
                ),
                Static(
                    recovery_copy,
                    id="library-conversation-action-state",
                    classes="ds-recovery-callout" if handoff_ready else "ds-recovery-callout is-blocked",
                ),
            )
        if self._active_mode == "import-export":
            return (
                Static("Import/Export actions", classes="destination-section"),
                Button(
                    "Open Ingest",
                    id="library-import-export-open-ingest",
                    classes="library-source-action",
                    tooltip=(
                        "Open Ingest for files, URLs, transcripts, and source packages. "
                        "Return to Library to see imported content."
                    ),
                ),
                Static(
                    "Route: Ingest. Return path: imported material appears in Library inventory.",
                    id="library-import-export-ingest-route-copy",
                ),
                Button(
                    "Open Media",
                    id="library-import-export-open-media",
                    classes="library-source-action",
                    tooltip="Open Media for full media ingestion, review, and analysis.",
                ),
                Static(
                    "Route: Media. Use when the task is media review, not generic source movement.",
                    id="library-import-export-media-route-copy",
                ),
                Button(
                    "Export Library sources",
                    id="library-import-export-export-sources",
                    classes="library-source-action",
                    disabled=True,
                    tooltip="Source-level Library export is not wired yet.",
                ),
                Static(
                    "Blocked: export from this Library panel is not wired yet. Use owner screens where available.",
                    id="library-import-export-action-blocked",
                    classes="ds-recovery-callout is-blocked",
                ),
            )
        if self._active_mode == "collections":
            return (
                Static("Collection item actions", classes="destination-section"),
                Static(
                    "Read/review collection items when a local item adapter is available.",
                    id="library-collection-actions-local-readiness",
                ),
                Static(
                    "Item actions unavailable until collection items exist.",
                    id="library-collection-actions-disabled-reason",
                    classes="ds-recovery-callout is-blocked",
                ),
                Static(
                    "Disabled: collection item Search/RAG, Study, Console handoff, "
                    "and server sync promotion are not wired yet.",
                    id="library-collection-actions-wip-reason",
                    classes="ds-recovery-callout is-blocked",
                ),
                Button(
                    "Study Dashboard",
                    id="library-open-study",
                    classes="library-source-action",
                    disabled=True,
                    tooltip="Collection-scoped Study is not available yet.",
                ),
                Button(
                    "Flashcards",
                    id="library-open-flashcards",
                    classes="library-source-action",
                    disabled=True,
                    tooltip="Collection-scoped Flashcards are not available yet.",
                ),
                Button(
                    "Quizzes",
                    id="library-open-quizzes",
                    classes="library-source-action",
                    disabled=True,
                    tooltip="Collection-scoped Quizzes are not available yet.",
                ),
                Button(
                    "Use in Console",
                    id="library-use-in-console",
                    classes="library-source-action",
                    disabled=True,
                    tooltip="Collection-scoped Console handoff is not available yet.",
                ),
            )
        if self._active_mode == "search":
            return ()
        if self._active_mode in LIBRARY_STUDY_HANDOFF_MODES:
            copy = self._study_handoff_copy()
            active_action_id = {
                "study": "library-open-study",
                "flashcards": "library-open-flashcards",
                "quizzes": "library-open-quizzes",
            }[self._active_mode]
            recovery_classes = (
                "ds-recovery-callout"
                if self._has_local_sources()
                else "ds-recovery-callout is-blocked"
            )

            def action_classes(button_id: str) -> str:
                classes = "library-source-action"
                if button_id == active_action_id:
                    classes = f"{classes} is-active"
                return classes

            return (
                Static(
                    f"{copy['label']} actions",
                    id="library-study-actions-title",
                    classes="destination-section",
                ),
                Static(
                    (
                        "Open with the current Library source snapshot."
                        if self._has_local_sources()
                        else "Open globally; no Library source snapshot is available."
                    ),
                    id="library-study-actions-summary",
                    classes=recovery_classes,
                ),
                Button(
                    "Study Dashboard",
                    id="library-open-study",
                    classes=action_classes("library-open-study"),
                    tooltip="Open Study globally or with the current Library source snapshot.",
                ),
                Button(
                    "Flashcards",
                    id="library-open-flashcards",
                    classes=action_classes("library-open-flashcards"),
                    tooltip="Open Flashcards globally or with the current Library source snapshot.",
                ),
                Button(
                    "Quizzes",
                    id="library-open-quizzes",
                    classes=action_classes("library-open-quizzes"),
                    tooltip="Open Quizzes globally or with the current Library source snapshot.",
                ),
                Button(
                    "Use in Console",
                    id="library-use-in-console",
                    classes="library-source-action",
                    disabled=handoff_disabled,
                    tooltip=handoff_tooltip,
                ),
            )
        if self._active_mode == "sources":
            return (
                Static(
                    "Hub actions",
                    id="library-hub-actions-title",
                    classes="destination-section",
                ),
                Static(
                    "Selected: none",
                    id="library-hub-actions-guidance",
                ),
                Static(
                    "Available now: open source modules and owner screens.",
                    id="library-hub-actions-selection",
                ),
                Static(
                    "Blocked: Use in Console requires workspace-eligible Library content.",
                    id="library-hub-actions-boundary",
                    classes="ds-recovery-callout is-blocked",
                ),
                Button(
                    "Use in Console",
                    id="library-use-in-console",
                    classes="library-source-action",
                    disabled=handoff_disabled,
                    tooltip=handoff_tooltip,
                ),
                Static(
                    "Next action: open a source mode, import content, or create a note.",
                    id="library-hub-actions-next",
                ),
            )
        return (
            Static("Knowledge workflow", classes="destination-section"),
            Static(
                (
                    "Collection-scoped Study, Flashcards, Quizzes, and Console "
                    "are later-stage."
                    if collection_scoped_actions_deferred
                    else "Turn Library material into study sessions, flashcards, and quizzes."
                ),
                id="library-study-purpose",
            ),
            Static(
                (
                    "Use Collections to organize source groups locally; scoped "
                    "execution remains deferred."
                    if collection_scoped_actions_deferred
                    else "Study generation entry uses the visible Library source snapshot."
                ),
                id="library-study-generation-entry",
            ),
            Button(
                "Study Dashboard",
                id="library-open-study",
                classes="library-source-action",
                disabled=collection_scoped_actions_deferred,
                tooltip=(
                    "Collection-scoped Study is not available yet."
                    if collection_scoped_actions_deferred
                    else "Open the Study dashboard for due cards, decks, quizzes, and resume actions."
                ),
            ),
            Button(
                "Flashcards",
                id="library-open-flashcards",
                classes="library-source-action",
                disabled=collection_scoped_actions_deferred,
                tooltip=(
                    "Collection-scoped Flashcards are not available yet."
                    if collection_scoped_actions_deferred
                    else "Open flashcards for selected or imported Library material."
                ),
            ),
            Button(
                "Quizzes",
                id="library-open-quizzes",
                classes="library-source-action",
                disabled=collection_scoped_actions_deferred,
                tooltip=(
                    "Collection-scoped Quizzes are not available yet."
                    if collection_scoped_actions_deferred
                    else "Open quizzes for selected or imported Library material."
                ),
            ),
            Button(
                "Use in Console",
                id="library-use-in-console",
                classes="library-source-action",
                disabled=handoff_disabled or collection_scoped_actions_deferred,
                tooltip=(
                    "Collection-scoped Console handoff is not available yet."
                    if collection_scoped_actions_deferred
                    else handoff_tooltip
                ),
            ),
        )

    def compose_content(self) -> ComposeResult:
        shell_input = self._build_library_shell_input()
        shell = build_library_shell_state(
            shell_input, selected_row_id=self._library_selected_row_id
        )
        self._library_selected_row_id = shell.selected_row_id
        preferences = self._library_rail_preferences()

        yield Static(
            shell.header_line,
            id="library-header-line",
            classes="destination-status-row",
        )
        shell_grid = Horizontal(
            id="library-shell-grid", classes="ds-panel destination-workbench"
        )
        shell_grid.styles.height = "1fr"
        shell_grid.styles.min_height = 12
        with shell_grid:
            rail = LibraryRail(
                shell,
                preferences,
                query=self._library_conversation_query,
                search_placeholder=self._library_rail_search_placeholder(),
                workspaces_body_factory=self._compose_workspaces_rail_body,
                id="library-rail",
                classes="destination-workbench-pane",
            )
            rail.styles.height = "100%"
            yield rail
            canvas_host = Vertical(
                id="library-canvas", classes="destination-workbench-pane"
            )
            canvas_host.styles.width = "13fr"
            canvas_host.styles.min_width = 40
            canvas_host.styles.height = "100%"
            with canvas_host:
                # Only the conversations, media, and notes canvases read the
                # local source snapshot directly, so only they can show a
                # false "no conversations"/"no media"/"no notes" empty state
                # while that snapshot is still loading. "mode" canvases
                # (Collections, Flashcards, Search/RAG, ...) and the
                # landing/empty canvas are unaffected and must not be
                # replaced by this loading/error copy.
                is_local_snapshot_canvas = shell.canvas_kind in (
                    "conversations",
                    "media",
                    "notes",
                )
                if (
                    is_local_snapshot_canvas
                    and not self._library_loaded
                    and not self._library_lookup_error
                ):
                    yield Static(
                        "Loading local Library sources…",
                        id="library-canvas-loading",
                        classes="destination-purpose",
                        markup=False,
                    )
                elif is_local_snapshot_canvas and self._library_lookup_error:
                    yield Static(
                        self._library_lookup_error,
                        id="library-canvas-error",
                        classes="destination-purpose",
                        markup=False,
                    )
                elif shell.canvas_kind == "conversations":
                    conversations_state = self._build_library_conversations_state()
                    self._selected_conversation_id = conversations_state.selected_id
                    yield LibraryConversationsCanvas(
                        conversations_state,
                        id="library-conversations-canvas",
                    )
                elif shell.canvas_kind == "media" and self._library_media_view == "viewer":
                    if self._library_media_detail is None:
                        yield Static(
                            "Loading media…",
                            id="library-media-viewer-loading",
                            classes="destination-purpose",
                            markup=False,
                        )
                    else:
                        yield LibraryMediaViewer(
                            build_library_media_viewer_state(self._library_media_detail),
                            editing=self._library_media_editing,
                            confirming_delete=self._library_media_confirming_delete,
                            highlights=build_library_media_highlight_rows(
                                self._library_media_highlights
                            ),
                            editing_analysis=self._library_media_editing_analysis,
                            content_query=self._library_media_content_query,
                            content_match_index=self._library_media_content_match_index,
                            id="library-media-viewer",
                        )
                elif shell.canvas_kind == "media":
                    media_state = self._build_library_media_state()
                    self._selected_media_id = media_state.selected_id
                    yield LibraryMediaCanvas(
                        media_state,
                        id="library-media-canvas",
                    )
                elif shell.canvas_kind == "notes" and self._library_notes_view == "editor":
                    if self._library_note_conflict_snapshot is not None:
                        # A save just lost an optimistic-lock race: recompose
                        # from the user's own kept text (never the stale
                        # ``_library_note_detail``) with the Overwrite/Reload
                        # actions surfaced.
                        yield LibraryNotesCanvas(
                            mode="editor",
                            editor_state=self._library_note_conflict_snapshot,
                            conflict=True,
                            id="library-notes-canvas",
                        )
                    elif self._library_note_detail is None:
                        yield Static(
                            "Loading note…",
                            id="library-note-loading",
                            classes="destination-purpose",
                            markup=False,
                        )
                    else:
                        # The Preview snapshot (when present) carries the
                        # live -- possibly unsaved -- text captured by the
                        # last Preview toggle, so restoring from preview (or
                        # any other recompose while it's live) never reverts
                        # to the stale on-disk ``_library_note_detail``.
                        editor_state = (
                            self._library_note_preview_snapshot
                            if self._library_note_preview_snapshot is not None
                            else build_library_note_editor_state(self._library_note_detail)
                        )
                        yield LibraryNotesCanvas(
                            mode="editor",
                            editor_state=editor_state,
                            confirming_delete=self._library_note_confirming_delete,
                            preview=self._library_note_preview,
                            id="library-notes-canvas",
                        )
                elif shell.canvas_kind == "notes" and self._library_notes_view == "sync":
                    yield LibraryNotesCanvas(
                        mode="sync",
                        sync_state=self._build_library_notes_sync_state(),
                        id="library-notes-canvas",
                    )
                elif shell.canvas_kind == "notes":
                    source_records = (
                        self._library_notes_filter_records
                        if self._library_notes_filter_records is not None
                        else self._local_source_records.get("notes", ())
                    )
                    notes_list_state = build_library_notes_list_state(
                        sort_notes_records(source_records, self._library_notes_sort),
                        filter_note=self._library_notes_filter,
                    )
                    yield LibraryNotesCanvas(
                        notes_list_state,
                        sort_mode=self._library_notes_sort,
                        filter_value=self._library_notes_filter,
                        id="library-notes-canvas",
                    )
                elif shell.canvas_kind == "notes-create":
                    yield LibraryNotesCanvas(
                        mode="create",
                        id="library-notes-canvas",
                    )
                elif shell.canvas_kind == "mode":
                    yield from self._compose_mode_canvas(shell.canvas_target)
                else:
                    yield Static(
                        shell.canvas_empty_copy,
                        id="library-canvas-landing",
                        classes="destination-purpose",
                        markup=False,
                    )

    def _build_library_shell_input(self) -> LibraryShellInput:
        """Build the pure shell input from live counts and runtime state.

        While the local source snapshot is still loading (``_library_loaded``
        is False) and no lookup error has been recorded yet, the media/notes/
        conversations counts are reported as ``None`` rather than the
        placeholder zeros seeded at construction time, so the rail does not
        render a misleading ``(0)`` before the real snapshot arrives.

        Returns:
            Adapter-provided Library shell input reflecting live counts and
            runtime state.
        """
        runtime_state = getattr(
            getattr(self.app_instance, "runtime_policy", None), "state", None
        )
        active_source = str(
            getattr(runtime_state, "active_source", "local") or "local"
        ).lower()
        server_label = None
        if active_source == "server":
            server_label = getattr(runtime_state, "last_known_server_label", None) or getattr(
                runtime_state, "active_server_id", None
            )
        collections_count = (
            len(self._library_collections_records)
            if self._library_collections_loaded
            else None
        )
        counts = self._local_source_counts
        known = self._local_source_total_known
        counts_known_yet = self._library_loaded or bool(self._library_lookup_error)
        return LibraryShellInput(
            media_count=counts.get("media") if counts_known_yet else None,
            media_known=known.get("media", True),
            conversations_count=counts.get("conversations") if counts_known_yet else None,
            conversations_known=known.get("conversations", True),
            notes_count=counts.get("notes") if counts_known_yet else None,
            notes_known=known.get("notes", True),
            collections_count=collections_count,
            runtime_source=active_source,
            server_label=str(server_label) if server_label else None,
            details_lines=self._library_details_lines(active_source, server_label),
        )

    def _library_details_lines(
        self, active_source: str, server_label: Any
    ) -> tuple[str, ...]:
        """Build the Status group's Details disclosure lines for the rail.

        Returns exactly two plain-text values: the runtime value (rendered
        by the rail with a dimmed "Runtime" label) and the local source
        counts, or a lookup-error/recovery block in place of the counts when
        the local source snapshot failed to load.
        """
        runtime_value = (
            "Local"
            if active_source != "server"
            else f"Server: {server_label or 'unknown'}"
        )
        counts = self._local_source_counts
        if self._library_lookup_error:
            counts_or_error = self._library_lookup_error
        else:
            counts_or_error = (
                f"Notes {counts.get('notes', 0)} · "
                f"Media {counts.get('media', 0)} · "
                f"Conversations {counts.get('conversations', 0)}"
            )
        return (runtime_value, counts_or_error)

    def _build_library_conversations_state(self):
        """Build the conversations canvas display state from local records."""
        return build_library_conversations_state(
            self._conversation_records(),
            query=self._library_conversation_query,
            selected_id=self._selected_conversation_id,
        )

    def _build_library_media_state(self) -> LibraryMediaCanvasState:
        """Build the media canvas display state from local records."""
        return build_library_media_state(
            self._local_source_records.get("media", ()),
            active_type=self._library_media_type_filter,
            selected_id=self._selected_media_id,
        )

    async def _refresh_library_media_detail(self, media_id: str) -> None:
        """Fetch and store the full detail for a selected Library media item.

        Mirrors ``_refresh_library_collections_snapshot``: guards against an
        unavailable media service, offloads the (possibly blocking) service
        call via ``_run_library_service_call``, and recomposes once the
        fetched detail (or a cleared/failed state) has been stored.

        Also fetches the item's reading highlights (see
        ``_fetch_library_media_highlights``) so both the detail and the
        highlights section are ready by the single recompose at the end,
        rather than the highlights section popping in on a second refresh.

        Triggered by ``handle_library_media_row`` on media-row selection;
        ``compose_content`` renders the stored ``_library_media_detail`` via
        ``build_library_media_viewer_state`` once this worker completes.

        Args:
            media_id: The Library media item id to fetch full detail for.
        """
        service = getattr(self.app_instance, "media_reading_scope_service", None)
        get_media_item = getattr(service, "get_media_item", None)
        if not callable(get_media_item):
            self._library_media_detail = None
            self._library_media_highlights = []
            if self.is_mounted:
                self.refresh(recompose=True)
            return
        try:
            detail = await self._run_library_service_call(
                get_media_item,
                mode="local",
                media_id=media_id,
                include_content=True,
                include_versions=True,
                isolate_in_worker=True,
            )
        except Exception:
            logger.warning(f"Failed to load Library media detail for {media_id!r}.", exc_info=True)
            detail = None
        # Discard out-of-order results: if the user has since selected a
        # different media row (or left the viewer), a slower in-flight fetch
        # for the previous selection must not overwrite the current one. The
        # highlights fetch is a second await point, so re-check after it too
        # and store detail + highlights together only when still current.
        if media_id != self._selected_media_id or self._library_media_view != "viewer":
            return
        highlights = await self._fetch_library_media_highlights(media_id)
        if media_id != self._selected_media_id or self._library_media_view != "viewer":
            return
        self._library_media_detail = detail if isinstance(detail, Mapping) else None
        self._library_media_highlights = highlights
        if self.is_mounted:
            self.refresh(recompose=True)

    async def _fetch_library_media_highlights(self, media_id: str) -> list[dict[str, Any]]:
        """Fetch reading highlights for a Library media item from the local scope service.

        Guards against an unavailable ``list_highlights`` service or a failed
        call the same way ``_refresh_library_media_detail`` guards the detail
        fetch: any failure yields an empty list rather than raising, so a
        highlights outage never blocks the rest of the viewer from loading.

        Args:
            media_id: The Library media item id to fetch highlights for.

        Returns:
            The fetched highlights list, or an empty list when the service
            is unavailable or the fetch fails.
        """
        service = getattr(self.app_instance, "media_reading_scope_service", None)
        list_highlights = getattr(service, "list_highlights", None)
        if not callable(list_highlights):
            return []
        try:
            highlights = await self._run_library_service_call(
                list_highlights,
                mode="local",
                item_id=media_id,
                isolate_in_worker=True,
            )
        except Exception:
            logger.warning(
                f"Failed to load Library media highlights for {media_id!r}.", exc_info=True
            )
            return []
        return list(highlights) if isinstance(highlights, list) else []

    async def _reload_library_media_highlights(self, media_id: str) -> None:
        """Re-fetch and store highlights after a highlight mutation, then recompose.

        Args:
            media_id: The Library media item id whose highlights changed.
        """
        self._library_media_highlights = await self._fetch_library_media_highlights(media_id)
        if self.is_mounted:
            self.refresh(recompose=True)

    async def _refresh_library_note_detail(self, note_id: str) -> None:
        """Fetch and store the full detail for a selected Library note.

        Mirrors ``_refresh_library_media_detail``: offloads the (possibly
        blocking) ``get_note_detail`` service call via
        ``_run_library_service_call`` and recomposes once the fetched detail
        (or a cleared/failed state) has been stored.

        Triggered by ``handle_library_notes_row`` on note-row selection;
        ``compose_content`` renders the stored ``_library_note_detail`` via
        ``build_library_note_editor_state`` once this worker completes.

        Args:
            note_id: The Library note id to fetch full detail for.
        """
        service = getattr(self.app_instance, "notes_scope_service", None)
        get_note_detail = getattr(service, "get_note_detail", None)
        if not callable(get_note_detail):
            self._library_note_detail = None
            self._library_note_version = None
            if self.is_mounted:
                self.refresh(recompose=True)
            return
        try:
            detail = await self._run_library_service_call(
                get_note_detail,
                scope="local_note",
                note_id=note_id,
                user_id=getattr(self.app_instance, "notes_user_id", None) or "default_user",
                isolate_in_worker=True,
            )
        except Exception:
            logger.warning(f"Failed to load Library note detail for {note_id!r}.", exc_info=True)
            detail = None
        # Discard out-of-order results: if the user has since selected a
        # different note (or left the editor), a slower in-flight fetch for
        # the previous selection must not overwrite the current one -- the
        # same stale-race guard as ``_refresh_library_media_detail``.
        if note_id != self._selected_note_id or self._library_notes_view != "editor":
            return
        if not isinstance(detail, Mapping):
            # The note no longer exists -- deleted elsewhere, or a stale
            # ghost row from an already-out-of-date list snapshot. Leaving
            # ``_library_note_detail`` at ``None`` here would strand the
            # canvas on the "Loading note…" placeholder forever (it never
            # becomes a real dict). Fall back to the list view instead of
            # that dead end, mirroring the delete-success flow's full
            # snapshot reload.
            logger.info(
                f"Library note {note_id!r} is no longer available; returning to list."
            )
            self._reset_library_note_editor_state()
            self._notify_library_note_missing_warning()
            self._refresh_local_source_snapshot()
            if self.is_mounted:
                self.refresh(recompose=True)
            return
        self._library_note_detail = detail
        keywords = await self._fetch_library_note_keywords(note_id)
        # Fetching keywords is a second await point: re-check freshness the
        # same way ``_refresh_library_media_detail`` re-checks after its own
        # second (highlights) fetch, so a switch that happened during that
        # fetch cannot land keywords for the wrong note.
        if note_id != self._selected_note_id or self._library_notes_view != "editor":
            return
        if keywords is not None and isinstance(self._library_note_detail, Mapping):
            enriched_detail = dict(self._library_note_detail)
            enriched_detail["keywords"] = keywords
            self._library_note_detail = enriched_detail
        editor_state = build_library_note_editor_state(self._library_note_detail)
        self._library_note_version = editor_state.version
        self._library_note_dirty = False
        self._library_note_autosave_state = "idle"
        self._library_note_conflict_snapshot = None
        self._library_note_preview = False
        self._library_note_preview_snapshot = None
        self._library_note_editor_armed = False
        if self.is_mounted:
            self.refresh(recompose=True)
            self.call_after_refresh(self._arm_library_note_editor)

    async def _fetch_library_note_keywords(self, note_id: str) -> list[Any] | None:
        """Fetch a Library note's keywords for the in-canvas editor.

        ``notes_scope_service.get_note_detail``'s local-scope shape is the
        raw ``notes`` table row and never carries a ``keywords`` field (see
        ``NotesInteropService.get_note_by_id``) -- keywords live in a
        separate join table. This fetches them through the standalone
        Notes screen's own seam, ``app.notes_service.get_keywords_for_note``
        (``NotesInteropService.get_keywords_for_note(user_id, note_id)``,
        which returns the ``ChaChaNotes_DB.get_keywords_for_note`` row
        shape -- each item a mapping with a ``keyword`` key), offloaded off
        the UI loop the same way every other Library service call is.

        Args:
            note_id: The Library note id to fetch keywords for.

        Returns:
            The fetched keyword records, or ``None`` when
            ``app.notes_service`` (or the method) is unavailable or the
            fetch fails -- callers should leave the detail's keywords field
            untouched in that case, the same quiet-absent behavior as
            before this enrichment existed.
        """
        notes_service = getattr(self.app_instance, "notes_service", None)
        get_keywords_for_note = getattr(notes_service, "get_keywords_for_note", None)
        if not callable(get_keywords_for_note):
            return None
        try:
            keywords = await self._run_library_service_call(
                get_keywords_for_note,
                self._library_notes_user_id(),
                note_id,
                isolate_in_worker=True,
            )
        except Exception:
            logger.warning(
                f"Failed to load keywords for Library note {note_id!r}.", exc_info=True
            )
            return None
        return list(keywords) if isinstance(keywords, list) else None

    def _arm_library_note_editor(self) -> None:
        """Enable dirty-tracking once the notes editor's mount-time
        ``Input.Changed`` (fired for the non-empty ``value=`` kwarg) has
        already been delivered, so it is never mistaken for a real edit.
        """
        self._library_note_editor_armed = True

    def _reset_library_note_editor_state(self) -> None:
        """Clear all in-canvas Library note editor/save state.

        Shared by the Back handler, note-row selection, and rail-row
        selection so every exit from the editor leaves save/autosave/
        conflict tracking in a clean ``idle`` state for the next note.
        """
        self._library_notes_view = "list"
        self._library_note_detail = None
        self._selected_note_id = ""
        self._library_note_version = None
        self._library_note_dirty = False
        self._library_note_autosave_state = "idle"
        self._library_note_conflict_snapshot = None
        self._library_note_confirming_delete = False
        self._library_note_preview = False
        self._library_note_preview_snapshot = None
        self._library_note_editor_armed = False
        if self._library_notes_autosave_timer is not None:
            self._library_notes_autosave_timer.stop()
            self._library_notes_autosave_timer = None

    def _reset_library_notes_sync_transient_state(self) -> None:
        """Clear the sync panel's run-scoped (non-persisted) state.

        Called on rail re-entry into Notes (extends the existing reset) so
        stale status/activity from a previous visit never reappears; the
        persisted direction/conflict/auto-sync preferences and the
        auto-sync timer are left untouched here -- the timer's lifetime is
        the whole Library screen's, not a single sync-panel visit (see
        ``handle_library_notes_sync_auto_toggle``).
        """
        self._library_notes_sync_status = "idle"
        self._library_notes_sync_activity = ()
        self._library_notes_sync_running = False

    def _ensure_library_notes_sync_config_loaded(self) -> None:
        """Seed sync direction/conflict/auto-sync from config on first entry.

        Idempotent: only reads config once per screen lifetime
        (``_library_notes_sync_config_loaded`` guards re-entry), so
        in-session cycling/toggling is never clobbered by a later sync-mode
        re-entry re-reading stale config.
        """
        if self._library_notes_sync_config_loaded:
            return
        self._library_notes_sync_config_loaded = True
        self._library_notes_sync_direction = str(
            get_cli_setting("notes", "sync_direction", "bidirectional")
            or "bidirectional"
        )
        self._library_notes_sync_conflict = str(
            get_cli_setting("notes", "sync_conflict_resolution", "newer_wins")
            or "newer_wins"
        )
        self._library_notes_sync_auto = bool(get_cli_setting("notes", "auto_sync", False))
        if self._library_notes_sync_auto:
            self._arm_library_notes_auto_sync_timer()

    def _library_notes_sync_folder(self) -> str:
        """Return the configured sync folder as text (unexpanded)."""
        return str(get_cli_setting("notes", "sync_directory", "~/Documents/Notes"))

    def _build_library_notes_sync_state(self) -> LibraryNotesSyncState:
        """Build the sync panel's display state from screen fields."""
        return LibraryNotesSyncState(
            folder=self._library_notes_sync_folder(),
            direction=self._library_notes_sync_direction,
            conflict=self._library_notes_sync_conflict,
            auto_sync=self._library_notes_sync_auto,
            status_line=self._library_notes_sync_status,
            activity_lines=self._library_notes_sync_activity,
        )

    def _resolve_library_notes_sync_db(self) -> Any:
        """Resolve the per-user ChaChaNotes DB the sync service writes to.

        Mirrors ``NotesSyncPane.on_mount`` (notes_workbench_panes.py) EXACTLY:
        prefer the app's ``chachanotes_db``, falling back to the notes
        service's own ``db`` attribute when that is unset.
        """
        notes_service = getattr(self.app_instance, "notes_service", None)
        return getattr(self.app_instance, "chachanotes_db", None) or getattr(
            notes_service, "db", None
        )

    def _arm_library_notes_auto_sync_timer(self) -> None:
        """Start the 300s auto-sync repeating timer if not already running.

        Scoped to this Library screen instance's lifetime (like the
        standalone ``NotesSyncPane``'s timer) -- it is never persisted or
        resumed across screen instances; only the ``auto_sync`` boolean
        preference is persisted, and is re-armed on the next sync-mode
        entry via ``_ensure_library_notes_sync_config_loaded``.
        """
        if self._library_notes_auto_sync_timer is not None:
            return
        self._library_notes_auto_sync_timer = self.set_interval(
            LIBRARY_NOTES_AUTO_SYNC_INTERVAL_SECONDS,
            self._library_notes_auto_sync_tick,
        )

    def _cancel_library_notes_auto_sync_timer(self) -> None:
        if self._library_notes_auto_sync_timer is not None:
            self._library_notes_auto_sync_timer.stop()
            self._library_notes_auto_sync_timer = None

    def _library_notes_auto_sync_tick(self) -> None:
        """Auto-sync timer callback: skip quietly when busy or misconfigured."""
        if self._library_notes_sync_running:
            return
        folder_value = self._library_notes_sync_folder()
        if not folder_value:
            return
        try:
            folder = validate_path_simple(
                Path(folder_value).expanduser(), require_exists=True
            )
        except ValueError:
            return
        if not folder.is_dir():
            return
        self.run_worker(
            self._run_library_notes_sync(folder),
            exclusive=True,
            group="library_notes_sync",
        )

    # ----- Notes editor: save, autosave, conflict policy -----------------

    def _mark_library_note_dirty(self) -> None:
        """Record an in-progress edit and (re)arm the autosave debounce.

        Ignored until ``_library_note_editor_armed`` is set (see that
        flag's docstring) and while a save conflict is being shown --
        autosaving against a version that is already known to be stale
        would just recreate the same conflict on a timer.
        """
        if not self._library_note_editor_armed:
            return
        self._library_note_dirty = True
        if self._library_note_autosave_state == "conflict":
            return
        if self._library_notes_autosave_timer is not None:
            self._library_notes_autosave_timer.stop()
        # Read as a bare module global (not a captured default) so tests can
        # monkeypatch LIBRARY_NOTES_AUTOSAVE_SECONDS to a short interval.
        self._library_notes_autosave_timer = self.set_timer(
            LIBRARY_NOTES_AUTOSAVE_SECONDS, self._fire_library_note_autosave
        )

    @on(Input.Changed, "#library-note-title")
    def handle_library_note_title_changed(self, event: Input.Changed) -> None:
        """Mark the open note dirty and (re)arm the autosave debounce.

        Args:
            event: Input change event emitted by the editor's title field.
        """
        self._mark_library_note_dirty()

    @on(TextArea.Changed, "#library-note-body")
    def handle_library_note_body_changed(self, event: TextArea.Changed) -> None:
        """Mark the open note dirty and (re)arm the autosave debounce.

        Args:
            event: Text change event emitted by the editor's body ``TextArea``.
        """
        self._mark_library_note_dirty()

    @on(Input.Changed, "#library-note-keywords")
    def handle_library_note_keywords_changed(self, event: Input.Changed) -> None:
        """Mark the open note dirty and (re)arm the autosave debounce.

        Args:
            event: Input change event emitted by the editor's keywords field.
        """
        self._mark_library_note_dirty()

    def _fire_library_note_autosave(self) -> None:
        """Debounce-timer callback: kick the actual autosave worker."""
        self._library_notes_autosave_timer = None
        if not self._library_note_dirty:
            return
        self.run_worker(
            self._save_library_note(explicit=False),
            exclusive=True,
            group="library_note_save",
        )

    @on(Button.Pressed, "#library-note-save")
    def handle_library_note_save(self, event: Button.Pressed) -> None:
        """Explicitly save the open note, bypassing the autosave debounce."""
        event.stop()
        if self._library_notes_autosave_timer is not None:
            self._library_notes_autosave_timer.stop()
            self._library_notes_autosave_timer = None
        self.run_worker(
            self._save_library_note(explicit=True),
            exclusive=True,
            group="library_note_save",
        )

    def _library_note_meta_base_line(self) -> str:
        """The static Created/Modified/version portion of the meta line."""
        return build_library_note_editor_state(self._library_note_detail).meta_line

    def _update_library_note_meta_static(self, *, content: str) -> None:
        """Targeted update of the meta line's autosave status suffix.

        Never recomposes: the ``#library-note-meta`` ``Static`` is updated
        in place, composing its already-known base meta line with the
        current autosave status text, so the ``TextArea``/``Input`` widget
        instances are left untouched by a save.

        Args:
            content: The just-saved (or attempted) note body, used only to
                compute the word count shown in the status text.
        """
        try:
            meta_static = self.query_one("#library-note-meta", Static)
        except (NoMatches, QueryError):
            return
        status_text = notes_autosave_status_text(
            self._library_note_autosave_state, word_count=self._note_word_count(content)
        )
        base_meta_line = self._library_note_meta_base_line()
        meta_static.update(f"{base_meta_line} · {status_text}" if base_meta_line else status_text)

    async def _save_library_note(self, *, explicit: bool) -> None:
        """Save the open Library note's current editor text.

        Reads title/content/keywords via ``_read_library_note_editor_fields``
        (which tolerates the Preview toggle, falling back to the live-capture
        snapshot for the body when ``#library-note-body`` isn't mounted),
        sanitizes the fields, and calls ``save_note`` through the offloaded
        service seam (``note_id``/``version`` supplied, so this is always the
        update path). A successful save bumps ``_library_note_version`` in memory
        and updates only the meta line -- it never recomposes, so the
        ``TextArea``/``Input`` widget instances stay identical across a
        save. A version conflict (a falsy result from the seam, or a
        ``ConflictError`` raised by the real local notes service on a
        stale ``expected_version``) cancels the autosave timer and
        recomposes into the conflict UI, re-seeded from the user's live
        widget text -- never from the stale ``_library_note_detail``.

        The save is awaited inline (see ``_flush_library_note_save``, which
        bypasses the exclusive ``library_note_save`` worker group), so
        another note can become selected while this call is still in
        flight. Every branch below therefore re-checks that the note this
        save was *for* is still the selected one -- and the editor is still
        showing -- before mutating any shared editor state, mirroring the
        stale-result guard in ``_refresh_library_note_detail`` and
        ``_resolve_library_note_conflict``.

        Args:
            explicit: Whether this save was triggered by the Save button
                (``True``) or the autosave debounce/flush (``False``). Not
                currently used to vary behavior, but kept for call-site
                clarity and future use (e.g. differentiated error copy).
        """
        if self._library_notes_view != "editor" or not self._selected_note_id:
            return
        note_id = self._selected_note_id
        fields = self._read_library_note_editor_fields()
        if fields is None:
            return
        raw_title, raw_content, raw_keywords_text = fields

        title = self._sanitize_media_field(raw_title, max_length=300)
        content = self._sanitize_note_content(raw_content, max_length=LIBRARY_NOTE_CONTENT_MAX_CHARS)
        keywords = self._library_note_keywords_from_input(raw_keywords_text)

        service = getattr(self.app_instance, "notes_scope_service", None)
        save_note = getattr(service, "save_note", None)
        if not callable(save_note):
            return

        self._library_note_autosave_state = "saving"

        try:
            result = await self._run_library_service_call(
                save_note,
                scope="local_note",
                title=title,
                content=content,
                note_id=note_id,
                version=self._library_note_version,
                user_id=self._library_notes_user_id(),
                keywords=keywords,
                isolate_in_worker=True,
            )
        except ConflictError:
            result = False
        except Exception:
            logger.warning(f"Library note save failed for {note_id!r}.", exc_info=True)
            if note_id != self._selected_note_id or self._library_notes_view != "editor":
                return
            self._library_note_autosave_state = "error"
            self._update_library_note_meta_static(content=raw_content)
            return

        # Discard a stale result: the user has since switched to a
        # different note (or left the editor) while this save was in
        # flight. Mutating shared state past this point would land this
        # note's save result -- version bump, meta line, or conflict UI --
        # on whatever note is now selected.
        if note_id != self._selected_note_id or self._library_notes_view != "editor":
            return

        if result:
            if isinstance(result, Mapping):
                version = result.get("version")
                if version is not None:
                    self._library_note_version = version
            else:
                self._library_note_version = (self._library_note_version or 0) + 1
            if isinstance(self._library_note_detail, dict):
                # Patch the just-saved fields into the cached detail mirror
                # too, not just ``version`` -- a save never recomposes the
                # editor itself, but a *later* recompose (entering the
                # delete-confirm state, toggling Preview, ...) rebuilds the
                # editor's widgets fresh from this detail. Leaving
                # title/content stale here would silently revert the
                # just-saved edit back to its pre-save text on that
                # recompose.
                self._library_note_detail["version"] = self._library_note_version
                self._library_note_detail["title"] = title
                self._library_note_detail["content"] = content
                if isinstance(result, Mapping) and "keywords" in result:
                    self._library_note_detail["keywords"] = result["keywords"]
                elif keywords is not None:
                    self._library_note_detail["keywords"] = keywords
            self._library_note_dirty = False
            self._library_note_autosave_state = "saved"
            self._update_library_note_meta_static(content=raw_content)
            return

        # Falsy result: another writer changed the note first (an
        # optimistic-lock version conflict). Stop any pending autosave and
        # recompose into the conflict UI, re-seeded from the user's live
        # widget text so nothing they typed is lost.
        self._library_note_autosave_state = "conflict"
        if self._library_notes_autosave_timer is not None:
            self._library_notes_autosave_timer.stop()
            self._library_notes_autosave_timer = None
        base_meta_line = self._library_note_meta_base_line()
        conflict_status = notes_autosave_status_text(
            "conflict", word_count=self._note_word_count(raw_content)
        )
        self._library_note_conflict_snapshot = LibraryNoteEditorState(
            note_id=note_id,
            title=raw_title,
            content=raw_content,
            keywords_text=raw_keywords_text,
            version=self._library_note_version,
            meta_line=(
                f"{base_meta_line} · {conflict_status}" if base_meta_line else conflict_status
            ),
            has_note=True,
        )
        self._library_note_preview = False
        self._library_note_preview_snapshot = None
        self._library_note_editor_armed = False
        if self.is_mounted:
            self.refresh(recompose=True)
            self.call_after_refresh(self._arm_library_note_editor)

    async def _flush_library_note_save(self) -> None:
        """Save any pending edit before the notes editor is left.

        Called at the top of the Back handler, note-row selection, and
        rail-row selection so a dirty edit is never silently discarded by
        navigating away.

        Cancels the pending autosave timer, then WAITS for any save already
        running in the ``library_note_save`` worker group (an autosave that
        fired just before this navigation) before deciding whether an inline
        save is still needed. Without the wait, this inline flush and the
        in-flight autosave both call ``save_note`` with the same
        not-yet-bumped version -- an optimistic-lock conflict that fires
        against the note's *own* autosave and pops a spurious "changed
        elsewhere" banner, aborting the navigation. After the in-flight save
        finishes it has already persisted the text and cleared the dirty
        flag, so the re-check below usually short-circuits; the inline save
        only runs when edits genuinely remain, and then against the bumped
        version.
        """
        if self._library_notes_autosave_timer is not None:
            self._library_notes_autosave_timer.stop()
            self._library_notes_autosave_timer = None
        for worker in list(self.workers):
            if worker.group == "library_note_save" and not worker.is_finished:
                try:
                    await worker.wait()
                except Exception:
                    logger.debug(
                        "In-flight note-save worker errored while flushing; continuing.",
                        exc_info=True,
                    )
        if not self._library_note_dirty:
            return
        await self._save_library_note(explicit=False)

    async def _resolve_library_note_conflict(self, *, overwrite: bool) -> None:
        """Resolve a shown save conflict via the Overwrite or Reload action.

        Both paths silently re-fetch the note's current server-side detail
        first (no "Loading…" placeholder -- the conflict UI stays put while
        this happens).

        * ``overwrite=True``: take only the fresh ``version`` from that
          detail and re-save the user's *live* editor text (read fresh from
          the widgets, not the recompose-time snapshot, so further edits
          made while the conflict banner was showing are not lost) with
          that version. On success, the local detail mirror is patched
          (not fully reloaded) with the saved fields so a normal-mode
          recompose renders the kept text, and the conflict state clears.
        * ``overwrite=False``: discard the local edits and recompose the
          editor from the freshly fetched detail.

        Either path falls back to the list view (with the same "no longer
        available" warning ``_refresh_library_note_detail`` shows) when the
        re-fetch discovers the note was deleted elsewhere entirely -- not
        just version-bumped again -- so neither action can strand the
        canvas on a permanent "Loading…" placeholder (Reload) or a
        conflict UI that can never be resolved (Overwrite).

        Args:
            overwrite: ``True`` for Overwrite, ``False`` for Reload.
        """
        note_id = self._selected_note_id
        if not note_id or self._library_note_autosave_state != "conflict":
            return
        service = getattr(self.app_instance, "notes_scope_service", None)
        get_note_detail = getattr(service, "get_note_detail", None)
        if not callable(get_note_detail):
            return
        try:
            detail = await self._run_library_service_call(
                get_note_detail,
                scope="local_note",
                note_id=note_id,
                user_id=self._library_notes_user_id(),
                isolate_in_worker=True,
            )
        except Exception:
            logger.warning(
                f"Failed to reload Library note {note_id!r} after a save conflict.",
                exc_info=True,
            )
            return
        if note_id != self._selected_note_id:
            return  # The user navigated away while the re-fetch was in flight.

        if not isinstance(detail, Mapping):
            # The note is gone entirely (deleted elsewhere) rather than
            # merely changed again -- neither Reload nor Overwrite has
            # anything left to reconcile against. Mirrors
            # ``_refresh_library_note_detail``'s missing-note fallback so
            # this never leaves the canvas stuck in the conflict UI or on
            # an unresolvable "Loading…" placeholder.
            logger.info(
                f"Library note {note_id!r} is no longer available; returning to list."
            )
            self._reset_library_note_editor_state()
            self._notify_library_note_missing_warning()
            self._refresh_local_source_snapshot()
            if self.is_mounted:
                self.refresh(recompose=True)
            return

        if not overwrite:
            self._library_note_detail = detail
            editor_state = build_library_note_editor_state(self._library_note_detail)
            self._library_note_version = editor_state.version
            self._library_note_conflict_snapshot = None
            self._library_note_dirty = False
            self._library_note_autosave_state = "idle"
            self._library_note_preview = False
            self._library_note_preview_snapshot = None
            self._library_note_editor_armed = False
            if self.is_mounted:
                self.refresh(recompose=True)
                self.call_after_refresh(self._arm_library_note_editor)
            return

        fresh_version = build_library_note_editor_state(detail).version
        if fresh_version is None:
            return
        try:
            title_widget = self.query_one("#library-note-title", Input)
            body_widget = self.query_one("#library-note-body", TextArea)
            keywords_widget = self.query_one("#library-note-keywords", Input)
        except (NoMatches, QueryError):
            return
        title = self._sanitize_media_field(title_widget.value, max_length=300)
        content = self._sanitize_note_content(body_widget.text, max_length=LIBRARY_NOTE_CONTENT_MAX_CHARS)
        keywords = self._library_note_keywords_from_input(keywords_widget.value)

        save_note = getattr(service, "save_note", None)
        if not callable(save_note):
            return
        try:
            result = await self._run_library_service_call(
                save_note,
                scope="local_note",
                title=title,
                content=content,
                note_id=note_id,
                version=fresh_version,
                user_id=self._library_notes_user_id(),
                keywords=keywords,
                isolate_in_worker=True,
            )
        except ConflictError:
            result = False
        except Exception:
            logger.warning(
                f"Failed to overwrite Library note {note_id!r} after a save conflict.",
                exc_info=True,
            )
            return
        if note_id != self._selected_note_id:
            return
        if not result:
            # Someone else won this race too; keep showing the conflict UI
            # (still seeded with the user's text) so they can try again.
            return

        if isinstance(result, Mapping):
            version = result.get("version")
            self._library_note_version = version if version is not None else fresh_version + 1
        else:
            self._library_note_version = fresh_version + 1

        patched_detail: dict[str, Any] = (
            dict(self._library_note_detail)
            if isinstance(self._library_note_detail, Mapping)
            else {}
        )
        patched_detail["id"] = note_id
        patched_detail["title"] = title
        patched_detail["content"] = content
        patched_detail["version"] = self._library_note_version
        if isinstance(detail, Mapping):
            for key in ("created_at", "last_modified", "updated_at"):
                if key in detail:
                    patched_detail[key] = detail[key]
        self._library_note_detail = patched_detail
        self._library_note_conflict_snapshot = None
        self._library_note_dirty = False
        self._library_note_autosave_state = "saved"
        self._library_note_preview = False
        self._library_note_preview_snapshot = None
        self._library_note_editor_armed = False
        if self.is_mounted:
            self.refresh(recompose=True)
            self.call_after_refresh(self._arm_library_note_editor)

    @on(Button.Pressed, "#library-note-conflict-overwrite")
    def handle_library_note_conflict_overwrite(self, event: Button.Pressed) -> None:
        """Resolve a shown save conflict by re-saving the kept local edits.

        Args:
            event: Button press event emitted by the conflict UI's
                "Overwrite" action.
        """
        event.stop()
        self.run_worker(
            self._resolve_library_note_conflict(overwrite=True),
            exclusive=True,
            group="library_note_save",
        )

    @on(Button.Pressed, "#library-note-conflict-reload")
    def handle_library_note_conflict_reload(self, event: Button.Pressed) -> None:
        """Resolve a shown save conflict by discarding local edits and reloading.

        Args:
            event: Button press event emitted by the conflict UI's
                "Reload" action.
        """
        event.stop()
        self.run_worker(
            self._resolve_library_note_conflict(overwrite=False),
            exclusive=True,
            group="library_note_save",
        )

    # ----- Notes editor: preview, export, copy, use-in-console -----------

    def _read_library_note_editor_fields(self) -> tuple[str, str, str] | None:
        """Read the note editor's current (possibly unsaved) title/content/keywords.

        Tolerates the Preview toggle: when ``#library-note-body`` is a
        read-only ``Markdown`` widget (no live text of its own), the body
        falls back to the live-capture snapshot taken when Preview was
        entered (see ``handle_library_note_preview_toggle``) instead of the
        stale ``_library_note_detail``.

        Returns:
            ``(title, content, keywords_text)`` read from the live widgets
            (and/or the preview snapshot), or ``None`` if the editor isn't
            mounted.
        """
        try:
            title = self.query_one("#library-note-title", Input).value
            keywords_text = self.query_one("#library-note-keywords", Input).value
        except (NoMatches, QueryError):
            return None
        if self._library_note_preview:
            content = (
                self._library_note_preview_snapshot.content
                if self._library_note_preview_snapshot is not None
                else ""
            )
        else:
            try:
                content = self.query_one("#library-note-body", TextArea).text
            except (NoMatches, QueryError):
                return None
        return title, content, keywords_text

    @on(Button.Pressed, "#library-note-preview")
    def handle_library_note_preview_toggle(self, event: Button.Pressed) -> None:
        """Toggle the note editor between edit and read-only Markdown preview.

        Captures the live editor text (via ``_read_library_note_editor_fields``,
        which already knows how to read through a prior Preview toggle)
        *before* flipping the flag and recomposing, so neither entering nor
        leaving preview silently drops in-progress edits -- the same
        live-widget-capture discipline ``_save_library_note``'s conflict
        branch already uses. A no-op while a save conflict is showing: that
        recompose branch ignores ``preview`` entirely (it always shows the
        live conflict text), so toggling here would have no visible effect
        beyond leaving a stale flag/snapshot behind.

        Args:
            event: Button press event emitted by the editor's Preview/Edit
                action.
        """
        event.stop()
        if self._library_notes_view != "editor" or self._library_note_autosave_state == "conflict":
            return
        fields = self._read_library_note_editor_fields()
        if fields is None:
            return
        title, content, keywords_text = fields
        self._library_note_preview_snapshot = LibraryNoteEditorState(
            note_id=self._selected_note_id,
            title=title,
            content=content,
            keywords_text=keywords_text,
            version=self._library_note_version,
            meta_line=self._library_note_meta_base_line(),
            has_note=True,
        )
        self._library_note_preview = not self._library_note_preview
        self._library_note_editor_armed = False
        self.refresh(recompose=True)
        self.call_after_refresh(self._arm_library_note_editor)

    async def _export_library_note(self, export_format: str) -> None:
        """Push the Export dialog for the open Library note.

        Mirrors ``notes_screen._export_current_note``'s dialog flow -- a
        ``FileSave`` prompt pre-filled with a sanitized default filename,
        whose callback writes the built export content once a path is
        chosen. The export reads the *live* editor widgets (via
        ``_read_library_note_editor_fields``), never the DB, so unlike
        Save there is nothing to flush first.

        Note: the real ``Third_Party.textual_fspicker.FileSave`` dialog
        only accepts ``location``/``title``/``default_file`` (not the
        ``default_filename``/``context`` kwargs ``notes_screen.py`` passes
        it, which would raise ``TypeError`` if that path ever actually
        ran) -- this uses the dialog's real constructor shape.

        Args:
            export_format: ``"markdown"`` for the frontmatter export
                (Export .md), or ``"text"`` for the plain-text export
                (Export .txt).
        """
        if self._library_notes_view != "editor" or not self._selected_note_id:
            return
        fields = self._read_library_note_editor_fields()
        if fields is None:
            return
        title, content, keywords_text = fields
        note_id = self._selected_note_id
        safe_title = "".join(
            char for char in (title.strip() or "note") if char.isalnum() or char in (" ", "-", "_")
        ).rstrip() or "note"
        default_filename = f"{safe_title}.md" if export_format == "markdown" else f"{safe_title}.txt"
        dialog_title = "Export Note as Markdown" if export_format == "markdown" else "Export Note as Text"
        await self.app.push_screen(
            FileSave(
                location=str(Path.home()),
                title=dialog_title,
                default_file=default_filename,
            ),
            callback=lambda path: self.call_after_refresh(
                self._write_library_note_export_file,
                path,
                export_format,
                title,
                content,
                keywords_text,
                note_id,
            ),
        )

    def _write_library_note_export_file(
        self,
        selected_path: Path | None,
        export_format: str,
        title: str,
        content: str,
        keywords_text: str,
        note_id: str,
    ) -> None:
        """Write the exported note content to the path chosen via ``FileSave``.

        Runs the dialog-returned path through ``validate_path_simple``
        before writing: ``FileSave`` lets the user pick any absolute
        destination, so there is no fixed base directory to constrain it
        to the way ``validate_path``/``safe_join_path`` require -- this is
        the same base-directory-free validator the rest of this codebase
        already uses for user-chosen save/output paths (e.g.
        ``notes_screen._import_note_from_path``, ``settings_screen``'s
        storage-location fields). It rejects null bytes and other
        shell-metacharacter/traversal patterns; a rejected path is a quiet
        warning notice with no write and no crash, same as any other
        failure in this method. This method awaits nothing (the write is a
        plain synchronous ``Path.write_text``), so it is a plain method
        rather than a coroutine -- Textual's ``call_after_refresh`` (its
        only caller, via ``_export_library_note``'s ``FileSave`` callback)
        accepts either.

        Args:
            selected_path: The chosen destination, or ``None`` if the
                dialog was cancelled.
            export_format: ``"markdown"`` or ``"text"`` (see
                ``build_note_export_content``).
            title: The note title captured when Export was pressed.
            content: The note body captured when Export was pressed.
            keywords_text: The note's keywords, as a comma-separated string.
            note_id: The note's id.
        """
        notify = getattr(self.app_instance, "notify", None)
        if not selected_path:
            if callable(notify):
                notify("Note export cancelled.", severity="information")
            return
        try:
            validated_path = validate_path_simple(selected_path, require_exists=False)
        except ValueError as exc:
            logger.warning(
                f"Rejected Library note export path {selected_path!r} for {note_id!r}: {exc}"
            )
            if callable(notify):
                notify(f"Rejected export path: {exc}", severity="warning")
            return
        try:
            validated_path.write_text(
                build_note_export_content(title, content, keywords_text, note_id, export_format),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning(
                f"Error exporting Library note {note_id!r} to '{validated_path}'.", exc_info=True
            )
            if callable(notify):
                notify(f"Error exporting note: {type(exc).__name__}", severity="error")
            return
        if callable(notify):
            notify(f"Note exported successfully to {validated_path.name}", severity="information")

    @on(Button.Pressed, "#library-note-export-md")
    async def handle_library_note_export_markdown(self, event: Button.Pressed) -> None:
        """Export the open note as Markdown via a ``FileSave`` dialog.

        Args:
            event: Button press event emitted by the editor's "Export .md" action.
        """
        event.stop()
        await self._export_library_note("markdown")

    @on(Button.Pressed, "#library-note-export-txt")
    async def handle_library_note_export_text(self, event: Button.Pressed) -> None:
        """Export the open note as plain text via a ``FileSave`` dialog.

        Args:
            event: Button press event emitted by the editor's "Export .txt" action.
        """
        event.stop()
        await self._export_library_note("text")

    @on(Button.Pressed, "#library-note-copy")
    def handle_library_note_copy(self, event: Button.Pressed) -> None:
        """Copy the open Library note to the clipboard as markdown.

        Uses the app's ``copy_to_clipboard`` seam (the same
        ``getattr``-gated pattern every other Library handoff/action in
        this screen already uses) rather than importing ``pyperclip``
        directly the way ``notes_screen._copy_current_note_to_clipboard``
        does -- that makes this testable via a recorded fake the way the
        rest of this screen's actions are, and doesn't add a hard runtime
        dependency on a package that isn't always installed/working.

        Args:
            event: Button press event emitted by the editor's Copy action.
        """
        event.stop()
        notify = getattr(self.app_instance, "notify", None)
        if self._library_notes_view != "editor" or not self._selected_note_id:
            return
        fields = self._read_library_note_editor_fields()
        if fields is None:
            return
        title, content, keywords_text = fields
        note_id = self._selected_note_id
        export_content = build_note_export_content(title, content, keywords_text, note_id, "markdown")
        copy_to_clipboard = getattr(self.app_instance, "copy_to_clipboard", None)
        if not callable(copy_to_clipboard):
            if callable(notify):
                notify("Clipboard copy is unavailable in this runtime.", severity="warning")
            return
        try:
            copy_to_clipboard(export_content)
        except Exception as exc:
            logger.warning(f"Failed to copy Library note {note_id!r} to clipboard.", exc_info=True)
            if callable(notify):
                notify(f"Error copying note: {type(exc).__name__}", severity="error")
            return
        if callable(notify):
            notify("Note copied to clipboard as markdown!", severity="information")

    def _selected_library_note_handoff_payload(self) -> ChatHandoffPayload | None:
        """Build the Console handoff payload for the open Library note.

        Mirrors ``_selected_media_handoff_payload``: reads the currently
        open note's live (possibly unsaved) editor fields rather than a
        list-row selection, matching ``notes_screen._build_note_chat_handoff_payload``'s
        local-note shape.

        Returns:
            A ``ChatHandoffPayload`` staging the open note as Console
            context, or ``None`` when no note is currently open.
        """
        if self._library_notes_view != "editor" or not self._selected_note_id:
            return None
        fields = self._read_library_note_editor_fields()
        if fields is None:
            return None
        title, content, keywords_text = fields
        note_id = self._selected_note_id
        keywords = self._library_note_keywords_from_input(keywords_text) or []
        return ChatHandoffPayload.from_source_content(
            source="notes",
            item_type="note",
            title=title.strip() or "Untitled Note",
            body=content,
            source_id=str(note_id),
            suggested_prompt="Use this note as context and help me work with it.",
            runtime_backend="local",
            source_owner="local",
            source_selector_state="local",
            discovery_owner="notes",
            discovery_entity_id=str(note_id),
            scope_type="global",
            metadata={
                "note_version": self._library_note_version,
                "keywords": keywords,
            },
        )

    def _open_selected_library_note_handoff(self) -> None:
        """Stage the note open in the editor into Console via the shared handoff.

        Mirrors ``_open_selected_media_handoff``: builds the payload, then
        guards on having an open note, the workspace context-handoff gate,
        and the app exposing ``open_chat_with_handoff`` at all.
        """
        workspace_state = self._library_workspace_depth_state()
        payload = self._selected_library_note_handoff_payload()
        notify = getattr(self.app_instance, "notify", None)
        if payload is None:
            if callable(notify):
                notify("Open a note before using it in Console.", severity="warning")
            return
        if not workspace_state.context_handoff_enabled:
            if callable(notify):
                notify(workspace_state.context_handoff_tooltip, severity="warning")
            return
        open_chat_with_handoff = getattr(self.app_instance, "open_chat_with_handoff", None)
        if not callable(open_chat_with_handoff):
            if callable(notify):
                notify("Console handoff is unavailable for Library Notes.", severity="warning")
            return
        open_chat_with_handoff(payload)

    @on(Button.Pressed, "#library-note-use-in-console")
    def handle_library_note_use_in_console(self, event: Button.Pressed) -> None:
        """Hand the open note off to Console as chat context.

        Args:
            event: Button press event emitted by the editor's
                "Use in Console" action.
        """
        event.stop()
        self._open_selected_library_note_handoff()

    def _compose_mode_canvas(self, mode: str) -> ComposeResult:
        """Render the canvas body for a mode row (moved middle-pane content)."""
        self._active_mode = mode
        active_mode = LIBRARY_MODES.get(mode, LIBRARY_MODES["sources"])
        active_mode_copy_visible = mode not in {
            "collections",
            "search",
            "sources",
            "workspaces",
        }
        if active_mode_copy_visible:
            yield Static(
                f"{active_mode['label']} mode",
                id="library-active-mode-title",
                classes="destination-section",
            )
            yield Static(
                active_mode["description"],
                id="library-active-mode-description",
            )
            yield Static(
                active_mode["next_action"],
                id="library-active-mode-next-action",
            )
        if mode in LIBRARY_STUDY_HANDOFF_MODES:
            yield self._study_handoff_detail_widget()
        elif mode == "search":
            yield LibrarySearchRagPanel(
                self._library_rag_panel_state(),
                id="library-search-rag-panel",
            )
        elif mode == "collections":
            yield LibraryCollectionsPanel(
                self._library_collections_panel_state(),
                name_value=self._library_collection_name_input,
                description_value=self._library_collection_description_input,
                delete_pending=bool(self._library_collection_pending_delete_id),
                id="library-collections-panel",
            )
        elif mode == "import-export":
            for row in self._import_export_workflow_rows():
                yield row

    def _library_rail_preferences(self):
        """Read persisted Library rail section preferences."""
        app_config = getattr(self.app_instance, "app_config", None)
        raw = None
        if isinstance(app_config, dict):
            library_config = app_config.get("library")
            if isinstance(library_config, dict):
                rail_state = library_config.get("rail_state")
                if isinstance(rail_state, dict):
                    raw = rail_state.get("sections")
        return coerce_library_rail_preferences(raw)

    def _set_library_rail_section(self, section_id: str, open_state: bool) -> None:
        """Persist one section preference and sync the rail body/header."""
        if section_id not in LIBRARY_RAIL_SECTION_IDS:
            return
        from dataclasses import replace as dataclass_replace

        preferences = dataclass_replace(
            self._library_rail_preferences(), **{f"{section_id}_open": open_state}
        )
        serialized = serialize_library_rail_preferences(preferences)
        app_config = getattr(self.app_instance, "app_config", None)
        if isinstance(app_config, dict):
            library_config = app_config.get("library")
            if not isinstance(library_config, dict):
                library_config = {}
                app_config["library"] = library_config
            rail_state = library_config.get("rail_state")
            if not isinstance(rail_state, dict):
                rail_state = {}
                library_config["rail_state"] = rail_state
            rail_state["sections"] = serialized
        self._save_library_rail_preferences(serialized)
        try:
            body = self.query_one(f"#library-rail-section-body-{section_id}")
            header = self.query_one(
                f"#library-rail-section-header-{section_id}", ConsoleRailSectionHeader
            )
        except Exception:
            return
        body.display = open_state
        header.sync_open(open_state)

    @work(thread=True)
    def _save_library_rail_preferences(self, serialized: dict[str, bool]) -> None:
        """Persist Library rail preferences without blocking the UI thread."""
        try:
            save_setting_to_cli_config("library.rail_state", "sections", serialized)
        except Exception:
            pass

    @on(Button.Pressed, ".library-rail-row")
    async def handle_library_rail_row(self, event: Button.Pressed) -> None:
        """Dispatch a Library rail row press: navigate, browse, or open a mode."""
        event.stop()
        button = event.button
        target_kind = str(getattr(button, "target_kind", "") or "")
        target_id = str(getattr(button, "target_id", "") or "")
        row_id = str(getattr(button, "row_id", "") or "")
        if target_kind == "screen":
            if target_id:
                self.post_message(NavigateToScreen(target_id))
            return
        if target_kind == "canvas":
            if target_id == "conversations":
                self._library_conversation_query = ""
            await self._select_library_rail_row(row_id, target_id or "conversations")
            return
        if target_kind == "mode":
            await self._select_library_rail_row(row_id, target_id)
            return
        # Unknown target kind: select the row and recompose from selection.
        await self._select_library_rail_row(row_id, self._active_mode)

    async def _select_library_rail_row(self, row_id: str, active_mode: str) -> None:
        """Apply a rail-row selection and recompose the canvas from it.

        Shared by the rail-row press handler and in-canvas mode shortcuts so
        that the single source of selection truth (``_library_selected_row_id``)
        always drives the recomposed canvas -- setting ``_active_mode`` alone is
        reverted by the next ``refresh(recompose=True)``.

        A dirty note edit is flushed first (awaited) so leaving via the rail
        never silently discards unsaved text; an unresolved save conflict
        aborts the row switch entirely so the user must resolve it first.
        """
        await self._flush_library_note_save()
        if self._library_note_autosave_state == "conflict":
            return
        self._library_selected_row_id = row_id
        self._active_mode = active_mode
        # A rail-row press is always a fresh entry into a content type, so
        # the media canvas must never resume a previously opened viewer
        # (e.g. Browse Media -> open item -> Browse Conversations -> Browse
        # Media again must show the list, not the stale viewer).
        self._library_media_view = "list"
        self._library_media_detail = None
        self._library_media_editing = False
        self._library_media_confirming_delete = False
        self._library_media_highlights = []
        self._library_media_editing_analysis = False
        self._library_media_content_query = ""
        self._library_media_content_match_index = 0
        self._library_notes_filter = ""
        self._library_notes_filter_records = None
        self._reset_library_note_editor_state()
        self._reset_library_notes_sync_transient_state()
        self._invalidate_library_workspace_depth_state()
        if self._active_mode == "collections" and not self._library_collections_loaded:
            # First Collections entry must load the snapshot the retired chip
            # flow ran; _sync_collections_panel recomposes once records arrive.
            await self._sync_collections_panel(refresh_snapshot=True)
            return
        self.refresh(recompose=True)

    @on(Button.Pressed, ".console-rail-section-toggle")
    def handle_library_rail_section_toggle(self, event: Button.Pressed) -> None:
        """Toggle a Library rail section and persist the preference.

        Args:
            event: Button press event emitted by a rail section's
                collapse/expand toggle.
        """
        button_id = event.button.id or ""
        prefix = f"{CONSOLE_RAIL_SECTION_TOGGLE_PREFIX}library-"
        if not button_id.startswith(prefix):
            return
        event.stop()
        section_id = button_id.removeprefix(prefix)
        currently_open = bool(
            getattr(self._library_rail_preferences(), f"{section_id}_open", True)
        )
        self._set_library_rail_section(section_id, not currently_open)

    @on(Button.Pressed, ".library-conversation-row")
    def handle_library_conversation_row(self, event: Button.Pressed) -> None:
        """Select a conversation row in the Library conversations canvas.

        Args:
            event: Button press event emitted by a conversation row button.
        """
        event.stop()
        conversation_id = str(getattr(event.button, "conversation_id", "") or "")
        if conversation_id:
            self._selected_conversation_id = conversation_id
        self._library_selected_row_id = LIBRARY_ROW_BROWSE_CONVERSATIONS
        self._active_mode = "conversations"
        self.refresh(recompose=True)

    @on(Button.Pressed, "#library-media-type-filter")
    def handle_library_media_type_filter_pressed(self, event: Button.Pressed) -> None:
        """Cycle the Library media canvas filter to the next available type.

        Advances through the authoritative ``type_options`` tuple built by
        ``_build_library_media_state`` (e.g. ``("All", "audio", "video")``),
        wrapping back to the first option after the last. Replaces the
        previous ``Select``-based filter, which did not render reliably in
        the deployed TUI; a ``Button.Pressed`` handler only fires on real
        user presses, so no mount-time-loop guard is needed here.

        Args:
            event: Button press event emitted by the media type filter.
        """
        event.stop()
        type_options = self._build_library_media_state().type_options
        if not type_options:
            return
        try:
            current_index = type_options.index(self._library_media_type_filter)
        except ValueError:
            current_index = 0
        next_index = (current_index + 1) % len(type_options)
        self._library_media_type_filter = type_options[next_index]
        self.refresh(recompose=True)

    @on(Button.Pressed, ".library-media-row")
    def handle_library_media_row(self, event: Button.Pressed) -> None:
        """Select a media row and open the full Library media viewer.

        Switches the media canvas from its list view to the in-canvas
        viewer, clears any stale detail, and kicks the async detail fetch
        (``_refresh_library_media_detail``); the viewer renders a loading
        line until that worker stores the fetched detail and recomposes.

        Args:
            event: Button press event emitted by a media row button.
        """
        event.stop()
        media_id = str(getattr(event.button, "media_id", "") or "")
        if media_id:
            self._selected_media_id = media_id
        self._library_selected_row_id = LIBRARY_ROW_BROWSE_MEDIA
        self._active_mode = "media"
        self._library_media_view = "viewer"
        self._library_media_detail = None
        self._library_media_editing = False
        self._library_media_confirming_delete = False
        self._library_media_highlights = []
        self._library_media_editing_analysis = False
        self._library_media_content_query = ""
        self._library_media_content_match_index = 0
        if media_id:
            # Exclusive in its own group so rapidly switching rows cancels the
            # previous in-flight detail fetch instead of letting a slower older
            # fetch finish and overwrite the newer selection's viewer.
            self.run_worker(
                self._refresh_library_media_detail(media_id),
                exclusive=True,
                group="library_media_detail",
            )
        self.refresh(recompose=True)

    @on(Button.Pressed, "#library-notes-sort")
    def handle_library_notes_sort(self, event: Button.Pressed) -> None:
        """Cycle the Library notes canvas sort mode (newest/oldest/title).

        Args:
            event: Button press event emitted by the notes sort control.
        """
        event.stop()
        self._library_notes_sort = next_notes_sort_mode(self._library_notes_sort)
        self.refresh(recompose=True)

    @on(Input.Submitted, "#library-notes-filter")
    def handle_library_notes_filter(self, event: Input.Submitted) -> None:
        """Apply the Library notes filter on Enter via the ``search_notes`` seam.

        Args:
            event: Input submission event emitted by the notes filter box.
        """
        event.stop()
        submitted = self._safe_text(event.value, max_length=200).strip()
        if submitted == self._library_notes_filter:
            return
        self._library_notes_filter = submitted
        if not submitted:
            self._library_notes_filter_records = None
            self.refresh(recompose=True)
            return
        self.run_worker(
            self._run_library_notes_filter(submitted),
            exclusive=True,
            group="library_notes_filter",
        )

    async def _run_library_notes_filter(self, query: str) -> None:
        """Fetch filtered notes from the ``search_notes`` seam and recompose.

        Clearing the filter (an empty submit) is handled synchronously by
        ``handle_library_notes_filter`` and never starts this worker, so it
        cannot cancel a slower, still-in-flight call for a *previous*
        non-empty query the ordinary way ``run_worker(exclusive=True)``
        would. Re-checking ``query`` against the current
        ``_library_notes_filter`` after the await closes that gap: a
        mismatch means the filter was cleared (or changed again) while
        this call was in flight, so the now-stale result is discarded
        instead of overwriting the cleared/changed state.

        Args:
            query: The submitted filter text.
        """
        service = getattr(self.app_instance, "notes_scope_service", None)
        if service is None:
            return
        try:
            records = await self._run_library_service_call(
                service.search_notes,
                scope="local_note",
                query=query,
                limit=LIBRARY_SOURCE_PAGE_SIZES["notes"],
                user_id=getattr(self.app_instance, "notes_user_id", None) or "default_user",
                isolate_in_worker=True,
            )
        except Exception:
            logger.warning("Library notes filter failed.", exc_info=True)
            return
        if query != self._library_notes_filter:
            return
        self._library_notes_filter_records = list(records or [])
        self.refresh(recompose=True)
        self.call_after_refresh(self._focus_library_notes_filter_input)

    def _focus_library_notes_filter_input(self) -> None:
        """Restore focus to the notes filter box after a filter recompose."""
        try:
            self.query_one("#library-notes-filter", Input).focus()
        except (NoMatches, QueryError):
            pass

    _LIBRARY_NOTE_IMPORT_TITLE_MAX_CHARS = 300

    @on(Button.Pressed, "#library-notes-import")
    def handle_library_notes_import(self, event: Button.Pressed) -> None:
        """Push a ``FileOpen`` dialog to import a local file as a new note.

        Mirrors ``notes_screen.handle_import_button``'s dialog flow exactly
        (the working ``FileOpen`` reference -- unlike ``FileSave``, whose
        constructor only accepts ``location``/``title``/``default_file``,
        ``FileOpen`` here is invoked the same simple ``title=``-only way the
        standalone screen already relies on). The callback resolves the
        chosen path (or ``None`` on cancel) through
        ``_import_library_note_from_path``, which validates, reads, parses,
        and hands off to the existing ``_create_library_note`` seam -- so a
        successful import lands in the editor with the snapshot/count
        refresh that seam already performs.

        Args:
            event: Button press event emitted by the "Import note" action.
        """
        event.stop()

        async def import_callback(selected_path: Path | None) -> None:
            await self._import_library_note_from_path(selected_path)

        self.app.push_screen(
            FileOpen(title="Import Note (TXT, MD, JSON, YAML)"),
            import_callback,
        )

    async def _import_library_note_from_path(self, selected_path: Path | None) -> None:
        """Validate, read, and parse a chosen file, then create a note from it.

        Cancelling the dialog (``selected_path is None``) is a silent no-op.
        Every other failure mode -- a path ``validate_path_simple`` rejects,
        a file that cannot be read/decoded, or one larger than
        ``LIBRARY_NOTE_CONTENT_MAX_CHARS`` -- is a quiet warning notice with
        no note created, matching every other Library note failure path in
        this screen. The file read is offloaded to a thread (mirroring
        ``notes_screen._import_note_from_path``); it is bounded by the same
        size cap enforced right after, so it can never block the UI loop on
        an unbounded read.

        Args:
            selected_path: The path chosen via the ``FileOpen`` dialog, or
                ``None`` if the dialog was cancelled.
        """
        if selected_path is None:
            return

        from tldw_chatbook.Event_Handlers.notes_events import _parse_note_from_file_content

        try:
            note_path = validate_path_simple(str(selected_path), require_exists=True)
        except ValueError:
            logger.warning(f"Rejected Library note import path {selected_path!r}.", exc_info=True)
            self._notify_library_note_create_warning("Could not import that file.")
            return

        try:
            file_content = await asyncio.to_thread(
                note_path.read_text, encoding="utf-8", errors="strict"
            )
        except (OSError, UnicodeDecodeError):
            logger.warning(f"Could not read Library note import file '{note_path}'.", exc_info=True)
            self._notify_library_note_create_warning("Could not import that file.")
            return

        if len(file_content) > LIBRARY_NOTE_CONTENT_MAX_CHARS:
            self._notify_library_note_create_warning("Could not import that file.")
            return

        title, content = _parse_note_from_file_content(note_path, file_content)
        title = sanitize_string(title or "", max_length=self._LIBRARY_NOTE_IMPORT_TITLE_MAX_CHARS)
        if not title:
            title = note_path.stem or "Imported note"
        content = (content or "").replace("\x00", "")

        await self._create_library_note(title=title, content=content)

    # ----- Notes sync panel ------------------------------------------------

    @on(Button.Pressed, "#library-notes-sync-open")
    async def handle_library_notes_sync_open(self, event: Button.Pressed) -> None:
        """Enter the in-canvas notes sync panel from the notes list header.

        Flushes any pending editor save first (mirrors every other exit
        from the notes list/editor) and seeds direction/conflict/auto-sync
        from config on first entry only.

        Args:
            event: Button press event emitted by the "Sync" action.
        """
        event.stop()
        await self._flush_library_note_save()
        if self._library_note_autosave_state == "conflict":
            return
        self._ensure_library_notes_sync_config_loaded()
        self._library_notes_view = "sync"
        self.refresh(recompose=True)

    @on(Button.Pressed, "#library-notes-sync-back")
    async def handle_library_notes_sync_back(self, event: Button.Pressed) -> None:
        """Return the Library notes canvas from the sync panel to its list view.

        Args:
            event: Button press event emitted by the "‹ Back to notes" action.
        """
        event.stop()
        await self._flush_library_note_save()
        self._library_notes_view = "list"
        self._reset_library_notes_sync_transient_state()
        self.refresh(recompose=True)

    @on(Input.Changed, "#library-notes-sync-folder")
    def handle_library_notes_sync_folder_changed(self, event: Input.Changed) -> None:
        """Persist the sync folder as the user edits it.

        Args:
            event: Input change event emitted by the sync folder box.
        """
        event.stop()
        save_setting_to_cli_config("notes", "sync_directory", event.value)

    @on(Button.Pressed, "#library-notes-sync-browse")
    async def handle_library_notes_sync_browse(self, event: Button.Pressed) -> None:
        """Open a directory picker and adopt the chosen folder.

        Args:
            event: Button press event emitted by the "Browse…" action.
        """
        event.stop()
        from ...Third_Party.textual_fspicker import SelectDirectory

        current = Path(self._library_notes_sync_folder()).expanduser()
        if not current.exists():
            current = Path.home()
        await self.app.push_screen(
            SelectDirectory(str(current), title="Select Notes Sync Folder"),
            callback=self._apply_library_notes_sync_folder,
        )

    def _apply_library_notes_sync_folder(self, path: Path | None) -> None:
        """Persist and render the folder chosen via ``SelectDirectory``."""
        if not path:
            return
        save_setting_to_cli_config("notes", "sync_directory", str(path))
        if self._library_notes_view == "sync":
            self.refresh(recompose=True)

    @on(Button.Pressed, "#library-notes-sync-direction")
    def handle_library_notes_sync_direction(self, event: Button.Pressed) -> None:
        """Cycle the sync direction and persist the new value.

        Args:
            event: Button press event emitted by the direction cycler.
        """
        event.stop()
        self._library_notes_sync_direction = next_sync_direction(
            self._library_notes_sync_direction
        )
        save_setting_to_cli_config(
            "notes", "sync_direction", self._library_notes_sync_direction
        )
        self.refresh(recompose=True)

    @on(Button.Pressed, "#library-notes-sync-conflict")
    def handle_library_notes_sync_conflict(self, event: Button.Pressed) -> None:
        """Cycle the conflict-resolution mode and persist the new value.

        Args:
            event: Button press event emitted by the conflict cycler.
        """
        event.stop()
        self._library_notes_sync_conflict = next_sync_conflict(
            self._library_notes_sync_conflict
        )
        save_setting_to_cli_config(
            "notes", "sync_conflict_resolution", self._library_notes_sync_conflict
        )
        self.refresh(recompose=True)

    @on(Button.Pressed, "#library-notes-sync-auto")
    def handle_library_notes_sync_auto_toggle(self, event: Button.Pressed) -> None:
        """Toggle auto-sync, persist it, and arm/cancel the repeating timer.

        The timer is scoped to this Library screen instance's lifetime --
        the same scope the standalone ``NotesSyncPane``'s timer had -- not
        persisted/resumed across screen instances; only the boolean
        preference persists, and is re-armed the next time sync mode is
        entered (``_ensure_library_notes_sync_config_loaded``).

        Args:
            event: Button press event emitted by the auto-sync toggle.
        """
        event.stop()
        self._library_notes_sync_auto = not self._library_notes_sync_auto
        save_setting_to_cli_config("notes", "auto_sync", self._library_notes_sync_auto)
        if self._library_notes_sync_auto:
            self._arm_library_notes_auto_sync_timer()
        else:
            self._cancel_library_notes_auto_sync_timer()
        self.refresh(recompose=True)

    @on(Button.Pressed, "#library-notes-sync-run")
    def handle_library_notes_sync_run(self, event: Button.Pressed) -> None:
        """Validate the folder and kick off a sync run as an exclusive worker.

        Args:
            event: Button press event emitted by the "Sync now" action.
        """
        event.stop()
        if self._library_notes_sync_running:
            return
        folder_value = self._library_notes_sync_folder()
        if not folder_value:
            self._notify_library_notes_sync_warning("Please select a folder to sync.")
            return
        try:
            folder = validate_path_simple(
                Path(folder_value).expanduser(), require_exists=True
            )
        except ValueError:
            self._notify_library_notes_sync_warning(
                "That sync folder does not exist."
            )
            return
        if not folder.is_dir():
            self._notify_library_notes_sync_warning(
                "Selected path is a file; choose a folder to sync."
            )
            return
        self.run_worker(
            self._run_library_notes_sync(folder),
            exclusive=True,
            group="library_notes_sync",
        )

    def _notify_library_notes_sync_warning(self, message: str) -> None:
        notify = getattr(self.app_instance, "notify", None)
        if callable(notify):
            notify(message, severity="warning")

    async def _run_library_notes_sync(self, folder: Path) -> None:
        """Run one notes sync pass against ``folder`` and report the outcome.

        Builds a fresh ``NotesSyncService`` per run (mirroring
        ``NotesSyncPane.on_mount`` -- see ``_resolve_library_notes_sync_db``)
        and calls ``sync_folder`` offloaded onto a worker thread via
        ``_run_library_service_call(..., isolate_in_worker=True)``, since
        the sync engine walks the filesystem and touches the DB
        synchronously in places.

        The engine's ``progress_callback`` fires from that worker thread
        (a plain function call, not a coroutine -- see
        ``NotesSyncEngine``), so it is never called directly here; it is
        marshaled onto the UI thread via ``self.app.call_from_thread``
        (Textual's own running-App property -- the same object as
        ``self.app_instance`` in production, but the one that actually
        matters for ``call_from_thread``'s event-loop lookup) and only
        ever performs targeted ``query_one(...).update(...)`` calls,
        guarded against the user having navigated away mid-run. Only the
        start and the final outcome trigger a recompose (also
        freshness-guarded).
        """
        from ...Notes.sync_engine import ConflictResolution, SyncDirection
        from ...Notes.sync_service import NotesSyncService

        notes_service = getattr(self.app_instance, "notes_service", None)
        db = self._resolve_library_notes_sync_db()
        if notes_service is None or db is None:
            self._notify_library_notes_sync_warning(
                "Sync service is unavailable in this runtime."
            )
            return

        try:
            direction = SyncDirection(self._library_notes_sync_direction)
        except ValueError:
            direction = SyncDirection.BIDIRECTIONAL
        try:
            resolution = ConflictResolution(self._library_notes_sync_conflict)
        except ValueError:
            resolution = ConflictResolution.NEWER_WINS

        self._library_notes_sync_running = True
        self._library_notes_sync_status = sync_status_line("syncing", processed=0, total=0)
        self._library_notes_sync_activity = append_activity(
            self._library_notes_sync_activity, f"Starting sync: {folder.name}"
        )
        self.refresh(recompose=True)

        def progress_callback(sync_progress: Any) -> None:
            def apply() -> None:
                if self._library_notes_view != "sync" or not self.is_mounted:
                    return
                total = getattr(sync_progress, "total_files", 0)
                processed = getattr(sync_progress, "processed_files", 0)
                self._library_notes_sync_status = sync_status_line(
                    "syncing", processed=processed, total=total
                )
                try:
                    self.query_one("#library-notes-sync-status", Static).update(
                        self._library_notes_sync_status
                    )
                except (NoMatches, QueryError):
                    pass

            # ``self.app`` (Textual's own running-App property), not
            # ``self.app_instance`` -- in production the two are the same
            # object (``screen_class(self)`` in app.py), but
            # ``call_from_thread`` needs the App whose event loop is
            # actually running this screen, which is what ``self.app``
            # always resolves to even where a test harness's ``app_instance``
            # is a separate, non-running object.
            try:
                self.app.call_from_thread(apply)
            except RuntimeError:
                # The app already finished shutting down mid-sync (or this
                # is being invoked outside a running app entirely) --
                # a missed progress tick must never surface as a sync
                # error for an otherwise-successful file.
                pass

        try:
            service = NotesSyncService(notes_service=notes_service, db=db)
            _session_id, results = await self._run_library_service_call(
                service.sync_folder,
                root_folder=folder,
                user_id=self._library_notes_user_id(),
                direction=direction,
                conflict_resolution=resolution,
                progress_callback=progress_callback,
                isolate_in_worker=True,
            )
            processed = len(results.created_notes) + len(results.updated_notes) + (
                len(results.created_files) + len(results.updated_files)
            )
            conflicts = len(results.conflicts)
            self._library_notes_sync_status = sync_status_line(
                "done", processed=processed, total=processed, conflicts=conflicts
            )
            summary_parts = []
            if results.created_notes:
                summary_parts.append(f"{len(results.created_notes)} notes created")
            if results.updated_notes:
                summary_parts.append(f"{len(results.updated_notes)} notes updated")
            if results.created_files:
                summary_parts.append(f"{len(results.created_files)} files created")
            if results.updated_files:
                summary_parts.append(f"{len(results.updated_files)} files updated")
            summary = ", ".join(summary_parts) if summary_parts else "No changes"
            self._library_notes_sync_activity = append_activity(
                self._library_notes_sync_activity, f"Sync complete: {summary}"
            )
            if conflicts:
                self._library_notes_sync_activity = append_activity(
                    self._library_notes_sync_activity,
                    f"{conflicts} conflicts recorded for review",
                )
            if results.errors:
                self._library_notes_sync_activity = append_activity(
                    self._library_notes_sync_activity,
                    f"{len(results.errors)} errors during sync",
                )
        except Exception as exc:
            logger.error(f"Library notes sync failed (folder={folder}): {exc}", exc_info=True)
            self._library_notes_sync_status = sync_status_line("failed", error=str(exc))
            self._library_notes_sync_activity = append_activity(
                self._library_notes_sync_activity, f"Sync failed: {exc}"
            )
        finally:
            self._library_notes_sync_running = False
            if self._library_notes_view == "sync" and self.is_mounted:
                self.refresh(recompose=True)

    @on(Button.Pressed, ".library-notes-row")
    async def handle_library_notes_row(self, event: Button.Pressed) -> None:
        """Select a note row and open the in-canvas Library note editor.

        Switches the notes canvas from its list view to the editor, clears
        any stale detail, and kicks the async detail fetch
        (``_refresh_library_note_detail``); ``compose_content`` renders a
        loading line until that worker stores the fetched detail and
        recomposes. Mirrors ``handle_library_media_row``.

        Flushes any dirty edit from a previously-open note first (awaited)
        so switching notes never silently discards unsaved text; an
        unresolved save conflict aborts the switch so it can be resolved.

        Args:
            event: Button press event emitted by a note row button.
        """
        event.stop()
        await self._flush_library_note_save()
        if self._library_note_autosave_state == "conflict":
            return
        note_id = str(getattr(event.button, "note_id", "") or "")
        if note_id:
            self._selected_note_id = note_id
        self._library_selected_row_id = LIBRARY_ROW_BROWSE_NOTES
        self._active_mode = "notes"
        self._library_notes_view = "editor"
        self._library_note_detail = None
        self._library_note_version = None
        self._library_note_dirty = False
        self._library_note_autosave_state = "idle"
        self._library_note_conflict_snapshot = None
        self._library_note_confirming_delete = False
        self._library_note_preview = False
        self._library_note_preview_snapshot = None
        self._library_note_editor_armed = False
        if note_id:
            # Exclusive in its own group so rapidly switching rows cancels the
            # previous in-flight detail fetch instead of letting a slower older
            # fetch finish and overwrite the newer selection's editor.
            self.run_worker(
                self._refresh_library_note_detail(note_id),
                exclusive=True,
                group="library_note_detail",
            )
        self.refresh(recompose=True)

    @on(Button.Pressed, "#library-note-back")
    async def handle_library_note_back(self, event: Button.Pressed) -> None:
        """Return the Library notes canvas from the editor to its list view.

        Flushes a dirty edit first (awaited) so Back never silently
        discards unsaved text; an unresolved save conflict aborts the
        navigation so the user resolves it via Overwrite/Reload first.

        Args:
            event: Button press event emitted by the "‹ Back to list" action.
        """
        event.stop()
        await self._flush_library_note_save()
        if self._library_note_autosave_state == "conflict":
            return
        self._reset_library_note_editor_state()
        self.refresh(recompose=True)

    @on(Button.Pressed, "#library-note-delete")
    async def handle_library_note_delete(self, event: Button.Pressed) -> None:
        """Enter the inline delete-confirmation state for the open note.

        Deleting is destructive, so the first press only swaps the normal
        action row for a "Delete" / "Cancel" confirm affordance (mirroring
        ``handle_library_media_delete``); the actual service call only
        happens once the confirm button (``#library-note-delete-confirm``)
        is pressed.

        Flushes a dirty edit first (awaited) so the version the confirmed
        delete sends is never stale; an unresolved save conflict aborts
        entering the confirm state so the user resolves it via
        Overwrite/Reload first, same as Back and note-row selection.

        Args:
            event: Button press event emitted by the editor's "Delete" action.
        """
        event.stop()
        await self._flush_library_note_save()
        if self._library_note_autosave_state == "conflict":
            return
        self._library_note_confirming_delete = True
        self._library_note_editor_armed = False
        self.refresh(recompose=True)
        self.call_after_refresh(self._arm_library_note_editor)

    @on(Button.Pressed, "#library-note-delete-cancel")
    def handle_library_note_delete_cancel(self, event: Button.Pressed) -> None:
        """Discard the pending delete confirmation and restore the normal action row.

        Args:
            event: Button press event emitted by the confirm affordance's
                "Cancel" action.
        """
        event.stop()
        self._library_note_confirming_delete = False
        self._library_note_editor_armed = False
        self.refresh(recompose=True)
        self.call_after_refresh(self._arm_library_note_editor)

    @on(Button.Pressed, "#library-note-delete-confirm")
    def handle_library_note_delete_confirm(self, event: Button.Pressed) -> None:
        """Hand the confirmed delete off to a worker that removes the note.

        Reads the currently selected note id/version synchronously (before
        any recompose can clear them) and defers the actual service call
        and state mutation to ``_delete_library_note`` -- mirroring
        ``handle_library_media_delete_confirm``.

        Args:
            event: Button press event emitted by the confirm affordance's
                "Delete" action.
        """
        event.stop()
        note_id = self._selected_note_id
        if not note_id:
            self._library_note_confirming_delete = False
            self.refresh(recompose=True)
            return
        self.run_worker(
            self._delete_library_note(note_id, version=self._library_note_version),
            exclusive=True,
            group="library_note_delete",
        )

    async def _delete_library_note(self, note_id: str, *, version: int | None) -> None:
        """Delete the selected Library note, then return to the list view.

        Calls ``delete_note`` through the offloaded service seam. The real
        local notes backend signals a stale (optimistic-lock) ``version``
        by raising ``ConflictError`` (mirroring ``update_note``'s
        stale-version signaling, see ``_save_library_note``), so that
        exception is normalized to the same falsy ``deleted`` outcome as an
        explicit ``False`` return -- both are quiet-warning, stay-in-editor
        outcomes here, never a crash.

        Guards against a missing ``delete_note`` service the same way
        ``_delete_library_media_item`` guards a missing
        ``delete_media_item``.

        On success, drops the note from the cached local-source snapshot by
        re-running the full snapshot reload (the same seam Task 6's create
        flow uses -- notes have no existing "patch the cached snapshot"
        mutation path the way media deletes do) so the list view and the
        rail's Notes count both reflect the deletion, resets the editor
        state, and returns to the list view.

        A stale result -- the user has since switched to a different note
        while this delete was in flight -- is discarded before mutating any
        shared editor state, mirroring the freshness guard in
        ``_save_library_note``.

        Args:
            note_id: The Library note id to delete.
            version: The note's in-memory version at confirm time.
        """
        service = getattr(self.app_instance, "notes_scope_service", None)
        delete_note = getattr(service, "delete_note", None)
        if not callable(delete_note):
            self._library_note_confirming_delete = False
            self._notify_library_note_delete_warning("Note deletion is unavailable.")
            self._library_note_editor_armed = False
            if self.is_mounted:
                self.refresh(recompose=True)
                self.call_after_refresh(self._arm_library_note_editor)
            return

        try:
            deleted = await self._run_library_service_call(
                delete_note,
                scope="local_note",
                note_id=note_id,
                version=version,
                user_id=self._library_notes_user_id(),
                isolate_in_worker=True,
            )
        except ConflictError:
            deleted = False
        except Exception:
            logger.warning(
                f"Failed to delete Library note {note_id!r}.", exc_info=True
            )
            if note_id != self._selected_note_id or self._library_notes_view != "editor":
                return
            self._library_note_confirming_delete = False
            self._notify_library_note_delete_warning("Could not delete this note.")
            self._library_note_editor_armed = False
            if self.is_mounted:
                self.refresh(recompose=True)
                self.call_after_refresh(self._arm_library_note_editor)
            return

        # Discard a stale result: the user has since switched to a different
        # note (or left the editor) while this delete was in flight.
        if note_id != self._selected_note_id or self._library_notes_view != "editor":
            return

        if not deleted:
            self._library_note_confirming_delete = False
            self._notify_library_note_delete_warning(
                "This note changed elsewhere — refresh and try again."
            )
            self._library_note_editor_armed = False
            if self.is_mounted:
                self.refresh(recompose=True)
                self.call_after_refresh(self._arm_library_note_editor)
            return

        self._reset_library_note_editor_state()
        # Clear any active filter (mirroring the create flow in
        # ``_create_library_note``): the filtered result set is now stale,
        # and leaving it in place would let the just-deleted note keep
        # rendering as a ghost row until the filter box is resubmitted.
        self._library_notes_filter = ""
        self._library_notes_filter_records = None
        # Reuses the same full local-source reload Task 6's create flow
        # uses (already its own exclusive worker via @work) so the list
        # view and the rail's Notes count both drop the deleted note.
        self._refresh_local_source_snapshot()
        if self.is_mounted:
            self.refresh(recompose=True)

    def _notify_library_note_delete_warning(self, message: str) -> None:
        """Surface a quiet warning notice for a failed Library note delete.

        Args:
            message: Human-readable warning text to notify with.
        """
        notify = getattr(self.app_instance, "notify", None)
        if callable(notify):
            notify(message, severity="warning")

    def _notify_library_note_missing_warning(self) -> None:
        """Surface a quiet warning when an opened note is no longer available.

        Used by ``_refresh_library_note_detail`` when the fetched detail
        resolves to nothing -- the note was deleted elsewhere, or the row
        pressed was already a stale ghost row.
        """
        notify = getattr(self.app_instance, "notify", None)
        if callable(notify):
            notify("That note is no longer available.", severity="warning")

    @on(Button.Pressed, "#library-notes-create-blank")
    def handle_library_notes_create_blank(self, event: Button.Pressed) -> None:
        """Create a blank local note from the in-canvas Create view.

        Args:
            event: Button press event emitted by the "Blank note" action.
        """
        event.stop()
        self.run_worker(
            self._create_library_note(title="Untitled", content=""),
            exclusive=True,
            group="library_note_create",
        )

    @on(Button.Pressed, ".library-notes-template-row")
    def handle_library_notes_create_template(self, event: Button.Pressed) -> None:
        """Create a local note pre-filled from the pressed template row.

        Args:
            event: Button press event emitted by a template row button.
        """
        event.stop()
        template_key = str(getattr(event.button, "template_key", "") or "")
        from tldw_chatbook.Event_Handlers.notes_events import NOTE_TEMPLATES

        template = NOTE_TEMPLATES.get(template_key)
        title, content = self._library_note_template_fields(template)
        # Templates carry keywords ("meeting, notes") that the standalone
        # screen applies on create -- parity requires passing them through.
        keywords = list(note_template_keywords(template))
        self.run_worker(
            self._create_library_note(title=title, content=content, keywords=keywords),
            exclusive=True,
            group="library_note_create",
        )

    @staticmethod
    def _library_note_template_fields(template: Any) -> tuple[str, str]:
        """Resolve a note template's title/content for the create flow.

        Malformed or unknown templates (not a mapping, or missing/non-string
        ``title``) degrade to the same "Untitled" / empty-body defaults the
        Blank note button uses. The real ``NOTE_TEMPLATES`` shape (bundled
        ``Config_Files/note_templates.json`` or a user override) always
        carries plain-string ``title``/``content`` values -- callables/
        "content builders" never occur in practice -- but a non-string,
        non-``None`` ``content`` is still coerced on a best-effort basis
        (calling it first if it is callable) rather than raising, so one
        odd template can never break the create view for the others.

        ``{date}``/``{time}``/``{datetime}`` placeholders are resolved
        against the current time, mirroring the standalone Notes screen's
        ``notes_screen._create_local_note_from_template`` substitution
        (same placeholder names, same ``strftime`` formats). Unlike that
        flow -- which notifies and aborts the create on a malformed
        placeholder -- resolution here is per-key: an unknown
        ``{placeholder}`` or a stray brace is left literal in the result,
        while every other *known* placeholder in the same template still
        gets substituted, so one broken template can never crash (or blank
        out) the create view for the others.

        Args:
            template: The raw ``NOTE_TEMPLATES[key]`` value, or ``None``
                when the key is unknown.

        Returns:
            A ``(title, content)`` tuple, always ``str``.
        """
        if not isinstance(template, Mapping):
            return "Untitled", ""
        raw_title = template.get("title")
        title = raw_title if isinstance(raw_title, str) and raw_title.strip() else "Untitled"
        raw_content = template.get("content")
        if isinstance(raw_content, str):
            content = raw_content
        elif raw_content is None:
            content = ""
        else:
            candidate = raw_content
            if callable(candidate):
                try:
                    candidate = candidate()
                except Exception:
                    candidate = ""
            try:
                content = str(candidate)
            except Exception:
                content = ""

        # Shared pure resolver (also drives the create view's row secondary
        # lines) -- per-field, so a title-only malformation still gets the
        # content substituted.
        title = resolve_note_template_placeholders(title)
        content = resolve_note_template_placeholders(content)
        return title, content

    async def _create_library_note(
        self, *, title: str, content: str, keywords: list[str] | None = None
    ) -> None:
        """Create a new local note from the in-canvas Create view and open it.

        Shared by the Blank note button and every template row: both
        resolve their title/content synchronously (see
        ``_library_note_template_fields``) and hand off to this single
        creation seam. Sanitizes the fields through the same Task 5
        boundary helpers ``_save_library_note`` uses, then calls
        ``save_note`` with ``note_id=None`` (the create path) offloaded via
        ``_run_library_service_call``.

        On success, switches straight into the editor for the newly
        created note: selects the Browse > Notes rail row, refreshes the
        local source snapshot (so the new note appears in both the notes
        list and the rail's count -- the same full-refresh the initial
        mount load uses, since there is no existing "append one record"
        mutation path for notes to reuse) and kicks the existing
        ``_refresh_library_note_detail`` worker.

        A missing/failed save leaves the create view in place with a quiet
        warning notice, mirroring ``_notify_library_media_edit_warning``.

        Args:
            title: The note's title (already resolved; not yet sanitized).
            content: The note's body (already resolved; not yet sanitized).
        """
        sanitized_title = self._sanitize_media_field(title, max_length=300)
        sanitized_content = self._sanitize_note_content(content, max_length=LIBRARY_NOTE_CONTENT_MAX_CHARS)
        sanitized_keywords = [
            sanitized
            for keyword in (keywords or [])
            if (sanitized := self._sanitize_media_field(keyword, max_length=100))
        ] or None

        service = getattr(self.app_instance, "notes_scope_service", None)
        save_note = getattr(service, "save_note", None)
        if not callable(save_note):
            self._notify_library_note_create_warning("Note creation is unavailable.")
            return
        try:
            result = await self._run_library_service_call(
                save_note,
                scope="local_note",
                title=sanitized_title,
                content=sanitized_content,
                note_id=None,
                user_id=self._library_notes_user_id(),
                keywords=sanitized_keywords,
                isolate_in_worker=True,
            )
        except Exception:
            logger.warning("Library note create failed.", exc_info=True)
            self._notify_library_note_create_warning("Could not create the note.")
            return

        created_id = result.get("id") if isinstance(result, Mapping) else result
        created_id = str(created_id) if created_id else ""
        if not created_id:
            self._notify_library_note_create_warning("Could not create the note.")
            return

        self._selected_note_id = created_id
        self._library_selected_row_id = LIBRARY_ROW_BROWSE_NOTES
        self._active_mode = "notes"
        self._library_notes_view = "editor"
        self._library_note_detail = None
        self._library_note_version = None
        self._library_note_dirty = False
        self._library_note_autosave_state = "idle"
        self._library_note_conflict_snapshot = None
        self._library_note_confirming_delete = False
        self._library_note_preview = False
        self._library_note_preview_snapshot = None
        self._library_note_editor_armed = False
        self._library_notes_filter = ""
        self._library_notes_filter_records = None
        # Reuses the same full local-source reload the initial mount load
        # runs (already its own exclusive worker via @work) rather than a
        # targeted append, since notes have no existing "patch the cached
        # snapshot" mutation path the way media edits/deletes do.
        self._refresh_local_source_snapshot()
        self.run_worker(
            self._refresh_library_note_detail(created_id),
            exclusive=True,
            group="library_note_detail",
        )
        if self.is_mounted:
            self.refresh(recompose=True)

    def _notify_library_note_create_warning(self, message: str) -> None:
        """Surface a quiet warning notice for a failed Library note create.

        Args:
            message: Human-readable warning text to notify with.
        """
        notify = getattr(self.app_instance, "notify", None)
        if callable(notify):
            notify(message, severity="warning")

    @on(Button.Pressed, "#library-media-back")
    def handle_library_media_back(self, event: Button.Pressed) -> None:
        """Return the Library media canvas from the viewer to its list view.

        Args:
            event: Button press event emitted by the "‹ Back to list" action.
        """
        event.stop()
        self._library_media_view = "list"
        self._library_media_editing = False
        self._library_media_confirming_delete = False
        self._library_media_editing_analysis = False
        self._library_media_content_query = ""
        self._library_media_content_match_index = 0
        self.refresh(recompose=True)

    @on(Button.Pressed, "#library-media-edit")
    def handle_library_media_edit(self, event: Button.Pressed) -> None:
        """Enter metadata edit mode for the open Library media viewer.

        Args:
            event: Button press event emitted by the viewer's "Edit" action.
        """
        event.stop()
        self._library_media_editing = True
        self.refresh(recompose=True)

    @on(Button.Pressed, "#library-media-edit-cancel")
    def handle_library_media_edit_cancel(self, event: Button.Pressed) -> None:
        """Discard in-progress metadata edits and return to the read-only viewer.

        Args:
            event: Button press event emitted by the edit form's "Cancel" action.
        """
        event.stop()
        self._library_media_editing = False
        self.refresh(recompose=True)

    @on(Button.Pressed, "#library-media-edit-save")
    def handle_library_media_edit_save(self, event: Button.Pressed) -> None:
        """Read the edit form's field values and hand the write off to a worker.

        Reads the four edit inputs directly (before any recompose removes
        them), splits the keywords input on commas, and defers the actual
        persistence to ``_save_library_media_edit`` -- a worker that mirrors
        how ``handle_library_media_row`` kicks off
        ``_refresh_library_media_detail`` for the initial detail fetch.

        Args:
            event: Button press event emitted by the edit form's "Save" action.
        """
        event.stop()
        media_id = self._selected_media_id
        if not media_id:
            self._library_media_editing = False
            self.refresh(recompose=True)
            return
        try:
            title = self.query_one("#library-media-edit-title", Input).value
            author = self.query_one("#library-media-edit-author", Input).value
            url = self.query_one("#library-media-edit-url", Input).value
            keywords_raw = self.query_one("#library-media-edit-keywords", Input).value
        except (NoMatches, QueryError):
            self._library_media_editing = False
            self.refresh(recompose=True)
            return
        # Validate/sanitize each user-entered field at the UI boundary before
        # it reaches the persistence service.
        title = self._sanitize_media_field(title, max_length=1000)
        author = self._sanitize_media_field(author, max_length=1000)
        url = self._sanitize_media_field(url, max_length=2000)
        keywords = [
            cleaned
            for item in keywords_raw.split(",")
            if (cleaned := self._sanitize_media_field(item, max_length=200).strip())
        ]
        self.run_worker(
            self._save_library_media_edit(
                media_id,
                title=title,
                author=author,
                url=url,
                keywords=keywords,
            )
        )

    async def _save_library_media_edit(
        self,
        media_id: str,
        *,
        title: str,
        author: str,
        url: str,
        keywords: list[str],
    ) -> None:
        """Persist metadata edits, then re-fetch detail and exit edit mode.

        Guards against a missing ``update_media_item`` service or a failed
        write by logging the failure and surfacing a quiet notice, but
        always re-fetches the current detail afterwards so the viewer never
        shows a stale/half-applied edit.

        Only the local backend's supported metadata fields (title, author,
        url, keywords) are sent -- notably ``version`` is NOT included.
        ``Client_Media_DB_v2.update_media_metadata`` performs its own
        optimistic-version check internally from the row it reads and takes
        no caller-supplied ``version`` argument, so omitting it here loses
        no locking guarantees while avoiding the local backend's metadata
        field allowlist rejecting the write outright.

        Args:
            media_id: The Library media item id being edited.
            title: New title field value.
            author: New author field value.
            url: New URL field value.
            keywords: New keywords, already split from the comma-separated
                edit input.
        """
        service = getattr(self.app_instance, "media_reading_scope_service", None)
        update_media_item = getattr(service, "update_media_item", None)
        if callable(update_media_item):
            try:
                await self._run_library_service_call(
                    update_media_item,
                    mode="local",
                    media_id=media_id,
                    title=title,
                    author=author,
                    url=url,
                    keywords=keywords,
                    isolate_in_worker=True,
                )
                # Keep the media list's local snapshot in step with the write
                # so navigating back shows the new title/author/url/keywords
                # immediately, not the pre-edit values until a full refetch.
                self._patch_local_media_record(
                    media_id, title=title, author=author, url=url, keywords=keywords
                )
            except Exception:
                logger.warning(
                    f"Failed to save Library media edit for {media_id!r}.", exc_info=True
                )
                self._notify_library_media_edit_warning(
                    "Could not save media changes; showing the latest saved version."
                )
        else:
            self._notify_library_media_edit_warning("Media editing is unavailable.")
        self._library_media_editing = False
        await self._refresh_library_media_detail(media_id)

    def _patch_local_media_record(
        self,
        media_id: str,
        *,
        title: str,
        author: str,
        url: str,
        keywords: list[str],
    ) -> None:
        """Update the cached media snapshot record after a saved metadata edit.

        The viewer re-fetches its detail from the backend, but the media
        *list* is rendered from ``_local_source_records['media']`` (seeded
        once at load), so without this the list keeps showing the pre-edit
        fields until a full snapshot refresh. Only the record whose id
        matches ``media_id`` is replaced; all others are passed through
        unchanged.

        Args:
            media_id: The edited media item's id.
            title: Saved title value.
            author: Saved author value.
            url: Saved URL value.
            keywords: Saved keywords list.
        """
        self._local_source_records["media"] = tuple(
            dict(record, title=title, author=author, url=url, keywords=list(keywords))
            if self._source_record_id(record) == media_id
            else record
            for record in self._local_source_records.get("media", ())
        )

    def _notify_library_media_edit_warning(self, message: str) -> None:
        """Surface a quiet warning notice for a failed media-edit save.

        Args:
            message: Human-readable warning text to notify with.
        """
        notify = getattr(self.app_instance, "notify", None)
        if callable(notify):
            notify(message, severity="warning")

    @on(Button.Pressed, "#library-media-delete")
    def handle_library_media_delete(self, event: Button.Pressed) -> None:
        """Enter the inline delete-confirmation state for the open media viewer.

        Deleting is destructive (it trashes the item), so the first press
        only swaps the normal action row for a "Delete" / "Cancel" confirm
        affordance; the actual service call only happens once the confirm
        button (``#library-media-delete-confirm``) is pressed.

        Args:
            event: Button press event emitted by the viewer's "Delete" action.
        """
        event.stop()
        self._library_media_confirming_delete = True
        self.refresh(recompose=True)

    @on(Button.Pressed, "#library-media-delete-cancel")
    def handle_library_media_delete_cancel(self, event: Button.Pressed) -> None:
        """Discard the pending delete confirmation and restore the normal action row.

        Args:
            event: Button press event emitted by the confirm affordance's
                "Cancel" action.
        """
        event.stop()
        self._library_media_confirming_delete = False
        self.refresh(recompose=True)

    @on(Button.Pressed, "#library-media-delete-confirm")
    def handle_library_media_delete_confirm(self, event: Button.Pressed) -> None:
        """Hand the confirmed delete off to a worker that trashes the item.

        Reads the currently selected media id synchronously (before any
        recompose can clear it) and defers the actual service call and list
        mutation to ``_delete_library_media_item`` -- mirroring how
        ``handle_library_media_edit_save`` defers to
        ``_save_library_media_edit``.

        Args:
            event: Button press event emitted by the confirm affordance's
                "Delete" action.
        """
        event.stop()
        media_id = self._selected_media_id
        if not media_id:
            self._library_media_confirming_delete = False
            self.refresh(recompose=True)
            return
        self.run_worker(self._delete_library_media_item(media_id))

    async def _delete_library_media_item(self, media_id: str) -> None:
        """Trash the selected Library media item, then return to the list view.

        Guards against a missing ``delete_media_item`` service or a failed
        write by logging the failure and surfacing a quiet notice; either
        way the pending confirmation is dismissed afterwards so a failed
        delete never strands the viewer in the confirm state.

        On success, the deleted item is dropped from the cached
        ``_local_source_records["media"]`` snapshot (matched via
        ``_source_record_id``, the same id-key precedence ``_study_source_items``
        uses) so the list view reflects the trash immediately, without
        waiting on a full snapshot re-fetch, and the canvas returns to its
        list view.

        Args:
            media_id: The Library media item id to delete.
        """
        service = getattr(self.app_instance, "media_reading_scope_service", None)
        delete_media_item = getattr(service, "delete_media_item", None)
        deleted = False
        if callable(delete_media_item):
            try:
                await self._run_library_service_call(
                    delete_media_item,
                    mode="local",
                    media_id=media_id,
                    isolate_in_worker=True,
                )
                deleted = True
            except Exception:
                logger.warning(
                    f"Failed to delete Library media item {media_id!r}.", exc_info=True
                )
                self._notify_library_media_delete_warning(
                    "Could not delete this media item."
                )
        else:
            self._notify_library_media_delete_warning("Media deletion is unavailable.")

        self._library_media_confirming_delete = False
        if deleted:
            self._local_source_records["media"] = tuple(
                record
                for record in self._local_source_records.get("media", ())
                if self._source_record_id(record) != media_id
            )
            self._library_media_view = "list"
            self._library_media_detail = None
            self._library_media_highlights = []
            self._library_media_editing_analysis = False
            self._library_media_content_query = ""
            self._library_media_content_match_index = 0
            self._selected_media_id = ""
        if self.is_mounted:
            self.refresh(recompose=True)

    def _notify_library_media_delete_warning(self, message: str) -> None:
        """Surface a quiet warning notice for a failed media-delete attempt.

        Args:
            message: Human-readable warning text to notify with.
        """
        notify = getattr(self.app_instance, "notify", None)
        if callable(notify):
            notify(message, severity="warning")

    @on(Button.Pressed, "#library-media-highlight-add")
    def handle_library_media_highlight_add(self, event: Button.Pressed) -> None:
        """Read the add-highlight form and hand the write off to a worker.

        Reads the three highlight inputs directly (before any recompose
        removes them) and silently ignores the press when the quote is
        blank -- mirroring how ``handle_library_media_edit_save`` reads its
        form inputs synchronously before deferring to a worker.

        Args:
            event: Button press event emitted by the highlights form's "Add
                highlight" action.
        """
        event.stop()
        media_id = self._selected_media_id
        if not media_id:
            return
        try:
            quote = self.query_one("#library-media-highlight-quote", Input).value
            note = self.query_one("#library-media-highlight-note", Input).value
            color = self.query_one("#library-media-highlight-color", Input).value
        except (NoMatches, QueryError):
            return
        # Validate/sanitize each user-entered field at the UI boundary before
        # it reaches the persistence service.
        quote = self._sanitize_media_field(quote, max_length=10000).strip()
        if not quote:
            return
        note = self._sanitize_media_field(note, max_length=10000).strip() or None
        color = self._sanitize_media_field(color, max_length=100).strip() or None
        # Exclusive in its own group so a rapid double-press cancels the first
        # add instead of inserting the same highlight twice (the add write is
        # not idempotent, unlike delete).
        self.run_worker(
            self._add_library_media_highlight(
                media_id,
                quote=quote,
                note=note,
                color=color,
            ),
            exclusive=True,
            group="library_media_highlight_add",
        )

    async def _add_library_media_highlight(
        self,
        media_id: str,
        *,
        quote: str,
        note: str | None,
        color: str | None,
    ) -> None:
        """Create a new reading highlight, then re-fetch the highlights list.

        Guards against a missing ``create_highlight`` service or a failed
        write by logging the failure and surfacing a quiet notice, but
        always re-fetches highlights afterwards so the section never shows a
        stale list.

        Args:
            media_id: The Library media item id to attach the highlight to.
            quote: The highlighted quote text.
            note: Optional note text, or None.
            color: Optional highlight color, or None.
        """
        service = getattr(self.app_instance, "media_reading_scope_service", None)
        create_highlight = getattr(service, "create_highlight", None)
        if callable(create_highlight):
            try:
                await self._run_library_service_call(
                    create_highlight,
                    mode="local",
                    item_id=media_id,
                    quote=quote,
                    note=note,
                    color=color,
                    isolate_in_worker=True,
                )
            except Exception:
                logger.warning(
                    f"Failed to add Library media highlight for {media_id!r}.", exc_info=True
                )
                self._notify_library_media_highlight_warning("Could not add this highlight.")
        else:
            self._notify_library_media_highlight_warning("Highlights are unavailable.")
        await self._reload_library_media_highlights(media_id)

    @on(Button.Pressed, ".library-media-highlight-delete")
    def handle_library_media_highlight_delete(self, event: Button.Pressed) -> None:
        """Read the pressed row's highlight id and hand the delete off to a worker.

        Args:
            event: Button press event emitted by a highlight row's delete
                action.
        """
        event.stop()
        media_id = self._selected_media_id
        highlight_id = getattr(event.button, "highlight_id", "")
        if not media_id or not highlight_id:
            return
        self.run_worker(self._delete_library_media_highlight(media_id, highlight_id))

    async def _delete_library_media_highlight(self, media_id: str, highlight_id: Any) -> None:
        """Delete a reading highlight, then re-fetch the highlights list.

        Guards against a missing ``delete_highlight`` service or a failed
        write by logging the failure and surfacing a quiet notice, but
        always re-fetches highlights afterwards so the section never shows a
        stale list.

        Args:
            media_id: The Library media item id the highlight belongs to.
            highlight_id: The highlight id to delete.
        """
        service = getattr(self.app_instance, "media_reading_scope_service", None)
        delete_highlight = getattr(service, "delete_highlight", None)
        if callable(delete_highlight):
            try:
                await self._run_library_service_call(
                    delete_highlight,
                    mode="local",
                    highlight_id=highlight_id,
                    isolate_in_worker=True,
                )
            except Exception:
                logger.warning(
                    f"Failed to delete Library media highlight {highlight_id!r}.", exc_info=True
                )
                self._notify_library_media_highlight_warning("Could not delete this highlight.")
        else:
            self._notify_library_media_highlight_warning("Highlights are unavailable.")
        await self._reload_library_media_highlights(media_id)

    def _notify_library_media_highlight_warning(self, message: str) -> None:
        """Surface a quiet warning notice for a failed highlight mutation.

        Args:
            message: Human-readable warning text to notify with.
        """
        notify = getattr(self.app_instance, "notify", None)
        if callable(notify):
            notify(message, severity="warning")

    @on(Input.Submitted, "#library-media-content-search")
    def handle_library_media_content_search_submitted(self, event: Input.Submitted) -> None:
        """Set the in-content search query and jump to the first match.

        Submitted (rather than Changed) is used deliberately: recomposing
        the whole screen on every keystroke would remount
        ``#library-media-content-search`` and drop focus/cursor position
        mid-typing, so the query only takes effect on Enter -- mirroring
        ``handle_library_search_submitted``'s rail search box. A no-op
        guard (mirroring ``update_library_rag_query``) skips the recompose
        entirely when the submitted text matches the current query.

        Args:
            event: Input submit event emitted by the content search box.
        """
        event.stop()
        # Strip once at the source so the status count, the body highlighting,
        # and prev/next navigation all search the exact same needle.
        submitted = event.value.strip()
        if submitted == self._library_media_content_query:
            return
        self._library_media_content_query = submitted
        self._library_media_content_match_index = 0
        self.refresh(recompose=True)
        self.call_after_refresh(self._focus_library_media_content_search_input)
        # Bring the first match into view on submit; otherwise the status line
        # claims "Match 1 of N" while the hit stays below the fold until the
        # user cycles Next all the way around.
        detail = self._library_media_detail if isinstance(self._library_media_detail, Mapping) else None
        content = build_library_media_viewer_state(detail).content if detail else ""
        matches = find_content_matches(content, self._library_media_content_query)
        if matches:
            self.call_after_refresh(self._scroll_library_media_content_to_line, matches[0])

    def _focus_library_media_content_search_input(self) -> None:
        """Re-focus the content search box after a submit-triggered recompose.

        Mirrors ``_focus_library_search_input``: the Submitted-driven
        recompose above remounts a brand-new
        ``#library-media-content-search``, so without this, focus falls
        back to the screen after every search.
        """
        try:
            self.query_one("#library-media-content-search", Input).focus()
        except (NoMatches, QueryError):
            pass

    @on(Button.Pressed, "#library-media-content-search-next")
    def handle_library_media_content_search_next(self, event: Button.Pressed) -> None:
        """Advance to the next in-content search match and scroll it into view.

        Args:
            event: Button press event emitted by the "Next" search action.
        """
        event.stop()
        self._advance_library_media_content_match(1)

    @on(Button.Pressed, "#library-media-content-search-prev")
    def handle_library_media_content_search_prev(self, event: Button.Pressed) -> None:
        """Return to the previous in-content search match and scroll it into view.

        Args:
            event: Button press event emitted by the "Prev" search action.
        """
        event.stop()
        self._advance_library_media_content_match(-1)

    def _advance_library_media_content_match(self, step: int) -> None:
        """Move the current content-search match index and scroll to it.

        No-ops when there is no open item or the query has no matches
        (the status line already reads "No matches" in that case).

        Args:
            step: ``1`` to move to the next match, ``-1`` for the previous
                one; wraps around the match count either direction.
        """
        detail = self._library_media_detail if isinstance(self._library_media_detail, Mapping) else None
        content = build_library_media_viewer_state(detail).content if detail else ""
        matches = find_content_matches(content, self._library_media_content_query)
        if not matches:
            return
        self._library_media_content_match_index = (
            self._library_media_content_match_index + step
        ) % len(matches)
        line_index = matches[self._library_media_content_match_index]
        self.refresh(recompose=True)
        self.call_after_refresh(self._scroll_library_media_content_to_line, line_index)

    def _scroll_library_media_content_to_line(self, line_index: int) -> None:
        """Scroll the content region so the given line index is visible.

        Uses the content ``VerticalScroll``'s ``scroll_to`` on the Y axis
        as an approximation of "the matched line" -- the content renders
        as a single ``Static``, so this is not pixel-perfect line
        targeting, but it reliably brings the matched line into (or near)
        view, which is the required bar for this feature.

        Args:
            line_index: 0-based line index within the content text to
                reveal.
        """
        try:
            content_scroll = self.query_one("#library-media-viewer-content", VerticalScroll)
        except (NoMatches, QueryError):
            return
        content_scroll.scroll_to(y=line_index, animate=False)

    @on(Button.Pressed, "#library-media-read-later")
    def handle_library_media_read_later(self, event: Button.Pressed) -> None:
        """Toggle the open media item's read-it-later state via a worker.

        Reads the currently known saved state from ``_library_media_detail``
        (already reflecting ``is_read_it_later`` from the last fetch) to
        decide whether to save or remove, mirroring how
        ``handle_library_media_delete_confirm`` reads state synchronously
        before deferring to a worker.

        Args:
            event: Button press event emitted by the viewer's "Read it
                later" / "Remove from read-it-later" action.
        """
        event.stop()
        media_id = self._selected_media_id
        if not media_id:
            return
        detail = (
            self._library_media_detail
            if isinstance(self._library_media_detail, Mapping)
            else {}
        )
        currently_saved = bool(detail.get("is_read_it_later"))
        self.run_worker(
            self._toggle_library_media_read_later(media_id, currently_saved=currently_saved)
        )

    async def _toggle_library_media_read_later(
        self, media_id: str, *, currently_saved: bool
    ) -> None:
        """Save or remove the read-it-later state, then re-fetch detail.

        Guards against a missing ``save_to_read_it_later``/
        ``remove_from_read_it_later`` service or a failed write by logging
        the failure and surfacing a quiet notice, but always re-fetches
        detail afterwards so the button's label never shows a stale state.

        Args:
            media_id: The Library media item id to toggle.
            currently_saved: Whether the item is currently saved for
                read-it-later (determines whether to save or remove).
        """
        service = getattr(self.app_instance, "media_reading_scope_service", None)
        method_name = "remove_from_read_it_later" if currently_saved else "save_to_read_it_later"
        method = getattr(service, method_name, None)
        if callable(method):
            try:
                await self._run_library_service_call(
                    method,
                    mode="local",
                    media_id=media_id,
                    isolate_in_worker=True,
                )
            except Exception:
                logger.warning(
                    f"Failed to toggle Library media read-it-later state for {media_id!r}.",
                    exc_info=True,
                )
                self._notify_library_media_read_later_warning(
                    "Could not update read-it-later status."
                )
        else:
            self._notify_library_media_read_later_warning("Read-it-later is unavailable.")
        await self._refresh_library_media_detail(media_id)

    def _notify_library_media_read_later_warning(self, message: str) -> None:
        """Surface a quiet warning notice for a failed read-it-later toggle.

        Args:
            message: Human-readable warning text to notify with.
        """
        notify = getattr(self.app_instance, "notify", None)
        if callable(notify):
            notify(message, severity="warning")

    @on(Button.Pressed, "#library-media-analysis-edit")
    def handle_library_media_analysis_edit(self, event: Button.Pressed) -> None:
        """Enter analysis edit mode for the open Library media viewer.

        Args:
            event: Button press event emitted by the analysis section's
                "Edit analysis" action.
        """
        event.stop()
        self._library_media_editing_analysis = True
        self.refresh(recompose=True)

    @on(Button.Pressed, "#library-media-analysis-cancel")
    def handle_library_media_analysis_cancel(self, event: Button.Pressed) -> None:
        """Discard in-progress analysis edits and return to the read-only view.

        Args:
            event: Button press event emitted by the analysis edit form's
                "Cancel" action.
        """
        event.stop()
        self._library_media_editing_analysis = False
        self.refresh(recompose=True)

    @on(Button.Pressed, "#library-media-analysis-save")
    def handle_library_media_analysis_save(self, event: Button.Pressed) -> None:
        """Read the analysis edit TextArea and hand the write off to a worker.

        Reads the edited analysis text directly (before any recompose
        removes the TextArea) and the current document content from the
        loaded detail -- ``save_analysis_version`` requires both, since it
        creates a new ``DocumentVersions`` row carrying the (unchanged)
        content alongside the edited analysis. Mirrors how
        ``handle_library_media_edit_save`` reads its form inputs
        synchronously before deferring to a worker.

        Args:
            event: Button press event emitted by the analysis edit form's
                "Save" action.
        """
        event.stop()
        media_id = self._selected_media_id
        if not media_id:
            self._library_media_editing_analysis = False
            self.refresh(recompose=True)
            return
        try:
            analysis_content = self.query_one(
                "#library-media-analysis-edit-text", TextArea
            ).text
        except (NoMatches, QueryError):
            self._library_media_editing_analysis = False
            self.refresh(recompose=True)
            return
        # Validate/sanitize the user-entered analysis at the UI boundary
        # before it reaches the persistence service.
        analysis_content = self._sanitize_media_field(
            analysis_content, max_length=100000
        )
        detail = (
            self._library_media_detail
            if isinstance(self._library_media_detail, Mapping)
            else {}
        )
        content = str(detail.get("content") or "")
        self.run_worker(
            self._save_library_media_analysis(
                media_id,
                content=content,
                analysis_content=analysis_content,
            )
        )

    async def _save_library_media_analysis(
        self, media_id: str, *, content: str, analysis_content: str
    ) -> None:
        """Persist an analysis edit as a new document version, then re-fetch detail.

        Guards against a missing ``save_analysis_version`` service or a
        failed write by logging the failure and surfacing a quiet notice,
        but always re-fetches detail afterwards so the viewer never shows a
        stale/half-applied edit. Analysis (re)generation via an LLM is
        explicitly out of scope -- this only persists caller-supplied text.

        Args:
            media_id: The Library media item id being edited.
            content: The current document content, sent unchanged alongside
                the edited analysis (``save_analysis_version`` requires it).
            analysis_content: The edited analysis text to persist.
        """
        service = getattr(self.app_instance, "media_reading_scope_service", None)
        save_analysis_version = getattr(service, "save_analysis_version", None)
        if callable(save_analysis_version):
            try:
                await self._run_library_service_call(
                    save_analysis_version,
                    mode="local",
                    media_id=media_id,
                    content=content,
                    analysis_content=analysis_content,
                    isolate_in_worker=True,
                )
            except Exception:
                logger.warning(
                    f"Failed to save Library media analysis for {media_id!r}.", exc_info=True
                )
                self._notify_library_media_analysis_warning(
                    "Could not save analysis changes; showing the latest saved version."
                )
        else:
            self._notify_library_media_analysis_warning("Analysis editing is unavailable.")
        self._library_media_editing_analysis = False
        await self._refresh_library_media_detail(media_id)

    def _notify_library_media_analysis_warning(self, message: str) -> None:
        """Surface a quiet warning notice for a failed analysis-edit save.

        Args:
            message: Human-readable warning text to notify with.
        """
        notify = getattr(self.app_instance, "notify", None)
        if callable(notify):
            notify(message, severity="warning")

    @on(Button.Pressed, "#library-media-open")
    def handle_library_media_open(self, event: Button.Pressed) -> None:
        """Hand off the selected media item to the Media screen.

        Args:
            event: Button press event emitted by the "Open in Media" action.
        """
        event.stop()
        self.post_message(NavigateToScreen("media"))

    @on(Input.Submitted, "#library-search-input")
    def handle_library_search_submitted(self, event: Input.Submitted) -> None:
        """Filter the conversations canvas from the rail search box.

        Args:
            event: Input submit event emitted by the rail's search box.
        """
        event.stop()
        self._library_conversation_query = self._safe_text(event.value, max_length=200)
        self._library_selected_row_id = LIBRARY_ROW_BROWSE_CONVERSATIONS
        self._active_mode = "conversations"
        self.refresh(recompose=True)
        self.call_after_refresh(self._focus_library_search_input)

    def _focus_library_search_input(self) -> None:
        """Re-focus the search box after a submit-triggered recompose.

        ``handle_library_search_submitted`` rebuilds the whole screen, which
        remounts a brand-new ``#library-search-input``; without this, focus
        silently falls back to the screen after every search.
        """
        try:
            self.query_one("#library-search-input", Input).focus()
        except (NoMatches, QueryError):
            pass

    def _library_rail_search_placeholder(self) -> str:
        """Placeholder for the rail search box, reflecting the active browse.

        The rail search only filters conversations (see
        ``handle_library_search_submitted``), so it is precise when the
        Conversations browse is active and a generic "Search Library…"
        otherwise -- it should not claim "conversations" while the user is
        browsing Media or another source. Making it actually search the
        active source is a tracked follow-up.

        Returns:
            The context-appropriate search placeholder text.
        """
        if self._library_selected_row_id == LIBRARY_ROW_BROWSE_CONVERSATIONS:
            return "Search conversations…"
        return "Search Library…"

    @on(Button.Pressed, ".library-mode-chip")
    async def switch_library_mode(self, event: Button.Pressed) -> None:
        mode_id = LIBRARY_MODE_BY_BUTTON_ID.get(event.button.id or "")
        if mode_id is None:
            return
        event.stop()
        await self._set_active_mode(mode_id)

    async def _set_active_mode(self, mode_id: str) -> None:
        if mode_id == self._active_mode:
            return
        self._active_mode = mode_id
        self._invalidate_library_workspace_depth_state()
        await self._refresh_active_mode_widgets()

    def _legacy_workbench_present(self) -> bool:
        """Whether the retired 3-pane workbench chrome is still mounted.

        The Library shell replaces the mode strip + contract grid with a rail +
        canvas. The legacy granular sync helpers below target ids that no longer
        render, so they early-return through this guard and navigation flows
        instead go through a full ``refresh(recompose=True)``.
        """
        return bool(self.query("#library-source-detail"))

    async def _refresh_active_mode_widgets(self) -> None:
        if not self._legacy_workbench_present():
            self.refresh(recompose=True)
            return
        active_mode = self._active_mode_contract()
        source_column_title, detail_column_title, inspector_column_title = self._active_column_titles()
        self.query_one("#library-status-row", Static).update(self._status_row_copy())
        self.query_one("#library-source-browser-title", Static).update(source_column_title)
        self.query_one("#library-source-detail-title", Static).update(detail_column_title)
        self.query_one("#library-source-inspector-title", Static).update(inspector_column_title)
        active_mode_copy_visible = self._active_mode not in {
            "collections",
            "search",
            "sources",
            "workspaces",
        }
        self.query_one("#library-active-mode-title", Static).update(
            f"{active_mode['label']} mode" if active_mode_copy_visible else ""
        )
        self.query_one("#library-active-mode-description", Static).update(
            active_mode["description"] if active_mode_copy_visible else ""
        )
        self.query_one("#library-active-mode-next-action", Static).update(
            active_mode["next_action"] if active_mode_copy_visible else ""
        )
        self.query_one("#library-active-mode-title", Static).display = active_mode_copy_visible
        self.query_one("#library-active-mode-description", Static).display = active_mode_copy_visible
        self.query_one("#library-active-mode-next-action", Static).display = active_mode_copy_visible
        local_snapshot_regions = list(self.query("#library-local-snapshot-region"))
        if local_snapshot_regions:
            local_snapshot_regions[0].display = self._should_show_local_snapshot_region()
        for mode_id, mode in LIBRARY_MODES.items():
            if not mode.get("show_in_strip", True):
                continue
            self.query_one(f"#{mode['button_id']}", Button).set_class(
                mode_id == self._active_mode,
                "is-active",
            )
        active_source_action_id = self._active_source_action_id()
        for button_id in (
            "library-open-notes",
            "library-open-media",
            "library-open-conversations",
            "library-open-import-export",
            "library-open-search",
            "library-open-collections",
        ):
            buttons = list(self.query(f"#{button_id}"))
            if buttons:
                buttons[0].set_class(button_id == active_source_action_id, "is-active")
        await self._sync_source_module_actions()
        workspace_depth_state = self._library_workspace_depth_state(refresh=True)
        self.query_one("#library-workspace-scope", Static).update(
            self._library_workspace_scope_label(workspace_depth_state)
        )
        await self._sync_local_snapshot_region(workspace_depth_state)
        await self._sync_study_handoff_detail()
        await self._sync_search_rag_panel(workspace_depth_state=workspace_depth_state)
        await self._sync_collections_panel(refresh_snapshot=True)
        await self._sync_workspaces_panel(workspace_depth_state)
        await self._sync_action_region(workspace_depth_state)

    async def _sync_source_module_actions(self) -> None:
        """Rebuild source-map actions so active-mode owned action IDs stay unique."""
        if not self.query("#library-source-browser"):
            return
        browser = self.query_one("#library-source-browser", Vertical)
        source_title = self.query_one("#library-source-browser-title", Static)
        quick_actions_title = self.query_one("#library-quick-actions-title", Static)
        children = list(browser.children)
        action_widgets: list[Any] = []
        collect = False
        for child in children:
            if child is source_title:
                collect = True
                continue
            if child is quick_actions_title:
                break
            if collect:
                action_widgets.append(child)
        for widget in action_widgets:
            await widget.remove()
        await browser.mount(*self._source_module_action_widgets(), before=quick_actions_title)

    async def _sync_search_rag_panel(
        self,
        *,
        workspace_depth_state: LibraryWorkspaceDepthState | None = None,
    ) -> None:
        if not self._legacy_workbench_present():
            return
        mounted_widgets = list(self.query("#library-search-rag-panel"))
        for widget in mounted_widgets:
            await widget.remove()
        if self._active_mode != "search":
            await self._sync_inspector_mode_region(
                None,
                workspace_depth_state=workspace_depth_state,
            )
            return
        panel_state = self._library_rag_panel_state()
        detail = self.query_one("#library-source-detail", Vertical)
        await detail.mount(
            LibrarySearchRagPanel(panel_state, id="library-search-rag-panel"),
            after="#library-source-detail-title",
        )
        await self._sync_inspector_mode_region(panel_state)

    async def _sync_study_handoff_detail(self) -> None:
        if not self._legacy_workbench_present():
            return
        mounted_widgets = list(self.query("#library-study-handoff-detail"))
        for widget in mounted_widgets:
            await widget.remove()
        if self._active_mode not in LIBRARY_STUDY_HANDOFF_MODES:
            return
        detail = self.query_one("#library-source-detail", Vertical)
        await detail.mount(
            self._study_handoff_detail_widget(),
            after="#library-active-mode-next-action",
        )

    async def _sync_inspector_mode_region(
        self,
        panel_state: LibraryRagPanelState | None,
        *,
        workspace_depth_state: LibraryWorkspaceDepthState | None = None,
    ) -> None:
        regions = list(self.query("#library-inspector-mode-region"))
        if not regions:
            return
        region = regions[0]
        await region.remove_children()
        if panel_state is not None:
            await region.mount(
                LibrarySearchRagInspectorPanel(
                    panel_state,
                    id="library-rag-inspector",
                    classes="library-rag-region",
                )
            )
            return
        if self._active_mode == "collections":
            for row in self._collections_inspector_rows(self._library_collections_panel_state()):
                await region.mount(row)
            return
        if self._active_mode == "workspaces":
            state = workspace_depth_state or self._library_workspace_depth_state()
            for row in self._workspaces_inspector_rows(state):
                await region.mount(row)
            return
        if self._active_mode == "conversations":
            state = workspace_depth_state or self._library_workspace_depth_state()
            selected = self._selected_conversation_record()
            await region.mount(
                Static("Conversation inspector", id="library-inspector-title", classes="destination-section")
            )
            if selected is None:
                await region.mount(
                    Static(
                        "No conversation selected.",
                        id="library-conversation-inspector-empty",
                    )
                )
                await region.mount(
                    Static(
                        "Select a saved conversation to inspect metadata and handoff eligibility.",
                        id="library-conversation-inspector-empty-next-action",
                    )
                )
                return
            _, record = selected
            for row in (
                Static(
                    self._source_title("conversations", record),
                    id="library-conversation-inspector-title",
                ),
                Static(
                    self._conversation_message_count_label(record),
                    id="library-conversation-inspector-message-count",
                ),
                Static(
                    "Source authority: local",
                    id="library-conversation-inspector-authority",
                ),
                Static(
                    self._conversation_handoff_label(state),
                    id="library-conversation-inspector-handoff",
                ),
                Static(
                    "Owner: Console/Conversations retains editing and saved-history management.",
                    id="library-conversation-inspector-owner",
                ),
            ):
                await region.mount(row)
            return
        if self._active_mode == "import-export":
            for row in self._import_export_inspector_rows():
                await region.mount(row)
            return
        state = workspace_depth_state or self._library_workspace_depth_state()
        for row in self._hub_inspector_rows(state):
            await region.mount(row)

    async def _sync_local_snapshot_region(
        self,
        workspace_depth_state: LibraryWorkspaceDepthState,
    ) -> None:
        regions = list(self.query("#library-local-snapshot-region"))
        if not regions:
            return
        region = regions[0]
        # Textual removes children asynchronously; wait before remounting reused IDs.
        await region.remove_children()
        if not self._should_show_local_snapshot_region():
            return
        if not self._library_loaded:
            await region.mount(
                Static(
                    "Loading local Library sources...",
                    id="library-source-loading",
                )
            )
            return
        if self._library_lookup_error:
            recovery_state = self._library_lookup_recovery_state
            await region.mount(
                Static(
                    self._library_lookup_error,
                    id=(
                        recovery_state.stable_selector
                        if recovery_state is not None
                        else "library-source-error"
                    ),
                )
            )
            return
        if self._active_mode == "conversations":
            for row in self._conversation_browser_rows(workspace_depth_state):
                await region.mount(row)
            return
        if self._active_mode == "import-export":
            for row in self._import_export_workflow_rows():
                await region.mount(row)
            return
        if not self._has_local_sources():
            await region.mount(Static(LIBRARY_EMPTY_COPY, id="library-source-empty"))
            await region.mount(
                Static(
                    LIBRARY_EMPTY_NEXT_ACTION_COPY,
                    id="library-source-empty-next-action",
                )
            )

    async def _sync_collections_panel(self, *, refresh_snapshot: bool = False) -> None:
        if not self._legacy_workbench_present():
            if self._active_mode != "collections":
                self._library_collection_pending_delete_id = ""
                return
            if refresh_snapshot:
                await self._refresh_library_collections_snapshot()
            self.refresh(recompose=True)
            return
        for widget in list(self.query("#library-collections-panel")):
            await widget.remove()
        if self._active_mode != "collections":
            self._library_collection_pending_delete_id = ""
            return
        if refresh_snapshot:
            await self._refresh_library_collections_snapshot()
        status_rows = list(self.query("#library-status-row"))
        if status_rows:
            status_rows[0].update(self._status_row_copy())
        panel_state = self._library_collections_panel_state()
        detail = self.query_one("#library-source-detail", Vertical)
        await detail.mount(
            LibraryCollectionsPanel(
                panel_state,
                name_value=self._library_collection_name_input,
                description_value=self._library_collection_description_input,
                delete_pending=bool(self._library_collection_pending_delete_id),
                id="library-collections-panel",
            ),
            after="#library-active-mode-next-action",
        )
        await self._sync_inspector_mode_region(None)

    async def _sync_workspaces_panel(
        self,
        workspace_depth_state: LibraryWorkspaceDepthState | None = None,
    ) -> None:
        if not self._legacy_workbench_present():
            return
        for widget in list(self.query("#library-workspaces-depth-panel")):
            await widget.remove()
        if self._active_mode != "workspaces":
            return
        state = workspace_depth_state or self._library_workspace_depth_state()
        detail = self.query_one("#library-source-detail", Vertical)
        panel = Vertical(id="library-workspaces-depth-panel")
        await detail.mount(panel, after="#library-source-detail-title")
        for row in self._workspaces_detail_rows(state):
            await panel.mount(row)
        await self._sync_inspector_mode_region(None)

    async def _sync_action_region(
        self,
        workspace_depth_state: LibraryWorkspaceDepthState | None = None,
    ) -> None:
        regions = list(self.query("#library-action-region"))
        if not regions:
            return
        region = regions[0]
        await region.remove_children()
        workspace_depth_state = workspace_depth_state or self._library_workspace_depth_state()
        handoff_disabled, handoff_tooltip = self._workspace_handoff_action_state(
            workspace_depth_state
        )
        for widget in self._library_action_widgets(
            workspace_depth_state=workspace_depth_state,
            collection_scoped_actions_deferred=self._active_mode == "collections",
            handoff_disabled=handoff_disabled,
            handoff_tooltip=handoff_tooltip,
            collections_panel_state=(
                self._library_collections_panel_state()
                if self._active_mode == "collections"
                else None
            ),
        ):
            await region.mount(widget)

    async def _refresh_collections_panel_action_state_widgets(self) -> None:
        if self._active_mode != "collections" or not list(self.query("#library-collections-panel")):
            return

        panel_state = self._library_collections_panel_state()
        for action in (
            panel_state.create_action,
            panel_state.rename_action,
            panel_state.delete_action,
        ):
            buttons = list(self.query(f"#{action.widget_id}"))
            if buttons:
                buttons[0].disabled = not action.enabled
                buttons[0].tooltip = action.tooltip

        confirm_buttons = list(self.query("#library-confirm-delete-collection"))
        if not self._library_collection_pending_delete_id:
            for button in confirm_buttons:
                await button.remove()
            return
        if not confirm_buttons:
            actions = list(self.query("#library-collection-actions"))
            if actions:
                await actions[0].mount(
                    Button(
                        "Confirm delete",
                        id="library-confirm-delete-collection",
                        tooltip="Delete the selected local Collection.",
                    )
                )

    async def _refresh_library_collections_snapshot(self) -> None:
        service = getattr(self.app_instance, "library_collections_service", None)
        list_collections = getattr(service, "list_collections", None)
        if not callable(list_collections):
            self._library_collections_records = ()
            self._library_sync_profile_summary = None
            self._library_collections_loaded = True
            self._library_collections_error = "Library Collections are unavailable in this runtime."
            return
        try:
            records = await self._run_library_service_call(list_collections)
        except Exception:
            logger.warning("Failed to load Library Collections.", exc_info=True)
            self._library_collections_records = ()
            self._library_sync_profile_summary = None
            self._library_collections_loaded = True
            self._library_collections_error = "Library Collections are unavailable."
            return
        self._library_collections_records = await self._decorate_library_collection_sync_records(
            tuple(records or ())
        )
        self._library_sync_profile_summary = await self._load_library_sync_profile_summary()
        self._library_collections_loaded = True
        self._library_collections_error = ""
        if (
            self._library_collections_selected_id
            and not any(
                _record_value(record, "collection_id") == self._library_collections_selected_id
                for record in self._library_collections_records
            )
        ):
            self._library_collections_selected_id = ""
        if not self._library_collections_selected_id and self._library_collections_records:
            self._library_collections_selected_id = (
                _record_value(self._library_collections_records[0], "collection_id") or ""
            )

    async def _decorate_library_collection_sync_records(
        self,
        records: Sequence[Any],
    ) -> tuple[Any, ...]:
        repository = getattr(self.app_instance, "sync_state_repository", None)
        if repository is None:
            return tuple(records)

        get_latest_mirror_report = getattr(repository, "get_latest_mirror_report", None)
        list_conflict_reports = getattr(repository, "list_conflict_reports", None)
        if not callable(get_latest_mirror_report) or not callable(list_conflict_reports):
            return tuple(records)

        sync_scope = _active_library_sync_scope(self.app_instance)
        try:
            latest_mirror_record, conflict_reports = await asyncio.gather(
                self._run_library_service_call(
                    get_latest_mirror_report,
                    source_authority=sync_scope["source_authority"],
                    server_profile_id=sync_scope["server_profile_id"],
                    authenticated_principal_id=sync_scope["authenticated_principal_id"],
                    workspace_scope=sync_scope["workspace_scope"],
                    domain="library_collections",
                    isolate_in_worker=True,
                ),
                self._run_library_service_call(
                    list_conflict_reports,
                    source_authority=sync_scope["source_authority"],
                    server_profile_id=sync_scope["server_profile_id"],
                    authenticated_principal_id=sync_scope["authenticated_principal_id"],
                    workspace_scope=sync_scope["workspace_scope"],
                    domain="library_collections",
                    limit=LIBRARY_COLLECTION_SYNC_CONFLICT_LIMIT,
                    isolate_in_worker=True,
                ),
            )
        except Exception:
            logger.warning("Failed to load Library Collections sync dry-run state.", exc_info=True)
            return tuple(records)

        latest_report = latest_mirror_record["report"] if latest_mirror_record else None
        readiness = build_sync_readiness_report(
            domain="library_collections",
            server_profile_id=sync_scope["server_profile_id"],
            workspace_id=sync_scope["workspace_scope"],
            registry=DEFAULT_SYNC_ELIGIBILITY_REGISTRY,
        )
        readiness_record = {
            "sync_eligible": readiness.sync_eligible,
            "write_enabled": readiness.write_enabled,
            "reason_codes": readiness.reason_codes,
            "details": dict(readiness.details),
        }

        decorated: list[dict[str, Any]] = []
        for record in records:
            collection_id = str(_record_value(record, "collection_id", ""))
            record_data = _library_collection_record_data(record)
            collection_report = _collection_scoped_mirror_report(latest_report, collection_id)
            collection_conflicts = _collection_scoped_conflicts(conflict_reports, collection_id)
            record_data["sync_mirror_report"] = collection_report or {}
            record_data["sync_readiness_report"] = readiness_record
            record_data["sync_conflicts"] = collection_conflicts
            explicit_status = str(_record_value(record, "sync_status", "local-only") or "").lower()
            if explicit_status in {"", "local-only"} or collection_report or collection_conflicts:
                promotion_state = build_sync_promotion_state(
                    domain="library_collections",
                    surface_label="Collections",
                    readiness=readiness,
                    latest_mirror_report=collection_report,
                    conflict_reports=collection_conflicts,
                    source_authority=sync_scope["source_authority"],
                    workspace_id=sync_scope["workspace_scope"],
                )
                record_data["sync_promotion_state"] = {
                    "authority_label": promotion_state.authority_label,
                    "sync_label": promotion_state.sync_label,
                    "review_label": promotion_state.review_label,
                    "conflict_label": promotion_state.conflict_label,
                    "rollback_label": promotion_state.rollback_label,
                    "mirror_label": promotion_state.mirror_label,
                    "primary_recovery": promotion_state.primary_recovery,
                    "mutation_allowed": promotion_state.mutation_allowed,
                }
            if collection_report:
                record_data["sync_status"] = ""
            elif (
                not readiness_record["sync_eligible"]
                and _record_value(record, "sync_status", "local-only") == "local-only"
            ):
                record_data["sync_status"] = ""
            decorated.append(record_data)
        return tuple(decorated)

    async def _load_library_sync_profile_summary(self) -> Mapping[str, Any] | None:
        sync_scope_service = getattr(self.app_instance, "sync_scope_service", None)
        get_summary = getattr(sync_scope_service, "get_sync_v2_profile_summary", None)
        if not callable(get_summary):
            return None

        runtime_policy = getattr(self.app_instance, "runtime_policy", None)
        runtime_state = getattr(runtime_policy, "state", None)
        if (
            not isinstance(runtime_state, RuntimeSourceState)
            or runtime_state.active_source != "server"
            or not runtime_state.server_configured
            or not runtime_state.active_server_id
        ):
            return None

        scope_provider = getattr(self.app_instance, "_server_notification_event_scope", None)
        scope = scope_provider() if callable(scope_provider) else {}
        scope_mapping = scope if isinstance(scope, Mapping) else {}
        raw_server_profile_id = scope_mapping.get("server_profile_id", runtime_state.active_server_id)
        server_profile_id = self._safe_sync_scope_text(raw_server_profile_id)
        if not server_profile_id:
            return None
        authenticated_principal_id = self._safe_sync_scope_text(
            scope_mapping.get("authenticated_principal_id")
        )
        if scope_mapping.get("authenticated_principal_id") is not None and authenticated_principal_id is None:
            return None
        workspace_scope = self._safe_sync_scope_text(scope_mapping.get("workspace_scope"))
        if scope_mapping.get("workspace_scope") is not None and workspace_scope is None:
            return None

        try:
            summary = await self._run_library_service_call(
                get_summary,
                server_profile_id=server_profile_id,
                authenticated_principal_id=authenticated_principal_id,
                workspace_scope=workspace_scope,
                isolate_in_worker=True,
            )
        except Exception:
            logger.warning("Failed to load Sync v2 profile summary.", exc_info=True)
            return None
        return summary if isinstance(summary, Mapping) else None

    @on(Input.Changed, "#library-rag-query-input")
    async def update_library_rag_query(self, event: Input.Changed) -> None:
        event.stop()
        if event.value == self._library_rag_query:
            return
        self._library_rag_query = event.value
        self._reset_library_rag_retrieval_state()
        await self._refresh_search_rag_panel_state_widgets()

    @on(Input.Submitted, "#library-rag-query-input")
    async def submit_library_rag_query(self, event: Input.Submitted) -> None:
        """Run Library Search/RAG from the query field for keyboard-only users."""
        event.stop()
        await self._start_library_rag_query()

    @on(Input.Changed, "#library-collection-name-input")
    async def update_library_collection_name_input(self, event: Input.Changed) -> None:
        event.stop()
        if event.value == self._library_collection_name_input:
            return
        self._library_collection_name_input = event.value
        await self._refresh_collections_panel_action_state_widgets()

    @on(Input.Changed, "#library-collection-description-input")
    async def update_library_collection_description_input(self, event: Input.Changed) -> None:
        event.stop()
        self._library_collection_description_input = event.value
        await self._refresh_collections_panel_action_state_widgets()

    def _reset_library_rag_retrieval_state(self) -> None:
        self._library_rag_results = ()
        self._library_rag_retrieval_status = ""
        self._library_rag_recovery_state = None
        self._library_rag_selected_result_id = ""

    @on(Button.Pressed, "#library-rag-run-query")
    async def run_library_rag_query(self, event: Button.Pressed) -> None:
        event.stop()
        await self._start_library_rag_query()

    @on(Button.Pressed, "#library-rag-open-import-export")
    async def open_import_export_from_library_rag(self, event: Button.Pressed) -> None:
        event.stop()
        # Drive the shell selection so the recomposed canvas resolves to the
        # Import/Export mode; flipping _active_mode alone reverts on recompose.
        await self._select_library_rail_row("ingest-import-export", "import-export")

    async def _start_library_rag_query(self) -> None:
        panel_state = self._library_rag_panel_state()
        run_action = panel_state.query_state.run_action
        if not run_action.enabled:
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify(run_action.disabled_reason, severity="warning")
            return

        request = LibraryRagSearchRequest(
            query=panel_state.query_state.query,
            source_types=panel_state.scope.selected_source_types,
            mode=panel_state.query_state.mode,
            top_k=panel_state.query_state.top_k,
            include_citations=panel_state.query_state.include_citations,
        )
        self._library_rag_results = ()
        self._library_rag_recovery_state = None
        self._library_rag_selected_result_id = ""
        self._library_rag_retrieval_status = "searching"
        await self._refresh_search_rag_panel_state_widgets()
        self._execute_library_rag_search(request)

    @on(Button.Pressed, "#library-create-collection")
    async def create_library_collection(self, event: Button.Pressed) -> None:
        event.stop()
        service = getattr(self.app_instance, "library_collections_service", None)
        create_collection = getattr(service, "create_collection", None)
        if not callable(create_collection):
            self._library_collections_error = "Library Collections are unavailable."
            await self._sync_collections_panel(refresh_snapshot=False)
            return
        try:
            created = await self._run_library_service_call(
                create_collection,
                self._library_collection_name_input,
                description=self._library_collection_description_input,
            )
        except LibraryCollectionsServiceError as exc:
            self._notify_library_collections_warning(str(exc))
            return
        self._library_collections_selected_id = getattr(created, "collection_id", "") or ""
        self._library_collection_name_input = ""
        self._library_collection_description_input = ""
        self._library_collection_pending_delete_id = ""
        await self._sync_collections_panel(refresh_snapshot=True)

    @on(Button.Pressed, "#library-rename-collection")
    async def rename_library_collection(self, event: Button.Pressed) -> None:
        event.stop()
        if not self._library_collections_selected_id:
            return
        service = getattr(self.app_instance, "library_collections_service", None)
        rename_collection = getattr(service, "rename_collection", None)
        if not callable(rename_collection):
            self._library_collections_error = "Library Collections are unavailable."
            await self._sync_collections_panel(refresh_snapshot=False)
            return
        try:
            renamed = await self._run_library_service_call(
                rename_collection,
                self._library_collections_selected_id,
                self._library_collection_name_input,
                description=self._library_collection_description_input,
            )
        except LibraryCollectionsServiceError as exc:
            self._notify_library_collections_warning(str(exc))
            return
        self._library_collections_selected_id = getattr(renamed, "collection_id", "") or ""
        self._library_collection_name_input = ""
        self._library_collection_description_input = ""
        self._library_collection_pending_delete_id = ""
        await self._sync_collections_panel(refresh_snapshot=True)

    @on(Button.Pressed, "#library-delete-collection")
    async def arm_library_collection_delete(self, event: Button.Pressed) -> None:
        event.stop()
        if not self._library_collections_selected_id:
            return
        self._library_collection_pending_delete_id = self._library_collections_selected_id
        await self._refresh_collections_panel_action_state_widgets()

    @on(Button.Pressed, "#library-confirm-delete-collection")
    async def confirm_library_collection_delete(self, event: Button.Pressed) -> None:
        event.stop()
        target_id = self._library_collection_pending_delete_id
        if not target_id:
            return
        service = getattr(self.app_instance, "library_collections_service", None)
        delete_collection = getattr(service, "delete_collection", None)
        if not callable(delete_collection):
            self._library_collections_error = "Library Collections are unavailable."
            await self._sync_collections_panel(refresh_snapshot=False)
            return
        try:
            deleted = await self._run_library_service_call(delete_collection, target_id)
        except LibraryCollectionsServiceError as exc:
            self._notify_library_collections_warning(str(exc))
            return
        if not deleted:
            self._notify_library_collections_warning("Failed to delete Collection.")
            return
        self._library_collections_selected_id = ""
        self._library_collection_pending_delete_id = ""
        await self._sync_collections_panel(refresh_snapshot=True)

    @on(Button.Pressed, ".library-collection-row")
    async def select_library_collection(self, event: Button.Pressed) -> None:
        event.stop()
        collection_id = getattr(event.button, "collection_id", "")
        if not collection_id:
            return
        self._library_collections_selected_id = collection_id
        self._library_collection_pending_delete_id = ""
        await self._sync_collections_panel(refresh_snapshot=False)

    def _notify_library_collections_warning(self, message: str) -> None:
        notify = getattr(self.app_instance, "notify", None)
        if callable(notify):
            notify(message or "Library Collections action failed.", severity="warning")

    @on(Button.Pressed, ".library-rag-result-action")
    async def select_library_rag_result(self, event: Button.Pressed) -> None:
        """Select an evidence row for inspector review and Console handoff."""
        event.stop()
        button_id = event.button.id or ""
        try:
            result_index = int(button_id.rsplit("-", 1)[-1])
        except ValueError:
            return
        if result_index < 0 or result_index >= len(self._library_rag_results):
            return
        self._library_rag_selected_result_id = self._library_rag_results[result_index].result_id
        await self._refresh_search_rag_panel_state_widgets()

    @on(Button.Pressed, "#library-rag-use-in-console")
    def use_library_rag_result_in_console(self, event: Button.Pressed) -> None:
        """Stage the selected Library Search/RAG evidence result in Console."""
        self._use_library_rag_result_in_console(event)

    @on(Button.Pressed, "#library-rag-use-selected-in-console")
    def use_selected_library_rag_result_in_console(
        self,
        event: Button.Pressed,
    ) -> None:
        """Stage selected evidence from the center results lane."""
        self._use_library_rag_result_in_console(event)

    def action_library_rag_use_in_console(self) -> None:
        """Keyboard shortcut for staging selected Search/RAG evidence in Console."""
        if self._active_mode != "search":
            return
        self._stage_library_rag_result_in_console()

    def _use_library_rag_result_in_console(self, event: Button.Pressed) -> None:
        """Shared implementation for inspector and results-lane handoff controls."""
        event.stop()
        self._stage_library_rag_result_in_console()

    def _stage_library_rag_result_in_console(self) -> None:
        """Stage the selected Search/RAG evidence result in Console."""
        panel_state = self._library_rag_panel_state()
        console_action = panel_state.use_in_console_action
        if not console_action.enabled or panel_state.selected_result is None:
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify(console_action.disabled_reason, severity="warning")
            return

        opener = getattr(self.app_instance, "open_console_for_live_work", None)
        if not callable(opener):
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify("Use in Console is unavailable for Library Search/RAG.", severity="warning")
            return

        opener(
            source="Library Search/RAG",
            title=panel_state.selected_result.title,
            payload=build_library_rag_console_live_work_payload(
                panel_state.selected_result,
                query=panel_state.query_state.query,
            ),
            status="staged",
            recovery="Review citations before sending.",
            action_label="Review evidence in Console",
        )

    @work(exclusive=True)
    async def _execute_library_rag_search(self, request: LibraryRagSearchRequest) -> None:
        outcome = await run_library_rag_search(self.app_instance, request)
        await self._apply_library_rag_search_outcome(request, outcome)

    async def _apply_library_rag_search_outcome(
        self,
        request: LibraryRagSearchRequest,
        outcome: LibraryRagSearchOutcome,
    ) -> None:
        if not self.is_mounted:
            return
        if self._active_mode != "search" or not self.query("#library-search-rag-panel"):
            return
        current_query = self._library_rag_panel_state().query_state.query
        if request.query != current_query:
            return
        self._library_rag_results = outcome.results
        self._library_rag_retrieval_status = outcome.status
        self._library_rag_recovery_state = outcome.recovery_state
        self._library_rag_selected_result_id = ""
        await self._refresh_search_rag_panel_state_widgets()

    async def _refresh_search_rag_panel_state_widgets(self) -> None:
        if self._active_mode != "search" or not self.query("#library-search-rag-panel"):
            return

        panel_state = self._library_rag_panel_state()
        status_rows = list(self.query("#library-status-row"))
        if status_rows:
            status_rows[0].update(self._status_row_copy())
        self.query_one("#library-rag-query-status", Static).update(
            f"Mode: {panel_state.query_state.mode_label} | Top {panel_state.query_state.top_k}"
        )
        self.query_one("#library-rag-query-blocked", Static).update(
            self._library_rag_query_blocked_summary(panel_state)
        )
        run_action = panel_state.query_state.run_action
        self.query_one("#library-rag-query-blocked-callout", Static).update(
            self._library_rag_query_blocked_callout(panel_state)
        )
        self.query_one("#library-rag-query-blocked-callout", Static).set_class(
            not run_action.enabled,
            "is-blocked",
        )
        self.query_one("#library-rag-query-blocked-callout", Static).set_class(
            run_action.enabled,
            "is-ready",
        )
        run_button = self.query_one("#library-rag-run-query", Button)
        run_button.disabled = not run_action.enabled
        run_button.tooltip = run_action.tooltip
        self.query_one("#library-rag-run-disabled-reason", Static).update(
            self._library_rag_run_disabled_reason(panel_state)
        )

        query_controls = self.query_one("#library-rag-query-controls", Vertical)
        recovery_widgets = list(self.query("#library-rag-query-recovery"))
        show_query_recovery = (
            bool(panel_state.query_state.recovery_copy)
            and panel_state.scope.status != "blocked"
        )
        query_controls.set_class(show_query_recovery, "has-recovery")
        if show_query_recovery:
            if recovery_widgets:
                recovery_widgets[0].update(panel_state.query_state.recovery_copy)
            else:
                await query_controls.mount(
                    Static(
                        panel_state.query_state.recovery_copy,
                        id="library-rag-query-recovery",
                    ),
                    after="#library-rag-query-shortcuts",
                )
        else:
            for widget in recovery_widgets:
                await widget.remove()

        scope_container = self.query_one("#library-rag-source-scope", Vertical)
        scope_container.set_class(bool(panel_state.scope.recovery_copy), "has-recovery")
        self.query_one("#library-rag-scope-summary", Static).update(
            self._library_rag_scope_summary(panel_state)
        )
        for widget_id, copy in self._library_rag_scope_rows(panel_state).items():
            self.query_one(f"#{widget_id}", Static).update(copy)
        scope_recovery_widgets = list(self.query("#library-rag-scope-recovery"))
        import_buttons = list(self.query("#library-rag-open-import-export"))
        if panel_state.scope.recovery_copy:
            if scope_recovery_widgets:
                scope_recovery_widgets[0].update(panel_state.scope.recovery_copy)
            else:
                await scope_container.mount(
                    Static(
                        panel_state.scope.recovery_copy,
                        id="library-rag-scope-recovery",
                    )
                )
            if not import_buttons:
                await scope_container.mount(
                    Button(
                        "Open Import/Export",
                        id="library-rag-open-import-export",
                        classes="library-rag-recovery-action",
                        tooltip="Open Library Import/Export to add sources.",
                    )
                )
        else:
            for widget in (*scope_recovery_widgets, *import_buttons):
                await widget.remove()

        self._refresh_library_rag_inspector(panel_state)
        await self._refresh_library_rag_results_widgets(panel_state)

    def _refresh_library_rag_inspector(
        self,
        panel_state: LibraryRagPanelState,
    ) -> None:
        inspector_widgets = list(self.query("#library-rag-inspector"))
        if not inspector_widgets:
            return
        inspector = inspector_widgets[0]
        if isinstance(inspector, LibrarySearchRagInspectorPanel):
            inspector.refresh_from_state(panel_state)

    async def _refresh_library_rag_results_widgets(
        self,
        panel_state: LibraryRagPanelState,
    ) -> None:
        results_container = self.query_one("#library-rag-results", Vertical)
        self.query_one("#library-rag-attribution-placeholder", Static).update(
            self._library_rag_attribution_placeholder(panel_state)
        )
        for child in list(results_container.children):
            if child.id in LIBRARY_RAG_RESULTS_STATIC_WIDGET_IDS:
                continue
            await child.remove()

        if panel_state.results:
            for index, result in enumerate(panel_state.results):
                score = "" if result.score is None else f" | score {result.score:.3f}"
                selected = result.result_id == panel_state.selected_result_id
                await results_container.mount(
                    Static(
                        f"{index + 1}. {result.title}{score}",
                        id=f"library-rag-result-{index}",
                        classes=(
                            "library-rag-result-row is-selected"
                            if selected
                            else "library-rag-result-row"
                        ),
                    )
                )
                await results_container.mount(
                    Button(
                        "Selected evidence" if selected else "Select evidence",
                        id=f"library-rag-select-result-{index}",
                        classes="library-rag-result-action",
                        tooltip="Select this evidence result for Console handoff.",
                    )
                )
                await results_container.mount(
                    Static(
                        result.row_badge_label,
                        id=f"library-rag-result-badges-{index}",
                        classes="library-rag-result-badges",
                    )
                )
                await results_container.mount(
                    Static(
                        result.snippet,
                        id=f"library-rag-result-snippet-{index}",
                    )
                )
                if result.citation_labels:
                    await results_container.mount(
                        Static(
                            f"Citations: {', '.join(result.citation_labels)}",
                            id=f"library-rag-result-citations-{index}",
                        )
                    )
                if selected:
                    await results_container.mount(
                        Button(
                            panel_state.use_in_console_action.label,
                            id="library-rag-use-selected-in-console",
                            classes=(
                                "library-rag-console-action "
                                "library-rag-center-console-action"
                            ),
                            disabled=not panel_state.use_in_console_action.enabled,
                            tooltip=panel_state.use_in_console_action.tooltip,
                        )
                    )
        elif panel_state.retrieval_status == "searching":
            await results_container.mount(
                Static("Searching Library sources...", id="library-rag-searching")
            )
        elif panel_state.recovery_copy and panel_state.recovery_selector:
            await results_container.mount(
                Static(
                    panel_state.recovery_copy,
                    id=panel_state.recovery_selector,
                )
            )
        else:
            await results_container.mount(
                Static(
                    "No evidence yet. Run Search/RAG to populate results.",
                    id="library-rag-results-empty",
                )
            )
            await results_container.mount(
                Static(
                    "Add or import sources, run a query, then select evidence for Console.",
                    id="library-rag-evidence-empty-guidance",
                    classes="library-rag-empty-guidance",
                )
            )

    @staticmethod
    def _library_rag_scope_summary(panel_state: LibraryRagPanelState) -> str:
        counts = {option.source_type: option.count for option in panel_state.scope.options}
        return (
            "Scope: all local"
            f" | Notes {counts.get('notes', 0)}"
            f" | Media {counts.get('media', 0)}"
            f" | Conversations {counts.get('conversations', 0)}"
        )

    @staticmethod
    def _library_rag_scope_rows(panel_state: LibraryRagPanelState) -> dict[str, str]:
        counts = {option.source_type: option.count for option in panel_state.scope.options}
        total = panel_state.scope.total_count
        selected = len(panel_state.scope.selected_source_types)
        return {
            "library-rag-scope-row-all": (
                f"All Library          | {total} sources    | Browse/search     | Add source"
            ),
            "library-rag-scope-row-workspace": (
                f"Workspace eligible   | {selected} scopes     | Stage after pick  | Select evidence"
            ),
            "library-rag-scope-row-notes": (
                f"Notes                | {counts.get('notes', 0)} sources    | Retrieval-ready   | Run query"
            ),
            "library-rag-scope-row-media": (
                f"Media                | {counts.get('media', 0)} sources    | Retrieval-ready   | Run query"
            ),
            "library-rag-scope-row-conversations": (
                "Conversations        | "
                f"{counts.get('conversations', 0)} sources    | Retrieval-ready   | Run query"
            ),
            "library-rag-scope-row-collections": (
                "Collections          | "
                f"{counts.get('collections', 0)} records    | Read/review WIP   | Open collection"
            ),
            "library-rag-scope-row-import-export": (
                "Import/Export recovery | add sources | Source intake      | Import source"
            ),
        }

    @staticmethod
    def _library_rag_query_blocked_summary(panel_state: LibraryRagPanelState) -> str:
        reason = panel_state.query_state.run_action.disabled_reason
        if not reason:
            return "Ready: run Search/RAG over selected Library sources."
        return f"Blocked: {reason[:1].lower()}{reason[1:]}"

    @staticmethod
    def _library_rag_query_blocked_callout(panel_state: LibraryRagPanelState) -> str:
        reason = panel_state.query_state.run_action.disabled_reason
        if not reason:
            return "Ready | Run retrieval over selected Library sources."
        if reason == "Enter a question or search query.":
            reason = "Enter a question before running retrieval."
        elif reason == "Select at least one Library source.":
            reason = "Select at least one Library source before running retrieval."
        return f"Blocked | {reason}"

    @staticmethod
    def _library_rag_run_disabled_reason(panel_state: LibraryRagPanelState) -> str:
        reason = panel_state.query_state.run_action.disabled_reason
        if not reason:
            return "Run ready: selected Library sources are queryable."
        return f"Run disabled: {reason[:1].lower()}{reason[1:]}"

    @staticmethod
    def _library_rag_attribution_placeholder(panel_state: LibraryRagPanelState) -> str:
        if panel_state.selected_result is None:
            return "Citation/snippet carry-through: reserved for selected evidence."
        return (
            "Citation/snippet carry-through placeholder: selected evidence preserves "
            "source, chunk, snippet, and citations."
        )

    @on(Button.Pressed, "#library-open-notes")
    def open_notes(self) -> None:
        self.post_message(NavigateToScreen("notes"))

    @on(Button.Pressed, "#library-open-media")
    def open_media(self) -> None:
        self.post_message(NavigateToScreen("media"))

    @on(Button.Pressed, "#library-open-conversations")
    async def open_conversations(self, event: Button.Pressed) -> None:
        event.stop()
        await self._set_active_mode("conversations")

    @on(Button.Pressed, "#library-conversations-open-console-empty")
    def open_console_from_empty_conversations(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(NavigateToScreen("chat"))

    @on(Button.Pressed, ".library-conversation-select")
    async def select_library_conversation(self, event: Button.Pressed) -> None:
        event.stop()
        raw_id = event.button.id or ""
        try:
            index = int(raw_id.rsplit("-", 1)[-1])
        except ValueError:
            return
        records = self._conversation_records()
        if index < 0 or index >= len(records):
            return
        self._selected_conversation_id = self._conversation_record_id(records[index], index)
        workspace_depth_state = self._library_workspace_depth_state(refresh=True)
        await self._sync_local_snapshot_region(workspace_depth_state)
        await self._sync_inspector_mode_region(None, workspace_depth_state=workspace_depth_state)
        await self._sync_action_region(workspace_depth_state)

    def _open_selected_conversation_handoff(self) -> None:
        workspace_state = self._library_workspace_depth_state()
        payload = self._selected_conversation_handoff_payload()
        notify = getattr(self.app_instance, "notify", None)
        if payload is None:
            if callable(notify):
                notify("Select a conversation before using it in Console.", severity="warning")
            return
        if not workspace_state.context_handoff_enabled:
            if callable(notify):
                notify(workspace_state.context_handoff_tooltip, severity="warning")
            return
        open_chat_with_handoff = getattr(self.app_instance, "open_chat_with_handoff", None)
        if not callable(open_chat_with_handoff):
            if callable(notify):
                notify("Console handoff is unavailable for Library Conversations.", severity="warning")
            return
        open_chat_with_handoff(payload)

    @on(Button.Pressed, "#library-conversation-open-console")
    def open_selected_conversation_in_console(self, event: Button.Pressed) -> None:
        event.stop()
        self._open_selected_conversation_handoff()

    @on(Button.Pressed, "#library-conversation-use-source")
    def use_selected_conversation_as_source(self, event: Button.Pressed) -> None:
        event.stop()
        self._open_selected_conversation_handoff()

    def _open_selected_media_handoff(self) -> None:
        """Stage the media item open in the viewer into Console via the shared handoff.

        Mirrors ``_open_selected_conversation_handoff``: builds the payload,
        then guards on having an open media item, the workspace
        context-handoff gate, and the app exposing ``open_chat_with_handoff``
        at all. The workspace gate is not conversation-specific --
        ``build_library_workspace_depth_state`` computes
        ``context_handoff_enabled`` across every visible Library source
        (notes, media, and conversations together), and the Library hub's
        own per-source-type readiness rows (``_hub_console_status``) already
        treat Media under that same gate -- so media handoff eligibility
        follows the identical workspace-staging policy as conversations.
        """
        workspace_state = self._library_workspace_depth_state()
        payload = self._selected_media_handoff_payload()
        notify = getattr(self.app_instance, "notify", None)
        if payload is None:
            if callable(notify):
                notify("Open a media item before using it in Console.", severity="warning")
            return
        if not workspace_state.context_handoff_enabled:
            if callable(notify):
                notify(workspace_state.context_handoff_tooltip, severity="warning")
            return
        open_chat_with_handoff = getattr(self.app_instance, "open_chat_with_handoff", None)
        if not callable(open_chat_with_handoff):
            if callable(notify):
                notify("Console handoff is unavailable for Library Media.", severity="warning")
            return
        open_chat_with_handoff(payload)

    @on(Button.Pressed, "#library-media-use-in-chat")
    def use_media_in_chat(self, event: Button.Pressed) -> None:
        """Handle the media viewer's "Use in Chat" action.

        Args:
            event: Button press event emitted by the viewer's "Use in Chat" action.
        """
        event.stop()
        self._open_selected_media_handoff()

    @on(Button.Pressed, "#library-open-import-export")
    async def open_import_export(self, event: Button.Pressed) -> None:
        event.stop()
        await self._set_active_mode("import-export")

    @on(Button.Pressed, "#library-import-export-open-ingest")
    def open_import_export_ingest(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(NavigateToScreen("ingest"))

    @on(Button.Pressed, "#library-import-export-open-media")
    def open_import_export_media(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(NavigateToScreen("media"))

    @on(Button.Pressed, "#library-workspace-import-sources")
    def open_workspace_import_sources(self) -> None:
        self.post_message(NavigateToScreen("ingest"))

    @on(Button.Pressed, "#library-create-local-workspace")
    async def create_local_workspace(self, event: Button.Pressed) -> None:
        """Create and activate a local-only Library workspace.

        Args:
            event: Button press event emitted by the Library Workspaces action.
        """
        event.stop()
        registry_service = getattr(self.app_instance, "workspace_registry_service", None)
        if registry_service is None:
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify("Workspace registry is not ready.", severity="warning")
            return
        try:
            workspace_id, workspace_name = self._next_local_workspace_identity()
            registry_service.create_workspace(
                workspace_id=workspace_id,
                name=workspace_name,
                description="Local workspace created from Library.",
            )
            registry_service.set_active_workspace(workspace_id)
        except Exception:
            logger.warning("Failed to create local Library workspace", exc_info=True)
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify("Local workspace could not be created.", severity="error")
            return

        self._invalidate_library_workspace_depth_state()
        await self._refresh_active_mode_widgets()
        notify = getattr(self.app_instance, "notify", None)
        if callable(notify):
            notify(f"Created local workspace {workspace_name}.", severity="information")

    @on(Button.Pressed, "#library-open-search")
    async def open_search_mode(self, event: Button.Pressed) -> None:
        event.stop()
        await self._set_active_mode("search")

    @on(Button.Pressed, "#library-open-collections")
    async def open_collections_mode(self, event: Button.Pressed) -> None:
        event.stop()
        await self._set_active_mode("collections")

    def _open_study_section(self, initial_section: str = "dashboard") -> None:
        open_study_screen = getattr(self.app_instance, "open_study_screen", None)
        if callable(open_study_screen):
            scope_context = self._source_study_context()
            if scope_context is None:
                open_study_screen(initial_section=initial_section)
            else:
                open_study_screen(scope_context, initial_section=initial_section)
            return
        self.post_message(NavigateToScreen("study"))

    @on(Button.Pressed, "#library-open-study")
    def open_study(self) -> None:
        self._open_study_section("dashboard")

    @on(Button.Pressed, "#library-open-flashcards")
    def open_flashcards(self) -> None:
        self._open_study_section("flashcards")

    @on(Button.Pressed, "#library-open-quizzes")
    def open_quizzes(self) -> None:
        self._open_study_section("quizzes")

    @on(Button.Pressed, "#library-use-in-console")
    def use_in_console(self, event: Button.Pressed) -> None:
        event.stop()
        if not self._has_local_sources():
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify(
                    self._library_lookup_error or LIBRARY_EMPTY_COPY,
                    severity="warning",
                )
            return
        workspace_state = self._library_workspace_depth_state()
        if not workspace_state.context_handoff_enabled:
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify(workspace_state.context_handoff_tooltip, severity="warning")
            return
        open_chat_with_handoff = getattr(self.app_instance, "open_chat_with_handoff", None)
        if not callable(open_chat_with_handoff):
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify(
                    "Console handoff is unavailable for Library in this runtime.",
                    severity="warning",
                )
            return
        open_chat_with_handoff(
            ChatHandoffPayload(
                source="library",
                item_type="library-source-snapshot",
                title="Local Library Sources",
                body=self._source_snapshot_body(),
                display_summary="Local Library source snapshot staged.",
                suggested_prompt="Use these Library sources as grounding context for my next question.",
                runtime_backend="local",
                source_owner="local",
                source_selector_state="local",
                metadata=self._source_snapshot_metadata(),
            )
        )
