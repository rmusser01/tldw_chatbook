"""Library destination shell for source material and Search/RAG."""

from __future__ import annotations

import asyncio
import inspect
import re
import threading
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
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
from textual.widgets import Button, Collapsible, Input, Static, TextArea

from ...Chat.chat_handoff_models import ChatHandoffPayload
from ...Chatbooks.chatbook_models import ContentType
from ...config import get_cli_setting, save_setting_to_cli_config
from ...Constants import (
    LIBRARY_MODE_CONVERSATIONS,
    LIBRARY_NAV_CONTEXT_CONVERSATION_ID,
    LIBRARY_NAV_CONTEXT_INGEST,
    LIBRARY_NAV_CONTEXT_MODE,
    LIBRARY_NAV_CONTEXT_NOTE_ID,
    LIBRARY_NAV_CONTEXT_NOTES_CREATE,
)
from ...DB.ChaChaNotes_DB import ConflictError
from ...Library.export_progress import (
    ExportProgressThrottle,
    format_export_progress_line,
)
from ...Library.library_collections_service import LibraryCollectionsServiceError
from ...Library.library_collections_state import LibraryCollectionsPanelState
from ...Library.library_conversations_state import build_library_conversations_state
from ...Library.library_export_scope import (
    ExportScope,
    count_export_scope,
    resolve_export_selections,
)
from ...Library.library_export_state import (
    DEFAULT_MEDIA_QUALITY,
    LibraryExportFormState,
    build_library_export_form_state,
    default_export_name,
    next_media_quality,
    normalize_export_destination,
)
from ...Library.library_ingest_jobs import LibraryIngestJob
from ...Library.library_ingest_state import (
    INGEST_UNAVAILABLE_COPY,
    LibraryIngestCanvasState,
    LibraryIngestFormState,
    build_library_ingest_state,
    clamp_chunk_size,
    parse_keywords,
)
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
    patch_note_records_after_save,
    resolve_note_template_placeholders,
    sort_notes_records,
)
from ...Library.library_notes_sync_state import (
    AUTO_SYNC_INTERVAL_SECONDS,
    SYNC_CONFLICTS,
    SYNC_DIRECTIONS,
    LibraryNotesSyncState,
    append_activity,
    count_noun,
    next_sync_conflict,
    next_sync_direction,
    sync_conflict_label,
    sync_status_line,
)
from ...Library.library_rag_service import (
    LibraryRagSearchOutcome,
    LibraryRagSearchRequest,
    run_library_rag_search,
)
from ...Library.library_rag_state import (
    LIBRARY_RAG_QUERY_MAX_LENGTH,
    LIBRARY_RAG_SCOPE_ALL_LOCAL_COPY,
    LIBRARY_RAG_SCOPE_TOGGLE_SOURCE_TYPES,
    LIBRARY_SEARCH_HISTORY_ENTRY_MAX_CHARS,
    LIBRARY_SEARCH_HISTORY_LIMIT,
    LibraryRagPanelState,
    update_search_history,
)
from ...Library.library_rail_state import (
    LIBRARY_RAIL_SECTION_IDS,
    coerce_library_rail_preferences,
    serialize_library_rail_preferences,
)
from ...Library.library_shell_state import (
    LIBRARY_EXPORT_SERVER_DISABLED_TOOLTIP,
    LIBRARY_ROW_BROWSE_COLLECTIONS,
    LIBRARY_ROW_BROWSE_CONVERSATIONS,
    LIBRARY_ROW_BROWSE_MEDIA,
    LIBRARY_ROW_BROWSE_NOTES,
    LIBRARY_ROW_BROWSE_SEARCH,
    LIBRARY_ROW_CREATE_NOTE,
    LIBRARY_ROW_INGEST_EXPORT,
    LIBRARY_ROW_INGEST_MEDIA,
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
    LibraryExportCanvas,
    LibraryIngestCanvas,
    LibraryMediaCanvas,
    LibraryMediaViewer,
    LibraryNotesCanvas,
    LibraryRail,
    LibrarySearchRagInspectorPanel,
    LibrarySearchRagPanel,
    library_dim_label_text,
    library_rag_history_children,
    library_rag_query_shows_full_recovery,
    library_rag_query_status_children,
    library_rag_results_body_children,
    library_rag_scope_recovery_children,
    library_rag_scope_shows_recovery,
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
# Navigation composes a FRESH LibraryScreen instance per visit (PR #595
# freeze fix), so a per-instance memo is useless -- the previous visit's
# snapshot is cached on the APP instance instead (see `on_mount` and
# `_refresh_local_source_snapshot`) so a repeat visit within this window
# renders instantly instead of showing the loading placeholder again. The
# cached snapshot is always applied THEN immediately reconciled with a
# fresh background fetch, so staleness is bounded to a single refresh
# cycle regardless of this TTL's length.
LIBRARY_SNAPSHOT_CACHE_TTL_SECONDS = 5.0
LIBRARY_NOTES_AUTOSAVE_SECONDS = 2.0
LIBRARY_NOTE_CONTENT_MAX_CHARS = 2_000_000
LIBRARY_COLLECTION_SYNC_CONFLICT_LIMIT = 200
LIBRARY_HANDOFF_LABEL_PREFIX = "Console/RAG handoff: "
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
LIBRARY_RAG_RESULTS_STATIC_WIDGET_IDS = frozenset({"library-rag-results-heading"})

LIBRARY_STUDY_HANDOFF_MODES = {
    "study": {
        # "header" mirrors the rail row title that opens this canvas
        # (LibraryRailRow "Study decks", library_shell_state.py) so the
        # canvas doesn't restate the mode name a second, differently-worded
        # way (L3b Task 8/9 follow-up: UX wave C/D, handoff copy
        # consolidation).
        "header": "Study decks",
        "action_label": "Study Dashboard",
        # UX wave L2: the button reads as a verb ("Continue in Study")
        # instead of restating the destination's own name -- action_label
        # still backs the header/purpose/recovery copy below.
        "button_label": "Continue in Study",
        "purpose": "Plan study decks from Library sources.",
    },
    "flashcards": {
        "header": "Flashcards",
        "action_label": "Flashcards",
        "button_label": "Continue in Study",
        "purpose": "Generate or review cards from Library sources.",
    },
    "quizzes": {
        "header": "Quizzes",
        "action_label": "Quizzes",
        "button_label": "Continue in Study",
        "purpose": "Generate or resume quizzes from Library sources.",
    },
}

# Single shared ownership line for all three handoff canvases: Library only
# prepares source context, Study owns everything downstream of "open".
LIBRARY_STUDY_HANDOFF_OWNERSHIP_COPY = "Generation and review run in Study."

# How many carried-forward source titles the handoff canvas names before
# collapsing the rest into an "and N more" count.
LIBRARY_STUDY_HANDOFF_TITLES_CAP = 3


def _library_carries_forward_line(titles: Sequence[str]) -> str:
    """Build the handoff canvas's capped, markup-escaped carries-forward line.

    Args:
        titles: Sampled source titles (notes/media/conversations) that will
            carry forward into Study. Must be non-empty -- callers render no
            line at all when there is no source context (see
            ``_study_handoff_copy``).

    Returns:
        ``"Carries forward: a, b, c"`` when there are at most
        ``LIBRARY_STUDY_HANDOFF_TITLES_CAP`` titles, else ``"Carries
        forward: a, b, c and N more."`` with the remaining count appended.
    """
    escaped_titles = [escape_markup(title) for title in titles]
    capped = escaped_titles[:LIBRARY_STUDY_HANDOFF_TITLES_CAP]
    joined = ", ".join(capped)
    remaining = len(escaped_titles) - len(capped)
    if remaining > 0:
        return f"Carries forward: {joined} and {remaining} more."
    return f"Carries forward: {joined}"


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

# Maps a Library navigation-context ``mode`` value to the shell rail row
# that selects that canvas -- covers exactly the mode values nav-context
# callers emit/support today (L3b Task 8 audit): ``conversations`` (Personas'
# conversations controller), ``search`` and ``collections`` (both directly
# tested contracts of ``apply_navigation_context``, though no live emitter
# currently sends them). ``notes`` is handled as its own dedicated branch in
# ``_apply_navigation_context_state`` below (``open_notes_workspace``'s
# route), not through this table. ``media`` has no navigation-context entry
# point at all (the retired mode-strip machinery never had a "media" mode
# either). Any other mode value -- including the retired
# ``study``/``flashcards``/``quizzes`` mode values (those rows are now
# "handoff" rows, not nav-context targets) and the retired
# ``sources``/``workspaces``/``import-export`` values -- degrades quietly,
# unchanged from before this table existed.
LIBRARY_NAV_MODE_TO_ROW_ID = {
    "conversations": LIBRARY_ROW_BROWSE_CONVERSATIONS,
    "collections": LIBRARY_ROW_BROWSE_COLLECTIONS,
    "search": LIBRARY_ROW_BROWSE_SEARCH,
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
        # Decorative Create-rail counts (study decks, flashcards due,
        # quizzes): None until the local source snapshot has loaded, and
        # None per-key thereafter when that count's service seam is absent
        # or its fetch failed -- see ``_study_count_or_none``. Unlike the
        # three browse sources, a missing/failed study count never carries
        # an error copy; the row just renders uncounted.
        self._library_study_counts: dict[str, int | None] = {
            "study_decks": None,
            "flashcards_due": None,
            "quizzes": None,
        }
        self._library_loaded = False
        self._library_rag_mode: str = "search"
        self._library_rag_query = ""
        self._library_rag_results = ()
        self._library_rag_retrieval_status = ""
        self._library_rag_recovery_state: DestinationRecoveryState | None = None
        self._library_rag_selected_result_id = ""
        # B2: source types the user has toggled OFF (deselected) in the
        # scope region. Empty = every available source is in scope (the
        # default). Persists across rail switches within the session, same
        # as mode, but is never written to config.
        self._library_rag_scope_deselected: set[str] = set()
        # D1: whether the `Recent searches` collapsible should render
        # collapsed. Only `_apply_library_rag_search_outcome` (the
        # results-arrival transition) is allowed to change this; every
        # other refresh must leave the user's manual expand/collapse alone.
        self._library_rag_history_collapsed: bool = False
        self._library_search_history: tuple[str, ...] = self._load_library_search_history()
        # Serializes history-collapsible content rebuilds: the "searching"
        # status refresh (called synchronously before the search worker is
        # scheduled) and that worker's own "outcome" refresh can otherwise
        # interleave mid-rebuild and mount duplicate row IDs.
        self._library_rag_history_refresh_lock = asyncio.Lock()
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
        # The folder box's live (possibly uncommitted) text. Typing updates
        # only this field -- persisting to the TOML config on every
        # Input.Changed meant a full config rewrite + cache reload per
        # keystroke. It commits to config on Enter, Browse…, or a validated
        # Sync now run. None = not edited this panel visit; fall back to the
        # persisted config value.
        self._library_notes_sync_folder_text: str | None = None
        # Ingest canvas form echo -- a single bundled mutable dataclass
        # (rather than a scatter of scalar fields like the sync panel
        # above) since every field here is reset together on rail
        # re-entry (see ``_reset_library_ingest_transient_state``); the
        # job queue itself is registry-owned, not screen state.
        self._library_ingest_form: LibraryIngestFormState = LibraryIngestFormState()
        # Dedupe counter for the "poke the source snapshot on transitions
        # into done" rule (Task 5's registry listener): only re-fetch when
        # the registry's done-job count has grown since the last time this
        # screen checked. Seeded from the live registry in ``on_mount``
        # (not here) so a re-mounted, cached screen instance never treats
        # jobs that finished in a previous mount as a fresh transition.
        self._library_ingest_last_done_count: int = 0
        # Export canvas state (F4 Task 2). ``_library_export_counts`` is
        # ``None`` until the counts worker lands a result for the current
        # scope (drives ``LibraryExportFormState.counts_loading`` --
        # deliberately not a separate boolean flag, so "loading" and "no
        # result yet" can never drift apart). ``_library_export_form`` is
        # a plain dict (not a dataclass, unlike the ingest form echo)
        # since Task 3 reads specific keys off it directly per the F4
        # plan's screen-attrs contract.
        self._library_export_scope: ExportScope = ExportScope(kind="everything")
        self._library_export_counts: dict[str, int] | None = None
        self._library_export_form: dict[str, Any] = self._default_library_export_form()
        self._library_export_running: bool = False
        self._library_export_error: str = ""
        # Task 3: the running export's quiet status line ("Exporting…
        # (N items)"); no backing field existed after Task 2 (its report
        # flagged this as the natural next attr). Cleared alongside
        # ``_library_export_error`` on every canvas reset and on run
        # completion.
        self._library_export_status: str = ""
        # Task 3 review fix: a monotonic token identifying the CURRENT
        # export attempt. Bumped both when a new export starts
        # (``handle_library_export_submit``) and whenever the export
        # canvas's transient state is reset out from under an in-flight
        # run (``_reset_library_export_transient_state`` -- reachable via
        # any rail-row switch or "Export…" section action while a worker
        # is still executing on its own OS thread, which cannot be
        # preempted mid-``asyncio.run`` by ``Worker.cancel()``). The
        # worker captures the token at dispatch time and the completion
        # handlers compare it back against the live value before mutating
        # ``_library_export_running``/``_library_export_error``/
        # ``_library_export_status`` or touching the DOM -- an orphaned
        # run's late completion still notifies (the export genuinely
        # happened) but can never stomp whatever the user is now looking
        # at, mirroring ``_apply_library_export_counts``'s scope-mismatch
        # staleness guard for the sibling counts worker.
        self._library_export_run_id: int = 0
        # Task 4: the current run's cancellation signal. Created fresh at
        # every submit (``handle_library_export_submit``); the worker reads
        # ``event.is_set`` as the service's ``cancel_check``. Nothing sets
        # it yet in this task -- the Cancel button and navigate-away wiring
        # land in Task 5.
        self._library_export_cancel_event: threading.Event | None = None

    def on_mount(self) -> None:
        """Populate the Library on entry, rendering instantly from cache.

        Arms the snapshot-timeout failsafe, then (166) if the app-scoped
        snapshot cache holds a recent result (within
        ``LIBRARY_SNAPSHOT_CACHE_TTL_SECONDS``) applies it synchronously so a
        returning visit paints immediately instead of showing the loading
        placeholder, and unconditionally kicks
        ``_refresh_local_source_snapshot`` to reconcile against the DB. Also
        seeds the ingest registry listener and runs any deferred deep-link
        loads (collections / note editor / media viewer) that
        ``apply_navigation_context`` could not run before mount.
        """
        super().on_mount()
        self.set_timer(
            LIBRARY_SOURCE_SNAPSHOT_TIMEOUT_SECONDS,
            self._apply_source_snapshot_timeout,
        )
        cached_snapshot = getattr(
            self.app_instance, "_library_source_snapshot_cache", None
        )
        cached_stamp = getattr(
            self.app_instance, "_library_source_snapshot_cache_stamp", None
        )
        if (
            cached_snapshot is not None
            and cached_stamp is not None
            and time.monotonic() - cached_stamp < LIBRARY_SNAPSHOT_CACHE_TTL_SECONDS
        ):
            # Instant-then-reconcile: paint the previous visit's snapshot
            # synchronously (this screen instance is brand new, so nothing
            # else has populated `_local_source_records` yet) so a
            # returning visit never shows the loading placeholder, then
            # still kick the real refresh below -- its completion re-applies
            # fresh data and refreshes the cache, so this can't drift more
            # than one refresh cycle stale.
            self._apply_local_source_snapshot(*cached_snapshot)
            # `_apply_local_source_snapshot`'s own `self.is_mounted`-guarded
            # `refresh(recompose=True)` is a no-op here: Textual only flips
            # `_is_mounted` True in the `finally` clause AFTER the Mount
            # event finishes dispatching (see
            # `MessagePump._pre_process`) -- i.e. strictly after this very
            # `on_mount` call returns -- so without an explicit recompose
            # here the cached attrs above would be set correctly but the
            # already-composed (stale, pre-cache) DOM would never actually
            # repaint. `Widget.refresh(recompose=True)` itself has no such
            # guard (it just schedules `_check_recompose` via
            # `call_next`), so calling it directly is safe mid-mount and is
            # what actually makes the cached snapshot visible at first
            # paint.
            self.refresh(recompose=True)
        self._refresh_local_source_snapshot()
        registry = self._library_ingest_registry()
        if registry is not None:
            counts_fn = getattr(registry, "counts", None)
            if callable(counts_fn):
                # Seed from whatever the registry already knows so a
                # freshly composed screen doesn't treat jobs that finished
                # during a previous Library visit as a brand-new
                # done-transition and fire a redundant snapshot refresh.
                self._library_ingest_last_done_count = counts_fn().get("done", 0)
            registry.add_listener(self._handle_library_ingest_registry_changed)
        if (
            self._library_selected_row_id == LIBRARY_ROW_BROWSE_COLLECTIONS
            and not self._library_collections_loaded
        ):
            # Deep-links that preset mode=collections call apply_navigation_context
            # BEFORE the screen is mounted (see app.py handle_screen_navigation),
            # so the is_mounted-guarded load there never fires. Kick the same
            # snapshot load here once the canvas has actually been composed.
            self.run_worker(self._sync_collections_panel(refresh_snapshot=True))
        if (
            self._library_notes_view == "editor"
            and self._selected_note_id
            and self._library_note_detail is None
        ):
            # Mirrors the collections case above: a note_id deep-link applied
            # before mount cannot run_worker yet, so the detail fetch is
            # deferred to here once the canvas has actually been composed.
            self.run_worker(
                self._refresh_library_note_detail(self._selected_note_id),
                exclusive=True,
                group="library_note_detail",
            )
        if (
            self._library_media_view == "viewer"
            and self._selected_media_id
            and self._library_media_detail is None
        ):
            # Cross-visit state restore (``restore_state``) sets the media
            # viewer's selection/view attrs before mount the same way a
            # note_id nav-context deep-link does above, but -- unlike that
            # case -- nothing else pre-mount kicks off the detail fetch:
            # ``handle_library_media_row`` (the only other caller of
            # ``_refresh_library_media_detail``) only runs from a live row
            # click. Without this, a restored viewer would render its
            # "Loading media…" placeholder forever. Deleted-record safety
            # mirrors the note case: ``_refresh_library_media_detail``
            # notifies and falls back to the list view when the id no
            # longer resolves.
            self.run_worker(
                self._refresh_library_media_detail(self._selected_media_id),
                exclusive=True,
                group="library_media_detail",
            )
        if (
            self._library_selected_row_id == LIBRARY_ROW_INGEST_EXPORT
            and self._library_export_counts is None
        ):
            # Same restored-placeholder class as the media viewer/notes
            # editor above: a cross-visit ``restore_state`` (or a tab
            # round-trip whose ``save_state`` persisted
            # ``_library_selected_row_id == LIBRARY_ROW_INGEST_EXPORT``)
            # lands a fresh instance on the export canvas with
            # ``_library_export_counts is None`` -> the scope line renders
            # "Counting…" and Export stays disabled. But the counts worker
            # is only kicked from the two LIVE entry points
            # (``_select_library_rail_row`` and
            # ``_open_library_export_canvas``), never from a restore --
            # so without this re-kick the form would stay stuck
            # "Counting…" with Export permanently disabled until the user
            # clicked another rail row and back. Mirrors
            # ``_select_library_rail_row``'s own post-recompose kick. The
            # ``is None`` guard keeps this from redundantly re-running when
            # counts already landed (they never survive a restore today --
            # ``save_state`` doesn't persist them -- but the guard makes
            # the intent explicit and is cheap insurance).
            self._start_library_export_counts_worker()

    def on_unmount(self) -> None:
        """Unregister the ingest registry listener registered in ``on_mount``.

        ``on_unmount`` (not ``on_screen_suspend``) is the correct pairing:
        listener add/remove must be symmetric with the mount/unmount
        cycle, not the temporary suspend/resume pair a screen gets while
        merely covered by another screen on the stack (suspend does not
        tear down this screen, so pairing removal with it would silently
        stop live updates while
        still fully composed and, per the plan brief, still able to
        resume). The registry itself is a plain in-memory object owned by
        the app, not this screen, and can keep firing mutations long after
        this screen is gone -- the listener body also guards on
        ``self.is_mounted`` (belt and braces, matching this file's
        established convention elsewhere), though note that in this
        Textual version ``is_mounted`` tracks "has been mounted at least
        once" rather than "currently mounted" (it is never reset back to
        ``False`` after removal) -- so this call is what actually closes
        the window, not the guard.
        """
        super().on_unmount()
        registry = self._library_ingest_registry()
        if registry is not None:
            registry.remove_listener(self._handle_library_ingest_registry_changed)

    def save_state(self) -> dict[str, Any]:
        """Persist Library selection/view state for the next visit.

        Only SELECTION and VIEW state is captured -- never bulk fetched
        snapshots (``_local_source_records`` and friends re-fetch fresh on
        the next mount's ``_refresh_local_source_snapshot``, and a restored
        id may be stale by then) or note editor text (``flush_pending_work``
        has already persisted any dirty edit to the DB before the app calls
        this). The ingest form/queue, rail collapse preferences, and search
        history are deliberately excluded here: they are already persisted
        elsewhere (the app-owned ingest job registry and the CLI config,
        respectively) and re-seeding them from this in-memory dict would
        fight those owners. The RAG results tuple (and its paired retrieval
        status / recovery state, set together by
        ``_apply_library_rag_search_outcome``) are safe to carry verbatim
        because their rows are frozen dataclasses -- copies are taken below
        only to avoid aliasing a live mutable set with the stashed dict.

        The four per-pane filter/sort values below (media type cycle, notes
        sort mode, notes substring filter, conversations query) are VIEW
        state exactly like the selection ids above -- they change what the
        canvas builders render, not what data is fetched -- so they belong
        here too (PR #595 shipped the selection/RAG half of this contract
        but left these out). ``_library_notes_filter_records`` (the
        substring filter's recomputed result cache) is deliberately NOT
        persisted -- it is a derived/bulk snapshot like
        ``_local_source_records``, and restore leaves it ``None`` so the
        canvas recomputes it fresh from ``_library_notes_filter`` on mount.
        """
        state = super().save_state()
        state["library_selected_row_id"] = self._library_selected_row_id
        state["selected_conversation_id"] = self._selected_conversation_id
        state["selected_note_id"] = self._selected_note_id
        state["library_notes_view"] = self._library_notes_view
        state["selected_media_id"] = self._selected_media_id
        state["library_media_view"] = self._library_media_view
        state["library_rag_query"] = self._library_rag_query
        state["library_rag_mode"] = self._library_rag_mode
        state["library_rag_scope_deselected"] = set(self._library_rag_scope_deselected)
        state["library_rag_results"] = tuple(self._library_rag_results)
        state["library_rag_selected_result_id"] = self._library_rag_selected_result_id
        state["library_rag_retrieval_status"] = self._library_rag_retrieval_status
        state["library_rag_recovery_state"] = self._library_rag_recovery_state
        state["library_media_type_filter"] = self._library_media_type_filter
        state["library_notes_sort"] = self._library_notes_sort
        state["library_notes_filter"] = self._library_notes_filter
        state["library_conversation_query"] = getattr(
            self, "_library_conversation_query", ""
        )
        return state

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore Library selection/view state saved by ``save_state``.

        The app calls this on a freshly-constructed, not-yet-mounted
        instance BEFORE ``switch_screen`` mounts it, so these attrs are
        exactly what ``on_mount`` and the first ``compose_content`` see.

        Stale-id safety (the record was deleted or is otherwise gone by the
        time the user comes back): the conversations and media LIST canvases
        already fall back to the first displayed row when the restored id is
        absent (``build_library_conversations_state`` /
        ``build_library_media_state``); the notes-editor and media-viewer
        deep-link fetches this triggers from ``on_mount`` notify the user and
        fall back to the list view when the id no longer resolves
        (``_refresh_library_note_detail`` / ``_refresh_library_media_detail``).
        A restored ``editor``/``viewer`` view with no matching selected id
        (should never happen, but a saved-state dict is not statically typed)
        degrades to the list view below rather than rendering a permanent
        loading placeholder.

        The four per-pane filter/sort values restored below are read by the
        canvas builders at mount time (``_build_library_media_state``,
        the notes canvas branch of ``compose_content``,
        ``_build_library_conversations_state``) -- setting them here, before
        ``switch_screen`` mounts this instance, is all that is needed for
        the first paint to already reflect them; no on_mount re-kick is
        required (unlike a fetched detail). The conversations query is
        user text re-sanitized through ``_safe_text`` here too -- it was
        already sanitized once when the user submitted it
        (``handle_library_conversations_filter_submitted``), but a saved-
        state dict is not statically typed, so this is defense against a
        corrupted/foreign dict rather than a real double-sanitization of
        trusted input.
        """
        super().restore_state(state)
        if not isinstance(state, dict):
            return

        self._library_selected_row_id = str(state.get("library_selected_row_id") or "")
        self._selected_conversation_id = str(state.get("selected_conversation_id") or "")

        selected_note_id = str(state.get("selected_note_id") or "")
        notes_view = str(state.get("library_notes_view") or "list")
        if notes_view == "editor" and not selected_note_id:
            notes_view = "list"
        self._selected_note_id = selected_note_id
        self._library_notes_view = notes_view

        selected_media_id = str(state.get("selected_media_id") or "")
        media_view = str(state.get("library_media_view") or "list")
        if media_view == "viewer" and not selected_media_id:
            media_view = "list"
        self._selected_media_id = selected_media_id
        self._library_media_view = media_view

        self._library_rag_query = str(state.get("library_rag_query") or "")
        rag_mode = state.get("library_rag_mode")
        self._library_rag_mode = rag_mode if rag_mode in ("search", "rag") else "search"
        scope_deselected = state.get("library_rag_scope_deselected")
        self._library_rag_scope_deselected = (
            set(scope_deselected)
            if isinstance(scope_deselected, (set, frozenset, list, tuple))
            else set()
        )
        rag_results = state.get("library_rag_results")
        self._library_rag_results = (
            tuple(rag_results) if isinstance(rag_results, (list, tuple)) else ()
        )
        self._library_rag_selected_result_id = str(
            state.get("library_rag_selected_result_id") or ""
        )
        self._library_rag_retrieval_status = str(
            state.get("library_rag_retrieval_status") or ""
        )
        recovery_state = state.get("library_rag_recovery_state")
        self._library_rag_recovery_state = (
            recovery_state if isinstance(recovery_state, DestinationRecoveryState) else None
        )

        media_type_filter = state.get("library_media_type_filter")
        self._library_media_type_filter = (
            media_type_filter if isinstance(media_type_filter, str) and media_type_filter else "All"
        )
        notes_sort = state.get("library_notes_sort")
        self._library_notes_sort = (
            notes_sort if isinstance(notes_sort, str) and notes_sort else "newest"
        )
        notes_filter = state.get("library_notes_filter")
        self._library_notes_filter = notes_filter if isinstance(notes_filter, str) else ""
        conversation_query = state.get("library_conversation_query")
        self._library_conversation_query = self._safe_text(
            conversation_query if isinstance(conversation_query, str) else "",
            "",
            max_length=200,
        )

    async def flush_pending_work(self) -> bool:
        """Persist pending note edits before the app navigates away.

        The app awaits this from ``handle_screen_navigation`` before
        discarding this screen instance -- without it, a note edit whose
        debounced autosave has not fired yet (the timer re-arms on every
        keystroke) would be destroyed with the screen when the user switches
        tabs mid-edit.

        Returns:
            False whenever unsaved edits survive the flush -- an unresolved
            save conflict (the user must resolve the banner) or a failed
            save (state "error"; the edits are still only in the editor).
            Both veto the navigation so the screen instance holding the
            edits is not discarded. True once nothing dirty remains.
        """
        await self._flush_library_note_save()
        return not self._library_note_dirty

    def apply_navigation_context(self, context: Mapping[str, Any]) -> None:
        """Apply route context supplied by shell navigation.

        Args:
            context: Navigation payload from ``NavigateToScreen``. A valid
                Library mode switches the active mode. A ``conversation_id``
                selects that conversation when the local source snapshot
                arrives, defaulting the mode to Conversations when no valid
                mode is supplied. A ``notes_create`` flag lands on the
                in-canvas Create > New note view (the retired Notes tab's
                "new note" deep link). A ``note_id`` opens that note's
                in-canvas editor directly (the retired Notes tab's
                chat-sidebar deep link); ``mode="notes"`` alone (no
                ``note_id``) lands on the Notes list instead. An
                ``ingest_media`` flag lands on the in-canvas Ingest >
                Import media view (Home's ingest-jobs "Open details"
                control, L3b Task 6).
        """
        if not isinstance(context, Mapping):
            return
        if self.is_mounted and self._library_note_dirty:
            # Defense in depth for direct callers: navigation always
            # composes a fresh (unmounted) screen, but a future palette
            # shortcut could invoke this on a live, mounted editor mid-edit. Applying it synchronously would
            # recompose the canvas out from under the pending debounced
            # autosave, destroying the #library-note-body it reads and dropping
            # the last edits. Flush first (awaited, off this sync nav path),
            # mirroring _select_library_rail_row; unsaved edits abort it.
            self.run_worker(
                self._apply_navigation_context_after_flush(context),
                exclusive=True,
                group="library_nav_context",
            )
            return
        self._apply_navigation_context_state(context)

    async def _apply_navigation_context_after_flush(
        self, context: Mapping[str, Any]
    ) -> None:
        """Flush a dirty note editor, then apply nav context on the UI loop.

        The mounted dirty-editor branch of ``apply_navigation_context`` routes
        here so the pending save is awaited before the recompose that tears the
        editor down. If unsaved edits survive the flush, the switch aborts and
        leaves the editor in place -- the same guard
        ``_select_library_rail_row`` applies.
        """
        await self._flush_library_note_save()
        if self._library_note_dirty:
            return
        self._apply_navigation_context_state(context)

    def _apply_navigation_context_state(self, context: Mapping[str, Any]) -> None:
        """Apply validated navigation context to canvas state and recompose.

        Split from ``apply_navigation_context`` so its mounted dirty-editor
        path can flush the pending save first (see
        ``_apply_navigation_context_after_flush``) while the pre-mount and
        clean-editor paths apply directly.
        """
        requested_mode = self._safe_text(
            context.get(LIBRARY_NAV_CONTEXT_MODE),
            max_length=64,
        )
        conversation_id = self._safe_text(
            context.get(LIBRARY_NAV_CONTEXT_CONVERSATION_ID),
            max_length=200,
        )
        note_id = self._safe_text(
            context.get(LIBRARY_NAV_CONTEXT_NOTE_ID),
            max_length=200,
        )
        notes_create = bool(context.get(LIBRARY_NAV_CONTEXT_NOTES_CREATE))
        ingest_media = bool(context.get(LIBRARY_NAV_CONTEXT_INGEST))
        target_mode = requested_mode if requested_mode in LIBRARY_NAV_MODE_TO_ROW_ID else ""
        if conversation_id and not target_mode:
            target_mode = LIBRARY_MODE_CONVERSATIONS
        if target_mode:
            selected_row_id = LIBRARY_NAV_MODE_TO_ROW_ID.get(target_mode)
            if selected_row_id:
                self._library_selected_row_id = selected_row_id
            self._invalidate_library_workspace_depth_state()
        if conversation_id:
            self._selected_conversation_id = conversation_id
            self._library_selected_row_id = LIBRARY_ROW_BROWSE_CONVERSATIONS
        if requested_mode == "notes" and not note_id:
            # "notes" is a canvas row, not a nav-context table entry (see
            # target_mode above), so it needs its own selection here --
            # mirrors handle_library_notes_row's list-view entry state.
            self._library_selected_row_id = LIBRARY_ROW_BROWSE_NOTES
        if notes_create:
            # Mirrors _select_library_rail_row(LIBRARY_ROW_CREATE_NOTE) --
            # the create-note rail row's own target_id. The rail row's
            # flush of a dirty editor is handled upstream by
            # apply_navigation_context's mounted dirty-editor branch; here we
            # only apply the selection the recompose reads. Reset the note
            # editor state FIRST (a mounted screen re-entered via this
            # deep link can still hold a previously opened note's
            # id/detail/version) then re-assert the create-note target state
            # AFTER, since the reset flips _library_notes_view back to
            # "list" -- same reset-then-set ordering as
            # _open_library_item_by_id's notes branch.
            self._reset_library_note_editor_state()
            self._library_selected_row_id = LIBRARY_ROW_CREATE_NOTE
        if ingest_media:
            # Home's ingest-jobs "Open details" control re-points here
            # (L3b Task 6): running/queued/failed Library ingest jobs
            # mirror into Home's Running and Needs Attention sections, and
            # this deep link is their one-hop route back to the in-canvas
            # ingest queue. Mirrors
            # _select_library_rail_row(LIBRARY_ROW_INGEST_MEDIA) -- unlike
            # collections/note_id above, the ingest canvas reads the job
            # registry directly on recompose, so no async data fetch (and
            # therefore no on_mount deferral) is needed even pre-mount.
            self._library_selected_row_id = LIBRARY_ROW_INGEST_MEDIA
            # Mirrors _select_library_rail_row's reset: a cached LibraryScreen
            # re-entered via this deep link (e.g. from Home's ingest-jobs
            # "Open details" control) must never show a stale half-filled
            # form left over from a previous Ingest visit.
            self._reset_library_ingest_transient_state()
        if note_id:
            # Forward-compat entry point: the retired Notes tab's chat-sidebar
            # deep link carried a note id, and this rebuilds the editor for it.
            # No caller in the tree emits a note_id context today (the surviving
            # open_notes_workspace route carries none, landing on the list), so
            # this is exercised only by tests until such a producer is wired --
            # not orphaned wiring.
            self._library_selected_row_id = LIBRARY_ROW_BROWSE_NOTES
            self._selected_note_id = note_id
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
            if self.is_mounted:
                self.run_worker(
                    self._refresh_library_note_detail(note_id),
                    exclusive=True,
                    group="library_note_detail",
                )
        if self.is_mounted:
            if (
                self._library_selected_row_id == LIBRARY_ROW_BROWSE_COLLECTIONS
                and not self._library_collections_loaded
            ):
                # Deep-link into Collections must load the snapshot the retired
                # chip flow ran; the panel shows the records once loaded.
                self.run_worker(self._sync_collections_panel(refresh_snapshot=True))
            else:
                self.refresh(recompose=True)

    # Own group, deliberately separate from the "default" group the plain
    # `self.run_worker(self._sync_collections_panel(...))` calls above use
    # (they take no explicit group either). Both were previously exclusive
    # in the SAME (default) group, so an ingest-completion poke into this
    # worker (see `_handle_library_ingest_registry_changed`) could cancel
    # an in-flight Collections load. A separate group makes the two
    # `exclusive=True` scopes independent.
    @work(exclusive=True, group="library_source_snapshot")
    async def _refresh_local_source_snapshot(self) -> None:
        (
            records,
            counts,
            total_known,
            lookup_error,
            recovery_state,
            study_counts,
        ) = await self._list_local_source_snapshot()
        # Refresh the app-scoped instant-repeat-visit cache (see `on_mount`)
        # with this fresh snapshot before applying it, so any other
        # LibraryScreen instance mounted from here on -- including a
        # concurrent one, since screens are recomposed per visit -- reads
        # this fetch's result rather than stale/no data.
        #
        # Only a SUCCESSFUL snapshot (``lookup_error is None``) is cached: an
        # error/service-unavailable result still applies to the current view
        # as usual below (unchanged), but must not become the next visit's
        # instant-apply seed -- otherwise a return visit within TTL would
        # flash the "services unavailable" banner for one frame before the
        # reconcile corrects it. Skipping the write leaves the previous good
        # snapshot (or nothing) in place, so the next visit does a normal
        # fresh fetch instead.
        if lookup_error is None:
            # Cache SHALLOW COPIES of the mutable containers, not the live
            # objects (Qodo review): ``_apply_local_source_snapshot`` below
            # aliases ``self._local_source_records = records``, and later
            # in-place key reassignments (e.g. ``self._local_source_records
            # ["media"] = ...`` after a media edit) would otherwise mutate
            # the cached dict too, so a return visit's instant-apply would
            # render a snapshot whose records no longer match its cached
            # counts/totals. The record tuples themselves are immutable, so a
            # one-level dict copy is enough to isolate the cache.
            self.app_instance._library_source_snapshot_cache = (
                dict(records),
                dict(counts) if isinstance(counts, dict) else counts,
                dict(total_known) if isinstance(total_known, dict) else total_known,
                lookup_error,
                recovery_state,
                dict(study_counts) if isinstance(study_counts, dict) else study_counts,
            )
            self.app_instance._library_source_snapshot_cache_stamp = time.monotonic()
        self._apply_local_source_snapshot(
            records, counts, total_known, lookup_error, recovery_state, study_counts
        )

    def _carry_selected_conversation_into_snapshot(
        self,
        records: dict[str, tuple[Mapping[str, Any], ...]],
    ) -> dict[str, tuple[Mapping[str, Any], ...]]:
        """Preserve an out-of-page selected conversation across a snapshot replace.

        (C3) A wholesale ``_local_source_records`` replace -- the periodic
        background refresh, not a user action -- can silently drop the
        currently-open conversation if it fell off the loaded page (the
        conversations snapshot is capped, see
        ``LIBRARY_SOURCE_PAGE_SIZES["conversations"]``) or was fetched
        out-of-band via ``_open_library_item_by_id`` and prepended into the
        OLD records. Without this, the next recompose would silently reset
        the selection to the first row (``_ensure_selected_conversation_id``)
        even though the user never navigated away -- the same class of race
        ``_open_library_item_by_id`` already guards against for its own
        out-of-snapshot fetch, just triggered by a background refresh
        instead of a user click.

        Pure in-memory merge: reads the OLD ``self._local_source_records``
        (not yet replaced) and the INCOMING ``records``, and -- only when the
        selected id is present in the old snapshot but missing from the new
        one -- prepends the old record into the new conversations tuple so
        the selection survives the replace.

        Args:
            records: The incoming snapshot about to replace
                ``self._local_source_records``.

        Returns:
            ``records``, unchanged, or with the selected conversation's
            record prepended into its ``"conversations"`` tuple.
        """
        selected_id = getattr(self, "_selected_conversation_id", "")
        if not selected_id:
            return records
        old_conversations = getattr(self, "_local_source_records", {}).get(
            "conversations", ()
        )
        old_index_by_id = {
            self._conversation_record_id(record, index): record
            for index, record in enumerate(old_conversations)
        }
        carried_record = old_index_by_id.get(selected_id)
        if carried_record is None:
            # Not present in the old snapshot either -- nothing to carry.
            return records
        new_conversations = records.get("conversations", ())
        new_ids = {
            self._conversation_record_id(record, index)
            for index, record in enumerate(new_conversations)
        }
        if selected_id in new_ids:
            # Still present in the incoming snapshot -- no carry-over needed.
            return records
        merged = dict(records)
        merged["conversations"] = (carried_record, *new_conversations)
        return merged

    def _apply_local_source_snapshot(
        self,
        records: dict[str, tuple[Mapping[str, Any], ...]],
        counts: dict[str, int],
        total_known: dict[str, bool],
        lookup_error: str | None = None,
        recovery_state: DestinationRecoveryState | None = None,
        study_counts: dict[str, int | None] | None = None,
    ) -> None:
        records = self._carry_selected_conversation_into_snapshot(records)
        self._local_source_records = records
        self._local_source_counts = counts
        self._local_source_total_known = total_known
        self._library_lookup_error = lookup_error
        self._library_lookup_recovery_state = recovery_state
        self._library_study_counts = (
            study_counts
            if study_counts is not None
            else {"study_decks": None, "flashcards_due": None, "quizzes": None}
        )
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
            {"study_decks": None, "flashcards_due": None, "quizzes": None},
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
            logger.opt(exception=True).warning("Failed to fetch exact local notes count; using sample count.")
            return None
        return result if isinstance(result, int) else None

    async def _study_count_or_none(self, count_callable: Any, label: str, **kwargs: Any) -> int | None:
        """Fetch a decorative Create-rail count, degrading quietly on failure.

        Runs inside the same ``asyncio.gather`` as the local source snapshot
        fetch (see ``_list_local_source_snapshot``). Unlike the three browse
        sources (notes/media/conversations), study/quiz counts are purely
        decorative rail badges: the underlying scope-service methods can
        raise ``PolicyDeniedError`` (via ``_enforce_policy``) or a plain
        ``ValueError`` (e.g. local backend unavailable) depending on the
        runtime, and none of that should ever surface as Library error copy
        or fail the snapshot fetch -- it just degrades to ``None``, which
        the rail renders as an uncounted row.

        Args:
            count_callable: The bound count method to invoke (e.g.
                ``study_scope_service.count_decks``).
            label: Human-readable label for the debug log on failure.
            **kwargs: Forwarded to ``count_callable``.

        Returns:
            The exact count, or ``None`` if the call failed or returned
            something other than an ``int``.
        """
        # SQLite ``:memory:`` connections are thread-local (``threading.local``
        # on ``CharactersRAGDB``) -- only the thread that created the DB has
        # the migrated schema. Forcing this single COUNT(*) query onto a
        # worker thread would open a brand-new, unmigrated in-memory
        # connection and fail. Same guard as
        # ``LibraryLocalRagSearchService._search_conversations`` and
        # ``_fetch_library_conversation_by_id``.
        chachanotes_db = getattr(self.app_instance, "chachanotes_db", None)
        isolate_in_worker = not bool(getattr(chachanotes_db, "is_memory_db", False))
        try:
            result = await self._run_library_service_call(
                count_callable, isolate_in_worker=isolate_in_worker, **kwargs
            )
        except Exception:
            logger.opt(exception=True).debug(f"Failed to fetch {label} count for Library create rail.")
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
        dict[str, int | None],
    ]:
        notes_service = getattr(self.app_instance, "notes_scope_service", None)
        media_service = getattr(self.app_instance, "media_reading_scope_service", None)
        conversation_service = getattr(self.app_instance, "chat_conversation_scope_service", None)
        study_service = getattr(self.app_instance, "study_scope_service", None)
        quiz_service = getattr(self.app_instance, "study_quiz_scope_service", None)
        list_notes = getattr(notes_service, "list_notes", None)
        list_media = getattr(media_service, "list_media_items", None)
        list_conversations = getattr(conversation_service, "list_conversations", None)
        count_notes = getattr(notes_service, "count_notes", None)
        count_notes_available = callable(count_notes)
        count_decks = getattr(study_service, "count_decks", None)
        count_due_flashcards = getattr(study_service, "count_due_flashcards", None)
        count_quizzes = getattr(quiz_service, "count_quizzes", None)
        notes_user_id = getattr(self.app_instance, "notes_user_id", None) or "default_user"

        empty_records: dict[str, tuple[Mapping[str, Any], ...]] = {
            "notes": (),
            "media": (),
            "conversations": (),
        }
        empty_counts = {"notes": 0, "media": 0, "conversations": 0}
        empty_total_known = {"notes": True, "media": True, "conversations": True}
        empty_study_counts: dict[str, int | None] = {
            "study_decks": None,
            "flashcards_due": None,
            "quizzes": None,
        }
        if not all(callable(call) for call in (list_notes, list_media, list_conversations)):
            return (
                empty_records,
                empty_counts,
                empty_total_known,
                LIBRARY_SERVICE_UNAVAILABLE_COPY,
                None,
                empty_study_counts,
            )

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
                # "all" spans global- and workspace-scoped conversations:
                # Console chats persisted inside a workspace session are
                # workspace-scoped and would be invisible (and uncounted)
                # under the service's default 'global' scope.
                scope_type="all",
                limit=LIBRARY_SOURCE_PAGE_SIZES["conversations"],
                offset=0,
                isolate_in_worker=True,
            ),
        ]
        # Optional decorative/exact counts are appended (and unpacked back)
        # by key so this stays simple as the number of optional seams
        # grows -- see ``_notes_true_count_or_none``/``_study_count_or_none``
        # for the per-count degrade-to-None contract.
        optional_calls: list[tuple[str, Any]] = []
        if count_notes_available:
            optional_calls.append(
                (
                    "notes_true_count",
                    self._notes_true_count_or_none(count_notes, scope="local_note", user_id=notes_user_id),
                )
            )
        if callable(count_decks):
            optional_calls.append(
                ("study_decks", self._study_count_or_none(count_decks, "study decks"))
            )
        if callable(count_due_flashcards):
            optional_calls.append(
                ("flashcards_due", self._study_count_or_none(count_due_flashcards, "flashcards due"))
            )
        if callable(count_quizzes):
            optional_calls.append(("quizzes", self._study_count_or_none(count_quizzes, "quizzes")))
        gathered_calls.extend(call for _, call in optional_calls)

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
                empty_study_counts,
            )
        except Exception:
            logger.opt(exception=True).warning(
                "Failed to load local Library source snapshot.",
            )
            return (
                empty_records,
                empty_counts,
                empty_total_known,
                LIBRARY_SERVICE_ERROR_COPY,
                None,
                empty_study_counts,
            )

        notes_result, media_result, conversation_result, *optional_results = gathered_results
        optional_values = dict(zip((key for key, _ in optional_calls), optional_results))

        notes_true_count = optional_values.get("notes_true_count")
        study_counts: dict[str, int | None] = {
            "study_decks": optional_values.get("study_decks"),
            "flashcards_due": optional_values.get("flashcards_due"),
            "quizzes": optional_values.get("quizzes"),
        }

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
            study_counts,
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
        a selected browser row -- the media viewer's "Use in Console" action
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

    def _study_handoff_copy(self, kind: str) -> dict[str, str]:
        mode = LIBRARY_STUDY_HANDOFF_MODES.get(
            kind,
            LIBRARY_STUDY_HANDOFF_MODES["study"],
        )
        titles = self._source_study_handoff_titles()
        has_context = self._has_source_study_context()
        action_label = mode["action_label"]
        if has_context and titles:
            context_copy = _library_carries_forward_line(titles)
        elif has_context:
            context_copy = "Carries forward: Library source snapshot (titles unavailable)"
        else:
            # No Library sources at all: the carries-forward line is omitted
            # entirely rather than stating the negative (the blocked
            # "recovery" line below already carries that signal).
            context_copy = ""
        return {
            "header": mode["header"],
            "action_label": action_label,
            "button_label": mode["button_label"],
            "purpose": mode["purpose"],
            "context": context_copy,
            "owner": LIBRARY_STUDY_HANDOFF_OWNERSHIP_COPY,
            "recovery": (
                "Source snapshot is ready."
                if has_context
                else (
                    "Import sources or create notes first, or open "
                    f"{action_label} globally without Library context."
                )
            ),
        }

    def _library_rag_panel_state(self) -> LibraryRagPanelState:
        # B2: explicit selection is every real source type NOT toggled off;
        # `LibraryRagScopeState.from_source_counts` intersects this with
        # actual availability, so a deselected-but-empty source can't
        # falsely count as "selected".
        selected_source_types = tuple(
            source_type
            for source_type in LIBRARY_RAG_SCOPE_TOGGLE_SOURCE_TYPES
            if source_type not in self._library_rag_scope_deselected
        )
        return LibraryRagPanelState.from_values(
            source_counts={
                "notes": self._local_source_counts.get("notes", 0),
                "media": self._local_source_counts.get("media", 0),
                "conversations": self._local_source_counts.get("conversations", 0),
                "workspaces": 0,
                "collections": 0,
            },
            query=self._library_rag_query,
            mode=self._library_rag_mode,
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
            # Deliberately always ready: the UI path never imports torch (or
            # any other optional Search/RAG dependency), and the retrieval
            # service double-guards missing runtimes/indexes at call time.
            dependencies_ready=True,
            index_ready=True,
            provider_ready=(getattr(self.app_instance, "_rag_service", None) is not None),
            selected_source_types=selected_source_types,
            history=self._library_search_history,
            history_collapsed=self._library_rag_history_collapsed,
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

    def _study_handoff_detail_widget(self, kind: str) -> Vertical:
        """Build the handoff canvas body: purpose, carried-forward sources,
        ownership, snapshot readiness, and the Open action -- five elements,
        down from the seven-line original (UX wave D1: no duplicated mode/
        purpose lines, no "Primary action:" line, no WIP roadmap callout).
        """
        copy = self._study_handoff_copy(kind)
        # D2: the ds-recovery-callout warning treatment is for the blocked
        # (no local sources) state only; ready renders as a plain Static.
        recovery_kwargs: dict[str, str] = (
            {} if self._has_local_sources() else {"classes": "ds-recovery-callout is-blocked"}
        )
        action_button_id = {
            "study": "library-open-study",
            "flashcards": "library-open-flashcards",
            "quizzes": "library-open-quizzes",
        }.get(kind, "library-open-study")
        handoff_toolbar = Horizontal(
            Button(
                copy["button_label"],
                id=action_button_id,
                # D3: the Open action is the canvas's primary control.
                classes="library-canvas-action console-action-primary",
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
        children: list[Static | Horizontal] = [
            Static(
                copy["purpose"],
                id="library-study-handoff-purpose",
            ),
        ]
        if copy["context"]:
            # D1: omitted entirely (not "No ... will be carried forward.")
            # when there is no Library source snapshot at all.
            children.append(
                Static(
                    copy["context"],
                    id="library-study-handoff-context",
                )
            )
        children.append(
            Static(
                copy["owner"],
                id="library-study-handoff-owner",
            )
        )
        children.append(
            Static(
                copy["recovery"],
                id="library-study-handoff-recovery",
                **recovery_kwargs,
            )
        )
        children.append(handoff_toolbar)
        return Vertical(
            *children,
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
                query=self._library_rag_query,
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
                elif shell.canvas_kind == "search":
                    yield LibrarySearchRagPanel(
                        self._library_rag_panel_state(),
                        id="library-search-rag-panel",
                    )
                elif shell.canvas_kind == "collections":
                    yield LibraryCollectionsPanel(
                        self._library_collections_panel_state(),
                        name_value=self._library_collection_name_input,
                        description_value=self._library_collection_description_input,
                        delete_pending=bool(self._library_collection_pending_delete_id),
                        id="library-collections-panel",
                    )
                elif shell.canvas_kind == "ingest-media":
                    yield LibraryIngestCanvas(
                        self._build_library_ingest_state(),
                        id="library-ingest-canvas",
                    )
                elif shell.canvas_kind == "export":
                    yield LibraryExportCanvas(
                        self._build_library_export_state(),
                        id="library-export-canvas",
                    )
                elif shell.canvas_kind == "handoff":
                    # Study/Flashcards/Quizzes rows (L3b Task 8): a first-class
                    # canvas kind of their own, sourced entirely from
                    # LIBRARY_STUDY_HANDOFF_MODES. UX wave D1 collapsed this
                    # to a single header (the row's own title -- "Flashcards"
                    # / "Study decks" / "Quizzes") plus the consolidated
                    # handoff detail widget below; the header no longer
                    # restates the mode name a second, differently-worded way
                    # (formerly "Flashcards mode" + a duplicated description
                    # + next-action line).
                    handoff_copy = LIBRARY_STUDY_HANDOFF_MODES.get(
                        shell.canvas_target, LIBRARY_STUDY_HANDOFF_MODES["study"]
                    )
                    yield Static(
                        handoff_copy["header"],
                        id="library-active-mode-title",
                        classes="destination-section",
                    )
                    yield self._study_handoff_detail_widget(shell.canvas_target)
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
            study_decks_count=self._library_study_counts.get("study_decks"),
            flashcards_due_count=self._library_study_counts.get("flashcards_due"),
            quizzes_count=self._library_study_counts.get("quizzes"),
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
            logger.opt(exception=True).warning(f"Failed to load Library media detail for {media_id!r}.")
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
        if (
            self._library_media_detail is None
            and media_id == self._selected_media_id
            and self._library_media_view == "viewer"
        ):
            # The record backing an opened item vanished between the click
            # and this fetch resolving (e.g. deleted elsewhere, or a stale
            # Search/RAG "Open" result) -- mirror the equivalent
            # "Conversation is unavailable." notify _open_library_item_by_id
            # gives its conversations branch, and fall back to the list
            # view instead of leaving an empty/stuck viewer.
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify("Media item is unavailable.", severity="warning")
            self._library_media_view = "list"
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
            logger.opt(exception=True).warning(
                f"Failed to load Library media highlights for {media_id!r}.")
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
            logger.opt(exception=True).warning(f"Failed to load Library note detail for {note_id!r}.")
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
            logger.opt(exception=True).warning(
                f"Failed to load keywords for Library note {note_id!r}.")
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
        # Typed-but-uncommitted folder text is visit-scoped, like the
        # status/activity above: re-entering the panel re-reads the
        # committed config value.
        self._library_notes_sync_folder_text = None

    def _reset_library_ingest_transient_state(self) -> None:
        """Clear the ingest canvas's form to defaults on rail re-entry.

        Called on every ``_select_library_rail_row`` switch (mirrors
        ``_reset_library_notes_sync_transient_state``'s placement) so a
        stale in-progress form from a previous Ingest visit never
        reappears when the user comes back to the canvas. The job queue
        itself is registry-owned and untouched by this reset -- only the
        local form echo resets.
        """
        self._library_ingest_form = LibraryIngestFormState()

    # ----- Export canvas -------------------------------------------------

    @staticmethod
    def _default_library_export_form() -> dict[str, Any]:
        """Build a fresh export form echo: today's stamped name, nothing else set."""
        return {
            "name": default_export_name(),
            "description": "",
            "quality": DEFAULT_MEDIA_QUALITY,
            "destination": "",
            "destination_exists": False,
        }

    def _reset_library_export_transient_state(self, scope: ExportScope | None = None) -> None:
        """Clear the export canvas's scope/counts/form to defaults on entry.

        Called from both entry points into the export canvas -- the rail
        row's own ``_select_library_rail_row`` switch (always the default
        Everything ``scope``) and the browse-canvas "Export…" section
        actions (``_open_library_export_canvas``, their own pre-scoped
        ``ExportScope``) -- so neither a stale form from a previous Export
        visit nor a stale scope/counts pairing from a different section
        ever reappears. The name field re-stamps today's local date every
        time (mirrors the ingest form's own from-scratch reset), never
        carrying a previous visit's edited name forward.

        Also invalidates any export run still executing on its own OS
        thread (bumps ``_library_export_run_id``) -- navigating away mid-
        run resets ``running`` to ``False`` for THIS fresh visit, but the
        abandoned worker keeps running regardless (it cannot be preempted
        mid-``asyncio.run``); bumping the token here ensures that worker's
        eventual completion is recognized as stale and cannot stomp
        whatever the user is looking at by the time it lands. See
        ``_library_export_run_id``'s docstring in ``__init__``.

        Args:
            scope: The scope to open the canvas with; defaults to
                ``ExportScope(kind="everything")`` when omitted.
        """
        self._library_export_scope = scope or ExportScope(kind="everything")
        self._library_export_counts = None
        self._library_export_form = self._default_library_export_form()
        self._library_export_running = False
        self._library_export_error = ""
        self._library_export_status = ""
        if self._library_export_cancel_event is not None:
            self._library_export_cancel_event.set()
        self._library_export_run_id += 1

    async def _open_library_export_canvas(self, scope: ExportScope) -> None:
        """Open the export canvas pre-scoped to a browse section's own filter.

        Wired to each browse canvas's "Export…" action (media/
        conversations/notes) -- mirrors ``_select_library_rail_row``'s
        dirty-note-flush discipline for switching canvases, but only
        touches the export-specific state (the rail row's own switch
        already resets everything else on the way past); the caller's
        ``scope`` survives untouched (unlike a plain rail-row switch,
        which always resets to Everything).

        Args:
            scope: The section-specific scope to open the form with (e.g.
                ``ExportScope(kind="media", media_type=...)``).
        """
        if self._library_export_is_server_mode():
            # The section "Export..." actions bypass the rail row's own
            # server-disabled gate, so re-check here (Qodo review): export
            # reads the LOCAL DBs, so running it while the Library is in
            # server runtime mode would package the wrong dataset.
            self.app_instance.notify(
                LIBRARY_EXPORT_SERVER_DISABLED_TOOLTIP, severity="warning"
            )
            return
        await self._flush_library_note_save()
        if self._library_note_autosave_state == "conflict":
            return
        self._library_selected_row_id = LIBRARY_ROW_INGEST_EXPORT
        self._reset_library_export_transient_state(scope)
        self.refresh(recompose=True)
        self._start_library_export_counts_worker()

    def _library_export_is_server_mode(self) -> bool:
        """True when the Library is in server runtime mode.

        Export packages LOCAL content only (it reads the local media /
        ChaChaNotes DBs), so both the rail Export row and the section
        "Export..." actions must refuse to run in server mode.
        """
        runtime_state = getattr(
            getattr(self.app_instance, "runtime_policy", None), "state", None
        )
        active_source = str(
            getattr(runtime_state, "active_source", "local") or "local"
        ).lower()
        return active_source == "server"

    def _resolve_library_export_chachanotes_db(self) -> Any:
        """Return the ChaChaNotes DB handle for export counts.

        Mirrors ``_resolve_library_notes_sync_db``'s exact access path
        (prefer ``app_instance.chachanotes_db``, fall back to
        ``notes_service.db``) -- the same canonical DB-access path this
        screen already uses elsewhere, per the F4 brief's requirement that
        the counts worker reach the DB the same way the rest of the
        screen does.
        """
        notes_service = getattr(self.app_instance, "notes_service", None)
        return getattr(self.app_instance, "chachanotes_db", None) or getattr(
            notes_service, "db", None
        )

    @staticmethod
    def _compute_library_export_counts(
        scope: ExportScope, media_db: Any, chachanotes_db: Any
    ) -> dict[str, int]:
        """Run the full-query, uncapped counts for ``scope`` (never a rendered snapshot).

        A quiet-degrade failure (a missing DB seam, an unexpected DB
        error) reports all-zero counts rather than raising -- the export
        canvas simply shows "Nothing to export in this scope." rather
        than crashing the recompose; the failure is still logged.
        """
        try:
            return count_export_scope(scope, media_db, chachanotes_db)
        except Exception:
            logger.opt(exception=True).warning(
                f"Library export counts failed for scope {scope!r}."
            )
            return {"media": 0, "conversations": 0, "notes": 0}

    def _start_library_export_counts_worker(self) -> None:
        """Kick off the export scope's full-query counts (Task 1's resolver).

        In-memory SQLite connections are thread-local -- only the thread
        that created/migrated the DB has a working connection (same guard
        as ``_fetch_library_conversation_by_id`` elsewhere on this
        screen).
        When either DB is memory-backed (the test suite's fixtures, and
        any other future in-memory deployment), the count runs inline on
        the calling (UI) thread instead of a real worker thread -- a
        worker thread would only ever see a blank, unmigrated connection.
        A real (file-backed) deployment always takes the
        ``group="library_export_counts"`` worker-thread path.
        """
        scope = self._library_export_scope
        media_db = getattr(self.app_instance, "media_db", None)
        chachanotes_db = self._resolve_library_export_chachanotes_db()
        if bool(getattr(media_db, "is_memory_db", False)) or bool(
            getattr(chachanotes_db, "is_memory_db", False)
        ):
            counts = self._compute_library_export_counts(scope, media_db, chachanotes_db)
            self._apply_library_export_counts(scope, counts)
            return
        self._run_library_export_counts_worker(scope, media_db, chachanotes_db)

    @work(thread=True, exclusive=True, group="library_export_counts")
    def _run_library_export_counts_worker(
        self, scope: ExportScope, media_db: Any, chachanotes_db: Any
    ) -> None:
        counts = self._compute_library_export_counts(scope, media_db, chachanotes_db)
        # ``self.app`` (Textual's own running-App property), not
        # ``self.app_instance`` -- ``call_from_thread`` needs the App whose
        # event loop is actually running this screen (see
        # ``_run_library_notes_sync``'s ``progress_callback`` for the full
        # reasoning). Guarded the same way: a shutdown mid-worker must
        # never surface as a crash.
        try:
            self.app.call_from_thread(self._apply_library_export_counts, scope, counts)
        except Exception:
            # A shutdown/detach mid-marshal can raise RuntimeError OR
            # Textual's NoApp (which subclasses Exception, not RuntimeError)
            # -- either way the worker thread must not crash on teardown.
            pass

    def _apply_library_export_counts(self, scope: ExportScope, counts: dict[str, int]) -> None:
        """Marshal a landed counts result onto the export form (UI thread).

        Guards against a stale result from a scope the user has since
        navigated away from (a second "Export…" press, or another rail
        row entirely, before the first counts worker finished) -- dropped
        rather than overwriting fresher (or absent) counts.

        Updates the mounted canvas via targeted DOM surgery, NEVER a
        recompose (mirrors ``handle_library_ingest_path_changed``'s
        targeted-update discipline): the user may be mid-keystroke in the
        name/description ``Input`` when the counts land -- on a large
        library (this feature's whole point) that window is real -- and a
        recompose would destroy and rebuild the ``Input``, silently
        dropping keyboard focus (the typed text survives via the form
        dict; focus does not). Only three widgets can change when counts
        land, and all three are always-mounted on the export canvas: the
        scope line's text, the empty-scope helper's text/visibility
        (display-toggled rather than conditionally composed in
        ``LibraryExportCanvas.compose`` for exactly this reason), and the
        Export button's disabled gate. The media/quality rows CANNOT
        change here: their visibility (``show_media_fields``) derives
        purely from ``scope.kind``, which is pinned before the worker
        ever starts -- never from counts.

        Args:
            scope: The scope the landed ``counts`` were computed for.
            counts: The landed counts (keys "media"/"conversations"/"notes").
        """
        if scope != self._library_export_scope:
            return
        self._library_export_counts = counts
        if not self.is_mounted or self._library_selected_row_id != LIBRARY_ROW_INGEST_EXPORT:
            return
        state = self._build_library_export_state()
        try:
            canvas = self.query_one("#library-export-canvas", LibraryExportCanvas)
        except (NoMatches, QueryError):
            # Canvas not mounted (yet) -- the state fields above are set,
            # so whatever composes it next renders the landed counts.
            return
        # Keep the canvas's own state snapshot in step with what the
        # targeted updates below render, so any later widget-level
        # recompose can never resurrect the stale "Counting…" state.
        canvas.state = state
        try:
            self.query_one("#library-export-scope-line", Static).update(state.scope_line)
            empty_line = self.query_one("#library-export-empty-line", Static)
            empty_line.update(state.empty_scope_line)
            empty_line.display = bool(state.empty_scope_line)
            self.query_one("#library-export-submit", Button).disabled = (
                not state.export_enabled
            )
        except (NoMatches, QueryError):
            pass

    def _build_library_export_state(self) -> LibraryExportFormState:
        """Build the export canvas's full display state from screen fields."""
        form = self._library_export_form
        return build_library_export_form_state(
            scope=self._library_export_scope,
            counts=self._library_export_counts,
            name=str(form.get("name", "")),
            description=str(form.get("description", "")),
            media_quality=str(form.get("quality", DEFAULT_MEDIA_QUALITY)),
            destination=str(form.get("destination", "")),
            destination_exists=bool(form.get("destination_exists", False)),
            running=self._library_export_running,
            status_line=self._library_export_status,
            error_line=self._library_export_error,
        )

    # ----- Export canvas: execution (Task 3) ------------------------------

    @on(Button.Pressed, "#library-export-submit")
    def handle_library_export_submit(self, event: Button.Pressed) -> None:
        """Validate and kick off the chatbook export worker.

        Re-validates on the UI thread (destination chosen, scope non-empty,
        not already running) rather than trusting the button's ``disabled``
        state alone. A second press while an export is already running is a
        guarded no-op here (``self._library_export_running``) -- on top of
        the button itself being disabled while running and the worker's own
        ``group="library_export"``/``exclusive=True`` single-flight, this is
        belt-and-suspenders against a stale/racing ``Pressed`` event.

        The transition INTO ``running`` is the one place this feature uses
        a full recompose rather than a targeted update (see
        ``_update_library_export_canvas_after_run``'s docstring for the
        reverse transition's targeted-update discipline): the user's last
        action was clicking this button, not typing, so nothing is
        mid-keystroke -- unlike the counts-landing case Task 2 fixed, or
        the run-completion case below, where the (long-running) wait window
        gives the user time to resume typing in the still-editable name/
        description fields.
        """
        event.stop()
        if self._library_export_running:
            return
        form = self._library_export_form
        destination = str(form.get("destination", "")).strip()
        counts = self._library_export_counts
        total = sum(counts.values()) if counts else 0
        if not destination or total <= 0:
            return
        if self._library_export_is_server_mode():
            # Defense in depth: the rail row and section actions already
            # gate on server mode, but re-check at submit in case the
            # runtime source flipped while the form was open (Qodo review).
            self.app_instance.notify(
                LIBRARY_EXPORT_SERVER_DISABLED_TOOLTIP, severity="warning"
            )
            return
        # Sanitize name/description at the UI boundary before they flow into
        # the export payload, chatbook manifest, and Artifacts registry
        # (Qodo review) -- bound length + strip unsafe content via the shared
        # input_validation helpers, mirroring the media-field path.
        name = self._safe_text(form.get("name", ""), "Chatbook", max_length=200)
        description = self._safe_text(form.get("description", ""), "", max_length=2000)
        media_quality = str(form.get("quality", DEFAULT_MEDIA_QUALITY))
        self._library_export_running = True
        self._library_export_error = ""
        self._library_export_status = f"Exporting… ({total} items)"
        self._library_export_run_id += 1
        run_id = self._library_export_run_id
        self._library_export_cancel_event = threading.Event()
        cancel_event = self._library_export_cancel_event
        self.refresh(recompose=True)
        self._start_library_export_worker(
            run_id=run_id,
            scope=self._library_export_scope,
            name=name,
            description=description,
            media_quality=media_quality,
            destination=destination,
            cancel_event=cancel_event,
        )

    @on(Button.Pressed, "#library-export-cancel")
    def handle_library_export_cancel(self, event: "Button.Pressed") -> None:
        """Request cancellation of the in-flight export.

        Sets the worker's cancel Event (idempotent) and flips the status line to
        "Cancelling…". Deliberately does NOT bump _library_export_run_id: the run
        is still the current, visible one until the worker reports back with the
        cancelled outcome (see _apply_library_export_cancelled).
        """
        if not self._library_export_running:
            return
        if event is not None:
            event.stop()
        event_obj = self._library_export_cancel_event
        if event_obj is not None:
            event_obj.set()
        self._library_export_status = "Cancelling…"
        self._refresh_library_export_status_line()

    def _start_library_export_worker(
        self,
        *,
        run_id: int,
        scope: ExportScope,
        name: str,
        description: str,
        media_quality: str,
        destination: str,
        cancel_event: threading.Event,
    ) -> None:
        """Resolve selections (memory-DB-safe), then dispatch the real export worker.

        Mirrors ``_start_library_export_counts_worker``'s memory-vs-file-
        backed DB branch for the id-resolution step specifically: a genuine
        OS worker thread only ever sees a blank, unmigrated connection for
        an in-memory-backed DB (``threading.local``), so when either DB is
        memory-backed, ``resolve_export_selections`` -- a pure, synchronous
        DB read with no ``asyncio`` involvement -- runs inline on the
        calling (UI) thread first, exactly like the counts worker's own
        inline fallback, and the resolved ids are handed to the worker.

        Unlike the counts worker, the ``@work(thread=True)`` dispatch below
        is never skipped: ``asyncio.run(service.export_chatbook(...))``
        would raise ("cannot be called from a running event loop") if
        invoked directly on the UI thread, which already owns Textual's
        own running event loop. A file-backed deployment defers
        ``resolve_export_selections`` into that same real thread instead
        (``preresolved_selections=None``), avoiding a synchronous full-
        library scan on the UI thread for the common (real, file-backed)
        case.
        """
        media_db = getattr(self.app_instance, "media_db", None)
        chachanotes_db = self._resolve_library_export_chachanotes_db()
        preresolved_selections: dict[ContentType, list[str]] | None = None
        if bool(getattr(media_db, "is_memory_db", False)) or bool(
            getattr(chachanotes_db, "is_memory_db", False)
        ):
            try:
                preresolved_selections = resolve_export_selections(
                    scope, media_db, chachanotes_db
                )
            except Exception as exc:
                logger.opt(exception=True).warning(
                    f"Library export selection resolution failed for scope {scope!r}."
                )
                self._apply_library_export_failure(
                    run_id, f"Failed to resolve export selections: {exc}"
                )
                return
        self._run_library_export_worker(
            run_id=run_id,
            scope=scope,
            name=name,
            description=description,
            media_quality=media_quality,
            destination=destination,
            media_db=media_db,
            chachanotes_db=chachanotes_db,
            preresolved_selections=preresolved_selections,
            cancel_event=cancel_event,
        )

    @staticmethod
    def _build_library_export_payload(
        *,
        name: str,
        description: str,
        selections: Mapping[ContentType, list[str]],
        destination: str,
        media_quality: str,
    ) -> dict[str, Any]:
        """Build the ``local_chatbook_service.export_chatbook`` request payload.

        ``include_media`` is spec-critical (F4 plan Global Constraints):
        it MUST be ``True`` whenever ``ContentType.MEDIA`` is present in
        ``selections`` -- ``ChatbookCreator`` silently skips all media
        content otherwise, even when media ids ARE present in
        ``content_selections``. Since ``resolve_export_selections`` omits
        a ``ContentType`` key entirely when that source resolves zero ids
        (see its docstring), keying off simple membership is automatically
        correct for every scope, including an "everything" scope whose
        library happens to have no media at all.
        """
        return {
            "name": name,
            "description": description,
            "content_selections": dict(selections),
            "output_path": destination,
            "media_quality": media_quality,
            "include_media": ContentType.MEDIA in selections,
        }

    @staticmethod
    def _run_library_export_via_service(
        service: Any,
        payload: dict[str, Any],
        *,
        name: str,
        description: str,
        progress_callback=None,
        cancel_check=None,
    ) -> dict[str, Any]:
        """Execute one export through ``service``, synchronously: zip first, registry only on success.

        Runs both of ``service``'s async-signature/sync-body methods
        through ``asyncio.run`` -- they never touch the app's own event
        loop, so this is only ever safe to call from a genuine OS thread
        (never the UI thread, which already owns a running loop). Exposed
        as its own (non-``@work``) static method so tests can call it
        directly with a fake ``service`` and assert call ordering /
        the include_media invariant without booting a real thread.

        ``create_chatbook`` (the registry record) is attempted ONLY when
        ``export_chatbook`` reports ``success`` -- the F4 plan's Global
        Constraints' "zip first, registry record only on success". A
        registry-recording failure AFTER a successful zip does not flip
        the overall outcome to failure (the artifact genuinely exists on
        disk; only the bookkeeping failed) -- ``registry_recorded``
        reports that separately for callers/tests that care.

        Returns a plain dict: ``success``, ``message``, ``path``,
        ``dependency_info``, ``registry_recorded``.
        """
        try:
            export_result = asyncio.run(service.export_chatbook(
                payload, progress_callback=progress_callback, cancel_check=cancel_check,
            ))
        except Exception as exc:
            logger.opt(exception=True).warning("Library export service call failed.")
            return {
                "success": False,
                "message": f"Export failed: {exc}",
                "path": "",
                "dependency_info": {},
                "registry_recorded": False,
                "cancelled": False,
            }

        if not export_result.get("success"):
            return {
                "success": False,
                "message": str(export_result.get("message") or "Export failed."),
                "path": export_result.get("path") or payload.get("output_path", ""),
                "dependency_info": export_result.get("dependency_info") or {},
                "registry_recorded": False,
                "cancelled": bool(export_result.get("cancelled", False)),
            }

        output_path = export_result.get("path") or payload.get("output_path", "")
        dependency_info = export_result.get("dependency_info") or {}
        registry_recorded = False
        try:
            asyncio.run(
                service.create_chatbook(
                    name=name,
                    description=description,
                    file_path=output_path,
                    tags=["library-export"],
                )
            )
            registry_recorded = True
        except Exception:
            logger.opt(exception=True).warning(
                f"Library export succeeded but registry recording failed for {output_path!r}."
            )

        return {
            "success": True,
            "message": export_result.get("message") or "",
            "path": output_path,
            "dependency_info": dependency_info,
            "registry_recorded": registry_recorded,
            "cancelled": False,
        }

    @work(thread=True, exclusive=True, group="library_export")
    def _run_library_export_worker(
        self,
        *,
        run_id: int,
        scope: ExportScope,
        name: str,
        description: str,
        media_quality: str,
        destination: str,
        media_db: Any,
        chachanotes_db: Any,
        preresolved_selections: dict[ContentType, list[str]] | None,
        cancel_event: threading.Event | None,
    ) -> None:
        if preresolved_selections is not None:
            selections = preresolved_selections
        else:
            try:
                selections = resolve_export_selections(scope, media_db, chachanotes_db)
            except Exception as exc:
                logger.opt(exception=True).warning(
                    f"Library export selection resolution failed for scope {scope!r}."
                )
                self._marshal_library_export_failure(
                    run_id, f"Failed to resolve export selections: {exc}"
                )
                return

        service = getattr(self.app_instance, "local_chatbook_service", None)
        if service is None:
            self._marshal_library_export_failure(
                run_id, "Chatbook export service unavailable."
            )
            return

        payload = self._build_library_export_payload(
            name=name,
            description=description,
            selections=selections,
            destination=destination,
            media_quality=media_quality,
        )
        throttle = ExportProgressThrottle()

        def _progress_cb(evt) -> None:
            try:
                if not throttle.should_emit(evt.phase, evt.current, evt.total, time.monotonic()):
                    return
                self.app.call_from_thread(
                    self._apply_library_export_progress, run_id, evt.phase, evt.current, evt.total,
                )
            except Exception:
                # NoApp/shutdown mid-marshal must not crash the worker.
                pass

        outcome = self._run_library_export_via_service(
            service, payload, name=name, description=description,
            progress_callback=_progress_cb,
            cancel_check=(cancel_event.is_set if cancel_event is not None else None),
        )
        if outcome.get("cancelled"):
            self._marshal_library_export_cancelled(run_id)
        elif outcome["success"]:
            self._marshal_library_export_success(
                run_id,
                outcome["path"],
                outcome["dependency_info"],
                bool(outcome["registry_recorded"]),
                outcome["message"],
            )
        else:
            self._marshal_library_export_failure(run_id, outcome["message"])

    def _marshal_library_export_success(
        self,
        run_id: int,
        path: str,
        dependency_info: Any,
        registry_recorded: bool,
        message: str = "",
    ) -> None:
        """Marshal a successful run onto the UI thread (called from the worker)."""
        try:
            self.app.call_from_thread(
                self._apply_library_export_success,
                run_id,
                path,
                dependency_info,
                registry_recorded,
                message,
            )
        except Exception:
            # A shutdown/detach mid-marshal can raise RuntimeError OR
            # Textual's NoApp (which subclasses Exception, not RuntimeError)
            # -- either way the worker thread must not crash on teardown.
            pass

    def _marshal_library_export_failure(self, run_id: int, message: str) -> None:
        """Marshal a failed run onto the UI thread (called from the worker)."""
        try:
            self.app.call_from_thread(
                self._apply_library_export_failure, run_id, message
            )
        except Exception:
            # A shutdown/detach mid-marshal can raise RuntimeError OR
            # Textual's NoApp (which subclasses Exception, not RuntimeError)
            # -- either way the worker thread must not crash on teardown.
            pass

    def _marshal_library_export_cancelled(self, run_id: int) -> None:
        """Marshal a cancelled run onto the UI thread (called from the worker)."""
        try:
            self.app.call_from_thread(self._apply_library_export_cancelled, run_id)
        except Exception:
            # A shutdown/detach mid-marshal can raise RuntimeError OR
            # Textual's NoApp (which subclasses Exception, not RuntimeError)
            # -- either way the worker thread must not crash on teardown.
            pass

    def _apply_library_export_cancelled(self, run_id: int) -> None:
        """UI-thread completion for a cancelled run: clear running, show cancelled, return to form."""
        if run_id != self._library_export_run_id:
            return
        self._library_export_running = False
        self._library_export_status = "Export cancelled."
        self._library_export_error = ""
        self._update_library_export_canvas_after_run()

    @staticmethod
    def _build_library_export_success_message(
        path: Any, dependency_info: Any, creator_message: Any = ""
    ) -> str:
        """Build the success notification text.

        Three pieces, in order:

        1. The destination path (always present), ``escape_markup``'d:
           Textual notifications render Rich console markup, so a
           user-chosen path containing ``[...]`` (legal in filenames on
           any platform) would otherwise mis-render or raise in the
           markup parser.
        2. The creator's own ``outcome["message"]`` detail (task-158):
           ``ChatbookCreator.create_chatbook`` returns a message carrying
           its own counts (e.g. missing-dependency warnings) that was
           previously discarded entirely by the caller. Its redundant
           ``"Chatbook created successfully at <path>"`` prefix -- the
           path is already the primary notify line above -- is stripped
           so only the actual detail remains; an unrecognized message
           shape (e.g. a different service implementation) is kept
           verbatim rather than guessed at.
        3. The ``dependency_info.get("auto_included")`` count suffix (the
           character ids ``ChatbookCreator`` pulled in automatically as
           conversation dependencies) -- BUT only when the creator detail
           above does not already state it. ``create_chatbook`` already
           puts an ``"Auto-included N character dependencies"`` clause
           into its own message (that clause and ``auto_included`` derive
           from the same ``self.auto_included_characters`` state), so
           emitting the suffix on top of a detail that carries that clause
           would restate the identical fact twice. The suffix therefore
           only fires when the auto-included count would otherwise go
           unstated (e.g. an empty creator message, or a creator message
           whose only detail is a missing-dependency warning).
        """
        message = f"Exported chatbook to {escape_markup(str(path))}"

        detail = str(creator_message or "").strip()
        known_prefix = f"Chatbook created successfully at {path}"
        if detail.startswith(known_prefix):
            detail = detail[len(known_prefix):].strip(" .")
        if detail:
            message += f": {escape_markup(detail)}"

        auto_included = (
            dependency_info.get("auto_included")
            if isinstance(dependency_info, dict)
            else None
        )
        # De-dup: skip the suffix when the surfaced detail already states
        # the auto-included count (see point 3 above).
        if auto_included and "auto-included" not in detail.lower():
            try:
                count = len(auto_included)
            except TypeError:
                count = auto_included
            message += f" ({count} characters auto-included)"

        return message

    def _apply_library_export_success(
        self,
        run_id: int,
        path: str,
        dependency_info: Any,
        registry_recorded: bool,
        message: str = "",
    ) -> None:
        """UI-thread completion: notify, clear running/error, update the form.

        See ``_build_library_export_success_message`` for how the
        notification text itself (path + creator detail + auto-included
        count) is assembled.

        ``registry_recorded=False`` (the zip succeeded but the
        ``create_chatbook`` registry step failed -- see
        ``_run_library_export_via_service``) fires a SECOND, warning-
        severity notification: without it the export silently never
        appears under Artifacts/Home and the user has no way to know why.
        It fires alongside the primary notification, BEFORE the staleness
        guard, deliberately: both report persistent facts about what
        actually happened on disk/in the registry, independent of which
        canvas the user is now looking at -- and the warning matters MOST
        for a superseded run, since a user who already navigated away
        would otherwise never learn the artifact is missing from
        Artifacts.

        ``run_id`` is compared against the live ``_library_export_run_id``
        BEFORE any state/DOM mutation: an export genuinely finished, so the
        notifications always fire, but a run the user has since navigated
        away from (see ``_library_export_run_id``'s docstring) must not
        stomp ``_library_export_running``/``_error``/``_status`` or the
        canvas DOM out from under whatever the user is now looking at.
        """
        notify_message = self._build_library_export_success_message(
            path, dependency_info, message
        )
        notify = getattr(self.app_instance, "notify", None)
        if callable(notify):
            notify(notify_message, severity="information")
            if not registry_recorded:
                notify(
                    "Export saved, but couldn't be registered — it won't "
                    "appear under Artifacts.",
                    severity="warning",
                )
        if run_id != self._library_export_run_id:
            return
        self._library_export_running = False
        self._library_export_error = ""
        self._library_export_status = ""
        self._update_library_export_canvas_after_run()

    def _apply_library_export_failure(self, run_id: int, message: str) -> None:
        """UI-thread completion: render the escaped error, clear running, re-enable Export.

        See ``_apply_library_export_success``'s docstring for the
        ``run_id`` staleness guard -- a superseded run's failure is
        dropped silently here (no error line to render it into, since the
        canvas may now belong to a different scope/visit entirely) rather
        than notified, since surfacing a failure banner for a run the user
        has already navigated away from and possibly re-run successfully
        would be actively misleading.
        """
        if run_id != self._library_export_run_id:
            logger.info(
                f"Library export run {run_id} failed after being superseded "
                f"(current run {self._library_export_run_id}): {message}"
            )
            return
        self._library_export_running = False
        self._library_export_status = ""
        self._library_export_error = escape_markup(str(message))
        self._update_library_export_canvas_after_run()

    def _apply_library_export_progress(
        self, run_id: int, phase: str, current: int, total: int
    ) -> None:
        """UI-thread progress tick: update the status line in place if this run is current."""
        if run_id != self._library_export_run_id or not self._library_export_running:
            return
        self._library_export_status = format_export_progress_line(phase, current, total)
        self._refresh_library_export_status_line()

    def _refresh_library_export_status_line(self) -> None:
        """Update only the #library-export-status-line widget (no recompose)."""
        if not self.is_mounted or self._library_selected_row_id != LIBRARY_ROW_INGEST_EXPORT:
            return
        try:
            widget = self.query_one("#library-export-status-line", Static)
            widget.update(self._library_export_status)
            widget.display = bool(self._library_export_status)
        except (NoMatches, QueryError):
            pass

    def _update_library_export_canvas_after_run(self) -> None:
        """Targeted DOM update once an export run finishes (success or failure).

        Mirrors ``_apply_library_export_counts``'s targeted-update
        discipline (Task 2's fix, commit 7793257e): the transition OUT of
        ``running`` must not recompose. Unlike the Export-press transition
        INTO ``running`` (a recompose is acceptable there -- see
        ``handle_library_export_submit``'s docstring), the running window
        itself can be long enough for the user to resume typing in the
        name/description ``Input`` while waiting (nothing disables those
        fields during ``running``) -- a recompose on completion would
        destroy and rebuild that ``Input`` out from under them, silently
        dropping keyboard focus. Only the status line, the error line, and
        the Export button's disabled gate can change here; both lines are
        unconditionally mounted by ``LibraryExportCanvas.compose`` (display-
        toggled, never conditionally yielded) specifically so this in-place
        update always finds them, mirroring the empty-scope helper's own
        always-mounted precedent from Task 2's fix.
        """
        if not self.is_mounted or self._library_selected_row_id != LIBRARY_ROW_INGEST_EXPORT:
            return
        state = self._build_library_export_state()
        try:
            canvas = self.query_one("#library-export-canvas", LibraryExportCanvas)
        except (NoMatches, QueryError):
            return
        canvas.state = state
        try:
            status_widget = self.query_one("#library-export-status-line", Static)
            status_widget.update(state.status_line)
            status_widget.display = bool(state.status_line)
            error_widget = self.query_one("#library-export-error-line", Static)
            error_widget.update(state.error_line)
            error_widget.display = bool(state.error_line)
            self.query_one("#library-export-submit", Button).disabled = (
                not state.export_enabled
            )
            self.query_one("#library-export-cancel", Button).display = bool(state.running)
        except (NoMatches, QueryError):
            pass

    def _library_ingest_registry(self) -> Any:
        """Return the app's ingest job registry, or ``None`` when absent."""
        return getattr(self.app_instance, "library_ingest_jobs", None)

    def _handle_library_ingest_registry_changed(self) -> None:
        """Registry listener: live-recompose the ingest canvas + poke the
        source snapshot when a job finishes (Task 5).

        Registered against ``self.app_instance.library_ingest_jobs`` in
        ``on_mount``, removed in ``on_unmount``. Per the registry's own
        contract (``LibraryIngestJobRegistry._notify_listeners``), this
        fires synchronously on the UI thread after every successful
        ``submit``/``mark_parsing``/``mark_writing``/``mark_done``/
        ``mark_failed``/``requeue`` -- from two different call shapes:

        - **Synchronously inside a message handler.** The "Start ingest"
          and "Retry" button handlers call ``submit_library_ingest_job``/
          ``retry_library_ingest_job`` directly, which mutate the registry
          (firing this listener) *before* the handler's own trailing
          ``self.refresh(recompose=True)`` runs.
        - **Marshaled from a background thread**, via ``call_from_thread``
          for ``mark_parsing``/``mark_writing`` (the F3 parse-pool
          coordinator, itself invoked from a pool callback thread) and
          ``mark_done``/``mark_failed`` (the writer's worker thread) --
          these land outside any message handler, as their own turn of the
          UI event loop.

        Both shapes are safe to handle with a plain, synchronous
        ``self.refresh(recompose=True)`` call (no ``call_after_refresh``
        indirection needed): ``Widget.refresh(recompose=True)`` never
        recomposes inline -- it only sets ``_recompose_required = True``
        and schedules the actual (async) ``_check_recompose`` via
        ``call_next``, which runs on a later turn of the event loop. That
        makes calling it redundant, or from inside another handler that
        will also call it, harmless: the flag is idempotent and the
        second scheduled check becomes a no-op once the first has already
        cleared it. (Verified by reading
        ``textual.widget.Widget.refresh``/``_check_recompose`` -- Textual
        8.2.7.)

        Behavior:

        - Recomposes the canvas ONLY when the ingest canvas is the
          currently selected rail row -- a job transition must never yank
          a user looking at a different canvas away from it.
        - Independently of the canvas recompose, pokes
          ``_refresh_local_source_snapshot()`` (which updates the rail's
          ``Media (N)`` count) whenever the registry's done-job count has
          grown since this screen last checked -- deduped via
          ``_library_ingest_last_done_count`` so a running/failed
          transition (or a second notification for the same completed
          job) never re-triggers the snapshot fetch. This fires
          regardless of which canvas is selected, since the rail is
          always visible.
        - A no-op when the screen isn't mounted -- belt-and-braces
          alongside ``on_unmount``'s removal (see that method's
          docstring for why removal can't simply happen earlier, e.g. on
          suspend). Note ``self.is_mounted`` never flips back to
          ``False`` after removal in this Textual version -- it only
          guards a callback that somehow fires before this screen's very
          first mount -- so ``on_unmount``'s ``remove_listener`` call is
          what actually prevents post-teardown notifications, not this
          guard.
        """
        if not self.is_mounted:
            return
        if self._library_selected_row_id == LIBRARY_ROW_INGEST_MEDIA:
            self.refresh(recompose=True)
        registry = self._library_ingest_registry()
        counts_fn = getattr(registry, "counts", None)
        done_count = counts_fn().get("done", 0) if callable(counts_fn) else 0
        if done_count != self._library_ingest_last_done_count:
            grew = done_count > self._library_ingest_last_done_count
            self._library_ingest_last_done_count = done_count
            if grew:
                self._refresh_local_source_snapshot()

    def _build_library_ingest_state(self) -> LibraryIngestCanvasState:
        """Build the ingest canvas's full display state from the live registry + form.

        Reads directly from ``self.app_instance.library_ingest_jobs`` via
        quiet-degrade ``getattr`` (never assuming the seam exists) rather
        than caching a screen-owned copy, so every render -- the canvas
        compose -- sees the registry's current truth, including
        transitions a live-update listener applies between renders. The
        Open in Library/Retry/Dismiss row-action handlers do NOT go
        through this method -- they resolve their target job directly by
        ``job_id`` from ``registry.jobs()`` (see
        ``_library_ingest_job_by_id``), never by re-deriving and indexing
        into a row snapshot, so an async queue mutation between render and
        click can never mis-target a different job (PR #591 review, F1).
        """
        registry = self._library_ingest_registry()
        jobs_fn = getattr(registry, "jobs", None)
        jobs = jobs_fn() if callable(jobs_fn) else ()
        runtime_state = getattr(
            getattr(self.app_instance, "runtime_policy", None), "state", None
        )
        runtime_source = str(getattr(runtime_state, "active_source", "local") or "local")
        return build_library_ingest_state(
            jobs,
            form=self._library_ingest_form,
            runtime_source=runtime_source,
            media_db_available=getattr(self.app_instance, "media_db", None) is not None,
            registry_available=registry is not None,
        )

    def _ensure_library_notes_sync_config_loaded(self) -> None:
        """Seed sync direction/conflict/auto-sync from config on first entry.

        Idempotent: only reads config once per screen lifetime
        (``_library_notes_sync_config_loaded`` guards re-entry), so
        in-session cycling/toggling is never clobbered by a later sync-mode
        re-entry re-reading stale config.

        A stale persisted conflict value not in ``SYNC_CONFLICTS`` (e.g. an
        old config still holding ``"ask"``, which this panel no longer
        offers) coerces to ``"newer_wins"``; likewise an unrecognized
        direction coerces to ``"bidirectional"``. This guarantees ``"ask"``
        can neither render nor reach the sync engine from this panel.
        """
        if self._library_notes_sync_config_loaded:
            return
        self._library_notes_sync_config_loaded = True
        direction = str(
            get_cli_setting("notes", "sync_direction", "bidirectional")
            or "bidirectional"
        )
        self._library_notes_sync_direction = (
            direction if direction in SYNC_DIRECTIONS else "bidirectional"
        )
        conflict = str(
            get_cli_setting("notes", "sync_conflict_resolution", "newer_wins")
            or "newer_wins"
        )
        self._library_notes_sync_conflict = (
            conflict if conflict in SYNC_CONFLICTS else "newer_wins"
        )
        self._library_notes_sync_auto = bool(get_cli_setting("notes", "auto_sync", False))
        if self._library_notes_sync_auto:
            self._arm_library_notes_auto_sync_timer()

    def _library_notes_sync_folder(self) -> str:
        """Return the sync folder as text (unexpanded).

        Prefers the folder box's live typed text (``Input.Changed`` keeps it
        in screen state without touching disk) so recomposes and Sync now
        always see what the user sees; falls back to the committed config
        value when the box hasn't been edited this panel visit.
        """
        if self._library_notes_sync_folder_text is not None:
            return self._library_notes_sync_folder_text
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
            running=self._library_notes_sync_running,
        )

    def _resolve_library_notes_sync_db(self) -> Any:
        """Resolve the per-user ChaChaNotes DB the sync service writes to.

        Mirrors the retired standalone Notes screen's sync-pane resolution
        EXACTLY: prefer the app's ``chachanotes_db``, falling back to the
        notes service's own ``db`` attribute when that is unset.
        """
        notes_service = getattr(self.app_instance, "notes_service", None)
        return getattr(self.app_instance, "chachanotes_db", None) or getattr(
            notes_service, "db", None
        )

    def _arm_library_notes_auto_sync_timer(self) -> None:
        """Start the 300s auto-sync repeating timer if not already running.

        Scoped to this Library screen instance's lifetime (like the retired
        standalone Notes screen's sync-pane timer) -- it is never persisted
        or resumed across screen instances; only the ``auto_sync`` boolean
        preference is persisted, and is re-armed on the next sync-mode
        entry via ``_ensure_library_notes_sync_config_loaded``.
        """
        if self._library_notes_auto_sync_timer is not None:
            return
        self._library_notes_auto_sync_timer = self.set_interval(
            AUTO_SYNC_INTERVAL_SECONDS,
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
            logger.opt(exception=True).warning(f"Library note save failed for {note_id!r}.")
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
            # Patch the cached notes-list snapshot too (title + a fresh
            # last_modified), not just the detail mirror: the list view is
            # rendered from these records on the next Back-to-list, and
            # without this it kept showing the pre-save title, stale
            # relative age, and stale Newest ordering until a full
            # snapshot refetch landed (2026-07 UAT finding).
            saved_stamp = datetime.now(timezone.utc).isoformat()
            self._local_source_records["notes"] = patch_note_records_after_save(
                self._local_source_records.get("notes", ()),
                note_id,
                title=title,
                modified_at=saved_stamp,
            )
            if self._library_notes_filter_records is not None:
                self._library_notes_filter_records = list(
                    patch_note_records_after_save(
                        self._library_notes_filter_records,
                        note_id,
                        title=title,
                        modified_at=saved_stamp,
                    )
                )
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
                    logger.opt(exception=True).debug(
                        "In-flight note-save worker errored while flushing; continuing.",
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
            logger.opt(exception=True).warning(
                f"Failed to reload Library note {note_id!r} after a save conflict.",
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
            logger.opt(exception=True).warning(
                f"Failed to overwrite Library note {note_id!r} after a save conflict.",
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

        Mirrors the retired standalone Notes screen's export dialog flow --
        a ``FileSave`` prompt pre-filled with a sanitized default filename,
        whose callback writes the built export content once a path is
        chosen. The export reads the *live* editor widgets (via
        ``_read_library_note_editor_fields``), never the DB, so unlike
        Save there is nothing to flush first.

        Note: the real ``Third_Party.textual_fspicker.FileSave`` dialog
        only accepts ``location``/``title``/``default_file`` (not the
        ``default_filename``/``context`` kwargs the retired screen passed
        it, which would have raised ``TypeError`` if that path ever
        actually ran) -- this uses the dialog's real constructor shape.

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
        already uses for user-chosen save/output paths (e.g. this screen's
        note import path, ``settings_screen``'s
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
            logger.opt(exception=True).warning(
                f"Error exporting Library note {note_id!r} to '{validated_path}'.")
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
        directly the way the retired standalone Notes screen's copy action
        did -- that makes this testable via a recorded fake the way the
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
            logger.opt(exception=True).warning(f"Failed to copy Library note {note_id!r} to clipboard.")
            if callable(notify):
                notify(f"Error copying note: {type(exc).__name__}", severity="error")
            return
        if callable(notify):
            notify("Note copied to clipboard as markdown!", severity="information")

    def _selected_library_note_handoff_payload(self) -> ChatHandoffPayload | None:
        """Build the Console handoff payload for the open Library note.

        Mirrors ``_selected_media_handoff_payload``: reads the currently
        open note's live (possibly unsaved) editor fields rather than a
        list-row selection, matching the local-note handoff shape the
        retired standalone Notes screen staged.

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
        open_chat_with_handoff(payload, action_label="Use in Console")

    @on(Button.Pressed, "#library-note-use-in-console")
    def handle_library_note_use_in_console(self, event: Button.Pressed) -> None:
        """Hand the open note off to Console as chat context.

        Args:
            event: Button press event emitted by the editor's
                "Use in Console" action.
        """
        event.stop()
        self._open_selected_library_note_handoff()

    def _library_rail_preferences(self):
        """Read persisted Library rail section preferences, defensively.

        (C4) Same restart-persistence gap as
        ``_load_library_search_history``: ``self.app_instance.app_config``
        (from ``load_settings()``) can come back without a ``library``
        section at all even when ``config.toml`` has persisted
        ``[library.rail_state]`` on disk -- so a freshly started app would
        otherwise always reopen every rail section at its hardcoded
        default instead of the user's last-chosen open/collapsed state.
        Falls back to a live ``get_cli_setting("library.rail_state")`` read
        of the CLI config file when ``app_config`` doesn't already carry a
        usable ``sections`` dict; ``app_config`` wins whenever it does.
        """
        app_config = getattr(self.app_instance, "app_config", None)
        raw = None
        if isinstance(app_config, dict):
            library_config = app_config.get("library")
            if isinstance(library_config, dict):
                rail_state = library_config.get("rail_state")
                if isinstance(rail_state, dict):
                    raw = rail_state.get("sections")
        if not isinstance(raw, dict):
            try:
                # Dotted 1-arg form, same shape as
                # `_load_library_search_history`'s CLI fallback:
                # `get_cli_setting("library.rail_state")` returns
                # `config["library"]["rail_state"]` (the rail_state
                # sub-dict), not the "sections" dict directly.
                cli_rail_state = get_cli_setting("library.rail_state")
            except Exception:
                cli_rail_state = None
            if isinstance(cli_rail_state, dict):
                raw = cli_rail_state.get("sections")
        return coerce_library_rail_preferences(raw)

    def _load_library_search_history(self) -> tuple[str, ...]:
        """Read persisted Library Search/RAG query history, defensively.

        Two sources are consulted, in order:

        1. `self.app_instance.app_config["library"]["search"]["history"]` --
           the in-memory config dict. This is the primary source: it is what
           pilots seed directly, and it reflects any history recorded during
           the current session (`_record_library_search_history` mutates it
           in place, alongside persisting to disk).
        2. `get_cli_setting("library.search")` -- a live re-read of
           `config.toml` via `load_cli_config_and_ensure_existence()`. This
           fallback exists because `app.app_config` comes from
           `load_settings()`, whose merged output does NOT reliably surface
           the `[library.search]` TOML table (it can come back empty even
           when `config.toml` has history on disk) -- so a freshly started
           app would otherwise always see empty history despite having
           persisted some in a prior session. `get_cli_setting` reads the
           CLI config file directly and does carry the value.

        Only source (1) is used when it already yields a list, so pilots
        that seed `app_config` directly stay authoritative and never touch
        disk. Missing keys or a malformed shape from either source quietly
        fall back to no history; entries are coerced to trimmed strings and
        capped to the same shape `update_search_history` produces (<= 10
        entries, <= 200 chars each).

        `config.toml` is user-editable, so each entry is also run through
        `_safe_text` (control-character stripping, dangerous-pattern
        removal, length validation) before it ever becomes a history
        `Button` label -- belt-and-suspenders alongside the markup escape
        `library_rag_history_children` applies at render time.
        """
        app_config = getattr(self.app_instance, "app_config", None)
        raw = None
        if isinstance(app_config, dict):
            library_config = app_config.get("library")
            if isinstance(library_config, dict):
                search_config = library_config.get("search")
                if isinstance(search_config, dict):
                    raw = search_config.get("history")
        if not isinstance(raw, list):
            try:
                # Dotted 1-arg form: get_cli_setting("library.search") splits
                # on the first '.' into section="library", key="search" and
                # returns config["library"]["search"] (the search sub-dict,
                # not the history list) -- deliberately NOT the 3-arg
                # ("library.search", "history", default) form, which treats
                # "library.search" as a single literal top-level section key
                # and never matches the nested TOML table.
                search_config = get_cli_setting("library.search")
            except Exception:
                search_config = None
            if isinstance(search_config, dict):
                raw = search_config.get("history")
        if not isinstance(raw, list):
            return ()
        entries = tuple(
            sanitized
            for entry in raw
            if (
                sanitized := self._safe_text(
                    entry, max_length=LIBRARY_SEARCH_HISTORY_ENTRY_MAX_CHARS
                )
            )
        )
        return entries[:LIBRARY_SEARCH_HISTORY_LIMIT]

    def _record_library_search_history(self, query: str) -> None:
        """Update in-memory and persisted Library Search/RAG query history."""
        self._library_search_history = update_search_history(
            self._library_search_history, query
        )
        self._persist_library_search_history(list(self._library_search_history))

    def _persist_library_search_history(self, history_list: list[str]) -> None:
        """Write `history_list` into the in-memory config and to disk.

        Shared by `_record_library_search_history` (append a new query) and
        `clear_library_search_history` (D1: empty the list) so both funnel
        through one persistence path.
        """
        app_config = getattr(self.app_instance, "app_config", None)
        if isinstance(app_config, dict):
            library_config = app_config.get("library")
            if not isinstance(library_config, dict):
                library_config = {}
                app_config["library"] = library_config
            search_config = library_config.get("search")
            if not isinstance(search_config, dict):
                search_config = {}
                library_config["search"] = search_config
            search_config["history"] = history_list
        self._save_library_search_history(history_list)

    @work(thread=True)
    def _save_library_search_history(self, history: list[str]) -> None:
        """Persist Library Search/RAG query history without blocking the UI thread."""
        try:
            save_setting_to_cli_config("library.search", "history", history)
        except Exception:
            pass

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
        """Dispatch a Library rail row press: navigate, browse, or open a canvas."""
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
            await self._select_library_rail_row(row_id)
            return
        if target_kind == "handoff":
            # Study/Flashcards/Quizzes rows (L3b Task 8): resolves to the
            # handoff canvas.
            await self._select_library_rail_row(row_id)
            return
        # Unknown target kind: select the row and recompose from selection.
        await self._select_library_rail_row(row_id)

    async def _select_library_rail_row(self, row_id: str) -> None:
        """Apply a rail-row selection and recompose the canvas from it.

        Shared by the rail-row press handler and in-canvas mode shortcuts so
        that the single source of selection truth (``_library_selected_row_id``)
        always drives the recomposed canvas.

        A dirty note edit is flushed first (awaited) so leaving via the rail
        never silently discards unsaved text; any unsaved edit surviving the
        flush aborts the row switch entirely.
        """
        await self._flush_library_note_save()
        if self._library_note_dirty:
            return
        self._library_selected_row_id = row_id
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
        self._reset_library_ingest_transient_state()
        # Always resets to the Everything scope (a plain rail-row press,
        # unlike a browse-canvas "Export…" action, never carries a
        # section-specific filter) -- see
        # ``_reset_library_export_transient_state``'s docstring.
        self._reset_library_export_transient_state()
        self._invalidate_library_workspace_depth_state()
        if (
            self._library_selected_row_id == LIBRARY_ROW_BROWSE_COLLECTIONS
            and not self._library_collections_loaded
        ):
            # First Collections entry must load the snapshot the retired chip
            # flow ran; _sync_collections_panel recomposes once records arrive.
            await self._sync_collections_panel(refresh_snapshot=True)
            return
        self.refresh(recompose=True)
        if self._library_selected_row_id == LIBRARY_ROW_INGEST_EXPORT:
            self._start_library_export_counts_worker()

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
        self._open_library_media_viewer(media_id)

    @on(Button.Pressed, "#library-media-open-viewer")
    def handle_library_media_open_viewer(self, event: Button.Pressed) -> None:
        """Open the browse summary's selected media item in the in-Library viewer.

        The browse canvas preview's primary action. Stays entirely inside
        Library (unlike the full viewer's "Open in Media manager" escape
        hatch, which navigates to the legacy Media screen) -- hence the
        "Open in viewer" label (2026-07 UAT relabel).

        Args:
            event: Button press event emitted by the "Open in viewer" action.
        """
        event.stop()
        self._open_library_media_viewer(self._selected_media_id)

    def _open_library_media_viewer(self, media_id: str) -> None:
        """Switch the media canvas to the in-canvas viewer for ``media_id``.

        Shared by media-row presses and the browse summary's "Open in
        viewer" action: resets per-item viewer state, kicks the async
        detail fetch, and recomposes into the viewer's loading line.

        Args:
            media_id: The media item to open; an empty id still switches to
                the viewer without kicking a fetch (mirrors the previous
                row-press behavior for a row missing its ``media_id``).
        """
        media_id = str(media_id or "")
        if media_id:
            self._selected_media_id = media_id
        self._library_selected_row_id = LIBRARY_ROW_BROWSE_MEDIA
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
            logger.opt(exception=True).warning("Library notes filter failed.")
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

        Mirrors the retired standalone Notes screen's import dialog flow
        exactly (the working ``FileOpen`` reference -- unlike ``FileSave``,
        whose constructor only accepts ``location``/``title``/``default_file``,
        ``FileOpen`` here is invoked the same simple ``title=``-only way the
        standalone screen already relied on). The callback resolves the
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
        this screen. The file read is offloaded to a thread (mirroring the
        retired standalone Notes screen's import) and is memory-bounded by a
        pre-read ``st_size`` guard: UTF-8 chars are at most 4 bytes, so any
        file over ``4 * LIBRARY_NOTE_CONTENT_MAX_CHARS`` bytes is guaranteed
        over the char cap and is rejected without reading it at all (no
        false rejections: a file that passes could still fail the exact
        char-level check after decoding, which stays in place).

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
            logger.opt(exception=True).warning(f"Rejected Library note import path {selected_path!r}.")
            self._notify_library_note_create_warning("Could not import that file.")
            return

        try:
            file_size = note_path.stat().st_size
        except OSError:
            logger.opt(exception=True).warning(f"Could not stat Library note import file '{note_path}'.")
            self._notify_library_note_create_warning("Could not import that file.")
            return
        if file_size > LIBRARY_NOTE_CONTENT_MAX_CHARS * 4:
            # See docstring: st_size > 4x the char cap proves the decoded
            # text exceeds the cap (UTF-8 is <= 4 bytes/char), so reject
            # BEFORE reading -- the char check below would otherwise slurp
            # an arbitrarily large file into memory first.
            self._notify_library_note_create_warning("Could not import that file.")
            return

        try:
            file_content = await asyncio.to_thread(
                note_path.read_text, encoding="utf-8", errors="strict"
            )
        except (OSError, UnicodeDecodeError):
            logger.opt(exception=True).warning(f"Could not read Library note import file '{note_path}'.")
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
        if self._library_note_dirty:
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
        """Track the sync folder text as the user edits it (state only).

        Deliberately does NOT persist: writing the TOML config here meant a
        full config rewrite + cache reload per keystroke. The typed value is
        committed by ``handle_library_notes_sync_folder_submitted`` (Enter),
        ``_apply_library_notes_sync_folder`` (Browse…), or a validated
        ``handle_library_notes_sync_run``.

        Args:
            event: Input change event emitted by the sync folder box.
        """
        event.stop()
        self._library_notes_sync_folder_text = event.value

    @on(Input.Submitted, "#library-notes-sync-folder")
    def handle_library_notes_sync_folder_submitted(self, event: Input.Submitted) -> None:
        """Commit the typed sync folder to config on Enter.

        Args:
            event: Input submit event emitted by the sync folder box.
        """
        event.stop()
        self._library_notes_sync_folder_text = event.value
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
        self._library_notes_sync_folder_text = str(path)
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
        the same scope the retired standalone Notes screen's sync-pane
        timer had -- not persisted/resumed across screen instances; only
        the boolean
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
        # A validated run is a commit point for a typed-but-unsubmitted
        # folder (see handle_library_notes_sync_folder_changed): the folder
        # a run actually used is always the one that persists.
        save_setting_to_cli_config("notes", "sync_directory", folder_value)
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

        Builds a fresh ``NotesSyncService`` per run (mirroring the retired
        standalone Notes screen's sync-pane setup -- see
        ``_resolve_library_notes_sync_db``)
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
            changes = len(results.created_notes) + len(results.updated_notes) + (
                len(results.created_files) + len(results.updated_files)
            )
            conflicts = len(results.conflicts)
            # The done line counts CHANGES the run made, not files scanned
            # (a no-op run over N files is "done · no changes", never
            # "done · 0 files" -- the PR reviewer's misleading-count flag).
            self._library_notes_sync_status = sync_status_line(
                "done", processed=changes, conflicts=conflicts
            )
            summary_parts = []
            if results.created_notes:
                summary_parts.append(f"{count_noun(len(results.created_notes), 'note')} created")
            if results.updated_notes:
                summary_parts.append(f"{count_noun(len(results.updated_notes), 'note')} updated")
            if results.created_files:
                summary_parts.append(f"{count_noun(len(results.created_files), 'file')} created")
            if results.updated_files:
                summary_parts.append(f"{count_noun(len(results.updated_files), 'file')} updated")
            summary = ", ".join(summary_parts) if summary_parts else "No changes"
            self._library_notes_sync_activity = append_activity(
                self._library_notes_sync_activity, f"Sync complete: {summary}"
            )
            if conflicts:
                # Honest-copy fix: this used to promise a "review" surface
                # that doesn't exist in this panel. State the resolved
                # policy instead -- what actually happened to the conflict.
                self._library_notes_sync_activity = append_activity(
                    self._library_notes_sync_activity,
                    f"{count_noun(conflicts, 'conflict')} resolved "
                    f"({sync_conflict_label(resolution.value)})",
                )
            if results.errors:
                self._library_notes_sync_activity = append_activity(
                    self._library_notes_sync_activity,
                    f"{count_noun(len(results.errors), 'error')} during sync",
                )
        except Exception as exc:
            logger.opt(exception=True).error(f"Library notes sync failed (folder={folder}): {exc}")
            self._library_notes_sync_status = sync_status_line("failed", error=str(exc))
            self._library_notes_sync_activity = append_activity(
                self._library_notes_sync_activity, f"Sync failed: {exc}"
            )
        finally:
            self._library_notes_sync_running = False
            if self._library_notes_view == "sync" and self.is_mounted:
                self.refresh(recompose=True)

    # ----- Ingest canvas -----------------------------------------------

    @on(Input.Changed, "#library-ingest-path")
    async def handle_library_ingest_path_changed(self, event: Input.Changed) -> None:
        """Track the ingest path text as the user types it (state only).

        Also live-updates the Start button's disabled state, AND the
        blank-path quiet line (L3b AB wave, A4), via targeted DOM surgery
        (mirroring ``update_library_collection_name_input``) rather than a
        full canvas recompose, so typing never disturbs the Input's cursor
        position. The quiet line ``Static`` is always mounted by
        ``LibraryIngestCanvas.compose`` with a fixed one-row height, so
        this handler only updates its text in place -- never mounts or
        removes it -- keeping the Start button's position stable across
        gate-state changes (2026-07 UAT: mount/remove shifted the button
        ~2 rows on every valid/blank transition).

        Args:
            event: Input change event emitted by the path field.
        """
        event.stop()
        self._library_ingest_form.path = event.value
        try:
            start_button = self.query_one("#library-ingest-start", Button)
        except (NoMatches, QueryError):
            return
        new_state = self._build_library_ingest_state()
        start_button.disabled = not new_state.start_enabled
        try:
            quiet_line = self.query_one("#library-ingest-start-quiet-line", Static)
        except (NoMatches, QueryError):
            return
        quiet_line.update(new_state.start_quiet_line)

    @on(Input.Changed, "#library-ingest-title")
    def handle_library_ingest_title_changed(self, event: Input.Changed) -> None:
        """Track the ingest title text as the user types it (state only)."""
        event.stop()
        self._library_ingest_form.title = event.value

    @on(Input.Changed, "#library-ingest-author")
    def handle_library_ingest_author_changed(self, event: Input.Changed) -> None:
        """Track the ingest author text as the user types it (state only)."""
        event.stop()
        self._library_ingest_form.author = event.value

    @on(Input.Changed, "#library-ingest-keywords")
    def handle_library_ingest_keywords_changed(self, event: Input.Changed) -> None:
        """Track the ingest keywords text as the user types it (state only)."""
        event.stop()
        self._library_ingest_form.keywords = event.value

    @on(Input.Changed, "#library-ingest-chunk-size")
    def handle_library_ingest_chunk_size_changed(self, event: Input.Changed) -> None:
        """Track the chunk-size text as typed (display-echo only).

        Parsed and clamped to ``[100, 5000]`` only at submit time (see
        ``clamp_chunk_size``) -- never here.
        """
        event.stop()
        self._library_ingest_form.chunk_size = event.value

    @on(Button.Pressed, "#library-ingest-browse")
    def handle_library_ingest_browse(self, event: Button.Pressed) -> None:
        """Push a ``FileOpen`` dialog to pick a local file to ingest.

        Mirrors ``handle_library_notes_import``'s dialog flow exactly (the
        working ``FileOpen`` reference, invoked the same simple
        ``title=``-only way). The callback writes the chosen path straight
        into the form and recomposes so the Input and the Start button's
        gate both reflect it immediately; validation still runs at Start
        so a path typed by hand (not picked via this dialog) is caught
        too.

        Args:
            event: Button press event emitted by the "Browse…" action.
        """
        event.stop()

        async def browse_callback(selected_path: Path | None) -> None:
            if selected_path is None:
                return
            self._library_ingest_form.path = str(selected_path)
            self.refresh(recompose=True)

        self.app.push_screen(
            FileOpen(title="Import Media"),
            browse_callback,
        )

    @on(Collapsible.Toggled, "#library-ingest-advanced")
    def sync_library_ingest_advanced_open(self, event: Collapsible.Toggled) -> None:
        """Track manual expand/collapse so recomposes preserve the user's choice.

        Mirrors ``sync_library_rag_history_collapsed`` exactly (see that
        handler's docstring for the full reasoning): ``Collapsible``'s
        ``collapsed`` reactive is defined with ``init=False``, so
        ``_watch_collapsed`` -- and therefore this ``Toggled`` message --
        fires only on an actual *change* of the reactive, never merely from
        ``compose()`` constructing a fresh ``Collapsible(collapsed=...)``
        with a value that happens to equal the reactive's own default.
        Concretely: the widget always passes
        ``collapsed=not state.form.advanced_open``, and the reactive's
        default is ``True`` -- so a compose only posts a spurious ``Toggled``
        when it constructs the panel already-expanded (``advanced_open`` is
        ``True``, i.e. ``collapsed=False`` differs from the ``True``
        default), which immediately reasserts the same ``True`` this
        handler already holds. Every recompose this handler must survive
        (the analyze/chunk toggles, a registry-listener-driven job
        transition) is triggered by something OTHER than a manual header
        click, so this handler is never invoked by them -- only a real
        user click (or a future programmatic ``collapsible.collapsed =``
        assignment) fires it, exactly like the history panel's precedent.
        """
        event.stop()
        self._library_ingest_form.advanced_open = not event.collapsible.collapsed

    @on(Button.Pressed, "#library-ingest-analyze-toggle")
    def handle_library_ingest_analyze_toggle(self, event: Button.Pressed) -> None:
        """Flip the "Analyze after ingest" form toggle.

        Args:
            event: Button press event emitted by the analyze toggle.
        """
        event.stop()
        self._library_ingest_form.analyze = not self._library_ingest_form.analyze
        self.refresh(recompose=True)

    @on(Button.Pressed, "#library-ingest-chunk-toggle")
    def handle_library_ingest_chunk_toggle(self, event: Button.Pressed) -> None:
        """Flip the "Chunk content" form toggle.

        Args:
            event: Button press event emitted by the chunk toggle.
        """
        event.stop()
        self._library_ingest_form.chunk = not self._library_ingest_form.chunk
        self.refresh(recompose=True)

    def _notify_library_ingest_warning(self, message: str) -> None:
        notify = getattr(self.app_instance, "notify", None)
        if callable(notify):
            notify(message, severity="warning")

    @on(Button.Pressed, "#library-ingest-start")
    def handle_library_ingest_start(self, event: Button.Pressed) -> None:
        """Validate the form and submit a new Library ingest job.

        Args:
            event: Button press event emitted by the "Start ingest" action.
        """
        event.stop()
        self._submit_library_ingest_form()

    @on(Input.Submitted, "#library-ingest-path")
    def handle_library_ingest_path_submitted(self, event: Input.Submitted) -> None:
        """Submit the ingest form when Enter is pressed in the path field.

        Mirrors the Start ingest button exactly, but only when the Start
        gate is open (``start_enabled``) -- Enter on a blank path (or with
        the registry/DB unavailable) stays quiet instead of nagging, since
        the always-visible gate line already explains the blocker
        (2026-07 UAT: Enter in a valid path field previously did nothing).

        Args:
            event: Input submission event emitted by the path field.
        """
        event.stop()
        if not self._build_library_ingest_state().start_enabled:
            return
        self._submit_library_ingest_form()

    def _submit_library_ingest_form(self) -> None:
        """Validate the ingest form and submit a new Library ingest job.

        Shared by the Start ingest button and Enter in the path field. An
        invalid/missing path is a quiet warning notice, matching every
        other Library form failure path in this screen; a missing
        ``submit_library_ingest_job`` seam (registry absent) gets the same
        treatment. On success, the path AND title fields clear (L3b AB
        wave, A1) -- title is per-file, so it must not silently reapply to
        the next file in a batch -- while author/keywords/advanced options
        persist, since those are batch metadata a user submitting several
        files in a row shouldn't have to retype for every submission.
        """
        form = self._library_ingest_form
        raw_path = form.path.strip()
        if not raw_path:
            self._notify_library_ingest_warning("Please choose a file to ingest.")
            return
        try:
            validated_path = validate_path_simple(
                Path(raw_path).expanduser(), require_exists=True
            )
        except ValueError:
            logger.opt(exception=True).warning(f"Rejected Library ingest path {raw_path!r}.")
            self._notify_library_ingest_warning("Could not find that file.")
            return
        submit = getattr(self.app_instance, "submit_library_ingest_job", None)
        if not callable(submit):
            self._notify_library_ingest_warning(INGEST_UNAVAILABLE_COPY)
            return
        submit(
            source_path=str(validated_path),
            title=self._safe_text(form.title, max_length=300),
            author=self._safe_text(form.author, max_length=200),
            keywords=parse_keywords(form.keywords),
            perform_analysis=form.analyze,
            chunk_enabled=form.chunk,
            chunk_size=clamp_chunk_size(form.chunk_size),
        )
        form.path = ""
        form.title = ""
        self.refresh(recompose=True)

    @staticmethod
    def _ingest_job_id_from_button(button_id: str | None, prefix: str) -> str | None:
        """Parse a job id from a Library-ingest row-action button id.

        Row-action buttons (``library-ingest-open-{job_id}``/``-retry-``/
        ``-dismiss-``) are keyed by the registry-assigned ``job_id``, NOT
        by row index (PR #591 review, F1): the queue mutates
        asynchronously between a render and a click (runner completions,
        retry-supersede, new submissions), so re-deriving a fresh row
        snapshot and indexing into it at click time can silently resolve
        to a DIFFERENT job than the one the user actually pressed. A
        prefix-strip is exact regardless of how the queue has shifted
        since the button was rendered.

        Args:
            button_id: The pressed button's ``id``.
            prefix: The button-id prefix to strip (e.g.
                ``"library-ingest-open-"``).

        Returns:
            The job id, or ``None`` when ``button_id`` is missing or
            doesn't carry the expected prefix (defensive only -- every
            real row-action button always does).
        """
        if not button_id or not button_id.startswith(prefix):
            return None
        job_id = button_id[len(prefix):]
        return job_id or None

    def _library_ingest_job_by_id(self, job_id: str) -> LibraryIngestJob | None:
        """Resolve a job by id from the live registry snapshot, or ``None``.

        Reads ``registry.jobs()`` fresh on every call (never a cached row
        list) so a click always resolves against the queue's current
        truth, including any transition that landed between render and
        click.
        """
        registry = self._library_ingest_registry()
        jobs_fn = getattr(registry, "jobs", None)
        jobs = jobs_fn() if callable(jobs_fn) else ()
        return next((job for job in jobs if job.job_id == job_id), None)

    @on(Button.Pressed, ".library-ingest-open")
    async def handle_library_ingest_open(self, event: Button.Pressed) -> None:
        """Open a done ingest job's resulting media item in the Library viewer.

        Args:
            event: Button press event emitted by an "Open in Library" row action.
        """
        event.stop()
        job_id = self._ingest_job_id_from_button(event.button.id, "library-ingest-open-")
        if job_id is None:
            return
        job = self._library_ingest_job_by_id(job_id)
        if job is None or job.media_id is None:
            return
        await self._open_library_item_by_id("media", str(job.media_id))

    @on(Button.Pressed, ".library-ingest-retry")
    def handle_library_ingest_retry(self, event: Button.Pressed) -> None:
        """Requeue a failed ingest job.

        Args:
            event: Button press event emitted by a "Retry" row action.
        """
        event.stop()
        job_id = self._ingest_job_id_from_button(event.button.id, "library-ingest-retry-")
        if job_id is None:
            return
        retry = getattr(self.app_instance, "retry_library_ingest_job", None)
        if callable(retry):
            # ``retry_library_ingest_job``/``LibraryIngestJobRegistry.requeue``
            # are already id-based and validate state themselves (FAILED,
            # not already superseded/dismissed) -- a stale or now-wrong-state
            # job id is a safe no-op, not a mis-targeted retry.
            retry(job_id)
        self.refresh(recompose=True)

    @on(Button.Pressed, ".library-ingest-dismiss")
    def handle_library_ingest_dismiss(self, event: Button.Pressed) -> None:
        """Dismiss a failed ingest job row (L3b AB wave, B2).

        A thin wrapper over ``LibraryIngestJobRegistry.dismiss`` -- valid
        only for a ``FAILED`` row; a quiet no-op (mirrors every other
        Library seam-absent path in this screen) when the registry itself
        is unavailable. The registry's own listener
        (``_handle_library_ingest_registry_changed``) already recomposes
        on a successful dismiss; the trailing ``refresh(recompose=True)``
        here is redundant-but-harmless belt-and-braces, matching
        ``handle_library_ingest_retry``.

        Args:
            event: Button press event emitted by a "Dismiss" row action.
        """
        event.stop()
        job_id = self._ingest_job_id_from_button(event.button.id, "library-ingest-dismiss-")
        if job_id is None:
            return
        registry = self._library_ingest_registry()
        dismiss = getattr(registry, "dismiss", None)
        if callable(dismiss):
            # Same id-based no-op safety as retry above -- ``dismiss`` only
            # ever acts on a currently-FAILED, not-yet-hidden job_id.
            dismiss(job_id)
        self.refresh(recompose=True)

    @on(Button.Pressed, "#library-ingest-clear-finished")
    def handle_library_ingest_clear_finished(self, event: Button.Pressed) -> None:
        """Clear every done+failed ingest job in one shot (L3b AB wave, B2).

        A thin wrapper over ``LibraryIngestJobRegistry.clear_finished``; a
        quiet no-op when the registry itself is unavailable (matching
        ``handle_library_ingest_dismiss``/``handle_library_ingest_retry``).

        Args:
            event: Button press event emitted by the "Clear finished" action.
        """
        event.stop()
        registry = self._library_ingest_registry()
        clear_finished = getattr(registry, "clear_finished", None)
        if callable(clear_finished):
            clear_finished()
        self.refresh(recompose=True)

    # ----- Export canvas: section entry points --------------------------

    @on(Button.Pressed, "#library-media-export")
    async def handle_library_media_export(self, event: Button.Pressed) -> None:
        """Open the export canvas scoped to the media list's current type filter.

        Args:
            event: Button press event emitted by the media canvas's
                "Export…" action.
        """
        event.stop()
        await self._open_library_export_canvas(
            ExportScope(kind="media", media_type=self._library_media_type_filter)
        )

    @on(Button.Pressed, "#library-conversations-export")
    async def handle_library_conversations_export(self, event: Button.Pressed) -> None:
        """Open the export canvas scoped to Conversations.

        Args:
            event: Button press event emitted by the conversations
                canvas's "Export…" action.
        """
        event.stop()
        await self._open_library_export_canvas(ExportScope(kind="conversations"))

    @on(Button.Pressed, "#library-notes-export")
    async def handle_library_notes_export(self, event: Button.Pressed) -> None:
        """Open the export canvas scoped to Notes.

        Args:
            event: Button press event emitted by the notes list canvas's
                "Export…" action.
        """
        event.stop()
        await self._open_library_export_canvas(ExportScope(kind="notes"))

    # ----- Export canvas: form fields ------------------------------------

    @on(Input.Changed, "#library-export-name")
    def handle_library_export_name_changed(self, event: Input.Changed) -> None:
        """Track the export name text as the user types it (state only)."""
        event.stop()
        self._library_export_form["name"] = event.value

    @on(Input.Changed, "#library-export-description")
    def handle_library_export_description_changed(self, event: Input.Changed) -> None:
        """Track the export description text as the user types it (state only)."""
        event.stop()
        self._library_export_form["description"] = event.value

    @on(Button.Pressed, "#library-export-quality")
    def handle_library_export_quality_cycle(self, event: Button.Pressed) -> None:
        """Cycle the media-quality control to its next option.

        Mirrors ``handle_library_media_type_filter_pressed``'s cycle-
        button convention -- see ``next_media_quality``'s docstring for
        why this isn't a ``Select``.

        Args:
            event: Button press event emitted by the quality control.
        """
        event.stop()
        self._library_export_form["quality"] = next_media_quality(
            str(self._library_export_form.get("quality", DEFAULT_MEDIA_QUALITY))
        )
        self.refresh(recompose=True)

    @on(Button.Pressed, "#library-export-destination")
    def handle_library_export_choose_destination(self, event: Button.Pressed) -> None:
        """Push a ``FileSave`` dialog to pick the export's destination path.

        Mirrors ``_export_library_note``'s dialog flow: a sanitized
        default filename derived from the export name field, callback via
        ``call_after_refresh`` so the write-path runs after this handler
        returns. ``FileSave`` DOES have overwrite handling of its own
        (``can_overwrite: bool = True`` -- ``False`` blocks picking an
        existing file outright), but its default imposes no friction, and
        more importantly it can only ever judge the RAW picked path: the
        creator coerces the suffix to ``.zip``, so the path that must be
        confirmed for overwrite is the *normalized* one, which the dialog
        never sees. The form therefore owns overwrite confirmation of the
        normalized path (see ``_apply_library_export_destination``), and
        the dialog is deliberately left at its permissive default rather
        than ``can_overwrite=False`` (which would wrongly block picking
        ``report.zip`` even though the user is knowingly replacing it,
        while failing to block picking ``report`` when ``report.zip``
        exists).

        Args:
            event: Button press event emitted by the "Choose destination…"
                action.
        """
        event.stop()
        raw_name = str(self._library_export_form.get("name", "")).strip() or "chatbook"
        safe_name = "".join(
            char for char in raw_name if char.isalnum() or char in (" ", "-", "_")
        ).rstrip() or "chatbook"
        self.app.push_screen(
            FileSave(
                location=str(Path.home()),
                title="Choose Export Destination",
                default_file=f"{safe_name}.zip",
            ),
            callback=lambda path: self.call_after_refresh(
                self._apply_library_export_destination, path
            ),
        )

    def _apply_library_export_destination(self, selected_path: Path | None) -> None:
        """Validate, ``.zip``-normalize, and apply a ``FileSave``-picked destination.

        Runs the dialog-returned path through ``validate_path_simple``
        (same base-directory-free validator ``_write_library_note_export_file``
        uses for any user-chosen save path) BEFORE normalizing its suffix
        to ``.zip`` -- and normalizes BEFORE checking whether it already
        exists, so the overwrite line the form shows always names the
        actual path that will be written, never the raw picked one (the
        F4 design spec's explicit ordering: "normalized to .zip BEFORE any
        overwrite confirmation").

        Args:
            selected_path: The chosen destination, or ``None`` if the
                dialog was cancelled.
        """
        if not selected_path:
            return
        try:
            validated_path = validate_path_simple(selected_path, require_exists=False)
        except ValueError as exc:
            logger.warning(
                f"Rejected Library export destination {selected_path!r}: {exc}"
            )
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify(f"Rejected export destination: {exc}", severity="warning")
            return
        normalized_path = normalize_export_destination(validated_path)
        self._library_export_form["destination"] = str(normalized_path)
        self._library_export_form["destination_exists"] = normalized_path.exists()
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
        unsaved edit surviving the flush aborts the switch.

        Args:
            event: Button press event emitted by a note row button.
        """
        event.stop()
        await self._flush_library_note_save()
        if self._library_note_dirty:
            return
        note_id = str(getattr(event.button, "note_id", "") or "")
        if note_id:
            self._selected_note_id = note_id
        self._library_selected_row_id = LIBRARY_ROW_BROWSE_NOTES
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
        discards unsaved text; an unsaved edit surviving the flush aborts the
        navigation.

        Also kicks the full local-source snapshot refetch (the same
        exclusive worker the delete/create flows already use) so the list's
        relative ages, ordering, and the rail's Notes badge reflect any
        edit saved during this editor visit from the DB's own truth -- the
        immediate recompose below renders the save-time in-memory patch
        (see ``_save_library_note``), and the refetch then confirms it.

        Args:
            event: Button press event emitted by the "‹ Back to list" action.
        """
        event.stop()
        await self._flush_library_note_save()
        if self._library_note_dirty:
            return
        self._reset_library_note_editor_state()
        self._refresh_local_source_snapshot()
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
        delete sends is never stale; an unsaved edit surviving the flush
        aborts entering the confirm state, same as Back and note-row selection.

        Args:
            event: Button press event emitted by the editor's "Delete" action.
        """
        event.stop()
        await self._flush_library_note_save()
        if self._library_note_dirty:
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
            logger.opt(exception=True).warning(
                f"Failed to delete Library note {note_id!r}.")
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
        against the current time, mirroring the retired standalone Notes
        screen's template substitution
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
            logger.opt(exception=True).warning("Library note create failed.")
            self._notify_library_note_create_warning("Could not create the note.")
            return

        created_id = result.get("id") if isinstance(result, Mapping) else result
        created_id = str(created_id) if created_id else ""
        if not created_id:
            self._notify_library_note_create_warning("Could not create the note.")
            return

        self._selected_note_id = created_id
        self._library_selected_row_id = LIBRARY_ROW_BROWSE_NOTES
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
                logger.opt(exception=True).warning(
                    f"Failed to save Library media edit for {media_id!r}.")
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
                logger.opt(exception=True).warning(
                    f"Failed to delete Library media item {media_id!r}.")
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
                logger.opt(exception=True).warning(
                    f"Failed to add Library media highlight for {media_id!r}.")
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
                logger.opt(exception=True).warning(
                    f"Failed to delete Library media highlight {highlight_id!r}.")
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
                logger.opt(exception=True).warning(
                    f"Failed to toggle Library media read-it-later state for {media_id!r}.",
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
                logger.opt(exception=True).warning(
                    f"Failed to save Library media analysis for {media_id!r}.")
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
        """Hand off to the legacy Media manager screen.

        Only the full in-Library viewer's action row carries this button
        now -- it genuinely navigates away from Library, so its "Open in
        Media manager" label is honest. The browse summary's in-Library
        action is ``#library-media-open-viewer`` ("Open in viewer") instead.

        Args:
            event: Button press event emitted by the "Open in Media manager" action.
        """
        event.stop()
        self.post_message(NavigateToScreen("media"))

    @on(Input.Submitted, "#library-search-input")
    async def handle_library_search_submitted(self, event: Input.Submitted) -> None:
        """Submit the rail-top query to the Search canvas (fast `search` mode).

        The rail search box is the single query truth for Library
        Search/RAG: submitting it seeds ``_library_rag_query``, selects the
        promoted Search canvas, and (for a non-blank query) runs it through
        the same exclusive-worker gate as the in-panel query box
        (``_start_library_rag_query``). A blank submit still lands on the
        Search canvas -- so a bare Enter always goes somewhere sensible --
        but never invokes the search service.

        Args:
            event: Input submit event emitted by the rail's search box.
        """
        event.stop()
        query = self._safe_text(event.value, max_length=LIBRARY_RAG_QUERY_MAX_LENGTH)
        self._library_rag_query = query
        self._library_rag_mode = "search"
        await self._select_library_rail_row(LIBRARY_ROW_BROWSE_SEARCH)
        if self._library_selected_row_id != LIBRARY_ROW_BROWSE_SEARCH:
            # A dirty note editor sitting in an unresolved save conflict
            # aborts the row switch (`_select_library_rail_row` returns
            # early without moving `_library_selected_row_id`) -- the rail
            # submit must not run a query against a canvas the user never
            # actually reached, and must not record a history entry for it.
            return
        if query.strip():
            await self._start_library_rag_query()
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
        """Placeholder for the rail search box.

        The rail box always feeds the Search canvas now (see
        ``handle_library_search_submitted``), never a source-specific
        filter, so the placeholder is unconditional regardless of which
        rail row is active.

        Returns:
            The rail search placeholder text.
        """
        return "Search Library…"

    @on(Input.Submitted, "#library-conversations-filter")
    def handle_library_conversations_filter_submitted(self, event: Input.Submitted) -> None:
        """Filter the conversations canvas from its in-canvas filter box.

        This is client-side substring filtering over the already-loaded
        conversations snapshot (up to ``LIBRARY_SOURCE_PAGE_SIZES["conversations"]``
        records) -- the same behavior the rail-top search box used to
        provide before it was rewired to feed the Search canvas. A
        service-backed FTS filter over the full conversation set (not just
        the loaded snapshot) is a tracked follow-up.

        Args:
            event: Input submit event emitted by the conversations canvas's
                filter box.
        """
        event.stop()
        self._library_conversation_query = self._safe_text(event.value, max_length=200)
        self.refresh(recompose=True)
        self.call_after_refresh(self._focus_library_conversations_filter)

    def _focus_library_conversations_filter(self) -> None:
        """Re-focus the conversations filter box after a submit-triggered recompose.

        Mirrors ``_focus_library_search_input``: the Submitted-driven
        recompose remounts a brand-new ``#library-conversations-filter``;
        without this, focus silently falls back to the screen after every
        filter submit.
        """
        try:
            self.query_one("#library-conversations-filter", Input).focus()
        except (NoMatches, QueryError):
            pass

    async def _sync_collections_panel(self, *, refresh_snapshot: bool = False) -> None:
        if self._library_selected_row_id != LIBRARY_ROW_BROWSE_COLLECTIONS:
            self._library_collection_pending_delete_id = ""
            return
        if refresh_snapshot:
            await self._refresh_library_collections_snapshot()
        self.refresh(recompose=True)

    async def _refresh_collections_panel_action_state_widgets(self) -> None:
        if (
            self._library_selected_row_id != LIBRARY_ROW_BROWSE_COLLECTIONS
            or not list(self.query("#library-collections-panel"))
        ):
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
            logger.opt(exception=True).warning("Failed to load Library Collections.")
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
            logger.opt(exception=True).warning("Failed to load Library Collections sync dry-run state.")
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
            logger.opt(exception=True).warning("Failed to load Sync v2 profile summary.")
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
        # Ingest canvas. The Import/Export row/mode this used to target is
        # retired -- the Ingest ▸ Import media canvas row is its only
        # surviving successor.
        await self._select_library_rail_row(LIBRARY_ROW_INGEST_MEDIA)

    @on(Button.Pressed, "#library-rag-mode-toggle")
    def cycle_library_rag_mode(self, event: Button.Pressed) -> None:
        """Cycle Library Search/RAG mode between keyword search and RAG answer."""
        event.stop()
        self._library_rag_mode = "rag" if self._library_rag_mode == "search" else "search"
        self._reset_library_rag_retrieval_state()
        self.refresh(recompose=True)

    @on(Button.Pressed, ".library-rag-scope-toggle")
    def toggle_library_rag_scope_source(self, event: Button.Pressed) -> None:
        """Toggle one source type in/out of the Search/RAG retrieval scope (B2).

        Unlike the mode toggle, this deliberately does NOT reset in-flight
        or already-landed retrieval state: scope only affects the NEXT run,
        so existing results/history stay visible. Still a transition (like
        the mode toggle), so the canvas recomposes to pick up the new
        toggle labels, run-gate state, and (if the scope is now empty) the
        A1 quiet line.
        """
        event.stop()
        button_id = event.button.id or ""
        source_type = button_id.removeprefix("library-rag-scope-toggle-")
        if source_type not in LIBRARY_RAG_SCOPE_TOGGLE_SOURCE_TYPES:
            return
        if source_type in self._library_rag_scope_deselected:
            self._library_rag_scope_deselected.discard(source_type)
        else:
            self._library_rag_scope_deselected.add(source_type)
        self.refresh(recompose=True)

    @on(Collapsible.Toggled, "#library-rag-history")
    def sync_library_rag_history_collapsed(self, event: Collapsible.Toggled) -> None:
        """Track manual expand/collapse so recomposes preserve the user's choice.

        `Collapsible._watch_collapsed` posts this message on every change of
        the `collapsed` reactive -- both the user clicking the title and the
        programmatic `collapsible.collapsed = force_collapsed` assignment in
        `_refresh_library_rag_history_widget` (the results-arrival
        force-collapse path). The latter is harmless to mirror here: that
        assignment always uses `panel_state.history_collapsed`, which is
        itself derived from `_library_rag_history_collapsed` moments after
        that field was just set at the results-arrival transition, so this
        handler only ever re-writes the field to the value it already holds.
        """
        event.stop()
        self._library_rag_history_collapsed = event.collapsible.collapsed

    @on(Button.Pressed, "#library-rag-history-clear")
    async def clear_library_search_history(self, event: Button.Pressed) -> None:
        """Clear all Library Search/RAG query history, in memory and on disk (D1)."""
        event.stop()
        self._library_search_history = ()
        self._persist_library_search_history([])
        await self._refresh_library_rag_history_widget(self._library_rag_panel_state())

    @on(Button.Pressed, ".library-rag-history-row")
    async def rerun_library_search_from_history(self, event: Button.Pressed) -> None:
        """Re-run a prior Library Search/RAG query selected from history."""
        event.stop()
        index = self._trailing_index(event.button.id)
        if index is None or index >= len(self._library_search_history):
            return
        query = self._library_search_history[index]
        self._library_rag_query = query
        # Repopulate the visible query input too -- otherwise it keeps
        # whatever text (or blank) it held before the history row was
        # clicked, even though the run underneath used the history entry.
        # Set this before starting the run: `update_library_rag_query`'s
        # `Input.Changed` handler is a no-op once its value already equals
        # `_library_rag_query` (true here), so it can't clobber the new
        # run's "searching" status either way, but setting it first keeps
        # the widget and state in lockstep from the start of the run.
        try:
            self.query_one("#library-rag-query-input", Input).value = query
        except (NoMatches, QueryError):
            pass
        await self._start_library_rag_query()

    @staticmethod
    def _trailing_index(button_id: str | None) -> int | None:
        """Parse the trailing `-{index}` integer from a button id, or None."""
        if not button_id:
            return None
        try:
            index = int(button_id.rsplit("-", 1)[-1])
        except ValueError:
            return None
        return index if index >= 0 else None

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
        self._record_library_search_history(request.query)
        self._library_rag_results = ()
        self._library_rag_recovery_state = None
        self._library_rag_selected_result_id = ""
        self._library_rag_retrieval_status = "searching"
        # The rail-top search box can invoke this mid-recompose -- it selects
        # the Search canvas via ``_select_library_rail_row`` and then runs the
        # query immediately after, before the scheduled recompose has mounted
        # ``#library-search-rag-panel``. The widget refresh is only attempted
        # when the panel is actually mounted; when it isn't, skipping is
        # non-fatal because the subsequent recompose renders the same state
        # (the status fields set above already carry it). This is an
        # explicit presence check, not a broad NoMatches/QueryError catch --
        # a prior version wrapped the whole refresh (results rows included)
        # in a blanket ``except (NoMatches, QueryError): pass``, which also
        # silently swallowed unrelated mid-rebuild query failures instead of
        # only tolerating the "panel not mounted yet" case it was meant for.
        if self.query("#library-search-rag-panel"):
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
        result_index = self._trailing_index(event.button.id)
        if result_index is None or result_index >= len(self._library_rag_results):
            return
        self._library_rag_selected_result_id = self._library_rag_results[result_index].result_id
        await self._refresh_search_rag_panel_state_widgets()

    @on(Button.Pressed, ".library-rag-result-open")
    async def open_library_rag_result(self, event: Button.Pressed) -> None:
        """Open a Search/RAG evidence result straight to its Library detail surface."""
        event.stop()
        index = self._trailing_index(event.button.id)
        rows = self._library_rag_results
        if index is None or not (0 <= index < len(rows)):
            return
        row = rows[index]
        await self._open_library_item_by_id(row.open_source_type, row.source_id)

    async def _open_library_item_by_id(self, source_type: str, record_id: str) -> None:
        """Open a Library item straight to its detail surface by id.

        Shared route for per-result Search/RAG "Open" actions -- unlike
        rail-row navigation (``_select_library_rail_row``), which always
        lands on the list/browse view for a content type, this jumps
        straight to the media viewer, the notes editor, or a specific
        conversation. Also the route the future ingest-queue "open result"
        actions reuse.

        Args:
            source_type: ``"media"``, ``"notes"``, or ``"conversations"``.
                Any other value (including empty) is a no-op -- defensive
                only, since the Open action is only rendered for rows with
                resolvable provenance (``LibraryRagResultRow.can_open``).
            record_id: The item's id within its source type.
        """
        if not record_id or source_type not in ("media", "notes", "conversations"):
            return

        if source_type == "media":
            await self._flush_library_note_save()
            if self._library_note_dirty:
                return
            # Mirrors handle_library_media_row's full state-set EXACTLY so
            # the recomposed canvas lands on a clean viewer, never a stale
            # one carried over from a previously opened item.
            self._selected_media_id = record_id
            self._library_selected_row_id = LIBRARY_ROW_BROWSE_MEDIA
            self._library_media_view = "viewer"
            self._library_media_detail = None
            self._library_media_editing = False
            self._library_media_confirming_delete = False
            self._library_media_highlights = []
            self._library_media_editing_analysis = False
            self._library_media_content_query = ""
            self._library_media_content_match_index = 0
            self.run_worker(
                self._refresh_library_media_detail(record_id),
                exclusive=True,
                group="library_media_detail",
            )
            self.refresh(recompose=True)
            return

        if source_type == "notes":
            await self._flush_library_note_save()
            if self._library_note_dirty:
                return
            # Reset first for a clean slate (also stops any autosave timer,
            # clears dirty/conflict/preview state), then apply the actual
            # open-target fields -- equivalent final state to the note_id
            # navigation-context branch's inline field-by-field reset.
            self._reset_library_note_editor_state()
            self._library_notes_view = "editor"
            self._selected_note_id = record_id
            self._library_selected_row_id = LIBRARY_ROW_BROWSE_NOTES
            self.run_worker(
                self._refresh_library_note_detail(record_id),
                exclusive=True,
                group="library_note_detail",
            )
            self.refresh(recompose=True)
            return

        # conversations: jump straight to a specific conversation, fetching
        # it by id when it isn't already in the loaded snapshot. Closes the
        # known deep-link caveat where an out-of-snapshot id silently fell
        # back to the first row (_ensure_selected_conversation_id).
        record_ids = {
            self._conversation_record_id(record, index)
            for index, record in enumerate(self._conversation_records())
        }
        if record_id not in record_ids:
            fetched = await self._fetch_library_conversation_by_id(record_id)
            if fetched is None:
                notify = getattr(self.app_instance, "notify", None)
                if callable(notify):
                    notify("Conversation is unavailable.", severity="warning")
                return
            self._local_source_records["conversations"] = (
                fetched,
                *self._local_source_records.get("conversations", ()),
            )
        self._selected_conversation_id = record_id
        # Opening a specific conversation must show it even if an in-canvas
        # filter would otherwise hide it -- handle_library_rail_row's own
        # canvas branch resets this same field for the same reason when
        # entering Conversations via the rail; _select_library_rail_row
        # itself does not touch it.
        self._library_conversation_query = ""
        await self._select_library_rail_row(LIBRARY_ROW_BROWSE_CONVERSATIONS)

    async def _fetch_library_conversation_by_id(
        self, conversation_id: str
    ) -> Mapping[str, Any] | None:
        """Fetch a single conversation record directly from ChaChaNotes by id.

        Used by ``_open_library_item_by_id`` when a conversation Open target
        is outside the loaded ``_local_source_records["conversations"]``
        snapshot -- ``chat_conversation_scope_service.list_conversations``
        only returns the loaded page, so a direct point lookup against the
        DB is needed instead.

        Args:
            conversation_id: The conversation id to fetch.

        Returns:
            The raw ``conversations`` table row as a mapping, or ``None``
            when the DB is unavailable, the lookup fails, or no matching
            (non-deleted) conversation exists.
        """
        db = getattr(self.app_instance, "chachanotes_db", None)
        get_conversation_by_id = getattr(db, "get_conversation_by_id", None)
        if not callable(get_conversation_by_id):
            return None
        try:
            if getattr(db, "is_memory_db", False):
                # In-memory SQLite connections are thread-local -- only the
                # thread that created the DB has the migrated schema, so
                # offloading to a worker thread would hit a blank connection.
                # Same guard as
                # LibraryLocalRagSearchService._search_conversations.
                record = get_conversation_by_id(conversation_id, include_deleted=False)
            else:
                record = await asyncio.to_thread(
                    get_conversation_by_id, conversation_id, include_deleted=False
                )
        except Exception:
            logger.opt(exception=True).warning(
                f"Failed to fetch Library conversation {conversation_id!r} by id.",
            )
            return None
        return record if isinstance(record, Mapping) else None

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
        if self._library_selected_row_id != LIBRARY_ROW_BROWSE_SEARCH:
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

    @work(exclusive=True, group="library_rag_search")
    async def _execute_library_rag_search(self, request: LibraryRagSearchRequest) -> None:
        outcome = await run_library_rag_search(self.app_instance, request)
        await self._apply_library_rag_search_outcome(request, outcome)

    async def _apply_library_rag_search_outcome(
        self,
        request: LibraryRagSearchRequest,
        outcome: LibraryRagSearchOutcome,
    ) -> None:
        """Resolve a completed Library Search/RAG worker's outcome into state.

        The state fields (results/status/recovery) always apply once the
        stale-query and stale-mode guards pass -- even if the user has since
        left the Search canvas (a different rail row, e.g. Media) -- so a
        dangling "searching" status can never survive: an outcome that lands
        while the user is elsewhere still resolves it, and re-entering the
        Search canvas composes from settled state instead of a stale
        in-flight line. Only the live widget refresh is skipped when the
        panel isn't mounted; there is nothing on screen to update.
        """
        if not self.is_mounted:
            return
        current_query = self._library_rag_panel_state().query_state.query
        if request.query != current_query:
            # Stale: a newer query has since replaced this one.
            return
        if request.mode != self._library_rag_mode:
            # Stale: the mode toggled mid-flight; this result belongs to
            # the mode the user has since left.
            return
        self._library_rag_results = outcome.results
        self._library_rag_retrieval_status = outcome.status
        self._library_rag_recovery_state = outcome.recovery_state
        self._library_rag_selected_result_id = ""
        # D1: the results-arrival transition is the ONLY place allowed to
        # force the `Recent searches` collapsible open/closed -- collapse it
        # once evidence lands (results take visual priority), expand it
        # when a search settles with nothing to show. Every other refresh
        # path leaves the user's manual expand/collapse alone.
        self._library_rag_history_collapsed = bool(self._library_rag_results)
        if (
            self._library_selected_row_id != LIBRARY_ROW_BROWSE_SEARCH
            or not self.query("#library-search-rag-panel")
        ):
            return
        await self._refresh_search_rag_panel_state_widgets(force_history_collapse=True)

    async def _refresh_search_rag_panel_state_widgets(
        self,
        *,
        force_history_collapse: bool = False,
    ) -> None:
        if (
            self._library_selected_row_id != LIBRARY_ROW_BROWSE_SEARCH
            or not self.query("#library-search-rag-panel")
        ):
            return

        panel_state = self._library_rag_panel_state()

        await self._refresh_library_rag_query_status_widgets(panel_state)

        scope_container = self.query_one("#library-rag-source-scope", Vertical)
        scope_container.set_class(
            library_rag_scope_shows_recovery(panel_state.scope), "has-recovery"
        )
        self.query_one("#library-rag-scope-summary", Static).update(
            self._library_rag_scope_summary(panel_state)
        )
        scope_recovery_widgets = list(self.query("#library-rag-scope-recovery"))
        import_buttons = list(self.query("#library-rag-open-import-export"))
        for widget in (*scope_recovery_widgets, *import_buttons):
            await widget.remove()
        for child in library_rag_scope_recovery_children(panel_state):
            await scope_container.mount(child)

        self._refresh_library_rag_inspector(panel_state)
        await self._refresh_library_rag_results_widgets(panel_state)
        await self._refresh_library_rag_history_widget(
            panel_state,
            force_collapsed=panel_state.history_collapsed if force_history_collapse else None,
        )
        # `force_history_collapse` is only set True from the results-arrival
        # transition in `_apply_library_rag_search_outcome` -- every other
        # refresh trigger (scope toggle, mode toggle, evidence selection)
        # passes the default False. Reuse that same signal (C2) to scroll
        # the Evidence heading back into view once results just landed.
        # Deliberately done LAST, after every widget mutation above
        # (results *and* history) has settled: mounting/removing the
        # history rows also changes the panel's virtual size, and a scroll
        # issued before that would just get overridden by it.
        if force_history_collapse and panel_state.results:
            try:
                self.query_one(
                    "#library-rag-results-heading", Static
                ).scroll_visible(animate=False)
            except NoMatches:
                pass

    async def _refresh_library_rag_query_status_widgets(
        self,
        panel_state: LibraryRagPanelState,
    ) -> None:
        """Sync the Run button and the query region's conditional status block.

        The quiet line / callout+recovery block is torn down and rebuilt
        from `library_rag_query_status_children` on every call -- it is at
        most two `Static` widgets, so a full rebuild is cheap and (unlike
        hand-written incremental mount/update/remove logic) can never drift
        from what `compose()` renders on a fresh mount.
        """
        query_controls = self.query_one("#library-rag-query-controls", Vertical)
        query_controls.set_class(
            library_rag_query_shows_full_recovery(panel_state.query_state), "has-recovery"
        )

        run_action = panel_state.query_state.run_action
        run_button = self.query_one("#library-rag-run-query", Button)
        run_button.label = run_action.label
        run_button.disabled = not run_action.enabled
        run_button.tooltip = run_action.tooltip

        for widget_id in (
            "library-rag-query-quiet-line",
            "library-rag-query-blocked-callout",
            "library-rag-query-recovery",
        ):
            for widget in list(self.query(f"#{widget_id}")):
                await widget.remove()
        anchor = "#library-rag-query-input"
        for child in library_rag_query_status_children(panel_state):
            await query_controls.mount(child, after=anchor)
            anchor = f"#{child.id}"

    async def _refresh_library_rag_history_widget(
        self,
        panel_state: LibraryRagPanelState,
        *,
        force_collapsed: bool | None = None,
    ) -> None:
        """Rebuild the `Recent searches` collapsible content from state.

        Mutates the compose-time `Collapsible` in place (its `collapsed`
        reactive, then its `Contents` children) rather than replacing the
        whole widget -- two refreshes can be triggered back to back (the
        synchronous "searching" status refresh, then the search worker's
        own "outcome" refresh), and remove-then-mount of the same fixed ID
        from overlapping calls raises `DuplicateIds`. The lock serializes
        those calls so one full rebuild always finishes before the next
        starts.

        `force_collapsed` (D1) is `None` for every caller except the
        results-arrival transition in `_apply_library_rag_search_outcome`:
        `None` leaves the live widget's `collapsed` reactive exactly as the
        user left it; a `bool` overwrites it. This is safe for full
        recomposes too (scope toggles, the mode toggle) -- not just in-place
        refreshes (query edits, evidence selection) -- because
        `sync_library_rag_history_collapsed` mirrors every live `collapsed`
        change (manual or programmatic) back into
        `_library_rag_history_collapsed`, so `compose()` always rebuilds the
        `Collapsible` from the user's last choice instead of a stale field.
        """
        async with self._library_rag_history_refresh_lock:
            history_widgets = list(self.query("#library-rag-history"))
            if not history_widgets:
                return
            collapsible = history_widgets[0]
            if not isinstance(collapsible, Collapsible):
                return
            if force_collapsed is not None:
                collapsible.collapsed = force_collapsed
            try:
                contents = collapsible.query_one(Collapsible.Contents)
            except (NoMatches, QueryError):
                # Defensive, mirroring the two guards above: an "exclusive"
                # search worker can be cancelled mid-refresh by a newer one
                # (e.g. re-running a history entry while a prior query is
                # still settling), which can catch this specific
                # `Collapsible` instance between un/remounting its own
                # `Contents` child. The next refresh (there is always one --
                # every query/scope/selection change triggers one) picks up
                # the settled state; there is nothing to safely rebuild here.
                return
            for child in list(contents.children):
                await child.remove()
            for row in library_rag_history_children(panel_state):
                await contents.mount(row)

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
        """Rebuild the Evidence region body from `library_rag_results_body_children`.

        Shared with `LibrarySearchRagPanel.compose()` (C1): both build rows,
        the searching line, recovery copy, and the empty state from the
        same function, closing the compose-vs-refresh duplication that
        previously let the two paths drift apart.
        """
        results_container = self.query_one("#library-rag-results", Vertical)
        self.query_one("#library-rag-results-heading", Static).update(
            f"Evidence · top {panel_state.query_state.top_k} per source"
        )
        for child in list(results_container.children):
            if child.id in LIBRARY_RAG_RESULTS_STATIC_WIDGET_IDS:
                continue
            await child.remove()
        for child in library_rag_results_body_children(panel_state):
            await results_container.mount(child)

    @staticmethod
    def _library_rag_scope_summary(panel_state: LibraryRagPanelState) -> str:
        return LIBRARY_RAG_SCOPE_ALL_LOCAL_COPY

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
        open_chat_with_handoff(payload, action_label="Use in Console")

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
        open_chat_with_handoff(payload, action_label="Use in Console")

    @on(Button.Pressed, "#library-media-use-in-chat")
    def use_media_in_chat(self, event: Button.Pressed) -> None:
        """Handle the media viewer's "Use in Console" action.

        Args:
            event: Button press event emitted by the viewer's "Use in Console" action.
        """
        event.stop()
        self._open_selected_media_handoff()

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
            logger.opt(exception=True).warning("Failed to create local Library workspace")
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify("Local workspace could not be created.", severity="error")
            return

        self._invalidate_library_workspace_depth_state()
        self.refresh(recompose=True)
        notify = getattr(self.app_instance, "notify", None)
        if callable(notify):
            notify(f"Created local workspace {workspace_name}.", severity="information")

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
            ),
            action_label="Use in Console",
        )
