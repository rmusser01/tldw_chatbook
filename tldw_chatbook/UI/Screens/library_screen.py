"""Library destination shell for source material and Search/RAG."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Mapping, Sequence
from typing import Any

from loguru import logger
from rich.markup import escape as escape_markup
from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, Static

from ...Chat.chat_handoff_models import ChatHandoffPayload
from ...Library.library_collections_service import LibraryCollectionsServiceError
from ...Library.library_collections_state import LibraryCollectionsPanelState
from ...Library.library_rag_service import (
    LibraryRagSearchOutcome,
    LibraryRagSearchRequest,
    run_library_rag_search,
)
from ...Library.library_rag_state import LibraryRagPanelState
from ...runtime_policy.server_event_scope import event_principal_id_from_active_context
from ...runtime_policy.types import PolicyDeniedError, RuntimeSourceState
from ...Sync_Interop.sync_promotion_state import build_sync_promotion_state
from ...Sync_Interop.sync_readiness import DEFAULT_SYNC_ELIGIBILITY_REGISTRY, build_sync_readiness_report
from ...Utils.input_validation import sanitize_string, validate_text_input
from ...Workspaces import LibraryWorkspaceDepthState, build_library_workspace_depth_state
from ...Widgets.Library import (
    LibraryCollectionsPanel,
    LibrarySearchRagInspectorPanel,
    LibrarySearchRagPanel,
)
from ...Widgets.destination_workbench import DestinationModeStrip
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
LIBRARY_SOURCE_PAGE_SIZE = 5
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
LIBRARY_COLLECTION_SYNC_CONFLICT_LIMIT = 200
LIBRARY_LOCAL_SNAPSHOT_MODES = frozenset({"sources", "conversations", "import-export"})
LIBRARY_WORKSPACE_SOURCE_COLUMN_WIDTH = 30
LIBRARY_WORKSPACE_SCOPE_COLUMN_WIDTH = 18
LIBRARY_WORKSPACE_VISIBLE_COLUMN_WIDTH = 7
LIBRARY_WORKSPACE_CONTEXT_COLUMN_WIDTH = 11
LIBRARY_HUB_MODULE_COLUMN_WIDTH = 15
LIBRARY_HUB_COUNT_COLUMN_WIDTH = 7
LIBRARY_HUB_BROWSE_COLUMN_WIDTH = 18
LIBRARY_HUB_RECENT_COLUMN_WIDTH = 32
LIBRARY_HUB_CONSOLE_COLUMN_WIDTH = 25
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
        # Collections owns its explanatory copy inside the reader and inspector.
        # Keeping these generic mode strings empty avoids stale terminal cells
        # overlapping the Collections workbench during Textual Web mode changes.
        "description": "",
        "next_action": "",
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

    def on_mount(self) -> None:
        super().on_mount()
        self.set_timer(
            LIBRARY_SOURCE_SNAPSHOT_TIMEOUT_SECONDS,
            self._apply_source_snapshot_timeout,
        )
        self._refresh_local_source_snapshot()

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

        empty_records: dict[str, tuple[Mapping[str, Any], ...]] = {
            "notes": (),
            "media": (),
            "conversations": (),
        }
        empty_counts = {"notes": 0, "media": 0, "conversations": 0}
        empty_total_known = {"notes": True, "media": True, "conversations": True}
        if not all(callable(call) for call in (list_notes, list_media, list_conversations)):
            return empty_records, empty_counts, empty_total_known, LIBRARY_SERVICE_UNAVAILABLE_COPY, None

        try:
            notes_result, media_result, conversation_result = await asyncio.wait_for(
                asyncio.gather(
                    self._run_library_service_call(
                        list_notes,
                        scope="local_note",
                        limit=LIBRARY_SOURCE_PAGE_SIZE,
                        offset=0,
                        user_id=getattr(self.app_instance, "notes_user_id", None) or "default_user",
                        isolate_in_worker=True,
                    ),
                    self._run_library_service_call(
                        list_media,
                        mode="local",
                        page=1,
                        results_per_page=LIBRARY_SOURCE_PAGE_SIZE,
                        include_keywords=False,
                        isolate_in_worker=True,
                    ),
                    self._run_library_service_call(
                        list_conversations,
                        mode="local",
                        limit=LIBRARY_SOURCE_PAGE_SIZE,
                        offset=0,
                        isolate_in_worker=True,
                    ),
                ),
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

        notes, notes_count, notes_total_known = self._response_records_and_count(notes_result)
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
        return f"{label} (showing up to {LIBRARY_SOURCE_PAGE_SIZE}): {count}"

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

    def _source_recent_label(self, source_type: str) -> str:
        recent = self._source_recent_value(source_type)
        return f"Recent: {recent}"

    def _hub_table_cell(self, value: str, width: int = LIBRARY_HUB_RECENT_COLUMN_WIDTH) -> str:
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

    def _hub_table_row(
        self,
        *,
        module: str,
        count: str,
        browse: str,
        recent: str,
        console: str,
    ) -> str:
        """Render terminal-native aligned columns without markdown table noise."""
        module_cell = self._hub_table_cell(module, LIBRARY_HUB_MODULE_COLUMN_WIDTH)
        count_cell = self._hub_table_cell(count, LIBRARY_HUB_COUNT_COLUMN_WIDTH)
        browse_cell = self._hub_table_cell(browse, LIBRARY_HUB_BROWSE_COLUMN_WIDTH)
        recent_cell = self._hub_table_cell(recent, LIBRARY_HUB_RECENT_COLUMN_WIDTH)
        console_cell = self._hub_table_cell(console, LIBRARY_HUB_CONSOLE_COLUMN_WIDTH)
        return (
            f"{module_cell:<{LIBRARY_HUB_MODULE_COLUMN_WIDTH}} "
            f"{count_cell:<{LIBRARY_HUB_COUNT_COLUMN_WIDTH}} "
            f"{browse_cell:<{LIBRARY_HUB_BROWSE_COLUMN_WIDTH}} "
            f"{recent_cell:<{LIBRARY_HUB_RECENT_COLUMN_WIDTH}} "
            f"{console_cell:<{LIBRARY_HUB_CONSOLE_COLUMN_WIDTH}}"
        )

    def _hub_table_border(self) -> str:
        return (
            "+"
            + "+".join(
                "-" * (width + 2)
                for width in (
                    LIBRARY_HUB_MODULE_COLUMN_WIDTH,
                    LIBRARY_HUB_COUNT_COLUMN_WIDTH,
                    LIBRARY_HUB_BROWSE_COLUMN_WIDTH,
                    LIBRARY_HUB_RECENT_COLUMN_WIDTH,
                    LIBRARY_HUB_CONSOLE_COLUMN_WIDTH,
                )
            )
            + "+"
        )

    def _hub_framed_table_row(
        self,
        *,
        module: str,
        count: str,
        browse: str,
        recent: str,
        console: str,
    ) -> str:
        """Render source status as a visible ASCII table row."""
        module_cell = self._hub_table_cell(module, LIBRARY_HUB_MODULE_COLUMN_WIDTH)
        count_cell = self._hub_table_cell(count, LIBRARY_HUB_COUNT_COLUMN_WIDTH)
        browse_cell = self._hub_table_cell(browse, LIBRARY_HUB_BROWSE_COLUMN_WIDTH)
        recent_cell = self._hub_table_cell(recent, LIBRARY_HUB_RECENT_COLUMN_WIDTH)
        console_cell = self._hub_table_cell(console, LIBRARY_HUB_CONSOLE_COLUMN_WIDTH)
        return (
            f"| {module_cell:<{LIBRARY_HUB_MODULE_COLUMN_WIDTH}} "
            f"| {count_cell:<{LIBRARY_HUB_COUNT_COLUMN_WIDTH}} "
            f"| {browse_cell:<{LIBRARY_HUB_BROWSE_COLUMN_WIDTH}} "
            f"| {recent_cell:<{LIBRARY_HUB_RECENT_COLUMN_WIDTH}} "
            f"| {console_cell:<{LIBRARY_HUB_CONSOLE_COLUMN_WIDTH}} |"
        )

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

    def _hub_source_card(
        self,
        *,
        source_type: str,
        label: str,
        owner: str,
        purpose: str,
        next_action: str,
        widget_id: str,
    ) -> Static:
        return Static(
            self._hub_framed_table_row(
                module=label,
                count=self._hub_source_count_value(source_type),
                browse=next_action,
                recent=self._source_recent_value(source_type),
                console=self._hub_console_status(source_type),
            ),
            markup=False,
            id=widget_id,
            classes="library-hub-card",
        )

    def _hub_spacer(self, widget_id: str) -> Static:
        return Static("", id=widget_id, classes="library-hub-spacer")

    def _content_hub_rows(self) -> tuple[Static, ...]:
        return (
            Static(
                "Library Content Hub",
                id="library-content-hub-title",
                classes="destination-section",
            ),
            Static(
                (
                    "Source overview, retrieval readiness, movement paths, and next action."
                ),
                id="library-content-hub-purpose",
            ),
            Static(
                self._hub_state_summary(),
                id="library-hub-state-summary",
                classes="library-hub-card",
            ),
            self._hub_section_rule(
                "Source Status",
                "library-hub-section-source-status",
            ),
            Static(
                "\n".join(
                    (
                        self._hub_table_border(),
                        self._hub_framed_table_row(
                            module="Source",
                            count="Count",
                            browse="Browse",
                            recent="Recent",
                            console="Console",
                        ),
                        self._hub_table_border(),
                    )
                ),
                id="library-content-hub-table-header",
                classes="destination-section",
            ),
            self._hub_source_card(
                source_type="notes",
                label="Notes",
                owner="Notes",
                purpose="Edit, sync, export notes",
                next_action="Open Notes",
                widget_id="library-notes-summary",
            ),
            self._hub_source_card(
                source_type="media",
                label="Media",
                owner="Media",
                purpose="Browse media library items",
                next_action="Browse media",
                widget_id="library-media-summary",
            ),
            self._hub_source_card(
                source_type="conversations",
                label="Conversations",
                owner="Conversations",
                purpose="Browse saved chats",
                next_action="Browse chats",
                widget_id="library-conversations-summary",
            ),
            Static(
                self._hub_table_border(),
                id="library-hub-source-table-bottom",
                classes="destination-section",
            ),
            self._hub_spacer("library-hub-spacer-after-source-table"),
            Static(
                (
                    "Owners: Notes edits/sync/export | Media browses library items | "
                    "Conversations resumes chats"
                ),
                id="library-hub-source-owner-summary",
                classes="library-hub-card",
            ),
            self._hub_spacer("library-hub-spacer-before-retrieval"),
            self._hub_section_rule(
                "Retrieval Readiness",
                "library-hub-section-retrieval-readiness",
            ),
            Static(
                "Library readiness",
                id="library-hub-readiness-title",
                classes="destination-section",
            ),
            Static(
                self._hub_readiness_summary(),
                markup=False,
                id="library-hub-readiness-summary",
                classes="library-hub-card",
            ),
            self._hub_spacer("library-hub-spacer-before-movement"),
            self._hub_section_rule(
                "Movement + Reuse",
                "library-hub-section-movement-reuse",
            ),
            Static(
                self._hub_key_value_row(
                    "Search/RAG",
                    "Query indexed content, inspect evidence, launch Console.",
                ),
                id="library-hub-search-card",
                classes="library-hub-card",
            ),
            Static(
                self._hub_key_value_row(
                    "Import/Export",
                    "Add or move content; imported material returns here.",
                ),
                id="library-hub-import-export-card",
                classes="library-hub-card",
            ),
            Static(
                self._hub_key_value_row(
                    "Collections",
                    "Read, review, reuse saved content.",
                ),
                id="library-hub-collections-card",
                classes="library-hub-card",
            ),
            self._hub_spacer("library-hub-spacer-before-learning"),
            self._hub_section_rule(
                "Learning",
                "library-hub-section-learning-paths",
            ),
            Static(
                self._hub_key_value_row(
                    "Study",
                    "Turn Library content into flashcards and quizzes.",
                ),
                id="library-hub-study-card",
                classes="library-hub-card",
            ),
            self._hub_spacer("library-hub-spacer-before-next-action"),
            self._hub_section_rule(
                "Next Action",
                "library-hub-section-next-action",
            ),
            Static(
                self._hub_key_value_row("Primary", "Import sources or create a note."),
                id="library-hub-next-primary",
                classes="library-hub-card",
            ),
            Static(
                self._hub_key_value_row(
                    "Then",
                    "Open Search/RAG after indexing or Collections after saving content.",
                ),
                id="library-hub-next-secondary",
                classes="library-hub-card",
            ),
        )

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

    def _workspaces_detail_rows(
        self,
        state: LibraryWorkspaceDepthState,
    ) -> tuple[Static, ...]:
        rows: list[Static] = [
            Static("Workspace Rules", classes="destination-section"),
            Static(
                Text.from_markup(escape_markup(state.workspace_label)),
                id="library-workspaces-active-workspace",
            ),
            Static(state.visibility_label, id="library-workspaces-visibility"),
            Static(
                "Workspace selection changes staging, not what you can browse or search.",
                id="library-workspaces-global-access-rule",
            ),
            Static(
                "Stage only active-workspace sources into Console, RAG, or agents.",
                id="library-workspaces-context-rule",
            ),
        ]
        if not state.source_rows:
            rows.extend(
                (
                    Static(
                        "No workspace sources yet.",
                        id="library-workspaces-empty-title",
                        classes="destination-section",
                    ),
                    Static(
                        "Browse/search still shows every Library and Notes item.",
                        id="library-workspaces-empty-browse",
                    ),
                    Static(
                        "Handoff unavailable until sources exist or are assigned here.",
                        id="library-workspaces-empty-handoff",
                    ),
                    Static(state.handoff_label, id="library-workspaces-handoff"),
                    Static(
                        state.collections_membership_label,
                        id="library-workspaces-collections-membership",
                    ),
                    Static(state.import_export_label, id="library-workspaces-import-export"),
                )
            )
            return tuple(rows)
        rows.append(
            Static(
                self._workspace_eligibility_header(),
                id="library-workspaces-eligibility-heading",
                classes="destination-section",
            )
        )
        for index, row in enumerate(state.source_rows):
            rows.append(
                Static(
                    self._workspace_eligibility_row(state, row),
                    id=f"library-workspaces-source-row-{index}",
                )
            )
        rows.extend(
            (
                Static(state.handoff_label, id="library-workspaces-handoff"),
                Static(
                    state.collections_membership_label,
                    id="library-workspaces-collections-membership",
                ),
                Static(state.import_export_label, id="library-workspaces-import-export"),
            )
        )
        return tuple(rows)

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

    def _workspace_eligibility_header(self) -> str:
        return (
            f"{'Source':<{LIBRARY_WORKSPACE_SOURCE_COLUMN_WIDTH}} "
            f"{'Workspace':<{LIBRARY_WORKSPACE_SCOPE_COLUMN_WIDTH}} "
            f"{'Visible':<{LIBRARY_WORKSPACE_VISIBLE_COLUMN_WIDTH}} "
            f"{'Console/RAG':<{LIBRARY_WORKSPACE_CONTEXT_COLUMN_WIDTH}} "
            "Recovery"
        )

    def _workspace_eligibility_row(
        self,
        state: LibraryWorkspaceDepthState,
        row: Any,
    ) -> str:
        return (
            f"{self._workspace_table_cell(row.title, LIBRARY_WORKSPACE_SOURCE_COLUMN_WIDTH, escape=True):<{LIBRARY_WORKSPACE_SOURCE_COLUMN_WIDTH}} "
            f"{self._workspace_table_cell(row.workspace_label, LIBRARY_WORKSPACE_SCOPE_COLUMN_WIDTH, escape=True):<{LIBRARY_WORKSPACE_SCOPE_COLUMN_WIDTH}} "
            f"{self._workspace_visible_label(row.visible):<{LIBRARY_WORKSPACE_VISIBLE_COLUMN_WIDTH}} "
            f"{self._workspace_context_status(row):<{LIBRARY_WORKSPACE_CONTEXT_COLUMN_WIDTH}} "
            f"{escape_markup(self._workspace_recovery_label(state, row))}"
        )

    @staticmethod
    def _workspace_table_cell(value: Any, width: int, *, escape: bool = False) -> str:
        text = str(value or "").strip()
        if len(text) <= width:
            return escape_markup(text) if escape else text
        if width <= 3:
            truncated = text[:width]
            return escape_markup(truncated) if escape else truncated
        truncated = f"{text[: width - 3]}..."
        return escape_markup(truncated) if escape else truncated

    def _workspace_visible_label(self, visible: bool) -> str:
        return "Yes" if visible else "No"

    def _workspace_context_status(self, row: Any) -> str:
        return "Eligible" if row.active_context_eligible else "Blocked"

    def _workspace_recovery_label(
        self,
        state: LibraryWorkspaceDepthState,
        row: Any,
    ) -> str:
        if row.active_context_eligible:
            return "Ready"
        workspace_name = state.workspace_name.strip()
        if workspace_name and workspace_name not in {"Local Default", "unavailable"}:
            return f"Copy/link to {workspace_name}"
        return "Assign to active workspace"

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
            id="library-study-handoff-detail",
            classes="library-rag-region",
        )

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
            widgets: list[Any] = [
                Static("Workspace actions", classes="destination-section"),
                Button(
                    "Create local workspace",
                    id="library-create-local-workspace",
                    classes="library-source-action",
                    tooltip=(
                        "Create a local-only workspace and make it active. "
                        "Server sync and ACP handoff remain WIP."
                    ),
                ),
                Static(
                    "Server sync: WIP/unavailable. Local workspace selection is active.",
                    id="library-workspace-create-local-copy",
                    classes="ds-recovery-callout",
                ),
            ]
            if workspace_depth_state.context_handoff_enabled:
                widgets.append(
                    Static(
                        "Ready: eligible sources can be staged in Console.",
                        id="library-workspace-action-ready",
                        classes="ds-recovery-callout",
                    )
                )
            elif not workspace_depth_state.source_rows:
                widgets.extend(
                    (
                        Button(
                            "Import sources",
                            id="library-workspace-import-sources",
                            classes="library-source-action",
                            tooltip="Open Library Import/Export to add workspace-eligible sources.",
                        ),
                        Static(
                            (
                                f"{self._workspace_handoff_blocked_label(workspace_depth_state)}\n"
                                f"{self._workspace_handoff_fix_label(workspace_depth_state)}"
                            ),
                            id="library-workspace-action-blocked",
                            classes="ds-recovery-callout is-blocked",
                        ),
                    )
                )
            else:
                widgets.append(
                    Static(
                        (
                            f"{self._workspace_handoff_blocked_label(workspace_depth_state)}\n"
                            f"{self._workspace_handoff_fix_label(workspace_depth_state)}"
                        ),
                        id="library-workspace-action-blocked",
                        classes="ds-recovery-callout is-blocked",
                    )
                )
            if workspace_depth_state.source_rows:
                widgets.append(
                    Static(
                        (
                            "Next: stage eligible sources in Console."
                            if workspace_depth_state.context_handoff_enabled
                            else f"Next: {self._workspace_handoff_recovery_label(workspace_depth_state)}."
                        ),
                        id="library-workspace-action-next-step",
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
            return tuple(widgets)
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
        has_sources = self._has_local_sources()
        handoff_disabled = True
        handoff_tooltip = "Stage Library source context after Library finishes loading."
        workspace_depth_state = self._library_workspace_depth_state(refresh=True)
        if self._library_lookup_error:
            recovery_state = self._library_lookup_recovery_state
            handoff_tooltip = (
                recovery_state.disabled_tooltip
                if recovery_state is not None
                else "Library source services are unavailable; retry Library later."
            )
        elif not has_sources:
            handoff_tooltip = "Stage Library source context after adding notes, media, or conversations."
        else:
            handoff_disabled = not workspace_depth_state.context_handoff_enabled
            handoff_tooltip = (
                workspace_depth_state.context_handoff_tooltip
                if handoff_disabled
                else "Stage Library source context in Console."
            )
        search_rag_panel_state = (
            self._library_rag_panel_state() if self._active_mode == "search" else None
        )
        collections_panel_state = (
            self._library_collections_panel_state()
            if self._active_mode == "collections"
            else None
        )
        collection_scoped_actions_deferred = self._active_mode == "collections"
        source_column_title, detail_column_title, inspector_column_title = self._active_column_titles()
        show_local_snapshot_region = self._should_show_local_snapshot_region()

        with Vertical(id="library-shell"):
            yield Static("Library", id="library-title", classes="ds-destination-header")
            yield Static(
                "Source material, notes, media, conversations, collections, imports/exports, and Search/RAG.",
                id="library-purpose",
                classes="destination-purpose",
            )
            yield Static(
                self._status_row_copy(),
                id="library-status-row",
                classes="destination-status-row",
            )
            with DestinationModeStrip(id="library-mode-bar", classes="destination-mode-strip"):
                yield Static(
                    "Modes:",
                    id="library-mode-label",
                    classes="destination-section",
                )
                for mode_id, mode in LIBRARY_MODES.items():
                    if not mode.get("show_in_strip", True):
                        continue
                    classes = "library-mode-chip"
                    if mode_id == self._active_mode:
                        classes = f"{classes} is-active"
                    yield Button(
                        mode["label"],
                        id=mode["button_id"],
                        classes=classes,
                        tooltip=mode["description"],
                    )

            with Horizontal(id="library-contract-grid", classes="ds-panel destination-workbench"):
                with Vertical(
                    id="library-source-browser",
                    classes="library-region destination-workbench-pane",
                ):
                    yield Static(
                        "Workspace Context",
                        id="library-workspace-context-title",
                        classes="destination-section",
                    )
                    yield Static(
                        self._library_workspace_scope_label(workspace_depth_state),
                        id="library-workspace-scope",
                    )
                    yield Static("Browse: all workspaces", id="library-workspace-browse-rule")
                    yield Static("Use/stage: active workspace only", id="library-workspace-use-rule")
                    yield Static(
                        source_column_title,
                        id="library-source-browser-title",
                        classes="destination-section",
                    )
                    yield from self._source_module_action_widgets()
                    yield Static(
                        "Quick Actions",
                        id="library-quick-actions-title",
                        classes="destination-section",
                    )
                    yield Static(
                        "Open a mode, then use the inspector for selected-item actions.",
                        id="library-quick-actions-guidance",
                    )

                with Vertical(
                    id="library-source-detail",
                    classes="library-region destination-workbench-pane",
                ):
                    yield Static(
                        detail_column_title,
                        id="library-source-detail-title",
                        classes="destination-section",
                    )
                    if self._active_mode == "workspaces":
                        with Vertical(id="library-workspaces-depth-panel"):
                            for row in self._workspaces_detail_rows(workspace_depth_state):
                                yield row
                    active_mode = self._active_mode_contract()
                    active_mode_copy_visible = self._active_mode not in {
                        "collections",
                        "sources",
                        "workspaces",
                    }
                    active_mode_title = Static(
                        f"{active_mode['label']} mode" if active_mode_copy_visible else "",
                        id="library-active-mode-title",
                        classes="destination-section",
                    )
                    active_mode_title.display = active_mode_copy_visible
                    yield active_mode_title
                    active_mode_description = Static(
                        active_mode["description"] if active_mode_copy_visible else "",
                        id="library-active-mode-description",
                    )
                    active_mode_description.display = active_mode_copy_visible
                    yield active_mode_description
                    active_mode_next_action = Static(
                        active_mode["next_action"] if active_mode_copy_visible else "",
                        id="library-active-mode-next-action",
                    )
                    active_mode_next_action.display = active_mode_copy_visible
                    yield active_mode_next_action
                    if self._active_mode in LIBRARY_STUDY_HANDOFF_MODES:
                        yield self._study_handoff_detail_widget()
                    if search_rag_panel_state is not None:
                        yield LibrarySearchRagPanel(
                            search_rag_panel_state,
                            id="library-search-rag-panel",
                        )
                    if collections_panel_state is not None:
                        yield LibraryCollectionsPanel(
                            collections_panel_state,
                            name_value=self._library_collection_name_input,
                            description_value=self._library_collection_description_input,
                            delete_pending=bool(self._library_collection_pending_delete_id),
                            id="library-collections-panel",
                        )
                    local_snapshot_region = Vertical(id="library-local-snapshot-region")
                    local_snapshot_region.display = show_local_snapshot_region
                    with local_snapshot_region:
                        if show_local_snapshot_region:
                            if not self._library_loaded:
                                yield Static(
                                    "Loading local Library sources...",
                                    id="library-source-loading",
                                )
                            elif self._library_lookup_error:
                                recovery_state = self._library_lookup_recovery_state
                                yield Static(
                                    self._library_lookup_error,
                                    id=(
                                        recovery_state.stable_selector
                                        if recovery_state is not None
                                        else "library-source-error"
                                    ),
                                )
                            elif self._active_mode == "conversations":
                                yield from self._conversation_browser_rows(workspace_depth_state)
                            elif self._active_mode == "import-export":
                                yield from self._import_export_workflow_rows()
                            elif not has_sources:
                                for hub_row in self._content_hub_rows():
                                    yield hub_row
                                yield Static(
                                    LIBRARY_EMPTY_COPY,
                                    id="library-source-empty",
                                )
                                yield Static(
                                    LIBRARY_EMPTY_NEXT_ACTION_COPY,
                                    id="library-source-empty-next-action",
                                )
                            else:
                                yield from self._content_hub_rows()

                with Vertical(
                    id="library-source-inspector",
                    classes="library-region destination-workbench-pane",
                ):
                    yield Static(
                        inspector_column_title,
                        id="library-source-inspector-title",
                        classes="destination-section",
                    )
                    action_region = Vertical(id="library-action-region")
                    action_region.styles.height = "auto"
                    with action_region:
                        yield from self._library_action_widgets(
                            workspace_depth_state=workspace_depth_state,
                            collection_scoped_actions_deferred=collection_scoped_actions_deferred,
                            handoff_disabled=handoff_disabled,
                            handoff_tooltip=handoff_tooltip,
                            collections_panel_state=collections_panel_state,
                        )
                    inspector_mode_region = Vertical(id="library-inspector-mode-region")
                    inspector_mode_region.styles.height = "1fr"
                    with inspector_mode_region:
                        if search_rag_panel_state is not None:
                            yield LibrarySearchRagInspectorPanel(
                                search_rag_panel_state,
                                id="library-rag-inspector",
                                classes="library-rag-region",
                            )
                        elif collections_panel_state is not None:
                            yield from self._collections_inspector_rows(collections_panel_state)
                        elif self._active_mode == "workspaces":
                            yield from self._workspaces_inspector_rows(workspace_depth_state)
                        elif self._active_mode == "conversations":
                            selected = self._selected_conversation_record()
                            yield Static(
                                "Conversation inspector",
                                id="library-inspector-title",
                                classes="destination-section",
                            )
                            if selected is None:
                                yield Static(
                                    "No conversation selected.",
                                    id="library-conversation-inspector-empty",
                                )
                                yield Static(
                                    "Select a saved conversation to inspect metadata and handoff eligibility.",
                                    id="library-conversation-inspector-empty-next-action",
                                )
                            else:
                                _, record = selected
                                yield Static(
                                    self._source_title("conversations", record),
                                    id="library-conversation-inspector-title",
                                )
                                yield Static(
                                    self._conversation_message_count_label(record),
                                    id="library-conversation-inspector-message-count",
                                )
                                yield Static(
                                    "Source authority: local",
                                    id="library-conversation-inspector-authority",
                                )
                                yield Static(
                                    self._conversation_handoff_label(workspace_depth_state),
                                    id="library-conversation-inspector-handoff",
                                )
                                yield Static(
                                    "Owner: Console/Conversations retains editing and saved-history management.",
                                    id="library-conversation-inspector-owner",
                                )
                        elif self._active_mode == "import-export":
                            yield from self._import_export_inspector_rows()
                        else:
                            yield from self._hub_inspector_rows(workspace_depth_state)

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

    async def _refresh_active_mode_widgets(self) -> None:
        active_mode = self._active_mode_contract()
        source_column_title, detail_column_title, inspector_column_title = self._active_column_titles()
        self.query_one("#library-status-row", Static).update(self._status_row_copy())
        self.query_one("#library-source-browser-title", Static).update(source_column_title)
        self.query_one("#library-source-detail-title", Static).update(detail_column_title)
        self.query_one("#library-source-inspector-title", Static).update(inspector_column_title)
        active_mode_copy_visible = self._active_mode not in {
            "collections",
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
            after="#library-active-mode-next-action",
        )
        await self._sync_inspector_mode_region(panel_state)

    async def _sync_study_handoff_detail(self) -> None:
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
        for hub_row in self._content_hub_rows():
            await region.mount(hub_row)
        if not self._has_local_sources():
            await region.mount(Static(LIBRARY_EMPTY_COPY, id="library-source-empty"))
            await region.mount(
                Static(
                    LIBRARY_EMPTY_NEXT_ACTION_COPY,
                    id="library-source-empty-next-action",
                )
            )

    async def _sync_collections_panel(self, *, refresh_snapshot: bool = False) -> None:
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
        await self._set_active_mode("import-export")

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
        run_action = panel_state.query_state.run_action
        run_button = self.query_one("#library-rag-run-query", Button)
        run_button.disabled = not run_action.enabled
        run_button.tooltip = run_action.tooltip

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
                    after="#library-rag-query-row",
                )
        else:
            for widget in recovery_widgets:
                await widget.remove()

        scope_container = self.query_one("#library-rag-source-scope", Vertical)
        scope_container.set_class(bool(panel_state.scope.recovery_copy), "has-recovery")
        self.query_one("#library-rag-scope-summary", Static).update(
            self._library_rag_scope_summary(panel_state)
        )
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
        for child in list(results_container.children)[1:]:
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
                Static(panel_state.next_action, id="library-rag-results-empty")
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
