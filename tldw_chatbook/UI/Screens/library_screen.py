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
from textual.geometry import Spacing
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
LIBRARY_EMPTY_COPY = "No local Library sources are available yet."
LIBRARY_EMPTY_NEXT_ACTION_COPY = "Import/Export Sources or Open Notes/Media to add content."
LIBRARY_INSPECTOR_EMPTY_COPY = "No source selected."
LIBRARY_INSPECTOR_EMPTY_NEXT_ACTION_COPY = (
    "Select a note, media item, conversation, collection, or RAG result to inspect."
)
LIBRARY_FRAME_BORDER = ("solid", "#6f7782")
LIBRARY_FRAME_PADDING = Spacing(1, 1, 1, 1)
LIBRARY_SOURCE_SNAPSHOT_TIMEOUT_SECONDS = 5.0
LIBRARY_COLLECTION_SYNC_CONFLICT_LIMIT = 200
LIBRARY_MODE_BAR_HEIGHT = 2
LIBRARY_MODE_LABEL_WIDTH = 7
LIBRARY_MODE_CHIP_MIN_WIDTH = 9
LIBRARY_MODE_CHIP_WIDTH_PADDING = 5
LIBRARY_MODES = {
    "sources": {
        "label": "Sources",
        "button_id": "library-mode-sources",
        "description": "Sources mode: browse notes, media, and conversations as reusable context.",
        "next_action": "Use in Console stages the visible source snapshot for grounded chat.",
    },
    "search": {
        "label": "Search/RAG",
        "button_id": "library-mode-search",
        "description": (
            "Search/RAG mode: ask over selected Library sources inside Library; "
            "standalone Search/RAG remains available from Source Browser."
        ),
        "next_action": "Run Search/RAG, inspect cited evidence, then Use in Console.",
    },
    "import-export": {
        "label": "Import/Export",
        "button_id": "library-mode-import-export",
        "description": "Import/Export mode: bring source material into Library or export it out.",
        "next_action": "Import/Export tools stay under Library, not Artifacts.",
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
        "description": "Collections mode: manage Library-owned reusable source sets.",
        "next_action": (
            "Create local source groups now; Collection-scoped Search/RAG citations/snippets, "
            "Study, and Console are later-stage."
        ),
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
    def _frame_library_region(widget: Any) -> Any:
        """Apply visible terminal workbench framing to Library panes."""
        widget.styles.border = LIBRARY_FRAME_BORDER
        widget.styles.padding = LIBRARY_FRAME_PADDING
        return widget

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

                return asyncio.run(await_result())
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

    def _source_sample_titles(self, source_type: str) -> list[str]:
        return [
            self._source_title(source_type, record)
            for record in self._local_source_records[source_type]
        ]

    @classmethod
    def _source_record_id(cls, record: Mapping[str, Any]) -> str | None:
        for key in ("id", "uuid", "record_id", "backing_id", "source_id"):
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

    def _source_study_context(self) -> StudyScopeContext | None:
        if not self._has_local_sources():
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

    def _active_mode_contract(self) -> Mapping[str, str]:
        return LIBRARY_MODES.get(self._active_mode, LIBRARY_MODES["sources"])

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
                Static("No Collection selected.", id="library-collection-inspector-empty"),
                Static(
                    "Select a Collection to inspect its source scope and read-only sync-safety labels.",
                    id="library-collection-inspector-empty-next-action",
                ),
            )
        return (
            Static("Selected Collection", id="library-inspector-title", classes="destination-section"),
            Static(selected.name, id="library-collection-inspector-name"),
            Static(selected.item_count_label, id="library-collection-inspector-item-count"),
            Static("What this means", classes="destination-section"),
            Static(
                "This is a read-only sync dry run. No server writes can run from this screen.",
                id="library-collection-inspector-sync-meaning",
            ),
            Static(selected.sync_status_label, id="library-collection-inspector-sync-status"),
            Static(selected.sync_status_detail, id="library-collection-inspector-sync-detail"),
        )

    def compose_content(self) -> ComposeResult:
        has_sources = self._has_local_sources()
        status_label = self._status_label()
        handoff_disabled = True
        handoff_tooltip = "Stage Library source context after Library finishes loading."
        search_rag_panel_state = (
            self._library_rag_panel_state() if self._active_mode == "search" else None
        )
        collections_panel_state = (
            self._library_collections_panel_state()
            if self._active_mode == "collections"
            else None
        )
        collection_scoped_actions_deferred = self._active_mode == "collections"

        with Vertical(id="library-shell"):
            yield Static("Library", id="library-title", classes="ds-destination-header")
            yield Static(
                "Source material, notes, media, conversations, collections, imports/exports, and Search/RAG.",
                id="library-purpose",
                classes="destination-purpose",
            )
            yield Static(
                f"Library | Sources, imports, Search/RAG, Workspaces, Collections, Study | {status_label} | Local",
                id="library-status-row",
                classes="destination-status-row",
            )
            mode_bar = DestinationModeStrip(id="library-mode-bar", classes="destination-mode-strip")
            mode_bar.styles.height = LIBRARY_MODE_BAR_HEIGHT
            mode_bar.styles.min_height = LIBRARY_MODE_BAR_HEIGHT
            with mode_bar:
                mode_label = Static(
                    "Modes:",
                    id="library-mode-label",
                    classes="destination-section",
                )
                mode_label.styles.width = LIBRARY_MODE_LABEL_WIDTH
                mode_label.styles.min_width = LIBRARY_MODE_LABEL_WIDTH
                mode_label.styles.height = LIBRARY_MODE_BAR_HEIGHT
                mode_label.styles.min_height = LIBRARY_MODE_BAR_HEIGHT
                yield mode_label
                for mode_id, mode in LIBRARY_MODES.items():
                    classes = "library-mode-chip"
                    if mode_id == self._active_mode:
                        classes = f"{classes} is-active"
                    mode_button = Button(
                        mode["label"],
                        id=mode["button_id"],
                        classes=classes,
                        tooltip=mode["description"],
                    )
                    mode_button.styles.width = max(
                        len(mode["label"]) + LIBRARY_MODE_CHIP_WIDTH_PADDING,
                        LIBRARY_MODE_CHIP_MIN_WIDTH,
                    )
                    mode_button.styles.min_width = 0
                    mode_button.styles.height = LIBRARY_MODE_BAR_HEIGHT
                    mode_button.styles.min_height = LIBRARY_MODE_BAR_HEIGHT
                    yield mode_button

            contract_grid = self._frame_library_region(
                Horizontal(id="library-contract-grid", classes="ds-panel destination-workbench")
            )
            with contract_grid:
                source_browser = self._frame_library_region(
                    Vertical(id="library-source-browser", classes="library-region destination-workbench-pane")
                )
                with source_browser:
                    yield Static("Source Browser", classes="destination-section")
                    yield Button(
                        "Open Notes",
                        id="library-open-notes",
                        classes="library-source-action",
                        tooltip="Open saved notes and workspaces.",
                    )
                    yield Button(
                        "Open Media",
                        id="library-open-media",
                        classes="library-source-action",
                        tooltip="Open ingested media and transcripts.",
                    )
                    yield Button(
                        "Open Conversations",
                        id="library-open-conversations",
                        classes="library-source-action",
                        tooltip="Open saved conversation browsing inside Library.",
                    )
                    yield Button(
                        "Import/Export Sources",
                        id="library-open-import-export",
                        classes="library-source-action",
                        tooltip="Open source import and export tools.",
                    )
                    yield Button(
                        "Search/RAG",
                        id="library-open-search",
                        classes="library-source-action",
                        tooltip="Search or ask over indexed sources.",
                    )
                    yield Button(
                        "Collections",
                        id="library-open-collections",
                        classes="library-source-action",
                        tooltip="Manage Library-owned reusable source sets.",
                    )
                    yield Static(
                        "Workspaces: all local sources until workspace scoping is selected.",
                        id="library-workspace-scope",
                    )

                source_detail = self._frame_library_region(
                    Vertical(id="library-source-detail", classes="library-region destination-workbench-pane")
                )
                with source_detail:
                    yield Static("Source Detail / Search Results", classes="destination-section")
                    active_mode = self._active_mode_contract()
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
                    yield Static("Local Library snapshot", classes="destination-section")
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
                        handoff_tooltip = (
                            recovery_state.disabled_tooltip
                            if recovery_state is not None
                            else "Library source services are unavailable; retry Library later."
                        )
                    elif not has_sources:
                        yield Static(
                            LIBRARY_EMPTY_COPY,
                            id="library-source-empty",
                        )
                        yield Static(
                            LIBRARY_EMPTY_NEXT_ACTION_COPY,
                            id="library-source-empty-next-action",
                        )
                        handoff_tooltip = "Stage Library source context after adding notes, media, or conversations."
                    else:
                        for source_type, label, widget_id in (
                            ("notes", "Notes", "library-notes-summary"),
                            ("media", "Media", "library-media-summary"),
                            ("conversations", "Conversations", "library-conversations-summary"),
                        ):
                            yield Static(
                                self._source_count_label(source_type, label),
                                id=widget_id,
                            )
                            for index, record in enumerate(self._local_source_records[source_type]):
                                yield Static(
                                    Text.from_markup(
                                        escape_markup(self._source_title(source_type, record))
                                    ),
                                    id=f"library-{source_type}-source-{index}",
                                )
                        handoff_disabled = False
                        handoff_tooltip = "Stage Library source context in Console."

                source_inspector = self._frame_library_region(
                    Vertical(id="library-source-inspector", classes="library-region destination-workbench-pane")
                )
                with source_inspector:
                    with Vertical(id="library-inspector-mode-region"):
                        if search_rag_panel_state is not None:
                            yield LibrarySearchRagInspectorPanel(
                                search_rag_panel_state,
                                id="library-rag-inspector",
                                classes="library-rag-region",
                            )
                        elif collections_panel_state is not None:
                            yield from self._collections_inspector_rows(collections_panel_state)
                        else:
                            yield Static("Inspector", id="library-inspector-title", classes="destination-section")
                            yield Static(LIBRARY_INSPECTOR_EMPTY_COPY, id="library-inspector-empty")
                            yield Static(
                                LIBRARY_INSPECTOR_EMPTY_NEXT_ACTION_COPY,
                                id="library-inspector-empty-next-action",
                            )
                    yield Static("Knowledge workflow", classes="destination-section")
                    yield Static(
                        (
                            "Collection-scoped Study, Flashcards, Quizzes, and Console "
                            "are later-stage."
                            if collection_scoped_actions_deferred
                            else "Turn Library material into study sessions, flashcards, and quizzes."
                        ),
                        id="library-study-purpose",
                    )
                    yield Static(
                        (
                            "Use Collections to organize source groups locally; scoped "
                            "execution remains deferred."
                            if collection_scoped_actions_deferred
                            else "Study generation entry uses the visible Library source snapshot."
                        ),
                        id="library-study-generation-entry",
                    )
                    yield Button(
                        "Study Dashboard",
                        id="library-open-study",
                        disabled=collection_scoped_actions_deferred,
                        tooltip=(
                            "Collection-scoped Study is not available yet."
                            if collection_scoped_actions_deferred
                            else "Open the Study dashboard for due cards, decks, quizzes, and resume actions."
                        ),
                    )
                    yield Button(
                        "Flashcards",
                        id="library-open-flashcards",
                        disabled=collection_scoped_actions_deferred,
                        tooltip=(
                            "Collection-scoped Flashcards are not available yet."
                            if collection_scoped_actions_deferred
                            else "Open flashcards for selected or imported Library material."
                        ),
                    )
                    yield Button(
                        "Quizzes",
                        id="library-open-quizzes",
                        disabled=collection_scoped_actions_deferred,
                        tooltip=(
                            "Collection-scoped Quizzes are not available yet."
                            if collection_scoped_actions_deferred
                            else "Open quizzes for selected or imported Library material."
                        ),
                    )
                    yield Button(
                        "Use in Console",
                        id="library-use-in-console",
                        disabled=handoff_disabled or collection_scoped_actions_deferred,
                        tooltip=(
                            "Collection-scoped Console handoff is not available yet."
                            if collection_scoped_actions_deferred
                            else handoff_tooltip
                        ),
                    )

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
        await self._refresh_active_mode_widgets()

    async def _refresh_active_mode_widgets(self) -> None:
        active_mode = self._active_mode_contract()
        self.query_one("#library-active-mode-title", Static).update(f"{active_mode['label']} mode")
        self.query_one("#library-active-mode-description", Static).update(active_mode["description"])
        self.query_one("#library-active-mode-next-action", Static).update(active_mode["next_action"])
        for mode_id, mode in LIBRARY_MODES.items():
            self.query_one(f"#{mode['button_id']}", Button).set_class(
                mode_id == self._active_mode,
                "is-active",
            )
        await self._sync_search_rag_panel()
        await self._sync_collections_panel(refresh_snapshot=True)
        self._sync_collection_scoped_action_buttons()

    async def _sync_search_rag_panel(self) -> None:
        mounted_widgets = list(self.query("#library-search-rag-panel"))
        for widget in mounted_widgets:
            await widget.remove()
        if self._active_mode != "search":
            await self._sync_inspector_mode_region(None)
            return
        panel_state = self._library_rag_panel_state()
        detail = self.query_one("#library-source-detail", Vertical)
        await detail.mount(
            LibrarySearchRagPanel(panel_state, id="library-search-rag-panel"),
            after="#library-active-mode-next-action",
        )
        await self._sync_inspector_mode_region(panel_state)

    async def _sync_inspector_mode_region(
        self,
        panel_state: LibraryRagPanelState | None,
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
        await region.mount(
            Static("Inspector", id="library-inspector-title", classes="destination-section")
        )
        await region.mount(Static(LIBRARY_INSPECTOR_EMPTY_COPY, id="library-inspector-empty"))
        await region.mount(
            Static(
                LIBRARY_INSPECTOR_EMPTY_NEXT_ACTION_COPY,
                id="library-inspector-empty-next-action",
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

    def _sync_collection_scoped_action_buttons(self) -> None:
        deferred = self._active_mode == "collections"
        for selector, deferred_tooltip, normal_tooltip in (
            (
                "#library-open-study",
                "Collection-scoped Study is not available yet.",
                "Open the Study dashboard for due cards, decks, quizzes, and resume actions.",
            ),
            (
                "#library-open-flashcards",
                "Collection-scoped Flashcards are not available yet.",
                "Open flashcards for selected or imported Library material.",
            ),
            (
                "#library-open-quizzes",
                "Collection-scoped Quizzes are not available yet.",
                "Open quizzes for selected or imported Library material.",
            ),
            (
                "#library-use-in-console",
                "Collection-scoped Console handoff is not available yet.",
                "Stage Library source context in Console.",
            ),
        ):
            buttons = list(self.query(selector))
            if buttons:
                if selector == "#library-use-in-console" and not deferred:
                    buttons[0].disabled = not self._has_local_sources()
                else:
                    buttons[0].disabled = deferred
                buttons[0].tooltip = deferred_tooltip if deferred else normal_tooltip
        purpose_widgets = list(self.query("#library-study-purpose"))
        if purpose_widgets:
            purpose_widgets[0].update(
                "Collection-scoped Study, Flashcards, Quizzes, and Console are later-stage."
                if deferred
                else "Turn Library material into study sessions, flashcards, and quizzes."
            )
        entry_widgets = list(self.query("#library-study-generation-entry"))
        if entry_widgets:
            entry_widgets[0].update(
                "Use Collections to organize source groups locally; scoped execution remains deferred."
                if deferred
                else "Study generation entry uses the visible Library source snapshot."
            )

    @on(Input.Changed, "#library-rag-query-input")
    async def update_library_rag_query(self, event: Input.Changed) -> None:
        event.stop()
        if event.value == self._library_rag_query:
            return
        self._library_rag_query = event.value
        self._reset_library_rag_retrieval_state()
        await self._refresh_search_rag_panel_state_widgets()

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
        event.stop()
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
        self.query_one("#library-rag-query-status", Static).update(
            f"Mode: {panel_state.query_state.mode_label} | Top {panel_state.query_state.top_k}"
        )
        run_action = panel_state.query_state.run_action
        run_button = self.query_one("#library-rag-run-query", Button)
        run_button.disabled = not run_action.enabled
        run_button.tooltip = run_action.tooltip

        recovery_widgets = list(self.query("#library-rag-query-recovery"))
        if panel_state.query_state.recovery_copy:
            if recovery_widgets:
                recovery_widgets[0].update(panel_state.query_state.recovery_copy)
            else:
                query_controls = self.query_one("#library-rag-query-controls", Vertical)
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
                await results_container.mount(
                    Static(
                        f"{index + 1}. {result.title}{score}",
                        id=f"library-rag-result-{index}",
                    )
                )
                await results_container.mount(
                    Button(
                        "Select evidence",
                        id=f"library-rag-select-result-{index}",
                        classes="library-rag-result-action",
                        tooltip="Select this evidence result for Console handoff.",
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

    @on(Button.Pressed, "#library-open-notes")
    def open_notes(self) -> None:
        self.post_message(NavigateToScreen("notes"))

    @on(Button.Pressed, "#library-open-media")
    def open_media(self) -> None:
        self.post_message(NavigateToScreen("media"))

    @on(Button.Pressed, "#library-open-conversations")
    def open_conversations(self) -> None:
        self.post_message(NavigateToScreen("conversation"))

    @on(Button.Pressed, "#library-open-import-export")
    def open_import_export(self) -> None:
        self.post_message(NavigateToScreen("ingest"))

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
