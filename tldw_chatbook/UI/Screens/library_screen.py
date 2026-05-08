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
from ...Library.library_rag_service import (
    LibraryRagSearchOutcome,
    LibraryRagSearchRequest,
    run_library_rag_search,
)
from ...Library.library_rag_state import LibraryRagPanelState
from ...runtime_policy.types import PolicyDeniedError
from ...Utils.input_validation import sanitize_string, validate_text_input
from ...Widgets.Library import LibrarySearchRagInspectorPanel, LibrarySearchRagPanel
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
        "next_action": "Collections can feed Search/RAG citations/snippets, study generation, and Console.",
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
LIBRARY_MODE_BY_BUTTON_ID = {
    mode["button_id"]: mode_id for mode_id, mode in LIBRARY_MODES.items()
}


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

    def on_mount(self) -> None:
        super().on_mount()
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

    @staticmethod
    async def _resolve_maybe_awaitable(value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

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
            notes_value = list_notes(
                scope="local_note",
                limit=LIBRARY_SOURCE_PAGE_SIZE,
                offset=0,
                user_id=getattr(self.app_instance, "notes_user_id", None) or "default_user",
            )
            media_value = list_media(
                mode="local",
                page=1,
                results_per_page=LIBRARY_SOURCE_PAGE_SIZE,
                include_keywords=False,
            )
            conversation_value = list_conversations(
                mode="local",
                limit=LIBRARY_SOURCE_PAGE_SIZE,
                offset=0,
            )
            notes_result, media_result, conversation_result = await asyncio.gather(
                self._resolve_maybe_awaitable(notes_value),
                self._resolve_maybe_awaitable(media_value),
                self._resolve_maybe_awaitable(conversation_value),
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
            return "Ready"
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

    def compose_content(self) -> ComposeResult:
        has_sources = self._has_local_sources()
        status_label = self._status_label()
        handoff_disabled = True
        handoff_tooltip = "Stage Library source context after Library finishes loading."
        search_rag_panel_state = (
            self._library_rag_panel_state() if self._active_mode == "search" else None
        )

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
            with Horizontal(id="library-mode-bar", classes="ds-panel"):
                mode_label = Static(
                    "Modes:",
                    id="library-mode-label",
                    classes="destination-section",
                )
                yield mode_label
                for mode_id, mode in LIBRARY_MODES.items():
                    classes = "library-mode-chip"
                    if mode_id == self._active_mode:
                        classes = f"{classes} is-active"
                    yield Button(
                        mode["label"],
                        id=mode["button_id"],
                        classes=classes,
                        tooltip=mode["description"],
                    )

            with Horizontal(id="library-contract-grid", classes="ds-panel"):
                with Vertical(id="library-source-browser", classes="library-region"):
                    yield Static("Source Browser", classes="destination-section")
                    yield Button("Open Notes", id="library-open-notes", tooltip="Open saved notes and workspaces.")
                    yield Button("Open Media", id="library-open-media", tooltip="Open ingested media and transcripts.")
                    yield Button(
                        "Open Conversations",
                        id="library-open-conversations",
                        tooltip="Open saved conversation browsing inside Library.",
                    )
                    yield Button(
                        "Import/Export Sources",
                        id="library-open-import-export",
                        tooltip="Open source import and export tools.",
                    )
                    yield Button("Search/RAG", id="library-open-search", tooltip="Search or ask over indexed sources.")
                    yield Static(
                        "Workspaces: all local sources until workspace scoping is selected.",
                        id="library-workspace-scope",
                    )

                with Vertical(id="library-source-detail", classes="library-region"):
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

                with Vertical(id="library-source-inspector", classes="library-region"):
                    yield Static("Source Inspector", classes="destination-section")
                    yield Static("Authority: local", id="library-source-authority")
                    yield Static(
                        "Search/RAG: query selected Library sources or stage evidence in Console.",
                        id="library-rag-entry-point",
                    )
                    if search_rag_panel_state is not None:
                        yield LibrarySearchRagInspectorPanel(
                            search_rag_panel_state,
                            id="library-rag-inspector",
                            classes="library-rag-region",
                        )
                    yield Static("Knowledge workflow", classes="destination-section")
                    yield Static(
                        "Turn Library material into study sessions, flashcards, and quizzes.",
                        id="library-study-purpose",
                    )
                    yield Static(
                        "Study generation entry uses the visible Library source snapshot.",
                        id="library-study-generation-entry",
                    )
                    yield Button(
                        "Study Dashboard",
                        id="library-open-study",
                        tooltip="Open the Study dashboard for due cards, decks, quizzes, and resume actions.",
                    )
                    yield Button(
                        "Flashcards",
                        id="library-open-flashcards",
                        tooltip="Open flashcards for selected or imported Library material.",
                    )
                    yield Button(
                        "Quizzes",
                        id="library-open-quizzes",
                        tooltip="Open quizzes for selected or imported Library material.",
                    )
                    yield Button(
                        "Use in Console",
                        id="library-use-in-console",
                        disabled=handoff_disabled,
                        tooltip=handoff_tooltip,
                    )

    @on(Button.Pressed, ".library-mode-chip")
    async def switch_library_mode(self, event: Button.Pressed) -> None:
        mode_id = LIBRARY_MODE_BY_BUTTON_ID.get(event.button.id or "")
        if mode_id is None:
            return
        event.stop()
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

    async def _sync_search_rag_panel(self) -> None:
        mounted_widgets = list(self.query("#library-search-rag-panel, #library-rag-inspector"))
        for widget in mounted_widgets:
            await widget.remove()
        if self._active_mode != "search":
            return
        panel_state = self._library_rag_panel_state()
        detail = self.query_one("#library-source-detail", Vertical)
        await detail.mount(
            LibrarySearchRagPanel(panel_state, id="library-search-rag-panel"),
            after="#library-active-mode-next-action",
        )
        inspector = self.query_one("#library-source-inspector", Vertical)
        await inspector.mount(
            LibrarySearchRagInspectorPanel(
                panel_state,
                id="library-rag-inspector",
                classes="library-rag-region",
            ),
            after="#library-rag-entry-point",
        )

    @on(Input.Changed, "#library-rag-query-input")
    async def update_library_rag_query(self, event: Input.Changed) -> None:
        event.stop()
        if event.value == self._library_rag_query:
            return
        self._library_rag_query = event.value
        self._reset_library_rag_retrieval_state()
        await self._refresh_search_rag_panel_state_widgets()

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

        for selector, value in (
            ("#library-rag-retrieval-status", f"Status: {panel_state.retrieval_status.title()}"),
            ("#library-rag-next-action", panel_state.next_action),
        ):
            widgets = list(self.query(selector))
            if widgets:
                widgets[0].update(value)
        await self._refresh_library_rag_inspector_selection(panel_state)

        await self._refresh_library_rag_results_widgets(panel_state)

        console_action = panel_state.use_in_console_action
        console_button = self.query_one("#library-rag-use-in-console", Button)
        console_button.disabled = not console_action.enabled
        console_button.tooltip = console_action.tooltip

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

    async def _refresh_library_rag_inspector_selection(
        self,
        panel_state: LibraryRagPanelState,
    ) -> None:
        inspector_widgets = list(self.query("#library-rag-inspector"))
        if not inspector_widgets:
            return

        selected_widgets = list(self.query("#library-rag-selected-result"))
        empty_widgets = list(self.query("#library-rag-inspector-empty"))
        if panel_state.selected_result is not None:
            selected_text = f"Selected: {panel_state.selected_result.title}"
            for widget in empty_widgets:
                await widget.remove()
            if selected_widgets:
                selected_widgets[0].update(selected_text)
            else:
                await inspector_widgets[0].mount(
                    Static(selected_text, id="library-rag-selected-result"),
                    before="#library-rag-use-in-console",
                )
            return

        for widget in selected_widgets:
            await widget.remove()
        disabled_reason = panel_state.use_in_console_action.disabled_reason
        if empty_widgets:
            empty_widgets[0].update(disabled_reason)
        else:
            await inspector_widgets[0].mount(
                Static(disabled_reason, id="library-rag-inspector-empty"),
                before="#library-rag-use-in-console",
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
    def open_search(self) -> None:
        self.post_message(NavigateToScreen("search"))

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
