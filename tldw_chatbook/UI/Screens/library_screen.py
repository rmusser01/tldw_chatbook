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
from textual.widgets import Button, Static

from ...Chat.chat_handoff_models import ChatHandoffPayload
from ...runtime_policy.types import PolicyDeniedError
from ...Utils.input_validation import sanitize_string, validate_text_input
from ..Navigation.base_app_screen import BaseAppScreen
from ..Navigation.main_navigation import NavigateToScreen
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
            "Search/RAG mode: ask over indexed Library sources or open the existing "
            "Search/RAG surface."
        ),
        "next_action": "Ask in Console by staging selected Library context with Use in Console.",
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

    def compose_content(self) -> ComposeResult:
        has_sources = self._has_local_sources()
        status_label = self._status_label()
        handoff_disabled = True
        handoff_tooltip = "Stage Library source context after Library finishes loading."

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
            with Horizontal(id="library-mode-bar", classes="ds-panel") as mode_bar:
                mode_bar.styles.height = 3
                mode_bar.styles.min_height = 3
                mode_label = Static(
                    "Modes:",
                    id="library-mode-label",
                    classes="destination-section",
                )
                mode_label.styles.width = 8
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
                    mode_button.styles.height = 1
                    mode_button.styles.min_height = 1
                    yield mode_button

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
    def switch_library_mode(self, event: Button.Pressed) -> None:
        mode_id = LIBRARY_MODE_BY_BUTTON_ID.get(event.button.id or "")
        if mode_id is None:
            return
        event.stop()
        self._active_mode = mode_id
        self.refresh(recompose=True)

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
