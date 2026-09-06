"""Operational Active/History switchboard for Console sessions (Ctrl+K)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha1
from typing import Any, ClassVar
from uuid import uuid4

from rich.markup import escape as escape_markup
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.events import DescendantFocus, MouseDown, MouseUp, Resize
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widgets import Button, Input, Static

from tldw_chatbook.Character_Chat.character_conversation_navigation import (
    CharacterConversationPage,
    CharacterConversationRow,
    CharacterKeywordIndexStatus,
    UnavailableCharacterReason,
)
from tldw_chatbook.Chat.console_conversation_activation import (
    CharacterConversationActivationRequest,
    ConsoleActivationPhase,
    ConsoleActivationResultKind,
    ConsoleConversationActivationResult,
)
from tldw_chatbook.Chat.console_switcher_state import (
    CONSOLE_SWITCHER_PAGE_LIMIT,
    ActivityGroup,
    ConsoleSwitcherActiveResult,
    ConsoleSwitcherCharacterResult,
    ConsoleSwitcherHistoryPage,
    SwitcherMode,
    UnavailableSessionNotice,
    build_console_character_results,
    build_console_switcher_entries,
    filter_console_active_results,
)
from tldw_chatbook.UI.character_display_text import sanitize_character_display_label
from tldw_chatbook.UI.Workbench.help import WorkbenchHelpPanel, WorkbenchHelpState
from tldw_chatbook.Utils.input_validation import (
    CONSOLE_SWITCHER_QUERY_MAX_LENGTH,
    validate_console_character_query,
    validate_console_switcher_query,
)
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin
from tldw_chatbook.Workspaces.conversation_browser_state import (
    ConsoleConversationBrowserInputRow,
)

_TITLE_LIMIT = 64
_SUBTITLE_LIMIT = 120
_GROUP_LABELS = {
    ActivityGroup.WAITING_FOR_YOU.value: "WAITING FOR YOU",
    ActivityGroup.WORKING.value: "WORKING",
    ActivityGroup.NEW_RESULTS.value: "NEW RESULTS",
    ActivityGroup.CURRENT.value: "CURRENT",
    ActivityGroup.OTHER_OPEN.value: "OTHER OPEN",
    "open": "OPEN AGENT TABS",
    "saved": "SAVED CHATS",
}
SEARCH_DEBOUNCE_SECONDS = 0.2
ACTIVE_PROJECTION_POLL_SECONDS = 0.2
RESULT_DISAPPEARED_COPY = (
    "The selected result is no longer available — selection moved."
)

HistoryLoader = Callable[..., Awaitable[ConsoleSwitcherHistoryPage]]
CharacterLoader = Callable[..., Awaitable[CharacterConversationPage]]
CharacterActivator = Callable[
    [CharacterConversationActivationRequest, asyncio.Event],
    Awaitable[ConsoleConversationActivationResult],
]
CharacterCommitWaiter = Callable[
    [CharacterConversationActivationRequest], Awaitable[None]
]
CharacterRecovery = Callable[[ConsoleSwitcherCharacterResult], Awaitable[bool]]
AuthoritySnapshot = Callable[[], tuple[str, str, int]]
ActiveProjectionLoader = Callable[
    [],
    tuple[
        tuple[ConsoleSwitcherActiveResult, ...],
        str,
        str,
        int,
        str,
    ],
]


@dataclass(frozen=True)
class ConsoleSwitcherChoice:
    """Immutable user intent returned by the switcher."""

    kind: str
    entry: ConsoleSwitcherActiveResult


ConsoleSwitcherResult = ConsoleSwitcherActiveResult | ConsoleSwitcherCharacterResult


@dataclass(frozen=True)
class _CommittedResultPayload:
    """One immutable row bound to an exact committed query presentation."""

    entry: ConsoleSwitcherResult
    generation: int
    mode: SwitcherMode
    query: str


@dataclass
class _ModeVisitState:
    """Transient per-open position for one switcher mode."""

    page_offset: int = 0
    selected_key: str = ""
    scroll_y: float = 0.0
    focused_result: bool = False
    explicit_navigation: bool = False


class ConsoleSessionSwitcherModal(
    SafeModalDismissMixin, ModalScreen["ConsoleSwitcherChoice | None"]
):
    """Switch among Active, History, and local Character conversations."""

    DEFAULT_CSS = """
    ConsoleSessionSwitcherModal { align: center middle; }
    #console-switcher-modal {
        width: 76; max-width: 100%; height: auto; max-height: 35;
        border: tall $surface-lighten-1;
        background: $panel; color: $text; padding: 1 1;
    }
    #console-switcher-mode-controls { height: 1; min-height: 1; }
    .console-switcher-mode {
        width: 1fr; min-width: 10; height: 1; min-height: 1;
        border: none; padding: 0 1; background: $panel; color: $text-muted;
    }
    .console-switcher-mode-current {
        background: $surface; color: $text; text-style: bold;
    }
    #console-switcher-character-mode { width: 19; min-width: 19; }
    .console-switcher-mode-divider { width: 1; height: 1; }
    #console-switcher-query {
        width: 100%; height: 1; min-height: 1; border: none; padding: 0;
    }
    #console-switcher-scope, #console-switcher-divider {
        width: 100%; height: 1; min-height: 1; overflow: hidden;
    }
    #console-switcher-results {
        height: auto; min-height: 2; max-height: 22; margin: 0;
        scrollbar-background: $panel; scrollbar-color: $text-muted;
    }
    .console-switcher-section {
        height: 1; color: $text-muted; text-style: bold;
    }
    #console-switcher-status { display: none; }
    #console-switcher-receipt-state {
        display: none; height: 1; color: $warning;
    }
    #console-switcher-feedback { display: none; }
    #console-switcher-selected-detail {
        height: 2; min-height: 2; max-height: 2; overflow: hidden;
        color: $text-muted;
    }
    #console-switcher-page-controls { height: 1; min-height: 1; }
    #console-switcher-page-status {
        width: 1fr; height: 1; content-align: center middle; color: $text-muted;
    }
    #console-switcher-previous-page, #console-switcher-next-page,
    #console-switcher-recovery, #console-switcher-confirm-mark-seen,
    #console-switcher-cancel {
        width: 10; min-width: 8; height: 1; min-height: 1;
        border: none; padding: 0 1;
    }
    #console-switcher-confirm-mark-seen { display: none; }
    #console-switcher-recovery { display: none; width: 19; }
    #console-switcher-footer { height: 1; min-height: 1; }
    #console-switcher-hints { width: 1fr; height: 1; color: $text-muted; overflow: hidden; }
    .console-switcher-result {
        width: 100%; height: 2; min-height: 2; margin: 0;
        content-align: left middle; text-align: left;
        border-left: solid $surface-lighten-1;
    }
    .console-switcher-result:focus {
        background: $surface; color: $text; text-style: bold underline;
    }
    .console-switcher-result-candidate { background: $surface; text-style: bold; }
    .console-switcher-result-waiting { border-left: solid $warning; }
    .console-switcher-result-working { border-left: solid $primary; }
    .console-switcher-result-new { border-left: solid $success; }
    .console-switcher-result-current { border-left: solid $accent; }
    .console-switcher-result-error { border-left: solid $error; }
    """

    SAFE_MODAL_CONTENT = "#console-switcher-modal"
    BINDINGS: ClassVar[list[tuple[str, str, str] | Binding]] = [
        ("escape", "request_safe_cancel", "Cancel"),
        ("f2", "rename_entry", "Rename"),
        ("f3", "toggle_mode", "Next mode"),
        Binding("down", "switcher_cursor_down", "Next result", priority=True),
        Binding("up", "switcher_cursor_up", "Previous result", priority=True),
        Binding("home", "switcher_cursor_home", "First result", priority=True),
        Binding("end", "switcher_cursor_end", "Last result", priority=True),
        Binding("pageup", "switcher_page_up", "Previous results", priority=True),
        Binding("pagedown", "switcher_page_down", "Next results", priority=True),
    ]

    def __init__(
        self,
        *,
        rows: tuple[ConsoleConversationBrowserInputRow, ...] = (),
        active_results: tuple[ConsoleSwitcherActiveResult, ...] | None = None,
        history_loader: HistoryLoader | None = None,
        character_loader: CharacterLoader | None = None,
        character_activate: CharacterActivator | None = None,
        character_commit_waiter: CharacterCommitWaiter | None = None,
        character_open_library: CharacterRecovery | None = None,
        initial_mode: SwitcherMode = SwitcherMode.ACTIVE,
        initial_character_query: str = "",
        preferred_native_session_id: str | None = None,
        profile_authority: str = "",
        authority_token: str = "",
        active_projection_generation: int = 0,
        authority_snapshot: AuthoritySnapshot | None = None,
        activity_receipt_state: str = "ready",
        active_projection_loader: ActiveProjectionLoader | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with an immediate Active snapshot and lazy History seam."""
        super().__init__(**kwargs)
        self._rows = rows
        self._legacy_rows = active_results is None
        self._active_results = (
            build_console_switcher_entries(rows, limit=CONSOLE_SWITCHER_PAGE_LIMIT)
            if active_results is None
            else tuple(active_results)
        )
        self._history_loader = history_loader
        self._character_loader = character_loader
        self._character_activate = character_activate
        self._character_commit_waiter = character_commit_waiter
        self._character_open_library = character_open_library
        self._preferred_native_session_id = str(
            preferred_native_session_id or ""
        ).strip()
        self._profile_authority = str(profile_authority or "")
        self._authority_token = str(authority_token or "")
        self._active_projection_generation = int(active_projection_generation)
        self._authority_snapshot = authority_snapshot
        self._active_projection_loader = active_projection_loader
        self._activity_receipt_state = str(activity_receipt_state or "ready")
        self._mode = initial_mode
        self._operational_query = ""
        self._character_query = self._validated_initial_character_query(
            initial_character_query
        )
        self._entries: tuple[ConsoleSwitcherResult, ...] = ()
        self._payload_by_widget_id: dict[str, ConsoleSwitcherResult] = {}
        self._committed_payload_by_widget_id: dict[str, _CommittedResultPayload] = {}
        self._committed_result_generation = 0
        self._mode_visits = {mode: _ModeVisitState() for mode in SwitcherMode}
        self._restoring_mode: SwitcherMode | None = initial_mode
        self._character_rows_by_key: dict[str, CharacterConversationRow] = {}
        self._candidate_index = 0
        self._rendered_query = ""
        self._page_offset = 0
        self._page_total = 0
        self._request_generation = 0
        self._instance_token = uuid4().hex
        self._query_pending = False
        self._pending_retained_key = ""
        self._pending_retained_index: int | None = None
        self._explicit_navigation = False
        self._selection_feedback = ""
        self._armed_mark_seen_key = ""
        self._widened_to_history = False
        self._character_data_revision = 0
        self._character_search_error = ""
        self._activation_phase = ConsoleActivationPhase.IDLE
        self._activation_cancellation: asyncio.Event | None = None
        self._activation_task: asyncio.Task[None] | None = None
        self._activation_failure_kind: ConsoleActivationResultKind | None = None
        self._committed_character_result: ConsoleSwitcherCharacterResult | None = None
        self._pointer_result_key = ""
        self._pointer_result: ConsoleSwitcherResult | None = None
        self._pointer_payload: _CommittedResultPayload | None = None
        self._ignore_next_result_pressed = False
        self._suppress_query_change = False
        self._compact_layout = False
        self._closed = False
        self._query_debounce_timer: Timer | None = None
        self._active_projection_timer: Timer | None = None

    def compose(self) -> ComposeResult:
        """Build the bounded switchboard structure."""
        with Vertical(id="console-switcher-modal"):
            with Horizontal(id="console-switcher-mode-controls"):
                yield Button(
                    "Active (0)",
                    id="console-switcher-active-mode",
                    classes="console-switcher-mode",
                    compact=True,
                )
                yield Static(
                    "|",
                    id="console-switcher-mode-divider",
                    classes="console-switcher-mode-divider",
                    markup=False,
                )
                yield Button(
                    "History",
                    id="console-switcher-history-mode",
                    classes="console-switcher-mode",
                    compact=True,
                )
                yield Static("|", classes="console-switcher-mode-divider", markup=False)
                yield Button(
                    "Character chats",
                    id="console-switcher-character-mode",
                    classes="console-switcher-mode",
                    compact=True,
                )
            yield Input(
                value=(
                    self._character_query
                    if self._mode is SwitcherMode.CHARACTER_CHATS
                    else self._operational_query
                ),
                placeholder="Search sessions, workspaces, waiting, running, or finished…",
                id="console-switcher-query",
                max_length=CONSOLE_SWITCHER_QUERY_MAX_LENGTH,
            )
            yield Static("", id="console-switcher-scope", markup=False)
            yield Static("─" * 48, id="console-switcher-divider", markup=False)
            yield VerticalScroll(id="console-switcher-results")
            yield Static("", id="console-switcher-receipt-state", markup=False)
            yield Static("", id="console-switcher-status", markup=False)
            yield Static("", id="console-switcher-feedback", markup=False)
            yield Static("", id="console-switcher-selected-detail", markup=False)
            with Horizontal(id="console-switcher-page-controls"):
                yield Button("Previous", id="console-switcher-previous-page")
                yield Static("", id="console-switcher-page-status", markup=False)
                yield Button("Refresh results", id="console-switcher-recovery")
                yield Button("Mark seen", id="console-switcher-confirm-mark-seen")
                yield Button("Next", id="console-switcher-next-page")
            with Horizontal(id="console-switcher-footer"):
                yield Static(
                    "Enter: switch · F3: History",
                    id="console-switcher-hints",
                    markup=False,
                )
                yield Button("Cancel", id="console-switcher-cancel")

    async def on_mount(self) -> None:  # type: ignore[override]
        """Paint Active immediately and leave History cold."""
        super().on_mount()
        self.query_one(
            "#console-switcher-modal", Vertical
        ).border_title = "Switch or resume"
        self._sync_modal_max_height()
        self._update_receipt_status()
        self.query_one("#console-switcher-query", Input).focus()
        initial_query = (
            self._character_query
            if self._mode is SwitcherMode.CHARACTER_CHATS
            else self._operational_query
        )
        await self._refresh_results(initial_query)
        if self._active_projection_loader is not None:
            self._active_projection_timer = self.set_interval(
                ACTIVE_PROJECTION_POLL_SECONDS,
                self._poll_active_projection,
            )

    def on_resize(self, event: Resize) -> None:
        """Keep the content-sized modal inside the live terminal viewport."""
        del event
        self._sync_modal_max_height()

    def _sync_modal_max_height(self) -> None:
        try:
            modal = self.query_one("#console-switcher-modal", Vertical)
            results = self.query_one("#console-switcher-results", VerticalScroll)
        except NoMatches:
            return
        viewport_height = self.app.size.height
        viewport_width = self.app.size.width
        self._compact_layout = viewport_height <= 20 or viewport_width <= 52
        modal.styles.max_height = min(35, viewport_height)
        if self._compact_layout:
            modal.styles.width = min(52, viewport_width)
            modal.styles.height = viewport_height
            results.styles.height = max(2, viewport_height - 12)
            results.styles.max_height = max(2, viewport_height - 12)
        else:
            modal.styles.width = min(76, viewport_width)
            section_count = len(
                {str(getattr(entry, "section", "") or "") for entry in self._entries}
            )
            result_rows = min(22, (2 * len(self._entries)) + section_count)
            estimated_rows = 12 + result_rows
            modal_height = min(35, viewport_height, max(14, estimated_rows))
            modal.styles.height = modal_height
            visible_result_rows = max(2, min(22, result_rows, modal_height - 12))
            results.styles.height = visible_result_rows
            results.styles.max_height = visible_result_rows
        self._update_receipt_status()

    @staticmethod
    def _validated_initial_character_query(value: object) -> str:
        return validate_console_character_query(value)

    def _set_results_pending(
        self,
        message: str,
        *,
        clear_pointer: bool = False,
        preserve_position: bool = True,
    ) -> None:
        """Remove the prior generation from the action surface immediately."""

        if preserve_position and not self._query_pending:
            self._pending_retained_key = self._focused_result_key()
            self._pending_retained_index = self._focused_result_index()
        elif not preserve_position:
            self._pending_retained_key = ""
            self._pending_retained_index = None
        self._query_pending = True
        if clear_pointer:
            self._pointer_payload = None
            self._pointer_result = None
            self._pointer_result_key = ""
        self._clear_mark_seen_confirmation()
        for button in self._result_buttons():
            button.disabled = True
        try:
            self.query_one(
                "#console-switcher-results", VerticalScroll
            ).styles.visibility = "hidden"
        except NoMatches:
            pass
        self._set_status(message)

    def on_unmount(self) -> None:
        """Invalidate late work without remounting into a closed screen."""
        self._closed = True
        self._request_generation += 1
        if (
            self._activation_phase is ConsoleActivationPhase.OPENING_CANCELLABLE
            and self._activation_cancellation is not None
        ):
            self._activation_cancellation.set()
        if self._query_debounce_timer is not None:
            self._query_debounce_timer.stop()
            self._query_debounce_timer = None
        if self._active_projection_timer is not None:
            self._active_projection_timer.stop()
            self._active_projection_timer = None
        # No super().on_unmount(): the dispatcher already invokes
        # SafeModalDismissMixin.on_unmount separately for this Unmount event
        # (TASK-31418).

    def _poll_active_projection(self) -> None:
        """Reconcile one memory-only live snapshot while the modal is open."""
        loader = self._active_projection_loader
        if self._closed or loader is None:
            return
        try:
            results, profile, token, generation, receipt_state = loader()
        except Exception:  # noqa: BLE001 - a polling seam must not break input
            return
        profile = str(profile or "")
        token = str(token or "")
        generation = int(generation)
        receipt_state = str(receipt_state or "ready")
        if profile != self._profile_authority or token != self._authority_token:
            self._query_pending = False
            self.dismiss(None)
            return
        if generation < self._active_projection_generation:
            return
        normalized_results = tuple(results)
        if (
            generation == self._active_projection_generation
            and normalized_results == self._active_results
            and receipt_state == self._activity_receipt_state
        ):
            return
        self.reconcile_active_results(
            normalized_results,
            profile_authority=profile,
            authority_token=token,
            projection_generation=generation,
            activity_receipt_state=receipt_state,
        )

    def reconcile_active_results(
        self,
        results: tuple[ConsoleSwitcherActiveResult, ...],
        *,
        profile_authority: str,
        authority_token: str,
        projection_generation: int,
        activity_receipt_state: str | None = None,
    ) -> None:
        """Accept a newer in-memory Active projection from this runtime only."""
        if (
            self._closed
            or profile_authority != self._profile_authority
            or authority_token != self._authority_token
            or projection_generation < self._active_projection_generation
        ):
            return
        self._active_projection_generation = projection_generation
        self._active_results = tuple(results)
        if activity_receipt_state is not None:
            self._activity_receipt_state = str(activity_receipt_state or "ready")
            self._update_receipt_status()
        try:
            query = self.query_one("#console-switcher-query", Input).value
        except NoMatches:
            return
        if self._mode is SwitcherMode.ACTIVE:
            focused_key = self._focused_result_key()
            if focused_key:
                filtered = filter_console_active_results(self._active_results, query)
                focused_index = next(
                    (
                        index
                        for index, entry in enumerate(filtered)
                        if entry.stable_result_key == focused_key
                    ),
                    None,
                )
                self._page_offset = (
                    (focused_index // CONSOLE_SWITCHER_PAGE_LIMIT)
                    * CONSOLE_SWITCHER_PAGE_LIMIT
                    if focused_index is not None
                    else 0
                )
        self.run_worker(
            self._refresh_results(query),
            exclusive=True,
            group="console-session-switcher-reconcile",
        )

    async def _refresh_results(self, query: str, *, reset_page: bool = False) -> bool:
        """Resolve one view and commit only when all captured fences match."""
        query = self._validated_query(query)
        if query is None:
            return False
        if reset_page:
            self._page_offset = 0
        self._request_generation += 1
        generation = self._request_generation
        captured = (
            self._instance_token,
            self._mode,
            self._page_offset,
            self._profile_authority,
            self._authority_token,
            self._active_projection_generation,
        )
        self._set_results_pending(
            "Searching local chats…"
            if self._mode is SwitcherMode.CHARACTER_CHATS and query.strip()
            else "Loading local chats…"
            if self._mode is SwitcherMode.CHARACTER_CHATS
            else "Searching…"
            if query.strip()
            else "Loading…"
        )
        self._widened_to_history = False
        self._character_search_error = ""
        try:
            self.query_one("#console-switcher-recovery", Button).display = False
        except NoMatches:
            pass

        page: ConsoleSwitcherHistoryPage | None = None
        character_page: CharacterConversationPage | None = None
        character_error = ""
        character_rows: tuple[CharacterConversationRow, ...] = ()
        widened_to_history = False
        if self._mode is SwitcherMode.ACTIVE:
            filtered = (
                build_console_switcher_entries(
                    self._rows,
                    query=query,
                    limit=max(CONSOLE_SWITCHER_PAGE_LIMIT, len(self._rows)),
                )
                if self._legacy_rows
                else filter_console_active_results(self._active_results, query)
            )
            entries = filtered[
                self._page_offset : self._page_offset + CONSOLE_SWITCHER_PAGE_LIMIT
            ]
            if query.strip() and not filtered and self._history_loader is not None:
                self._set_status("Searching History…")
                page = await self._load_history(query=query, offset=self._page_offset)
                entries = tuple(page.entries)
                widened_to_history = True
            elif len(filtered) > CONSOLE_SWITCHER_PAGE_LIMIT or self._page_offset:
                page = ConsoleSwitcherHistoryPage(
                    tuple(entries),
                    self._page_offset,
                    CONSOLE_SWITCHER_PAGE_LIMIT,
                    len(filtered),
                )
        elif self._mode is SwitcherMode.HISTORY:
            self._set_status(
                "Searching History…" if query.strip() else "Loading History…"
            )
            page = await self._load_history(query=query, offset=self._page_offset)
            entries = tuple(page.entries)
        else:
            self._set_status(
                "Searching local chats…" if query.strip() else "Loading local chats…"
            )
            character_page = await self._load_character(
                query=query, offset=self._page_offset
            )
            character_error = self._character_keyword_error(character_page)
            character_rows = tuple(character_page.rows)
            entries = (
                ()
                if character_error
                else build_console_character_results(
                    character_page.rows,
                    now=datetime.now(UTC),
                    limit=CONSOLE_SWITCHER_PAGE_LIMIT,
                )
            )

        if not self._request_is_current(generation, captured, query):
            return False
        self._query_pending = False
        self._widened_to_history = widened_to_history
        self._character_search_error = character_error
        if character_page is not None:
            self._character_data_revision = character_page.data_revision
            self._character_rows_by_key = {
                row.row_key: row for row in character_rows if row.row_key
            }
        self._page_total = (
            character_page.total
            if character_page is not None
            else page.total
            if page is not None
            else len(entries)
        )
        self._rendered_query = query
        await self._commit_entries(
            entries,
            page=page,
            character_page=character_page,
        )
        if self._character_search_error:
            self._set_status(self._character_search_error)
            recovery = self.query_one("#console-switcher-recovery", Button)
            recovery.label = "Refresh results"
            recovery.display = True
            self._update_selected_detail()
        return True

    @staticmethod
    def _character_keyword_error(page: CharacterConversationPage) -> str:
        return {
            CharacterKeywordIndexStatus.ABSENT: "Character source changed",
            CharacterKeywordIndexStatus.BUILDING: "Keyword search rebuilding",
            CharacterKeywordIndexStatus.FAILED: "Keyword search unavailable",
        }.get(page.keyword_status, "")

    async def _load_history(
        self, *, query: str, offset: int
    ) -> ConsoleSwitcherHistoryPage:
        if self._history_loader is None:
            return ConsoleSwitcherHistoryPage(
                (), offset, CONSOLE_SWITCHER_PAGE_LIMIT, 0
            )
        try:
            return await self._history_loader(
                query=query,
                offset=offset,
                limit=CONSOLE_SWITCHER_PAGE_LIMIT,
            )
        except Exception:  # noqa: BLE001 - History failure stays inside modal
            return ConsoleSwitcherHistoryPage(
                (),
                offset,
                CONSOLE_SWITCHER_PAGE_LIMIT,
                0,
                error=(
                    "History is temporarily unavailable. "
                    "Active agents are still usable."
                ),
            )

    async def _load_character(
        self, *, query: str, offset: int
    ) -> CharacterConversationPage:
        if self._character_loader is None:
            return CharacterConversationPage((), 0, None, 0)
        try:
            return await self._character_loader(
                query=query,
                offset=offset,
                limit=CONSOLE_SWITCHER_PAGE_LIMIT,
            )
        except Exception:  # noqa: BLE001 - local search failure stays recoverable
            return CharacterConversationPage(
                (), 0, None, 0, CharacterKeywordIndexStatus.FAILED
            )

    def _request_is_current(
        self, generation: int, captured: tuple[Any, ...], query: str
    ) -> bool:
        current = (
            self._instance_token,
            self._mode,
            self._page_offset,
            self._profile_authority,
            self._authority_token,
            self._active_projection_generation,
        )
        if (
            self._closed
            or generation != self._request_generation
            or captured != current
        ):
            return False
        try:
            if self.query_one("#console-switcher-query", Input).value != query:
                return False
        except NoMatches:
            return False
        if self._authority_snapshot is not None:
            profile, token, projection = self._authority_snapshot()
            if (profile, token, projection) != current[3:]:
                self._query_pending = False
                self._set_status("Activity changed — refreshing…")
                if self._active_projection_loader is not None:
                    self.call_after_refresh(self._poll_active_projection)
                return False
        return True

    async def _commit_entries(
        self,
        entries: tuple[ConsoleSwitcherResult, ...],
        *,
        page: ConsoleSwitcherHistoryPage | None,
        character_page: CharacterConversationPage | None = None,
    ) -> None:
        self._clear_mark_seen_confirmation()
        restore_visit = (
            self._mode_visits[self._mode]
            if self._restoring_mode is self._mode
            else None
        )
        previous_key = self._candidate_key()
        focused_key = self._pending_retained_key or self._focused_result_key()
        retained_key = (
            restore_visit.selected_key
            if restore_visit is not None
            else focused_key or previous_key
        )
        previous_index = (
            self._pending_retained_index
            if self._pending_retained_index is not None
            else self._focused_result_index()
            if focused_key
            else self._candidate_index
        )
        self._pending_retained_key = ""
        self._pending_retained_index = None
        had_previous_entries = bool(self._entries)
        self._entries = entries[:CONSOLE_SWITCHER_PAGE_LIMIT]
        retained_disappeared = bool(
            had_previous_entries
            and focused_key
            and self._index_for_key(retained_key) is None
        )
        results = self.query_one("#console-switcher-results", VerticalScroll)
        await results.remove_children()
        self._payload_by_widget_id.clear()
        self._committed_payload_by_widget_id.clear()
        self._committed_result_generation += 1
        committed_generation = self._committed_result_generation

        if not self._entries:
            await results.mount(
                Static(
                    self._empty_copy(page),
                    id="console-switcher-empty",
                    markup=False,
                )
            )
            self._candidate_index = 0
        else:
            if restore_visit is not None and retained_key:
                restored_index = self._index_for_key(retained_key)
                self._candidate_index = restored_index or 0
            elif retained_disappeared:
                self._candidate_index = min(previous_index, len(self._entries) - 1)
            else:
                self._candidate_index = self._choose_candidate(retained_key)
            widgets: list[Static | Button] = []
            previous_section = ""
            for index, entry in enumerate(self._entries):
                section = str(getattr(entry, "section", "") or "")
                if (
                    section
                    and section != previous_section
                    and not self._compact_layout
                    and self._mode is not SwitcherMode.CHARACTER_CHATS
                ):
                    widgets.append(
                        Static(
                            _GROUP_LABELS.get(section, section.upper()),
                            id=f"console-switcher-section-{len(widgets)}",
                            classes="console-switcher-section",
                            markup=False,
                        )
                    )
                    previous_section = section
                widget_id = self._result_widget_id(index, entry)
                button = Button(
                    self._entry_label(index, entry),
                    id=widget_id,
                    classes=" ".join(self._result_classes(entry)),
                    compact=True,
                )
                button.set_class(
                    bool(getattr(entry, "is_active", False)),
                    "console-switcher-result-active",
                )
                button.set_class(
                    index == self._candidate_index,
                    "console-switcher-result-candidate",
                )
                button.tooltip = self._entry_tooltip(entry)
                self._payload_by_widget_id[widget_id] = entry
                self._committed_payload_by_widget_id[widget_id] = (
                    _CommittedResultPayload(
                        entry,
                        committed_generation,
                        self._mode,
                        self._rendered_query,
                    )
                )
                widgets.append(button)
            await results.mount_all(widgets)
        results.styles.visibility = "visible"

        self._update_mode_controls()
        self._update_page_controls(page, character_page=character_page)
        if page is not None and page.error:
            self._set_status(page.error)
        elif self._widened_to_history:
            self._update_selection_status(prefix="Active · showing History matches")
        else:
            self._update_selection_status()

        self._update_selected_detail()

        if retained_disappeared:
            self._selection_feedback = RESULT_DISAPPEARED_COPY
            if self._entries:
                buttons = self._result_buttons()
                self._focus_candidate(buttons)
            else:
                self.query_one("#console-switcher-query", Input).focus()
            self._set_status(self._selection_feedback)
            self.notify(RESULT_DISAPPEARED_COPY, severity="warning")
        elif restore_visit is not None:
            if restore_visit.focused_result:
                restored = self._button_for_key(retained_key)
                if restored is not None:
                    restored.focus()
            results.scroll_to(
                y=restore_visit.scroll_y,
                animate=False,
                immediate=True,
                force=True,
            )
            self.call_after_refresh(
                results.scroll_to,
                y=restore_visit.scroll_y,
                animate=False,
                immediate=True,
                force=True,
            )
        elif focused_key:
            focused = self._button_for_key(focused_key)
            if focused is not None:
                focused.focus()
        self._restoring_mode = None
        self._sync_modal_max_height()

    def _empty_copy(self, page: ConsoleSwitcherHistoryPage | None) -> str:
        query = self._rendered_query.strip()
        if page is not None and page.error:
            return page.error
        if self._mode is SwitcherMode.CHARACTER_CHATS:
            if self._character_search_error:
                return self._character_search_error
            return (
                "No Keyword matches"
                if query
                else "Type a Keyword to search local Character chats"
            )
        if self._mode is SwitcherMode.ACTIVE and not query and not self._active_results:
            return (
                "No active agents yet. Ctrl+T creates an agent tab. Use F3 for "
                "saved conversation History."
            )
        if query:
            return (
                "No matches. Try a title, workspace:<name>, or is:waiting, "
                "is:working, is:new, is:failed, or is:saved."
            )
        return "No saved conversations yet. Return to Active or start a new agent tab."

    def _choose_candidate(self, retained_key: str) -> int:
        query = self.query_one("#console-switcher-query", Input).value
        if query.strip():
            return 0
        if self._explicit_navigation and retained_key:
            retained = self._index_for_key(retained_key)
            if retained is not None:
                return retained
        if self._mode is SwitcherMode.ACTIVE and not self._legacy_rows:
            preferred = next(
                (
                    index
                    for index, entry in enumerate(self._entries)
                    if getattr(entry, "native_session_id", None)
                    == self._preferred_native_session_id
                ),
                None,
            )
            if preferred is not None:
                return preferred
            other_open = next(
                (
                    index
                    for index, entry in enumerate(self._entries)
                    if getattr(entry, "native_session_id", None)
                    and not bool(getattr(entry, "is_active", False))
                ),
                None,
            )
            if other_open is not None:
                return other_open
        if self._legacy_rows and self._preferred_native_session_id:
            preferred = next(
                (
                    index
                    for index, entry in enumerate(self._entries)
                    if getattr(entry, "native_session_id", None)
                    == self._preferred_native_session_id
                ),
                None,
            )
            if preferred is not None:
                return preferred
        retained = self._index_for_key(retained_key)
        return retained if retained is not None else 0

    def _result_widget_id(self, index: int, entry: ConsoleSwitcherResult) -> str:
        if self._legacy_rows:
            return f"console-switcher-result-{index}"
        digest = sha1(entry.stable_result_key.encode("utf-8")).hexdigest()[:16]
        return f"console-switcher-result-{digest}"

    def _character_state(self, entry: ConsoleSwitcherCharacterResult) -> str:
        """Derive truthful UI state without widening the frozen result contract."""

        committed = self._committed_character_result
        if (
            committed is not None
            and committed.stable_result_key == entry.stable_result_key
        ):
            if self._activation_phase is ConsoleActivationPhase.OPENING_CANCELLABLE:
                return "OPENING"
            if self._activation_phase is ConsoleActivationPhase.COMMITTING:
                return "FINISHING"
            if (
                self._activation_phase is ConsoleActivationPhase.FAILURE_VISIBLE
                and self._activation_failure_kind
                is ConsoleActivationResultKind.CHARACTER_UNAVAILABLE
            ):
                return "CHARACTER UNAVAILABLE"
        row = self._character_rows_by_key.get(entry.row_key)
        if entry.target is None:
            reason = row.unavailable_reason if row is not None else None
            return {
                UnavailableCharacterReason.DELETED_CARD: "DELETED CARD",
                UnavailableCharacterReason.MISSING_CHARACTER_AUTHORITY_LINK: (
                    "CHARACTER SOURCE CHANGED"
                ),
                UnavailableCharacterReason.AMBIGUOUS_LEGACY_LINK: (
                    "CHARACTER SOURCE CHANGED"
                ),
                UnavailableCharacterReason.MISSING_CARD: "CHARACTER UNAVAILABLE",
            }.get(reason, "CHARACTER UNAVAILABLE")
        matching_open = next(
            (
                active
                for active in self._active_results
                if (active_target := getattr(active, "target", None)) is not None
                and getattr(active, "native_session_id", None)
                and active_target.profile_authority == self._profile_authority
                and active_target.authority_token == self._authority_token
                and active_target.session_id
                == getattr(active, "native_session_id", None)
                and active_target.conversation_id == entry.target.conversation_id
                and getattr(active, "conversation_id", None)
                == entry.target.conversation_id
            ),
            None,
        )
        if bool(matching_open and getattr(matching_open, "is_active", False)):
            return "CURRENT TAB"
        if matching_open is not None:
            return "OPEN TAB"
        return "RESUME CHAT"

    def _entry_action(self, entry: ConsoleSwitcherResult) -> str:
        if isinstance(entry, UnavailableSessionNotice):
            return "MARK SEEN"
        if isinstance(entry, ConsoleSwitcherCharacterResult):
            state = self._character_state(entry)
            if state in {"OPENING", "FINISHING"}:
                return state
            if entry.target is None or state == "CHARACTER UNAVAILABLE":
                return "VIEW DETAILS"
            if state in {"CURRENT TAB", "OPEN TAB"}:
                return "OPEN TAB"
            return "RESUME CHAT"
        return "OPEN TAB" if getattr(entry, "native_session_id", None) else "OPEN"

    def _entry_tooltip(self, entry: ConsoleSwitcherResult) -> str:
        return escape_markup(f"{self._entry_action(entry)}: {entry.title}")

    def _entry_label(self, index: int, entry: ConsoleSwitcherResult) -> Text:
        available_width = max(20, min(70, self.app.size.width - 4))
        marker = "▸" if index == self._candidate_index else " "
        if isinstance(entry, ConsoleSwitcherCharacterResult):
            state = self._character_state(entry)
            title = (
                sanitize_character_display_label(
                    entry.title, max_characters=_TITLE_LIMIT * 2
                )
                or "Untitled conversation"
            )
            first = Text(f"{marker} [{state}] {title}")
            first.truncate(available_width, overflow="ellipsis", pad=False)
            character = (
                sanitize_character_display_label(
                    entry.character_label, max_characters=_TITLE_LIMIT
                )
                or "Unavailable character"
            )
            second = Text(f"  {character} · Local · {entry.relative_time}")
            second.truncate(available_width, overflow="ellipsis", pad=False)
            return Text.assemble(first, "\n", second)
        title_limit = max(8, min(_TITLE_LIMIT, available_width - 22))
        display_title = (
            sanitize_character_display_label(entry.title, max_characters=title_limit)
            or "Untitled conversation"
        )
        state = sanitize_character_display_label(
            str(getattr(entry, "state_label", "") or self._fallback_state(entry)),
            max_characters=18,
        )
        metadata = sanitize_character_display_label(
            self._entry_metadata(entry), max_characters=_SUBTITLE_LIMIT
        )
        label = f"{marker} {state:<18} {display_title}"
        if metadata:
            label = f"{label}\n  {metadata}"
        return Text(label)

    @staticmethod
    def _fallback_state(entry: ConsoleSwitcherResult) -> str:
        if isinstance(entry, UnavailableSessionNotice):
            return entry.primary_status.upper()
        if getattr(entry, "native_session_id", None):
            return "OPEN AGENT"
        return "SAVED CHAT"

    @staticmethod
    def _entry_metadata(entry: ConsoleSwitcherResult) -> str:
        if isinstance(entry, UnavailableSessionNotice):
            count = len(entry.receipts)
            return f"{count} unseen {'update' if count == 1 else 'updates'}"

        parts: list[str] = []
        workspace = str(getattr(entry, "workspace_label", "") or "").strip()
        if workspace:
            parts.append(workspace)
        parts.append(
            "Console tab" if getattr(entry, "native_session_id", None) else "Saved chat"
        )

        recency = ""
        for candidate in reversed(
            str(getattr(entry, "subtitle", "") or "").split(" · ")
        ):
            value = candidate.strip()
            lowered = value.casefold()
            if (
                lowered in {"now", "today", "yesterday", "older", "previous 7 days"}
                or lowered.endswith(" ago")
                or (
                    len(lowered) > 1
                    and lowered[:-1].isdigit()
                    and lowered[-1] in "smhdwy"
                )
            ):
                recency = value
                break
        if recency:
            parts.append(recency)
        multiplicity = int(getattr(entry, "multiplicity", 0) or 0)
        if multiplicity:
            parts.append(f"{multiplicity + 1} updates")
        return " · ".join(parts)

    @staticmethod
    def _result_classes(entry: ConsoleSwitcherResult) -> tuple[str, ...]:
        classes = ["console-switcher-result"]
        if isinstance(entry, ConsoleSwitcherCharacterResult):
            if entry.target is None:
                classes.append("console-switcher-result-error")
            return tuple(classes)
        state = str(
            getattr(entry, "activity_state", "") or getattr(entry, "primary_status", "")
        ).casefold()
        if state in {"failed", "error", "stuck", "stopped", "cancelled"}:
            classes.append("console-switcher-result-error")
            return tuple(classes)
        group = getattr(entry, "group", None)
        group_classes = {
            ActivityGroup.WAITING_FOR_YOU: "console-switcher-result-waiting",
            ActivityGroup.WORKING: "console-switcher-result-working",
            ActivityGroup.NEW_RESULTS: "console-switcher-result-new",
            ActivityGroup.CURRENT: "console-switcher-result-current",
        }
        group_class = group_classes.get(group)
        if group_class:
            classes.append(group_class)
        return tuple(classes)

    @on(Input.Changed, "#console-switcher-query")
    def _query_changed(self, event: Input.Changed) -> None:
        event.stop()
        if self._suppress_query_change:
            return
        if self._activation_phase in {
            ConsoleActivationPhase.OPENING_CANCELLABLE,
            ConsoleActivationPhase.COMMITTING,
        }:
            self._restore_owned_query()
            return
        if self._mode is SwitcherMode.CHARACTER_CHATS:
            self._character_query = event.value
        else:
            self._operational_query = event.value
        self._selection_feedback = ""
        self._clear_mark_seen_confirmation()
        self._cancel_query_debounce()
        self._set_results_pending(
            "Searching local chats…"
            if self._mode is SwitcherMode.CHARACTER_CHATS
            else "Searching…",
            clear_pointer=True,
            preserve_position=False,
        )
        query = self._validated_query(event.value)
        if query is None:
            return
        self._query_debounce_timer = self.set_timer(
            SEARCH_DEBOUNCE_SECONDS,
            lambda: self.run_worker(
                self._refresh_results(query, reset_page=True),
                exclusive=True,
                group="console-session-switcher-search",
            ),
        )

    def _cancel_query_debounce(self) -> None:
        if self._query_debounce_timer is not None:
            self._query_debounce_timer.stop()
            self._query_debounce_timer = None

    def _validated_query(self, value: object) -> str | None:
        """Validate one UI query and expose bounded recovery copy."""
        try:
            if self._mode is SwitcherMode.CHARACTER_CHATS:
                return validate_console_character_query(value)
            return validate_console_switcher_query(value)
        except ValueError as error:
            self._query_pending = False
            if self._mode is SwitcherMode.CHARACTER_CHATS:
                self._set_feedback(str(error))
                self.query_one("#console-switcher-selected-detail", Static).update(
                    str(error)
                )
                return None
            self._set_feedback(
                f"Search is limited to {CONSOLE_SWITCHER_QUERY_MAX_LENGTH} characters."
            )
            return None

    @on(Input.Submitted, "#console-switcher-query")
    async def _query_submitted(self, event: Input.Submitted) -> None:
        """Commit the exact query before applying its explicit candidate."""
        event.stop()
        self._cancel_query_debounce()
        query = self._validated_query(event.value)
        if query is None:
            return
        if query != self._rendered_query or self._query_pending:
            committed = await self._refresh_results(query, reset_page=True)
            if not committed:
                return
        if not self._entries:
            return
        index = self._candidate_index
        if not query.strip() and not self._explicit_navigation:
            index = self._choose_candidate(self._candidate_key())
            self._candidate_index = index
            self._sync_candidate_labels()
        if not 0 <= index < len(self._entries):
            return
        entry = self._entries[index]
        payload = self._committed_payload_for_key(entry.stable_result_key)
        if payload is None or not self._payload_is_current(payload):
            return
        if isinstance(entry, ConsoleSwitcherCharacterResult):
            self._begin_character_activation(entry)
            return
        if isinstance(entry, UnavailableSessionNotice):
            button = self._button_for_key(entry.stable_result_key)
            if button is not None:
                self._arm_mark_seen(entry, button, input_name="Enter")
            return
        self._activate_choice("activate", entry)

    def _result_buttons(self) -> list[Button]:
        try:
            results = self.query_one("#console-switcher-results", VerticalScroll)
        except NoMatches:
            return []
        return [
            button
            for button in results.query(Button)
            if button.has_class("console-switcher-result")
        ]

    def _focused_result_index(self) -> int | None:
        focused = self.app.focused
        for index, button in enumerate(self._result_buttons()):
            if button is focused:
                return index
        return None

    def _focused_result_key(self) -> str:
        focused = self.app.focused
        if not isinstance(focused, Button):
            return ""
        entry = self._payload_by_widget_id.get(focused.id or "")
        return entry.stable_result_key if entry is not None else ""

    def _candidate_key(self) -> str:
        if 0 <= self._candidate_index < len(self._entries):
            return self._entries[self._candidate_index].stable_result_key
        return ""

    def _index_for_key(self, key: str) -> int | None:
        return next(
            (
                index
                for index, entry in enumerate(self._entries)
                if entry.stable_result_key == key
            ),
            None,
        )

    def _button_for_key(self, key: str) -> Button | None:
        for button in self._result_buttons():
            entry = self._payload_by_widget_id.get(button.id or "")
            if entry is not None and entry.stable_result_key == key:
                return button
        return None

    def _entry_for_key(self, key: str) -> ConsoleSwitcherResult | None:
        index = self._index_for_key(key)
        return self._entries[index] if index is not None else None

    def _committed_payload_for_key(self, key: str) -> _CommittedResultPayload | None:
        return next(
            (
                payload
                for payload in self._committed_payload_by_widget_id.values()
                if payload.entry.stable_result_key == key
            ),
            None,
        )

    def _payload_is_current(self, payload: _CommittedResultPayload) -> bool:
        """Require the exact row to belong to the visible committed query."""

        if (
            self._closed
            or self._query_pending
            or payload.generation != self._committed_result_generation
            or payload.mode is not self._mode
            or payload.query != self._rendered_query
        ):
            return False
        try:
            visible_query = self.query_one("#console-switcher-query", Input).value
        except NoMatches:
            return False
        current = self._committed_payload_for_key(payload.entry.stable_result_key)
        return (
            visible_query == payload.query
            and current is not None
            and current.generation == payload.generation
        )

    def _pointer_payload_can_activate(self, payload: _CommittedResultPayload) -> bool:
        """Preserve same-query pointer identity while rejecting query churn."""

        current = self._committed_payload_for_key(payload.entry.stable_result_key)
        if current is None or not self._payload_is_current(current):
            return False
        return current.mode is payload.mode and current.query == payload.query

    @property
    def _activation_in_flight(self) -> bool:
        return self._activation_phase in {
            ConsoleActivationPhase.OPENING_CANCELLABLE,
            ConsoleActivationPhase.COMMITTING,
        }

    def action_switcher_cursor_down(self) -> None:
        if self._activation_in_flight or self._query_pending:
            return
        buttons = self._result_buttons()
        if not buttons:
            return
        index = self._focused_result_index()
        self._explicit_navigation = True
        self._selection_feedback = ""
        self._clear_mark_seen_confirmation()
        if index is None:
            self._candidate_index = 0
        elif index + 1 < len(buttons):
            self._candidate_index = index + 1
        else:
            return
        self._focus_candidate(buttons)

    def action_switcher_cursor_up(self) -> None:
        if self._activation_in_flight or self._query_pending:
            return
        buttons = self._result_buttons()
        index = self._focused_result_index()
        if index is None or index == 0:
            try:
                self.query_one("#console-switcher-query", Input).focus()
            except NoMatches:
                pass
            return
        self._explicit_navigation = True
        self._selection_feedback = ""
        self._clear_mark_seen_confirmation()
        self._candidate_index = index - 1
        self._focus_candidate(buttons)

    def action_switcher_cursor_home(self) -> None:
        self._move_candidate(0)

    def action_switcher_cursor_end(self) -> None:
        buttons = self._result_buttons()
        if buttons:
            self._move_candidate(len(buttons) - 1, buttons=buttons)

    def action_switcher_page_up(self) -> None:
        self._move_candidate_by_page(-1)

    def action_switcher_page_down(self) -> None:
        self._move_candidate_by_page(1)

    def _move_candidate_by_page(self, direction: int) -> None:
        buttons = self._result_buttons()
        if not buttons:
            return
        results = self.query_one("#console-switcher-results", VerticalScroll)
        step = max(1, results.content_region.height // 2)
        current = self._focused_result_index()
        if current is None:
            current = self._candidate_index
        self._move_candidate(current + (step * direction), buttons=buttons)

    def _move_candidate(
        self, index: int, *, buttons: list[Button] | None = None
    ) -> None:
        if self._activation_in_flight or self._query_pending:
            return
        mounted = buttons if buttons is not None else self._result_buttons()
        if not mounted:
            return
        self._explicit_navigation = True
        self._selection_feedback = ""
        self._clear_mark_seen_confirmation()
        self._candidate_index = min(max(0, index), len(mounted) - 1)
        self._focus_candidate(mounted)

    def _focus_candidate(self, buttons: list[Button]) -> None:
        if not 0 <= self._candidate_index < len(buttons):
            return
        button = buttons[self._candidate_index]
        button.focus()
        self._sync_candidate_labels(buttons)
        self._update_selection_status()
        button.scroll_visible(animate=False, immediate=True)

    def _sync_candidate_labels(self, buttons: list[Button] | None = None) -> None:
        mounted = buttons if buttons is not None else self._result_buttons()
        for index, button in enumerate(mounted):
            entry = self._payload_by_widget_id.get(button.id or "")
            if entry is None:
                continue
            button.set_class(
                index == self._candidate_index,
                "console-switcher-result-candidate",
            )
            button.label = self._entry_label(index, entry)
            button.tooltip = self._entry_tooltip(entry)

    def on_descendant_focus(self, event: DescendantFocus) -> None:
        widget = event.widget
        if (
            not isinstance(widget, Button)
            or not widget.has_class("console-switcher-result")
            or widget is not self.app.focused
        ):
            return
        if self._query_pending:
            return
        entry = self._payload_by_widget_id.get(widget.id or "")
        if entry is None:
            return
        if self._activation_in_flight:
            committed = self._committed_character_result
            if committed is not None:
                button = self._button_for_key(committed.stable_result_key)
                if button is not None and button is not widget:
                    self.call_after_refresh(button.focus)
            return
        index = self._index_for_key(entry.stable_result_key)
        if index is None:
            return
        if (
            self._armed_mark_seen_key
            and entry.stable_result_key != self._armed_mark_seen_key
        ):
            self._clear_mark_seen_confirmation()
        self._explicit_navigation = True
        self._candidate_index = index
        self._sync_candidate_labels()
        self._update_selection_status()

    def on_mouse_down(self, event: MouseDown) -> None:
        """Freeze the payload identity at pointer press, before any await."""
        widget = event.widget
        if not isinstance(widget, Button) or not widget.has_class(
            "console-switcher-result"
        ):
            self._pointer_result_key = ""
            self._pointer_result = None
            self._pointer_payload = None
            return
        payload = self._committed_payload_by_widget_id.get(widget.id or "")
        if payload is None or not self._payload_is_current(payload):
            self._pointer_payload = None
            return
        entry = payload.entry if payload is not None else None
        self._pointer_payload = payload
        self._pointer_result = entry
        self._pointer_result_key = entry.stable_result_key if entry is not None else ""

    def on_mouse_up(self, event: MouseUp) -> None:
        """Honor the immutable press payload if a refresh repainted the row."""
        pressed_payload = self._pointer_payload
        widget = event.widget
        current_payload = (
            self._committed_payload_by_widget_id.get(widget.id or "")
            if isinstance(widget, Button)
            and widget.has_class("console-switcher-result")
            else None
        )
        if (
            pressed_payload is None
            or current_payload is None
            or (
                pressed_payload.generation == current_payload.generation
                and pressed_payload.entry.stable_result_key
                == current_payload.entry.stable_result_key
            )
        ):
            return
        pressed = pressed_payload.entry
        self._pointer_payload = None
        self._pointer_result = None
        self._pointer_result_key = ""
        self._ignore_next_result_pressed = True
        if not self._pointer_payload_can_activate(pressed_payload):
            self._selection_feedback = RESULT_DISAPPEARED_COPY
            self._set_status(RESULT_DISAPPEARED_COPY)
            return
        self._apply_result_activation(pressed, input_name="click")

    @on(Button.Pressed, ".console-switcher-result")
    def _result_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if self._ignore_next_result_pressed:
            self._ignore_next_result_pressed = False
            return
        if self._activation_in_flight:
            return
        pointer_payload = self._pointer_payload
        payload = pointer_payload or self._committed_payload_by_widget_id.get(
            event.button.id or ""
        )
        self._pointer_payload = None
        self._pointer_result = None
        self._pointer_result_key = ""
        if payload is None or not self._payload_is_current(payload):
            return
        entry = payload.entry
        self._apply_result_activation(entry, input_name="click", button=event.button)

    def _apply_result_activation(
        self,
        entry: ConsoleSwitcherResult,
        *,
        input_name: str,
        button: Button | None = None,
    ) -> None:
        """Apply one already-authorized committed result interaction."""

        if isinstance(entry, ConsoleSwitcherCharacterResult):
            self._clear_mark_seen_confirmation()
            self._begin_character_activation(entry)
            return
        if isinstance(entry, UnavailableSessionNotice):
            if self._armed_mark_seen_key != entry.stable_result_key:
                target_button = button or self._button_for_key(entry.stable_result_key)
                if target_button is not None:
                    self._arm_mark_seen(entry, target_button, input_name=input_name)
                return
            kind = "mark_seen"
        else:
            self._clear_mark_seen_confirmation()
            kind = "activate"
        self._activate_choice(kind, entry)

    def _arm_mark_seen(
        self,
        entry: UnavailableSessionNotice,
        button: Button,
        *,
        input_name: str,
    ) -> None:
        self._explicit_navigation = True
        self._armed_mark_seen_key = entry.stable_result_key
        index = self._index_for_key(entry.stable_result_key)
        if index is not None:
            self._candidate_index = index
        button.focus()
        self._sync_candidate_labels()
        try:
            self.query_one("#console-switcher-confirm-mark-seen", Button).display = True
        except NoMatches:
            pass
        self._sync_modal_max_height()
        self._set_feedback(
            f"{input_name.capitalize()} again or use Mark seen; Esc keeps it unseen."
        )

    def _clear_mark_seen_confirmation(self) -> None:
        self._armed_mark_seen_key = ""
        try:
            self.query_one(
                "#console-switcher-confirm-mark-seen", Button
            ).display = False
        except NoMatches:
            pass
        self._sync_modal_max_height()

    @on(Button.Pressed, "#console-switcher-confirm-mark-seen")
    def _confirm_mark_seen(self, event: Button.Pressed) -> None:
        event.stop()
        payload = self._committed_payload_for_key(self._armed_mark_seen_key)
        if payload is None or not self._payload_is_current(payload):
            self._clear_mark_seen_confirmation()
            return
        index = self._index_for_key(self._armed_mark_seen_key)
        if index is None:
            self._clear_mark_seen_confirmation()
            self._set_feedback("The unavailable result changed; select it again.")
            return
        entry = self._entries[index]
        if not isinstance(entry, UnavailableSessionNotice):
            self._clear_mark_seen_confirmation()
            return
        self._activate_choice("mark_seen", entry)

    def _activate_choice(self, kind: str, entry: ConsoleSwitcherResult) -> None:
        if isinstance(entry, ConsoleSwitcherCharacterResult):
            if kind == "activate":
                self._begin_character_activation(entry)
            return
        if kind == "activate" and not bool(getattr(entry, "openable", False)):
            self._set_feedback("This conversation is unavailable and cannot be opened.")
            return
        self._request_generation += 1
        self._cancel_query_debounce()
        self.dismiss(ConsoleSwitcherChoice(kind, entry))

    @on(Button.Pressed, "#console-switcher-active-mode")
    def _active_mode_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._set_mode(SwitcherMode.ACTIVE)

    @on(Button.Pressed, "#console-switcher-history-mode")
    def _history_mode_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._set_mode(SwitcherMode.HISTORY)

    @on(Button.Pressed, "#console-switcher-character-mode")
    def _character_mode_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._set_mode(SwitcherMode.CHARACTER_CHATS)

    def action_toggle_mode(self) -> None:
        if self._activation_phase in {
            ConsoleActivationPhase.OPENING_CANCELLABLE,
            ConsoleActivationPhase.COMMITTING,
        }:
            return
        target = {
            SwitcherMode.ACTIVE: SwitcherMode.HISTORY,
            SwitcherMode.HISTORY: SwitcherMode.CHARACTER_CHATS,
            SwitcherMode.CHARACTER_CHATS: SwitcherMode.ACTIVE,
        }[self._mode]
        self._set_mode(target)

    def _snapshot_mode_visit(self) -> None:
        """Capture stable position only from this mode's committed generation."""

        payload = self._committed_payload_for_key(self._candidate_key())
        if self._query_pending:
            return
        if payload is not None and payload.mode is not self._mode:
            return
        try:
            visible_query = self.query_one("#console-switcher-query", Input).value
        except NoMatches:
            return
        if visible_query != self._rendered_query:
            return
        try:
            results = self.query_one("#console-switcher-results", VerticalScroll)
        except NoMatches:
            scroll_y = 0.0
        else:
            scroll_y = float(results.scroll_y)
        self._mode_visits[self._mode] = _ModeVisitState(
            page_offset=self._page_offset,
            selected_key=self._candidate_key(),
            scroll_y=scroll_y,
            focused_result=self._focused_result_index() is not None,
            explicit_navigation=self._explicit_navigation,
        )

    def _set_mode(self, mode: SwitcherMode) -> None:
        if (
            self._closed
            or mode is self._mode
            or self._activation_phase
            in {
                ConsoleActivationPhase.OPENING_CANCELLABLE,
                ConsoleActivationPhase.COMMITTING,
            }
        ):
            return
        self._snapshot_mode_visit()
        current_query = self.query_one("#console-switcher-query", Input).value
        if self._mode is SwitcherMode.CHARACTER_CHATS:
            self._character_query = current_query
        else:
            self._operational_query = current_query
        self._mode = mode
        visit = self._mode_visits[mode]
        self._page_offset = visit.page_offset
        self._explicit_navigation = visit.explicit_navigation
        self._restoring_mode = mode
        self._selection_feedback = ""
        self._clear_mark_seen_confirmation()
        query = (
            self._character_query
            if mode is SwitcherMode.CHARACTER_CHATS
            else self._operational_query
        )
        input_widget = self.query_one("#console-switcher-query", Input)
        self._cancel_query_debounce()
        self._suppress_query_change = True
        input_widget.value = query
        self._suppress_query_change = False
        self._update_mode_controls()
        self.run_worker(
            self._refresh_results(query),
            exclusive=True,
            group="console-session-switcher-mode",
        )

    @on(Button.Pressed, "#console-switcher-previous-page")
    def _previous_page(self, event: Button.Pressed) -> None:
        event.stop()
        if self._page_offset <= 0:
            return
        self._page_offset = max(0, self._page_offset - CONSOLE_SWITCHER_PAGE_LIMIT)
        self._load_current_page()

    @on(Button.Pressed, "#console-switcher-next-page")
    def _next_page(self, event: Button.Pressed) -> None:
        event.stop()
        if self._page_offset + len(self._entries) >= self._page_total:
            return
        self._page_offset += CONSOLE_SWITCHER_PAGE_LIMIT
        self._load_current_page()

    def _load_current_page(self) -> None:
        if self._activation_phase in {
            ConsoleActivationPhase.OPENING_CANCELLABLE,
            ConsoleActivationPhase.COMMITTING,
        }:
            return
        query = self.query_one("#console-switcher-query", Input).value
        self._explicit_navigation = False
        self.run_worker(
            self._refresh_results(query),
            exclusive=True,
            group="console-session-switcher-page",
        )

    def _update_mode_controls(self) -> None:
        try:
            active = self.query_one("#console-switcher-active-mode", Button)
            history = self.query_one("#console-switcher-history-mode", Button)
            character = self.query_one("#console-switcher-character-mode", Button)
        except NoMatches:
            return
        count = len(self._active_results)
        active.label = f"Active ({count})"
        history.label = "History"
        character.label = "Character chats"
        try:
            query = self.query_one("#console-switcher-query", Input)
            query.placeholder = (
                "Search local Character chats by Keyword…"
                if self._mode is SwitcherMode.CHARACTER_CHATS
                else ("Search sessions, workspaces, waiting, running, or finished…")
            )
        except NoMatches:
            pass
        history_is_current = (
            self._mode is SwitcherMode.HISTORY or self._widened_to_history
        )
        active.set_class(
            self._mode is SwitcherMode.ACTIVE and not history_is_current,
            "console-switcher-mode-current",
        )
        history.set_class(history_is_current, "console-switcher-mode-current")
        character.set_class(
            self._mode is SwitcherMode.CHARACTER_CHATS,
            "console-switcher-mode-current",
        )
        for control in (active, history, character):
            control.disabled = self._activation_in_flight

    def _update_page_controls(
        self,
        page: ConsoleSwitcherHistoryPage | None,
        *,
        character_page: CharacterConversationPage | None = None,
    ) -> None:
        try:
            controls = self.query_one("#console-switcher-page-controls", Horizontal)
            previous = self.query_one("#console-switcher-previous-page", Button)
            following = self.query_one("#console-switcher-next-page", Button)
            status = self.query_one("#console-switcher-page-status", Static)
        except NoMatches:
            return
        source = character_page if character_page is not None else page
        offset = (
            self._page_offset
            if character_page is not None
            else page.offset
            if page
            else 0
        )
        visible = source is not None and (
            source.total > CONSOLE_SWITCHER_PAGE_LIMIT or offset > 0
        )
        controls.display = True
        previous.display = visible
        following.display = visible
        if not visible:
            status.update("")
            return
        assert source is not None
        previous.disabled = offset <= 0
        item_count = (
            len(source.rows)
            if isinstance(source, CharacterConversationPage)
            else len(source.entries)
        )
        following.disabled = offset + item_count >= source.total
        first = offset + 1 if item_count else 0
        last = offset + item_count
        status.update(f"{first}–{last} of {source.total}")

    def _update_receipt_status(self) -> None:
        """Expose only content-free local activity storage readiness."""
        try:
            status = self.query_one("#console-switcher-receipt-state", Static)
        except NoMatches:
            return
        degraded = self._activity_receipt_state == "degraded"
        status.display = degraded and not self._compact_layout
        status.update(
            "Local activity updates unavailable — retrying; switching and "
            "History still work."
            if degraded
            else ""
        )

    def _update_selection_status(self, *, prefix: str = "") -> None:
        if self._selection_feedback:
            self._set_status(self._selection_feedback)
            return
        if not self._entries:
            mode = {
                SwitcherMode.ACTIVE: "Active",
                SwitcherMode.HISTORY: "History",
                SwitcherMode.CHARACTER_CHATS: "Character chats",
            }[self._mode]
            self._set_status(f"{mode}: no results")
            self._update_hints(None)
            self._update_selected_detail()
            return
        index = min(self._candidate_index, len(self._entries) - 1)
        title = sanitize_character_display_label(
            self._entries[index].title,
            max_characters=_TITLE_LIMIT,
        )
        entry = self._entries[index]
        if isinstance(entry, ConsoleSwitcherCharacterResult):
            consequence = self._entry_action(entry)
        elif isinstance(entry, UnavailableSessionNotice):
            consequence = "Enter marks seen"
        elif getattr(entry, "native_session_id", None):
            consequence = "Enter switches to"
        else:
            consequence = "Enter opens"
        message = f"{consequence}: {title} · {index + 1} of {len(self._entries)}"
        if prefix:
            message = f"{prefix} · {message}"
        self._set_status(message)
        self._update_hints(entry)
        self._update_selected_detail()

    def _update_hints(self, entry: ConsoleSwitcherResult | None) -> None:
        try:
            hints = self.query_one("#console-switcher-hints", Static)
        except NoMatches:
            return
        if isinstance(entry, ConsoleSwitcherCharacterResult):
            primary = f"Enter:{self._entry_action(entry)}"
        elif isinstance(entry, UnavailableSessionNotice):
            primary = "Enter:mark seen"
        elif entry is not None and getattr(entry, "native_session_id", None):
            primary = (
                "Enter:switch" if self._compact_layout else "Enter:switch F2: rename"
            )
        elif entry is not None:
            primary = "Enter:open"
        else:
            primary = "No result selected"
        if self._activation_phase is ConsoleActivationPhase.OPENING_CANCELLABLE:
            hints.update("Opening…  ·  Esc: cancel")
        elif self._activation_phase is ConsoleActivationPhase.COMMITTING:
            hints.update("Finishing…")
        else:
            hints.update(
                f"{primary} F3:mode Esc:close"
                if self._compact_layout
                else f"{primary} · ↑↓:move · F3:mode · Esc:close"
            )

    async def action_show_workbench_help(self) -> None:
        """Expose full selected identity without shadowing app-global F1."""

        entry: ConsoleSwitcherResult | None = self._committed_character_result
        if entry is None and self._entries:
            index = min(self._candidate_index, len(self._entries) - 1)
            entry = self._entries[index]
        selected = (
            (
                (
                    "Selected result",
                    (
                        ("Title", entry.title),
                        ("Action", self._entry_action(entry)),
                    ),
                ),
            )
            if entry is not None
            else ()
        )
        self.app.push_screen(
            WorkbenchHelpPanel(
                WorkbenchHelpState(
                    route_id="console-session-switcher",
                    title="Switch or resume",
                    shortcut_groups=(
                        *selected,
                        (
                            "Keyboard",
                            (
                                ("Enter", "apply selected action"),
                                ("F2", "rename eligible open Console tab"),
                                ("F3", "next mode"),
                                ("Esc", "close or cancel precommit open"),
                                (
                                    "Library Back",
                                    "returns to Console Context Character",
                                ),
                            ),
                        ),
                    ),
                )
            )
        )

    def _update_selected_detail(self) -> None:
        try:
            detail = self.query_one("#console-switcher-selected-detail", Static)
        except NoMatches:
            return
        if self._mode is SwitcherMode.CHARACTER_CHATS and self._character_search_error:
            detail.update(
                f"{self._character_search_error}\nRefresh results · This profile · Local chats"
            )
            return
        if self._mode is not SwitcherMode.CHARACTER_CHATS or not self._entries:
            detail.update("")
            return
        entry = self._committed_character_result
        if entry is None:
            index = min(self._candidate_index, len(self._entries) - 1)
            candidate = self._entries[index]
            entry = (
                candidate
                if isinstance(candidate, ConsoleSwitcherCharacterResult)
                else None
            )
        if entry is None:
            detail.update("")
            return
        excerpt = sanitize_character_display_label(
            entry.selected_excerpt, max_characters=_SUBTITLE_LIMIT * 2
        )
        state = self._character_state(entry)
        if self._activation_phase is ConsoleActivationPhase.OPENING_CANCELLABLE:
            state = "Opening…"
        elif self._activation_phase is ConsoleActivationPhase.COMMITTING:
            state = "Finishing…"
        elif self._activation_phase is ConsoleActivationPhase.FAILURE_VISIBLE:
            state = str(self.query_one("#console-switcher-status", Static).renderable)
        first = Text(excerpt or "No matching excerpt")
        first.truncate(max(20, self.app.size.width - 4), overflow="ellipsis")
        second = Text(
            f"{entry.absolute_time} · {state}"
            if self._activation_phase is ConsoleActivationPhase.IDLE
            else state
        )
        second.truncate(max(20, self.app.size.width - 4), overflow="ellipsis")
        detail.update(Text.assemble(first, "\n", second))

    def _restore_owned_query(self) -> None:
        try:
            query = self.query_one("#console-switcher-query", Input)
        except NoMatches:
            return
        expected = (
            self._character_query
            if self._mode is SwitcherMode.CHARACTER_CHATS
            else self._operational_query
        )
        self._suppress_query_change = True
        query.value = expected
        self._suppress_query_change = False

    def _begin_character_activation(
        self, entry: ConsoleSwitcherCharacterResult
    ) -> None:
        if self._activation_in_flight:
            return
        self._committed_character_result = entry
        if entry.target is None:
            self._show_activation_failure(
                ConsoleActivationResultKind.CHARACTER_UNAVAILABLE
            )
            return
        if self._character_activate is None:
            self._show_activation_failure(ConsoleActivationResultKind.FAILED)
            return
        request = CharacterConversationActivationRequest(
            target=entry.target,
            data_authority_id=entry.target.character.data_authority_id,
            data_revision=self._character_data_revision,
        )
        self._activation_cancellation = asyncio.Event()
        self._activation_failure_kind = None
        self._activation_phase = ConsoleActivationPhase.OPENING_CANCELLABLE
        self._set_activation_controls_disabled(True)
        self._set_status("Opening…")
        self._sync_candidate_labels()
        self._update_hints(entry)
        self._update_selected_detail()
        self._activation_task = asyncio.create_task(
            self._run_character_activation(request, self._activation_cancellation)
        )

    async def _run_character_activation(
        self,
        request: CharacterConversationActivationRequest,
        cancellation: asyncio.Event,
    ) -> None:
        activator = self._character_activate
        if activator is None:
            self._show_activation_failure(ConsoleActivationResultKind.FAILED)
            return
        activation = asyncio.create_task(activator(request, cancellation))
        commit_waiter: asyncio.Task[None] | None = None
        if self._character_commit_waiter is not None:
            commit_waiter = asyncio.create_task(self._character_commit_waiter(request))
        try:
            if commit_waiter is not None:
                done, _ = await asyncio.wait(
                    {activation, commit_waiter},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if commit_waiter in done and not activation.done():
                    try:
                        commit_waiter.result()
                    except Exception:  # noqa: BLE001 - activation result remains authority
                        commit_signal_received = False
                    else:
                        commit_signal_received = True
                    if commit_signal_received:
                        self._activation_phase = ConsoleActivationPhase.COMMITTING
                        self._set_activation_controls_disabled(True)
                        self._set_status("Finishing…")
                        self._sync_candidate_labels()
                        self._update_hints(self._committed_character_result)
                        self._update_selected_detail()
            result = await activation
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - adapter failure stays visible in modal
            self._show_activation_failure(ConsoleActivationResultKind.FAILED)
            return
        finally:
            if commit_waiter is not None and not commit_waiter.done():
                commit_waiter.cancel()

        if (
            result.kind is ConsoleActivationResultKind.OPENED
            and result.target == request.target
            and result.commit_started
        ):
            self._activation_phase = ConsoleActivationPhase.COMMITTING
            self._set_activation_controls_disabled(True)
            self._set_status("Finishing…")
            self._sync_candidate_labels()
            self._request_generation += 1
            self._cancel_query_debounce()
            self.dismiss_safe_once(None)
            return
        if result.kind is ConsoleActivationResultKind.CANCELLED_PRECOMMIT:
            self._activation_phase = ConsoleActivationPhase.IDLE
            self._activation_cancellation = None
            self._committed_character_result = None
            self._set_activation_controls_disabled(False)
            self._set_status("Open cancelled")
            self._sync_candidate_labels()
            self._update_selection_status()
            return
        self._show_activation_failure(result.kind)

    def _show_activation_failure(self, kind: ConsoleActivationResultKind) -> None:
        copy = {
            ConsoleActivationResultKind.NOT_FOUND: "Conversation no longer exists",
            ConsoleActivationResultKind.DATA_PROFILE_CHANGED: "Profile changed",
            ConsoleActivationResultKind.CHARACTER_UNAVAILABLE: "Character unavailable",
            ConsoleActivationResultKind.FAILED: "Could not open chat",
            ConsoleActivationResultKind.OPENED: "Could not open chat",
            ConsoleActivationResultKind.CANCELLED_PRECOMMIT: "Open cancelled",
        }[kind]
        action = {
            ConsoleActivationResultKind.NOT_FOUND: "Refresh results",
            ConsoleActivationResultKind.DATA_PROFILE_CHANGED: "Refresh results",
            ConsoleActivationResultKind.CHARACTER_UNAVAILABLE: "Open Library",
        }.get(kind, "Retry")
        self._activation_phase = ConsoleActivationPhase.FAILURE_VISIBLE
        self._activation_failure_kind = kind
        self._activation_cancellation = None
        self._set_activation_controls_disabled(False)
        self._set_status(copy)
        self._sync_candidate_labels()
        try:
            recovery = self.query_one("#console-switcher-recovery", Button)
            recovery.label = action
            recovery.display = True
        except NoMatches:
            pass
        self._update_hints(self._committed_character_result)
        self._update_selected_detail()

    def _set_activation_controls_disabled(self, disabled: bool) -> None:
        try:
            self.query_one("#console-switcher-query", Input).disabled = disabled
        except NoMatches:
            pass
        self._update_mode_controls()
        for selector in (
            "#console-switcher-previous-page",
            "#console-switcher-next-page",
        ):
            try:
                self.query_one(selector, Button).disabled = disabled
            except NoMatches:
                pass
        try:
            cancel = self.query_one("#console-switcher-cancel", Button)
            committing = self._activation_phase is ConsoleActivationPhase.COMMITTING
            cancel.disabled = committing
            cancel.label = "Finishing" if committing else "Cancel"
        except NoMatches:
            pass

    @on(Button.Pressed, "#console-switcher-recovery")
    async def _recover_character_activation(self, event: Button.Pressed) -> None:
        event.stop()
        recovery = event.button
        failure = self._activation_failure_kind
        retry_entry = self._committed_character_result
        if (
            failure is ConsoleActivationResultKind.CHARACTER_UNAVAILABLE
            and self._character_open_library is not None
            and retry_entry is not None
        ):
            recovery.disabled = True
            self._set_status("Opening Library…")
            try:
                accepted = await self._character_open_library(retry_entry)
            except Exception:  # noqa: BLE001 - destination rejection stays visible
                accepted = False
            if accepted:
                self.dismiss_safe_once(None)
                return
            recovery.disabled = False
            recovery.display = True
            self._set_status("Character unavailable")
            self._sync_candidate_labels()
            self._update_selected_detail()
            return
        recovery.display = False
        self._activation_failure_kind = None
        self._activation_phase = ConsoleActivationPhase.IDLE
        self._committed_character_result = None
        self._set_activation_controls_disabled(False)
        if (
            failure is ConsoleActivationResultKind.FAILED
            and retry_entry is not None
            and retry_entry.target is not None
        ):
            self._begin_character_activation(retry_entry)
            return
        self._load_current_page()

    async def _perform_safe_cancel(self, *, source: str) -> None:
        del source
        if self._activation_phase is ConsoleActivationPhase.OPENING_CANCELLABLE:
            cancellation = self._activation_cancellation
            if cancellation is not None:
                cancellation.set()
            self._set_status("Cancelling…")
            return
        if self._activation_phase is ConsoleActivationPhase.COMMITTING:
            return
        self._request_generation += 1
        self._cancel_query_debounce()
        self.dismiss_safe_once(None)

    @on(Button.Pressed, "#console-switcher-cancel")
    async def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="button")

    def action_rename_entry(self) -> None:
        """F2 acts only on an explicitly focused native result."""
        if self._mode is SwitcherMode.CHARACTER_CHATS:
            self._set_feedback("Character chats cannot be renamed here.")
            return
        focused = self.focused
        if not isinstance(focused, Button) or not focused.has_class(
            "console-switcher-result"
        ):
            self._set_feedback("Focus an open agent result to rename it.")
            return
        entry = self._payload_by_widget_id.get(focused.id or "")
        payload = self._committed_payload_by_widget_id.get(focused.id or "")
        if (
            entry is None
            or payload is None
            or not self._payload_is_current(payload)
            or isinstance(entry, UnavailableSessionNotice)
        ):
            self._set_feedback("Focus an open agent result to rename it.")
            return
        if not getattr(entry, "native_session_id", None):
            self._set_feedback("Saved chats cannot be renamed here; open one first.")
            return
        self._activate_choice("rename", entry)

    def _set_status(self, message: str) -> None:
        try:
            self.query_one("#console-switcher-status", Static).update(message)
        except NoMatches:
            pass
        try:
            scope = self.query_one("#console-switcher-scope", Static)
        except NoMatches:
            return
        if self._mode is SwitcherMode.CHARACTER_CHATS:
            scope.update("This profile · Local chats")
            self.query_one("#console-switcher-divider", Static).update(
                message
                if self._activation_phase is not ConsoleActivationPhase.IDLE
                else "─" * 48
            )
        else:
            self.query_one("#console-switcher-divider", Static).update("─" * 48)
            scope.update(
                f"Activity updates unavailable · {message}"
                if self._compact_layout and self._activity_receipt_state == "degraded"
                else message
            )

    def _set_feedback(self, message: str) -> None:
        try:
            self.query_one("#console-switcher-feedback", Static).update(message)
        except NoMatches:
            pass
        self._set_status(message)

    @staticmethod
    def _result_index_from_widget_id(widget_id: str) -> int | None:
        """Retain the legacy index parser for compatibility tests only."""
        prefix = "console-switcher-result-"
        if not widget_id.startswith(prefix):
            return None
        try:
            return int(widget_id[len(prefix) :])
        except ValueError:
            return None
