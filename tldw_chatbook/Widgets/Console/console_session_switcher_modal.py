"""Operational Active/History switchboard for Console sessions (Ctrl+K)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from hashlib import sha1
from typing import Any
from uuid import uuid4

from rich.markup import escape as escape_markup
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.events import DescendantFocus
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widgets import Button, Input, Static

from tldw_chatbook.Chat.console_switcher_state import (
    CONSOLE_SWITCHER_PAGE_LIMIT,
    ActivityGroup,
    ConsoleSwitcherActiveResult,
    ConsoleSwitcherHistoryPage,
    SwitcherMode,
    UnavailableSessionNotice,
    build_console_switcher_entries,
    filter_console_active_results,
)
from tldw_chatbook.UI.character_display_text import sanitize_character_display_label
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


class ConsoleSessionSwitcherModal(
    SafeModalDismissMixin, ModalScreen["ConsoleSwitcherChoice | None"]
):
    """Switch among live agents or bounded local conversation History."""

    DEFAULT_CSS = """
    ConsoleSessionSwitcherModal { align: center middle; }
    #console-switcher-modal {
        width: 76; max-width: 100%; height: 100%; max-height: 35;
        border: tall gray; background: black; padding: 1 2;
    }
    #console-switcher-mode-controls { height: 1; min-height: 1; }
    .console-switcher-mode {
        width: auto; min-width: 10; height: 1; min-height: 1;
        border: none; padding: 0 1;
    }
    #console-switcher-mode-divider { width: 1; height: 1; }
    #console-switcher-results {
        height: 1fr; min-height: 3; margin: 1 0 0 0;
    }
    .console-switcher-section {
        height: 1; color: gray; text-style: bold;
    }
    #console-switcher-status { height: 1; color: yellow; }
    #console-switcher-receipt-state {
        display: none; height: 1; color: yellow;
    }
    #console-switcher-feedback { display: none; }
    #console-switcher-page-controls { height: 3; min-height: 3; }
    #console-switcher-page-status {
        width: 1fr; height: 1; content-align: center middle; color: gray;
    }
    #console-switcher-previous-page, #console-switcher-next-page,
    #console-switcher-cancel {
        width: 10; min-width: 10; height: 3; min-height: 3;
    }
    #console-switcher-hints { height: auto; color: gray; }
    .console-switcher-result {
        width: 100%; height: 2; min-height: 2; margin: 0;
    }
    .console-switcher-result-candidate { text-style: bold; }
    .console-switcher-result-active { text-style: underline; }
    """

    SAFE_MODAL_CONTENT = "#console-switcher-modal"
    BINDINGS = [
        ("escape", "request_safe_cancel", "Cancel"),
        ("f2", "rename_entry", "Rename"),
        ("f3", "toggle_mode", "Active / History"),
        Binding("down", "switcher_cursor_down", "Next result", priority=True),
        Binding("up", "switcher_cursor_up", "Previous result", priority=True),
    ]

    def __init__(
        self,
        *,
        rows: tuple[ConsoleConversationBrowserInputRow, ...] = (),
        active_results: tuple[ConsoleSwitcherActiveResult, ...] | None = None,
        history_loader: HistoryLoader | None = None,
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
            build_console_switcher_entries(
                rows, limit=CONSOLE_SWITCHER_PAGE_LIMIT
            )
            if active_results is None
            else tuple(active_results)
        )
        self._history_loader = history_loader
        self._preferred_native_session_id = str(
            preferred_native_session_id or ""
        ).strip()
        self._profile_authority = str(profile_authority or "")
        self._authority_token = str(authority_token or "")
        self._active_projection_generation = int(active_projection_generation)
        self._authority_snapshot = authority_snapshot
        self._active_projection_loader = active_projection_loader
        self._activity_receipt_state = str(activity_receipt_state or "ready")
        self._mode = SwitcherMode.ACTIVE
        self._entries: tuple[ConsoleSwitcherActiveResult, ...] = ()
        self._payload_by_widget_id: dict[str, ConsoleSwitcherActiveResult] = {}
        self._candidate_index = 0
        self._rendered_query = ""
        self._page_offset = 0
        self._page_total = 0
        self._request_generation = 0
        self._instance_token = uuid4().hex
        self._query_pending = False
        self._explicit_navigation = False
        self._selection_feedback = ""
        self._widened_to_history = False
        self._closed = False
        self._query_debounce_timer: Timer | None = None
        self._active_projection_timer: Timer | None = None

    def compose(self) -> ComposeResult:
        """Build the bounded switchboard structure."""
        with Vertical(id="console-switcher-modal"):
            yield Static("Switch Session", classes="console-modal-header")
            with Horizontal(id="console-switcher-mode-controls"):
                yield Button(
                    "Active (0) — selected",
                    id="console-switcher-active-mode",
                    classes="console-switcher-mode",
                    compact=True,
                )
                yield Static("|", id="console-switcher-mode-divider", markup=False)
                yield Button(
                    "History",
                    id="console-switcher-history-mode",
                    classes="console-switcher-mode",
                    compact=True,
                )
            yield Input(
                placeholder=(
                    "Search title, workspace:<name>, or is:waiting / "
                    "is:working / is:new…"
                ),
                id="console-switcher-query",
            )
            yield VerticalScroll(id="console-switcher-results")
            yield Static("", id="console-switcher-receipt-state", markup=False)
            yield Static("", id="console-switcher-status", markup=False)
            yield Static("", id="console-switcher-feedback", markup=False)
            with Horizontal(id="console-switcher-page-controls"):
                yield Button("Previous", id="console-switcher-previous-page")
                yield Static("", id="console-switcher-page-status", markup=False)
                yield Button("Next", id="console-switcher-next-page")
            yield Static(
                "Enter: open/mark seen  |  F2: rename focused tab  |  "
                "F3: Active/History  |  Up/Down: navigate  |  Esc: close",
                id="console-switcher-hints",
                markup=False,
            )
            yield Button("Cancel", id="console-switcher-cancel")

    async def on_mount(self) -> None:  # type: ignore[override]
        """Paint Active immediately and leave History cold."""
        super().on_mount()
        self._update_receipt_status()
        self.query_one("#console-switcher-query", Input).focus()
        await self._refresh_results("")
        if self._active_projection_loader is not None:
            self._active_projection_timer = self.set_interval(
                ACTIVE_PROJECTION_POLL_SECONDS,
                self._poll_active_projection,
            )

    def on_unmount(self) -> None:
        """Invalidate late work without remounting into a closed screen."""
        self._closed = True
        self._request_generation += 1
        if self._query_debounce_timer is not None:
            self._query_debounce_timer.stop()
            self._query_debounce_timer = None
        if self._active_projection_timer is not None:
            self._active_projection_timer.stop()
            self._active_projection_timer = None
        super().on_unmount()

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
        self._query_pending = True
        self._widened_to_history = False

        page: ConsoleSwitcherHistoryPage | None = None
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
        else:
            self._set_status(
                "Searching History…" if query.strip() else "Loading History…"
            )
            page = await self._load_history(query=query, offset=self._page_offset)
            entries = tuple(page.entries)

        if not self._request_is_current(generation, captured, query):
            return False
        self._query_pending = False
        self._widened_to_history = widened_to_history
        self._page_total = page.total if page is not None else len(entries)
        self._rendered_query = query
        await self._commit_entries(entries, page=page)
        return True

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
        entries: tuple[ConsoleSwitcherActiveResult, ...],
        *,
        page: ConsoleSwitcherHistoryPage | None,
    ) -> None:
        previous_key = self._candidate_key()
        focused_key = self._focused_result_key()
        retained_key = focused_key or previous_key
        previous_index = (
            self._focused_result_index()
            if focused_key
            else self._candidate_index
        )
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
            if retained_disappeared:
                self._candidate_index = min(previous_index, len(self._entries) - 1)
            else:
                self._candidate_index = self._choose_candidate(retained_key)
            widgets: list[Static | Button] = []
            previous_section = ""
            for index, entry in enumerate(self._entries):
                section = str(getattr(entry, "section", "") or "")
                if section != previous_section:
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
                    classes="console-switcher-result",
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
                verb = (
                    "Mark seen"
                    if isinstance(entry, UnavailableSessionNotice)
                    else "Open"
                )
                button.tooltip = escape_markup(f"{verb}: {entry.title}")
                self._payload_by_widget_id[widget_id] = entry
                widgets.append(button)
            await results.mount_all(widgets)

        self._update_mode_controls()
        self._update_page_controls(page)
        if page is not None and page.error:
            self._set_status(page.error)
        elif self._widened_to_history:
            self._set_status("No active matches — showing History")
        else:
            self._update_selection_status()

        if retained_disappeared:
            self._selection_feedback = RESULT_DISAPPEARED_COPY
            if self._entries:
                buttons = self._result_buttons()
                self._focus_candidate(buttons)
            else:
                self.query_one("#console-switcher-query", Input).focus()
            self._set_status(self._selection_feedback)
            self.notify(RESULT_DISAPPEARED_COPY, severity="warning")
        elif focused_key:
            focused = self._button_for_key(focused_key)
            if focused is not None:
                focused.focus()

    def _empty_copy(self, page: ConsoleSwitcherHistoryPage | None) -> str:
        query = self._rendered_query.strip()
        if page is not None and page.error:
            return page.error
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

    def _result_widget_id(
        self, index: int, entry: ConsoleSwitcherActiveResult
    ) -> str:
        if self._legacy_rows:
            return f"console-switcher-result-{index}"
        digest = sha1(entry.stable_result_key.encode("utf-8")).hexdigest()[:16]
        return f"console-switcher-result-{digest}"

    def _entry_label(
        self, index: int, entry: ConsoleSwitcherActiveResult
    ) -> Text:
        display_title = (
            sanitize_character_display_label(
                entry.title,
                max_characters=_TITLE_LIMIT,
            )
            or "Untitled conversation"
        )
        display_subtitle = sanitize_character_display_label(
            entry.subtitle,
            max_characters=_SUBTITLE_LIMIT,
        )
        marker = "▸" if index == self._candidate_index else " "
        action = "Mark seen" if isinstance(entry, UnavailableSessionNotice) else ""
        title = f"{display_title} — {action}" if action else display_title
        label = f"{marker} {title}"
        if display_subtitle:
            label = f"{label}\n  {display_subtitle}"
        return Text(label)

    @on(Input.Changed, "#console-switcher-query")
    def _query_changed(self, event: Input.Changed) -> None:
        event.stop()
        self._selection_feedback = ""
        self._cancel_query_debounce()
        self._query_pending = True
        query = event.value
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

    @on(Input.Submitted, "#console-switcher-query")
    async def _query_submitted(self, event: Input.Submitted) -> None:
        """Commit the exact query before applying its explicit candidate."""
        event.stop()
        self._cancel_query_debounce()
        if event.value != self._rendered_query or self._query_pending:
            committed = await self._refresh_results(event.value, reset_page=True)
            if not committed:
                return
        if not self._entries:
            return
        index = self._candidate_index
        if not event.value.strip() and not self._explicit_navigation:
            index = self._choose_candidate(self._candidate_key())
            self._candidate_index = index
            self._sync_candidate_labels()
        if not 0 <= index < len(self._entries):
            return
        entry = self._entries[index]
        if isinstance(entry, UnavailableSessionNotice):
            button = self._button_for_key(entry.stable_result_key)
            if button is not None:
                self._explicit_navigation = True
                button.focus()
                self._set_feedback(
                    "Press Enter again to Mark seen, or Esc to keep it unseen."
                )
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

    def action_switcher_cursor_down(self) -> None:
        buttons = self._result_buttons()
        if not buttons:
            return
        index = self._focused_result_index()
        self._explicit_navigation = True
        self._selection_feedback = ""
        if index is None:
            self._candidate_index = 0
        elif index + 1 < len(buttons):
            self._candidate_index = index + 1
        else:
            return
        self._focus_candidate(buttons)

    def action_switcher_cursor_up(self) -> None:
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
        self._candidate_index = index - 1
        self._focus_candidate(buttons)

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

    def on_descendant_focus(self, event: DescendantFocus) -> None:
        widget = event.widget
        if (
            not isinstance(widget, Button)
            or not widget.has_class("console-switcher-result")
            or widget is not self.app.focused
        ):
            return
        entry = self._payload_by_widget_id.get(widget.id or "")
        if entry is None:
            return
        index = self._index_for_key(entry.stable_result_key)
        if index is None:
            return
        self._explicit_navigation = True
        self._candidate_index = index
        self._update_selection_status()

    @on(Button.Pressed, ".console-switcher-result")
    def _result_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        entry = self._payload_by_widget_id.get(event.button.id or "")
        if entry is None:
            return
        kind = (
            "mark_seen"
            if isinstance(entry, UnavailableSessionNotice)
            else "activate"
        )
        self._activate_choice(kind, entry)

    def _activate_choice(
        self, kind: str, entry: ConsoleSwitcherActiveResult
    ) -> None:
        if kind == "activate" and not bool(getattr(entry, "openable", False)):
            self._set_feedback(
                "This conversation is unavailable and cannot be opened."
            )
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

    def action_toggle_mode(self) -> None:
        target = (
            SwitcherMode.HISTORY
            if self._mode is SwitcherMode.ACTIVE
            else SwitcherMode.ACTIVE
        )
        self._set_mode(target)

    def _set_mode(self, mode: SwitcherMode) -> None:
        if self._closed or mode is self._mode:
            return
        self._mode = mode
        self._page_offset = 0
        self._explicit_navigation = False
        self._selection_feedback = ""
        query = self.query_one("#console-switcher-query", Input).value
        self._update_mode_controls()
        self.run_worker(
            self._refresh_results(query, reset_page=True),
            exclusive=True,
            group="console-session-switcher-mode",
        )

    @on(Button.Pressed, "#console-switcher-previous-page")
    def _previous_page(self, event: Button.Pressed) -> None:
        event.stop()
        if self._page_offset <= 0:
            return
        self._page_offset = max(
            0, self._page_offset - CONSOLE_SWITCHER_PAGE_LIMIT
        )
        self._load_current_page()

    @on(Button.Pressed, "#console-switcher-next-page")
    def _next_page(self, event: Button.Pressed) -> None:
        event.stop()
        if self._page_offset + len(self._entries) >= self._page_total:
            return
        self._page_offset += CONSOLE_SWITCHER_PAGE_LIMIT
        self._load_current_page()

    def _load_current_page(self) -> None:
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
        except NoMatches:
            return
        count = len(self._active_results)
        active.label = (
            f"Active ({count}) — selected"
            if self._mode is SwitcherMode.ACTIVE
            else f"Active ({count})"
        )
        history.label = (
            "History — selected"
            if self._mode is SwitcherMode.HISTORY
            else "History"
        )

    def _update_page_controls(
        self, page: ConsoleSwitcherHistoryPage | None
    ) -> None:
        try:
            controls = self.query_one(
                "#console-switcher-page-controls", Horizontal
            )
            previous = self.query_one("#console-switcher-previous-page", Button)
            following = self.query_one("#console-switcher-next-page", Button)
            status = self.query_one("#console-switcher-page-status", Static)
        except NoMatches:
            return
        visible = page is not None and (page.total > page.limit or page.offset > 0)
        controls.display = visible
        if not visible:
            status.update("")
            return
        previous.disabled = page.offset <= 0
        following.disabled = not page.has_more
        first = page.offset + 1 if page.entries else 0
        last = page.offset + len(page.entries)
        status.update(f"{first}–{last} of {page.total}")

    def _update_receipt_status(self) -> None:
        """Expose only content-free local activity storage readiness."""
        try:
            status = self.query_one("#console-switcher-receipt-state", Static)
        except NoMatches:
            return
        degraded = self._activity_receipt_state == "degraded"
        status.display = degraded
        status.update(
            "Local activity updates unavailable — retrying; switching and "
            "History still work."
            if degraded
            else ""
        )

    def _update_selection_status(self) -> None:
        if self._selection_feedback:
            self._set_status(self._selection_feedback)
            return
        if not self._entries:
            mode = (
                "Active" if self._mode is SwitcherMode.ACTIVE else "History"
            )
            self._set_status(f"{mode}: no results")
            return
        index = min(self._candidate_index, len(self._entries) - 1)
        title = sanitize_character_display_label(
            self._entries[index].title,
            max_characters=_TITLE_LIMIT,
        )
        self._set_status(
            f"Selection: {index + 1} of {len(self._entries)} — {title}"
        )

    async def _perform_safe_cancel(self, *, source: str) -> None:
        del source
        self._request_generation += 1
        self._cancel_query_debounce()
        self.dismiss_safe_once(None)

    @on(Button.Pressed, "#console-switcher-cancel")
    async def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="button")

    def action_rename_entry(self) -> None:
        """F2 acts only on an explicitly focused native result."""
        focused = self.focused
        if not isinstance(focused, Button) or not focused.has_class(
            "console-switcher-result"
        ):
            self._set_feedback("Focus an open agent result to rename it.")
            return
        entry = self._payload_by_widget_id.get(focused.id or "")
        if entry is None or isinstance(entry, UnavailableSessionNotice):
            self._set_feedback("Focus an open agent result to rename it.")
            return
        if not entry.native_session_id:
            self._set_feedback(
                "Saved chats cannot be renamed here; open one first."
            )
            return
        self._activate_choice("rename", entry)

    def _set_status(self, message: str) -> None:
        try:
            self.query_one("#console-switcher-status", Static).update(message)
        except NoMatches:
            pass

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
