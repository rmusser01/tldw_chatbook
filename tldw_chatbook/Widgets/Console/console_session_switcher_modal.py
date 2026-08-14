"""Console fuzzy session switcher modal (Ctrl+K)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rich.markup import escape as escape_markup
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widgets import Button, Input, Static

from tldw_chatbook.Chat.console_switcher_state import (
    ConsoleSwitcherEntry,
    build_console_switcher_entries,
)
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin
from tldw_chatbook.Workspaces.conversation_browser_state import (
    ConsoleConversationBrowserInputRow,
)
from tldw_chatbook.UI.character_display_text import sanitize_character_display_label


_SWITCHER_TITLE_MAX_CHARACTERS = 500
_SWITCHER_SUBTITLE_MAX_CHARACTERS = 500


#: Debounce for the search `Input` -- mirrors the console picker family's
#: 0.2 s shape (`console_prompt_picker_modal.py`). A full refresh removes
#: and remounts one `Button` per matching entry (up to
#: `CONSOLE_SWITCHER_RESULT_LIMIT`), which should not happen on every
#: keystroke (task-15476).
SEARCH_DEBOUNCE_SECONDS = 0.2


@dataclass(frozen=True)
class ConsoleSwitcherChoice:
    """Result returned by the session switcher modal."""

    kind: str
    entry: ConsoleSwitcherEntry


class ConsoleSessionSwitcherModal(
    SafeModalDismissMixin, ModalScreen["ConsoleSwitcherChoice | None"]
):
    """Fuzzy-find and activate a Console session or persisted conversation."""

    DEFAULT_CSS = """
    ConsoleSessionSwitcherModal {
        align: center middle;
    }

    #console-switcher-modal {
        width: 72;
        height: auto;
        max-height: 30;
        border: tall gray;
        background: black;
        padding: 1 2;
    }

    #console-switcher-results {
        height: auto;
        max-height: 20;
        margin: 1 0 0 0;
    }

    #console-switcher-hints {
        height: auto;
        margin: 1 0 0 0;
        color: gray;
    }

    #console-switcher-cancel {
        width: 10;
        min-width: 10;
        height: 3;
        min-height: 3;
    }

    .console-switcher-result {
        width: 100%;
        height: 2;
        min-height: 2;
        margin: 0;
    }
    """

    SAFE_MODAL_CONTENT = "#console-switcher-modal"
    BINDINGS = [
        ("escape", "request_safe_cancel", "Cancel"),
        ("f2", "rename_entry", "Rename"),
        ("down", "switcher_cursor_down", "Next result"),
        ("up", "switcher_cursor_up", "Previous result"),
    ]

    def __init__(
        self,
        *,
        rows: tuple[ConsoleConversationBrowserInputRow, ...],
        **kwargs: Any,
    ) -> None:
        """Initialize the switcher with the browser rows to search over.

        Args:
            rows: Console conversation browser input rows to build the
                fuzzy-search result list from.
            **kwargs: Forwarded to ``ModalScreen``.
        """
        super().__init__(**kwargs)
        self._rows = rows
        self._entries: tuple[ConsoleSwitcherEntry, ...] = ()
        self._query_debounce_timer: Timer | None = None

    def compose(self) -> ComposeResult:
        """Build the search input and results container."""
        with Vertical(id="console-switcher-modal"):
            yield Static("Switch Session", classes="console-modal-header")
            yield Input(
                placeholder="Search conversations…",
                id="console-switcher-query",
            )
            yield Vertical(id="console-switcher-results")
            # DS-08 (TASK-2154.15): in-modal key hints -- these bindings were
            # documented only in F1 before. Only keys that actually work are
            # listed (no Ctrl+Enter: that spec'd binding was never implemented).
            yield Static(
                "Enter: open  |  F2: rename  |  Up/Down: navigate  |  Esc: close",
                id="console-switcher-hints",
            )
            yield Button("Cancel", id="console-switcher-cancel")

    async def on_mount(self) -> None:  # type: ignore[override]
        """Focus the search input and populate the initial (unfiltered) results."""
        self.query_one("#console-switcher-query", Input).focus()
        await self._refresh_results("")

    async def _refresh_results(self, query: str) -> None:
        """Recompute entries and fully replace the results children.

        This is awaited to completion within a single handler invocation
        (no ``call_later`` deferral) so that Textual's serialized message
        pump cannot interleave two refresh/mount cycles and mount
        duplicate widget ids.
        """
        # Update entries synchronously first: Enter-activates-first-result
        # reads self._entries[0] and must never observe a stale value.
        self._entries = build_console_switcher_entries(self._rows, query=query)
        results = self.query_one("#console-switcher-results", Vertical)

        await results.remove_children()

        if not self._entries:
            await results.mount(
                Static("No matches.", id="console-switcher-empty", markup=False)
            )
        else:
            buttons = []
            for index, entry in enumerate(self._entries):
                display_title = sanitize_character_display_label(
                    entry.title,
                    max_characters=_SWITCHER_TITLE_MAX_CHARACTERS,
                ) or "Untitled conversation"
                display_subtitle = sanitize_character_display_label(
                    entry.subtitle,
                    max_characters=_SWITCHER_SUBTITLE_MAX_CHARACTERS,
                )
                label = (
                    display_title
                    if not display_subtitle
                    else f"{display_title}\n  {display_subtitle}"
                )
                button = Button(
                    Text(label),
                    id=f"console-switcher-result-{index}",
                    classes="console-switcher-result",
                    compact=True,
                )
                button.set_class(entry.is_active, "console-switcher-result-active")
                button.tooltip = escape_markup(f"Switch to {display_title}")
                buttons.append(button)
            await results.mount_all(buttons)

    @on(Input.Changed, "#console-switcher-query")
    def _query_changed(self, event: Input.Changed) -> None:
        """Recompute results as the search query changes (debounced).

        Args:
            event: The search input's change event.
        """
        event.stop()
        query = event.value
        self._cancel_query_debounce()
        self._query_debounce_timer = self.set_timer(
            SEARCH_DEBOUNCE_SECONDS,
            lambda: self.run_worker(
                self._refresh_results(query),
                exclusive=True,
                group="console-session-switcher-search",
            ),
        )

    def _cancel_query_debounce(self) -> None:
        if self._query_debounce_timer is not None:
            self._query_debounce_timer.stop()
            self._query_debounce_timer = None

    @on(Input.Submitted, "#console-switcher-query")
    def _query_submitted(self, event: Input.Submitted) -> None:
        """Activate the top result when the search query is submitted.

        Args:
            event: The search input's submit event.
        """
        event.stop()
        if self._entries:
            self._cancel_query_debounce()
            self.dismiss(ConsoleSwitcherChoice("activate", self._entries[0]))

    def _result_buttons(self) -> list[Button]:
        """Return mounted result buttons in display order."""
        try:
            results = self.query_one("#console-switcher-results", Vertical)
        except NoMatches:
            return []
        return [
            button
            for button in results.query(Button)
            if button.has_class("console-switcher-result")
        ]

    def _focused_result_index(self) -> int | None:
        focused = self.app.focused
        buttons = self._result_buttons()
        for index, button in enumerate(buttons):
            if button is focused:
                return index
        return None

    def action_switcher_cursor_down(self) -> None:
        """ArrowDown: search field -> first result -> next result (TASK-358)."""
        buttons = self._result_buttons()
        if not buttons:
            return
        index = self._focused_result_index()
        if index is None:
            buttons[0].focus()
            return
        if index + 1 < len(buttons):
            buttons[index + 1].focus()

    def action_switcher_cursor_up(self) -> None:
        """ArrowUp: previous result; from the first, back to the search field."""
        buttons = self._result_buttons()
        index = self._focused_result_index()
        if index is None or index == 0:
            try:
                self.query_one("#console-switcher-query", Input).focus()
            except NoMatches:
                pass
            return
        buttons[index - 1].focus()

    @on(Button.Pressed, ".console-switcher-result")
    def _result_pressed(self, event: Button.Pressed) -> None:
        """Activate the clicked result entry.

        Args:
            event: The result button's press event.
        """
        event.stop()
        index = self._result_index_from_widget_id(event.button.id or "")
        if index is not None and 0 <= index < len(self._entries):
            self._cancel_query_debounce()
            self.dismiss(ConsoleSwitcherChoice("activate", self._entries[index]))

    async def _perform_safe_cancel(self, *, source: str) -> None:
        del source
        self._cancel_query_debounce()
        self.dismiss_safe_once(None)

    @on(Button.Pressed, "#console-switcher-cancel")
    async def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="button")

    def action_rename_entry(self) -> None:
        """Request a rename for the focused result, or the first native entry (F2).

        Prefers the entry backing the currently focused result button, so
        that F2 renames whatever the user is looking at rather than always
        the first result. Falls back to the first entry with a
        ``native_session_id`` when focus isn't on a result button, or that
        button's entry isn't a native (renameable) session.
        """
        focused_index = self._result_index_from_widget_id(
            getattr(self.focused, "id", None) or ""
        )
        if focused_index is not None and 0 <= focused_index < len(self._entries):
            focused_entry = self._entries[focused_index]
            if focused_entry.native_session_id:
                self._cancel_query_debounce()
                self.dismiss(ConsoleSwitcherChoice("rename", focused_entry))
                return
        for entry in self._entries:
            if entry.native_session_id:
                self._cancel_query_debounce()
                self.dismiss(ConsoleSwitcherChoice("rename", entry))
                return

    @staticmethod
    def _result_index_from_widget_id(widget_id: str) -> int | None:
        """Parse the result index out of a ``console-switcher-result-N`` id.

        Args:
            widget_id: Candidate widget id.

        Returns:
            The parsed index, or ``None`` if ``widget_id`` doesn't match the
            expected result-button id shape.
        """
        prefix = "console-switcher-result-"
        if not widget_id.startswith(prefix):
            return None
        try:
            return int(widget_id[len(prefix) :])
        except ValueError:
            return None
