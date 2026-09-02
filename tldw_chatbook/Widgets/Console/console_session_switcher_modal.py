"""Console fuzzy session switcher modal (Ctrl+K)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rich.markup import escape as escape_markup
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.events import DescendantFocus
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


_SWITCHER_TITLE_MAX_CHARACTERS = 64
_SWITCHER_SUBTITLE_MAX_CHARACTERS = 120


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
        width: 76;
        max-width: 100%;
        height: 100%;
        max-height: 35;
        border: tall gray;
        background: black;
        padding: 1 2;
    }

    #console-switcher-results {
        height: 1fr;
        min-height: 3;
        margin: 1 0 0 0;
    }

    .console-switcher-section {
        height: 1;
        color: gray;
        text-style: bold;
    }

    #console-switcher-feedback {
        height: 1;
        color: yellow;
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

    .console-switcher-result-candidate {
        text-style: bold;
    }

    .console-switcher-result-active {
        text-style: underline;
    }
    """

    SAFE_MODAL_CONTENT = "#console-switcher-modal"
    BINDINGS = [
        ("escape", "request_safe_cancel", "Cancel"),
        ("f2", "rename_entry", "Rename"),
        Binding(
            "down",
            "switcher_cursor_down",
            "Next result",
            priority=True,
        ),
        Binding(
            "up",
            "switcher_cursor_up",
            "Previous result",
            priority=True,
        ),
    ]

    def __init__(
        self,
        *,
        rows: tuple[ConsoleConversationBrowserInputRow, ...],
        preferred_native_session_id: str | None = None,
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
        self._preferred_native_session_id = str(
            preferred_native_session_id or ""
        ).strip()
        self._entries: tuple[ConsoleSwitcherEntry, ...] = ()
        self._candidate_index = 0
        self._rendered_query = ""
        self._query_debounce_timer: Timer | None = None

    def compose(self) -> ComposeResult:
        """Build the search input and results container."""
        with Vertical(id="console-switcher-modal"):
            yield Static("Switch Session", classes="console-modal-header")
            yield Input(
                placeholder="Search title, workspace, or state…",
                id="console-switcher-query",
            )
            yield VerticalScroll(id="console-switcher-results")
            yield Static("", id="console-switcher-feedback", markup=False)
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
        previous_key = (
            self._entries[self._candidate_index].row_key
            if 0 <= self._candidate_index < len(self._entries)
            else ""
        )
        self._entries = build_console_switcher_entries(self._rows, query=query)
        self._rendered_query = query
        results = self.query_one("#console-switcher-results", VerticalScroll)

        await results.remove_children()

        if not self._entries:
            copy = (
                "No agent tabs yet. Ctrl+T creates an agent tab; saved chats "
                "appear after your first message."
                if not self._rows and not query.strip()
                else "No matches. Try a title, workspace, or state such as "
                "running, approval, queued, failed, or is:saved."
            )
            await results.mount(Static(copy, id="console-switcher-empty", markup=False))
            self._candidate_index = 0
        else:
            if query.strip():
                self._candidate_index = 0
            else:
                preferred_index = next(
                    (
                        index
                        for index, entry in enumerate(self._entries)
                        if entry.native_session_id == self._preferred_native_session_id
                    ),
                    None,
                )
                retained_index = next(
                    (
                        index
                        for index, entry in enumerate(self._entries)
                        if entry.row_key == previous_key
                    ),
                    None,
                )
                self._candidate_index = (
                    preferred_index
                    if preferred_index is not None
                    else retained_index
                    if retained_index is not None
                    else 0
                )
            widgets = []
            previous_section = ""
            for index, entry in enumerate(self._entries):
                if entry.section != previous_section:
                    widgets.append(
                        Static(
                            "OPEN AGENT TABS"
                            if entry.section == "open"
                            else "SAVED CHATS",
                            id=f"console-switcher-section-{entry.section}",
                            classes="console-switcher-section",
                            markup=False,
                        )
                    )
                    previous_section = entry.section
                button = Button(
                    self._entry_label(index, entry),
                    id=f"console-switcher-result-{index}",
                    classes="console-switcher-result",
                    compact=True,
                )
                button.set_class(entry.is_active, "console-switcher-result-active")
                button.set_class(
                    index == self._candidate_index,
                    "console-switcher-result-candidate",
                )
                button.tooltip = escape_markup(f"Switch to {entry.title}")
                widgets.append(button)
            await results.mount_all(widgets)

    def _entry_label(self, index: int, entry: ConsoleSwitcherEntry) -> Text:
        display_title = (
            sanitize_character_display_label(
                entry.title,
                max_characters=_SWITCHER_TITLE_MAX_CHARACTERS,
            )
            or "Untitled conversation"
        )
        display_subtitle = sanitize_character_display_label(
            entry.subtitle,
            max_characters=_SWITCHER_SUBTITLE_MAX_CHARACTERS,
        )
        marker = "▸" if index == self._candidate_index else " "
        label = f"{marker} {display_title}"
        if display_subtitle:
            label = f"{label}\n  {display_subtitle}"
        return Text(label)

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
        self._cancel_query_debounce()
        entries = (
            self._entries
            if event.value == self._rendered_query
            else build_console_switcher_entries(self._rows, query=event.value)
        )
        if not entries:
            return
        index = self._candidate_index if entries is self._entries else 0
        if not event.value.strip() and self._preferred_native_session_id:
            index = next(
                (
                    candidate_index
                    for candidate_index, entry in enumerate(entries)
                    if entry.native_session_id == self._preferred_native_session_id
                ),
                index,
            )
        if 0 <= index < len(entries):
            entry = entries[index]
            if entry.openable:
                self.dismiss(ConsoleSwitcherChoice("activate", entry))
            else:
                self._set_feedback(
                    "This saved chat is unavailable and cannot be opened."
                )

    def _result_buttons(self) -> list[Button]:
        """Return mounted result buttons in display order."""
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
            self._focus_candidate(buttons)
            return
        if index + 1 < len(buttons):
            self._candidate_index = index + 1
            self._focus_candidate(buttons)

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
        self._candidate_index = index - 1
        self._focus_candidate(buttons)

    def _focus_candidate(self, buttons: list[Button]) -> None:
        if not 0 <= self._candidate_index < len(buttons):
            return
        button = buttons[self._candidate_index]
        button.focus()
        self._sync_candidate_labels(buttons)
        button.scroll_visible(animate=False, immediate=True)

    def _sync_candidate_labels(self, buttons: list[Button] | None = None) -> None:
        mounted = buttons if buttons is not None else self._result_buttons()
        for index, button in enumerate(mounted):
            if not 0 <= index < len(self._entries):
                continue
            button.set_class(
                index == self._candidate_index,
                "console-switcher-result-candidate",
            )
            button.label = self._entry_label(index, self._entries[index])

    def on_descendant_focus(self, event: DescendantFocus) -> None:
        widget = event.widget
        if not isinstance(widget, Button) or not widget.has_class(
            "console-switcher-result"
        ):
            return
        if widget is not self.app.focused:
            return
        index = self._result_index_from_widget_id(widget.id or "")
        if index is None or not 0 <= index < len(self._entries):
            return
        self._candidate_index = index
        self._sync_candidate_labels()
        widget.scroll_visible(animate=False, immediate=True)

    @on(Button.Pressed, ".console-switcher-result")
    def _result_pressed(self, event: Button.Pressed) -> None:
        """Activate the clicked result entry.

        Args:
            event: The result button's press event.
        """
        event.stop()
        index = self._result_index_from_widget_id(event.button.id or "")
        if index is not None and 0 <= index < len(self._entries):
            if not self._entries[index].openable:
                self._set_feedback(
                    "This saved chat is unavailable and cannot be opened."
                )
                return
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
        """Request a rename only for the explicit focused/highlighted entry."""
        focused_index = self._result_index_from_widget_id(
            getattr(self.focused, "id", None) or ""
        )
        index = focused_index if focused_index is not None else self._candidate_index
        if not 0 <= index < len(self._entries):
            self._set_feedback("Choose an open agent tab to rename.")
            return
        entry = self._entries[index]
        if not entry.native_session_id:
            self._set_feedback("Saved chats cannot be renamed here; open one first.")
            return
        self._cancel_query_debounce()
        self.dismiss(ConsoleSwitcherChoice("rename", entry))

    def _set_feedback(self, message: str) -> None:
        try:
            feedback = self.query_one("#console-switcher-feedback", Static)
        except NoMatches:
            return
        feedback.update(message)

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
