"""Console fuzzy session switcher modal (Ctrl+K)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static

from tldw_chatbook.Chat.console_switcher_state import (
    ConsoleSwitcherEntry,
    build_console_switcher_entries,
)
from tldw_chatbook.Workspaces.conversation_browser_state import (
    ConsoleConversationBrowserInputRow,
)


@dataclass(frozen=True)
class ConsoleSwitcherChoice:
    """Result returned by the session switcher modal."""

    kind: str
    entry: ConsoleSwitcherEntry


class ConsoleSessionSwitcherModal(ModalScreen["ConsoleSwitcherChoice | None"]):
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

    .console-switcher-result {
        width: 100%;
        height: 2;
        min-height: 2;
        margin: 0;
    }
    """

    BINDINGS = [
        ("escape", "dismiss_switcher", "Cancel"),
        ("f2", "rename_entry", "Rename"),
    ]

    def __init__(
        self,
        *,
        rows: tuple[ConsoleConversationBrowserInputRow, ...],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._rows = rows
        self._entries: tuple[ConsoleSwitcherEntry, ...] = ()

    def compose(self) -> ComposeResult:
        with Vertical(id="console-switcher-modal"):
            yield Static("Switch Session", classes="console-modal-header")
            yield Input(
                placeholder="Search conversations…",
                id="console-switcher-query",
            )
            yield Vertical(id="console-switcher-results")

    def on_mount(self) -> None:
        self.query_one("#console-switcher-query", Input).focus()
        self._refresh_results("")

    def _refresh_results(self, query: str) -> None:
        self._entries = build_console_switcher_entries(self._rows, query=query)
        results = self.query_one("#console-switcher-results", Vertical)

        # Clear existing children
        for child in list(results.children):
            child.remove()

        # Schedule remounting after the event loop processes removals
        self.app.call_later(self._mount_results_content)

    def _mount_results_content(self) -> None:
        """Mount result buttons after clearing old ones."""
        results = self.query_one("#console-switcher-results", Vertical)
        if not self._entries:
            results.mount(
                Static("No matches.", id="console-switcher-empty", markup=False)
            )
        else:
            for index, entry in enumerate(self._entries):
                label = entry.title if not entry.subtitle else f"{entry.title}\n  {entry.subtitle}"
                button = Button(
                    label,
                    id=f"console-switcher-result-{index}",
                    classes="console-switcher-result",
                    compact=True,
                )
                button.set_class(entry.is_active, "console-switcher-result-active")
                button.tooltip = f"Switch to {entry.title}"
                results.mount(button)

    @on(Input.Changed, "#console-switcher-query")
    def _query_changed(self, event: Input.Changed) -> None:
        event.stop()
        self._refresh_results(event.value)

    @on(Input.Submitted, "#console-switcher-query")
    def _query_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        if self._entries:
            self.dismiss(ConsoleSwitcherChoice("activate", self._entries[0]))

    @on(Button.Pressed, ".console-switcher-result")
    def _result_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        button_id = event.button.id or ""
        try:
            index = int(button_id.rsplit("-", 1)[-1])
        except ValueError:
            return
        if 0 <= index < len(self._entries):
            self.dismiss(ConsoleSwitcherChoice("activate", self._entries[index]))

    def action_dismiss_switcher(self) -> None:
        self.dismiss(None)

    def action_rename_entry(self) -> None:
        for entry in self._entries:
            if entry.native_session_id:
                self.dismiss(ConsoleSwitcherChoice("rename", entry))
                return
