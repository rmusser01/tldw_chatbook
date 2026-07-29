"""Console "View full log" modal (TASK-870).

Read-only viewer for one agent run's full, untruncated run log -- the
counterpart to the Console's configurable tool-result DISPLAY cap (see
``Chat.console_agent_bridge._console_tool_result_display_cap``): where that
setting governs how much of a tool result the transcript/rail SHOW, this
modal is how a user reaches everything the run actually recorded, including
whatever the display cap trimmed away. Only ever opened when the caller has
already confirmed a log exists for the run (``ConsoleAgentBridge.run_log_
available``) -- this widget itself does no existence checking, so it never
needs to render an empty/error state.
"""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches, QueryError
from textual.screen import ModalScreen
from textual.widgets import Button, Static, TextArea

MODAL_ID = "console-run-log-modal"
TEXT_AREA_ID = "console-run-log-text"
CLOSE_BUTTON_ID = "console-run-log-close"


class ConsoleRunLogModal(ModalScreen[None]):
    """Show one run's full run log text, read-only.

    Dismisses with ``None`` on Close/Escape -- there is nothing to apply,
    this widget never mutates anything.
    """

    BINDINGS = [("escape", "dismiss_viewer", "Close")]

    def __init__(self, *, run_id: str, log_text: str) -> None:
        """Initialize the viewer.

        Args:
            run_id: The run whose log this is, shown in the header so the
                user can tell which run they are looking at (relevant when
                drilled into a sub-agent run).
            log_text: The full rendered log text (already produced by
                ``ConsoleAgentBridge.load_run_log_text`` -- this widget does
                not fetch or format it itself).
        """
        super().__init__()
        self._run_id = run_id
        self._log_text = log_text

    def compose(self) -> ComposeResult:
        """Build the header, read-only text area, and Close action."""
        with Vertical(id=MODAL_ID):
            yield Static(
                f"Full run log — {self._run_id}",
                classes="console-modal-header",
                markup=False,
            )
            yield TextArea(
                self._log_text,
                id=TEXT_AREA_ID,
                read_only=True,
                soft_wrap=True,
            )
            with Horizontal(id="console-run-log-actions"):
                yield Button("Close", id=CLOSE_BUTTON_ID, variant="primary")

    def on_mount(self) -> None:
        """Focus the text area so the log is immediately scrollable."""
        try:
            self.query_one(f"#{TEXT_AREA_ID}", TextArea).focus()
        except (NoMatches, QueryError):
            pass

    def action_dismiss_viewer(self) -> None:
        """Dismiss, bound to the Escape key."""
        self.dismiss(None)

    @on(Button.Pressed, f"#{CLOSE_BUTTON_ID}")
    def _close(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)
