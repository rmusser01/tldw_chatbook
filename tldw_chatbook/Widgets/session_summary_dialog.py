"""Quit-time session summary dialog (issue #365).

Shown once by the quit worker after approved-quit cleanup completes and
before App.exit(). Reads only the snapshot passed in; never recomputes.
Spec: Docs/superpowers/specs/2026-09-22-quit-time-session-summary-design.md
"""

from __future__ import annotations

import time

from textual.app import ComposeResult
from textual.containers import Container
from textual.events import Key
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widgets import Static

from ..Chat.session_usage import SessionUsageSnapshot

__all__ = ["SessionSummaryDialog"]


def _format_elapsed(seconds: float) -> str:
    total_hours = int(seconds // 3600)
    days, hours = divmod(total_hours, 24)
    if days:
        return f"{days}d {hours}h session"
    minutes = int(seconds // 60) % 60
    if hours:
        return f"{hours}h {minutes:02d}m session"
    return f"{minutes}m session"


class SessionSummaryDialog(ModalScreen[None]):
    """Brief session usage summary; auto-closes, any key skips.

    Follows the SplashScreen timed-close idiom (strong timer reference,
    idempotent close that stops the timer) and the ConfirmationDialog
    small-modal structure. No BINDINGS: any key skips (ADR-031 -- no
    terminal-convention keys bound; ctrl+q stays app-global). Styling uses
    legacy semantic variables because ``$ds-*`` tokens do not resolve in
    Python-side DEFAULT_CSS (only in the bundled tcss).
    """

    DEFAULT_CSS = """
    SessionSummaryDialog {
        align: center middle;
    }
    #session-summary-dialog {
        width: 44;
        height: auto;
        background: $panel;
        border: round $secondary;
        padding: 1 2;
    }
    .session-summary-title {
        text-style: bold;
        color: $accent;
    }
    .session-summary-line {
        color: $text;
    }
    .session-summary-hint {
        color: $text-muted;
    }
    """

    def __init__(
        self,
        snapshot: SessionUsageSnapshot,
        *,
        started_at: float,
        duration_seconds: float,
    ) -> None:
        super().__init__()
        self.snapshot = snapshot
        self._started_at = started_at  # time.perf_counter() stamp
        self._duration = max(0.05, float(duration_seconds))
        self._closed = False
        self._auto_close_timer: Timer | None = None

    def compose(self) -> ComposeResult:
        with Container(id="session-summary-dialog"):
            yield Static("Session summary", classes="session-summary-title")
            if self.snapshot.calls:
                line = f"{self.snapshot.total_tokens:,} tokens"
                if self.snapshot.estimated_tokens > 0:
                    line += " · includes estimates"
                yield Static(line, classes="session-summary-line")
            else:
                yield Static(
                    "No usage recorded this session",
                    classes="session-summary-line",
                )
            elapsed = max(0.0, time.perf_counter() - self._started_at)
            yield Static(_format_elapsed(elapsed), classes="session-summary-line")
            if self.snapshot.embeddings_tokens > 0:
                yield Static(
                    f"{self.snapshot.embeddings_tokens:,} embedding tokens",
                    classes="session-summary-line",
                )
            yield Static("press any key to exit", classes="session-summary-hint")

    def on_mount(self) -> None:
        self._auto_close_timer = self.set_timer(self._duration, self._close)

    def on_key(self, event: Key) -> None:
        """Any key skips: the key's whole job here is to dismiss (splash
        precedent -- consume it)."""
        event.stop()
        event.prevent_default()
        self._close()

    def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._auto_close_timer is not None:
            self._auto_close_timer.stop()
        self.dismiss(None)
