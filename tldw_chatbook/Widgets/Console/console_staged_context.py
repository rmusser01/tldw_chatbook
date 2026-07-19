"""Console-native staged context tray."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static

from tldw_chatbook.Chat.console_display_state import ConsoleStagedContextState


_STATUS_CLASS_MAP = {
    "ready": {"ready", "available", "attached"},
    "running": {"retrieving", "running", "stale"},
    "blocked": {"blocked", "missing", "unavailable"},
}


def _normalize_source_status(status: str) -> str:
    """Map a raw source status to one of the UI status classes.

    Args:
        status: Raw status value from the display state row.

    Returns:
        One of ``ready``, ``running``, ``blocked``, or ``muted``.
    """
    normalized = str(status or "").strip().lower()
    for class_name, synonyms in _STATUS_CLASS_MAP.items():
        if normalized in synonyms:
            return class_name
    return "muted"


class ConsoleStagedContextTray(Vertical):
    """Render staged handoff/live-work provenance in the Console shell.

    The tray shows the current staged-context heading, summary, structured
    provenance rows, and recovery guidance supplied by the pure Console
    display-state contract.
    """

    def __init__(self, state: ConsoleStagedContextState, **kwargs: Any) -> None:
        """Initialize the staged-context tray.

        Args:
            state: Staged-context display-state snapshot to render.
            **kwargs: Additional Textual widget arguments.
        """
        super().__init__(**kwargs)
        self.state = state

    def compose(self) -> ComposeResult:
        with Horizontal(classes="console-staged-context-header"):
            yield Static(
                "Sources",
                id="console-staged-context-title",
                classes="console-rail-section-title",
            )
            yield Static(
                str(len(self.state.rows)),
                id="console-staged-context-count",
                classes="console-staged-context-count",
            )

        if self.state.summary:
            yield Static(
                self.state.summary,
                id="console-staged-context-summary",
                classes="console-staged-context-summary",
            )

        if self.state.rows:
            for index, row in enumerate(self.state.rows):
                status_class = _normalize_source_status(row.status)
                with Vertical(
                    id=f"console-staged-context-row-{index}",
                    classes="console-staged-source-row",
                ):
                    yield Static(
                        str(row.value),
                        id=f"console-staged-source-name-{index}",
                        classes="console-staged-source-name",
                        markup=False,
                    )
                    yield Static(
                        status_class,
                        id=f"console-staged-source-status-{index}",
                        classes=f"console-staged-source-status {status_class}",
                        markup=False,
                    )
        else:
            yield Static(
                "No sources attached. Stage sources from Library.",
                id="console-staged-context-empty",
                classes="console-staged-context-empty",
            )

        if self.state.recovery:
            yield Static(
                self.state.recovery,
                id="console-staged-context-recovery",
                classes="console-staged-context-recovery",
            )

    def sync_state(self, state: ConsoleStagedContextState) -> None:
        """Refresh the mounted tray from a new staged-context snapshot.

        Equality-guarded like the other Console tray widgets; a real change
        recomposes only this widget (row count, Attach button, and recovery
        line presence all vary with the state), never the owning screen.

        Args:
            state: Staged-context display-state snapshot to render.
        """
        if state == self.state:
            return
        self.state = state
        self.refresh(recompose=True)
