"""Console-native staged context tray."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches, QueryError
from textual.widgets import Static

from tldw_chatbook.Chat.console_display_state import ConsoleStagedContextState
from tldw_chatbook.Widgets.recompose_capture_guard import RecomposeCaptureGuard
from tldw_chatbook.Widgets.Console.console_bounded_section import (
    ConsoleBoundedSection,
)


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


class ConsoleStagedContextTray(RecomposeCaptureGuard, Vertical):
    """Render staged handoff/live-work provenance in the Console shell.

    The tray shows the current staged-context heading, summary, structured
    provenance rows, and recovery guidance supplied by the pure Console
    display-state contract.
    """

    def __init__(
        self,
        state: ConsoleStagedContextState,
        *,
        on_reconcile: Callable[[], None] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the staged-context tray.

        Args:
            state: Staged-context display-state snapshot to render.
            **kwargs: Additional Textual widget arguments.
        """
        super().__init__(**kwargs)
        self.state = state
        self._on_reconcile = on_reconcile

    def _body_widgets(self) -> list[Static | Vertical]:
        """Build the body while keeping the stable header outside its viewport."""

        body: list[Static | Vertical] = []
        if self.state.summary:
            body.append(
                Static(
                    self.state.summary,
                    id="console-staged-context-summary",
                    classes="console-staged-context-summary",
                    markup=False,
                )
            )

        if self.state.rows:
            for index, row in enumerate(self.state.rows):
                status_class = _normalize_source_status(row.status)
                body.append(
                    Vertical(
                        Static(
                            str(row.value),
                            id=f"console-staged-source-name-{index}",
                            classes="console-staged-source-name",
                            markup=False,
                        ),
                        Static(
                            status_class,
                            id=f"console-staged-source-status-{index}",
                            classes=f"console-staged-source-status {status_class}",
                            markup=False,
                        ),
                        id=f"console-staged-context-row-{index}",
                        classes="console-staged-source-row",
                    )
                )
        else:
            body.append(
                Static(
                    "No sources attached. Stage sources from Library.",
                    id="console-staged-context-empty",
                    classes="console-staged-context-empty",
                )
            )

        if self.state.recovery:
            body.append(
                Static(
                    self.state.recovery,
                    id="console-staged-context-recovery",
                    classes="console-staged-context-recovery",
                )
            )
        return body

    def compose(self) -> ComposeResult:
        with Horizontal(classes="console-staged-context-header"):
            yield Static(
                "Sources",
                id="console-staged-context-title",
                classes="console-rail-section-title",
            )
            yield Static(
                str(self.state.source_count),
                id="console-staged-context-count",
                classes="console-staged-context-count",
            )

        yield ConsoleBoundedSection(*self._body_widgets(), section_id="sources")

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
        self.call_after_refresh(self._request_section_reconcile)

    def _request_section_reconcile(self) -> None:
        """Settle local demand before invalidating the Inspector owner."""

        try:
            section = self.query_one(
                "#console-bounded-section-sources", ConsoleBoundedSection
            )
        except (NoMatches, QueryError):
            return
        section.request_reconcile()
        if self._on_reconcile is not None:
            self._on_reconcile()
