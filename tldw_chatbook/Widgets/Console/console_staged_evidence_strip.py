"""Console staged-evidence strip (directly above the composer).

RAG-40: before this widget, the only surface that listed staged Library-RAG
evidence was the Inspector-rail tray, and Console's own "Run Library RAG"
success path never opened that rail -- so with the rail closed the single
"Sources: N staged" chip was the entire feedback for a bundle that was about
to be prepended to the user's next prompt. This strip puts the staged rows on
the main surface, one keystroke from the composer, and carries the only
un-stage control in the Console.

The strip renders three shapes, all decided by the pure state contract in
``Chat/console_display_state.py``: hidden, staged rows, or the one-send
"evidence sent" notice.
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_display_state import (
    CONSOLE_STAGED_EVIDENCE_UNSTAGE_ID,
    ConsoleStagedEvidenceStripState,
)
from tldw_chatbook.Widgets.recompose_capture_guard import RecomposeCaptureGuard


class ConsoleStagedEvidenceStrip(RecomposeCaptureGuard, Vertical):
    """Render the evidence staged for the next Console send.

    Every user-supplied string reaching this widget (library titles, source
    labels) is already display-escaped by the state builder, and every
    ``Static`` here is rendered with ``markup=False`` so escaping stays the
    TERMINAL step of the pipeline -- a retrieved document titled
    ``[bold]...`` can never reach a console-markup parser.
    """

    def __init__(self, state: ConsoleStagedEvidenceStripState, **kwargs: Any) -> None:
        """Initialize the staged-evidence strip.

        Args:
            state: Staged-evidence display-state snapshot to render.
            **kwargs: Additional Textual widget arguments.
        """
        super().__init__(**kwargs)
        self.state = state
        self.display = state.visible

    def compose(self) -> ComposeResult:
        if not self.state.visible:
            return

        if self.state.notice:
            yield Static(
                self.state.notice,
                id="console-staged-evidence-notice",
                classes="console-staged-evidence-notice",
                markup=False,
            )
            return

        with Horizontal(classes="console-staged-evidence-header"):
            yield Static(
                self.state.heading,
                id="console-staged-evidence-heading",
                classes="console-staged-evidence-heading",
                markup=False,
            )
            yield Button(
                self.state.unstage_label,
                id=CONSOLE_STAGED_EVIDENCE_UNSTAGE_ID,
                classes="console-staged-evidence-unstage",
            )

        for index, row in enumerate(self.state.rows):
            yield Static(
                f"{row.title} — {row.source}",
                id=f"console-staged-evidence-row-{index}",
                classes=f"console-staged-evidence-row {row.status}",
                markup=False,
            )

        if self.state.overflow:
            yield Static(
                self.state.overflow,
                id="console-staged-evidence-overflow",
                classes="console-staged-evidence-overflow",
                markup=False,
            )

    def sync_state(self, state: ConsoleStagedEvidenceStripState) -> None:
        """Refresh the mounted strip from a new staged-evidence snapshot.

        Equality-guarded like the other Console trays: an unchanged state is
        a no-op, and a real change recomposes only this widget (row count,
        the un-stage button, and the notice line all vary with the state),
        never the owning screen.

        Args:
            state: Staged-evidence display-state snapshot to render.
        """
        if state == self.state:
            return
        self.state = state
        self.display = state.visible
        self.refresh(recompose=True)
