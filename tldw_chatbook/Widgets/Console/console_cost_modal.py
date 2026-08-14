"""Console cost-breakdown modal (task-5, PR3 cost ticker).

Read-only per-message cost breakdown opened from the cost chip
(``ConsoleCostChip.ConsoleCostChipPressed``, see ``console_status_chips.py``).
Rows are precomputed by ``console_cost_tracker.build_cost_rows``/
``build_cost_rows_totals`` and handed in at construction -- mirrors
``ConsoleRunLogModal``'s "already computed, just render it" shape, since
there is nothing here that needs a live refresh worker. Dismissal follows
``ConsoleContextModal``'s idiom: Escape and the Close button both call
``self.dismiss(None)``, and CSS lives inline on ``DEFAULT_CSS`` rather than
a source ``.tcss`` module (the other half of that same precedent).
"""

from __future__ import annotations

from typing import Sequence

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_cost_tracker import ConsoleCostRow, ConsoleCostRowTotals
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin

MODAL_ID = "console-cost-modal"
CLOSE_BUTTON_ID = "console-cost-modal-close"


class ConsoleCostModal(SafeModalDismissMixin, ModalScreen[None]):
    """Show the active Console session's per-message cost breakdown."""

    DEFAULT_CSS = """
    ConsoleCostModal { align: center middle; }
    #console-cost-modal {
        width: 96; max-width: 95%; height: 36; max-height: 90%;
        border: tall gray; padding: 1 2;
    }
    #console-cost-modal-header { height: auto; }
    #console-cost-modal-rows { height: 1fr; margin-top: 1; }
    .console-cost-modal-row { height: auto; }
    #console-cost-modal-totals { height: auto; margin-top: 1; text-style: bold; }
    #console-cost-modal-actions { height: auto; margin-top: 1; }
    """

    BINDINGS = [("escape", "request_safe_cancel", "Close")]
    SAFE_MODAL_CONTENT = "#console-cost-modal"

    def __init__(
        self,
        rows: Sequence[ConsoleCostRow],
        totals: ConsoleCostRowTotals,
    ) -> None:
        """Initialize the breakdown viewer.

        Args:
            rows: Precomputed per-message rows (``build_cost_rows``'s
                output) -- this widget never queries the store or the
                pricing catalog itself.
            totals: Precomputed aggregate row (``build_cost_rows_totals``'s
                output) for the same ``rows``.
        """
        super().__init__()
        self._rows = list(rows)
        self._totals = totals

    def compose(self) -> ComposeResult:
        """Build the header, scrollable row list, totals line, and Close action."""
        with Vertical(id=MODAL_ID):
            yield Static("Cost breakdown", id="console-cost-modal-header", markup=False)
            with VerticalScroll(id="console-cost-modal-rows"):
                if not self._rows:
                    yield Static("No priced or estimated messages yet.", markup=False)
                for row in self._rows:
                    yield Static(
                        self._format_row(row),
                        classes="console-cost-modal-row",
                        markup=False,
                    )
            yield Static(
                self._format_totals(self._totals),
                id="console-cost-modal-totals",
                markup=False,
            )
            with Horizontal(id="console-cost-modal-actions"):
                yield Button("Close", id=CLOSE_BUTTON_ID, variant="primary")

    @staticmethod
    def _format_row(row: ConsoleCostRow) -> str:
        """Pure ``str`` render for one breakdown row.

        task-2390: ``row.cost_usd`` already folds in any audio/
        transcription dollar contribution (see ``ConsoleCostRow``'s own
        docstring), so a realtime row's audio-token and transcription-
        duration usage is appended here as its own segment -- omitted
        entirely for a non-realtime row (all three fields 0) -- rather
        than left invisible inside that one total.
        """
        cost_text = "unpriced" if row.cost_usd is None else f"${row.cost_usd:.4f}"
        if row.estimated:
            cost_text = f"~{cost_text}"
        text = (
            f"[{row.index}] {row.role} ({row.model or 'unknown'}) -- "
            f"in:{row.uncached_input} cache_r:{row.cache_read} "
            f"cache_w:{row.cache_write} out:{row.output}"
        )
        if row.audio_input or row.audio_output:
            text += f" audio_in:{row.audio_input} audio_out:{row.audio_output}"
        if row.transcription_seconds:
            text += f" transcribe:{row.transcription_seconds:g}s"
        return f"{text} -- {cost_text}"

    @staticmethod
    def _format_totals(totals: ConsoleCostRowTotals) -> str:
        """Pure ``str`` render for the aggregate totals row."""
        if totals.total_cost_usd is None:
            cost_text = "unpriced"
        else:
            cost_text = f"${totals.total_cost_usd:.4f}"
            if totals.has_estimated_entries:
                cost_text = f"~{cost_text} (includes estimated rows)"
        return (
            f"Total -- {totals.total_tokens} tokens -- {cost_text} "
            f"({totals.row_count} rows)"
        )

    async def action_dismiss(self) -> None:
        """Dismiss, bound to the Escape key."""
        await self.request_safe_cancel(source="visible")

    @on(Button.Pressed, f"#{CLOSE_BUTTON_ID}")
    async def _close(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="visible")
