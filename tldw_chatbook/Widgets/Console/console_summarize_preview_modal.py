# console_summarize_preview_modal.py
"""Confirm dialog for manual conversation summarization (TASK-26017).

Shows what the summarize WILL do -- turns summarized, turns retained, the
estimated context change -- before any model call happens. The numbers come
from the same ``ManualMemoryPlan`` the commit path executes, so what is
previewed is what happens (AC#4). Dismisses the (possibly empty) focus
topic string on confirm and ``None`` on cancel; the caller re-runs
planning on commit, so a conversation that changed in between is
re-validated rather than trusted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static

from tldw_chatbook.Chat.console_context_compaction import (
    MAX_SUMMARY_FOCUS_CHARS,
)

from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin

if TYPE_CHECKING:
    from tldw_chatbook.Chat.console_context_compaction import ManualSummaryPreview


class ConsoleSummarizePreviewModal(
    SafeModalDismissMixin, ModalScreen["str | None"]
):
    """Preview one manual summarize; confirm or discard.

    Dismisses ``None`` on cancel; on confirm, the (possibly empty) focus
    topic string typed into the optional input (TASK-26018) -- empty means
    an unsteered summary, exactly today's behavior.
    """

    BUNDLED_CSS = """
    ConsoleSummarizePreviewModal {
        align: center middle;
    }

    #console-summarize-preview-modal {
        width: 64;
        max-width: 95%;
        height: auto;
        border: tall gray;
        background: black;
        padding: 1 2;
    }

    #console-summarize-preview-actions {
        height: auto;
        margin-top: 1;
        align-horizontal: right;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape", "request_safe_cancel", "Cancel", show=False)
    ]
    SAFE_MODAL_CONTENT = "#console-summarize-preview-modal"

    def __init__(self, preview: "ManualSummaryPreview") -> None:
        super().__init__()
        self._preview = preview

    def compose(self) -> ComposeResult:
        preview = self._preview
        turns = preview.turns_summarized
        retained = preview.turns_retained
        with Vertical(id="console-summarize-preview-modal"):
            yield Static(
                (
                    "Summarize from that message to the end?"
                    if preview.from_here
                    else "Summarize everything before that message?"
                ),
                markup=False,
            )
            yield Static(
                f"Summarized: {turns} turn{'s' if turns != 1 else ''}",
                id="console-summarize-preview-selected",
                markup=False,
            )
            yield Static(
                f"Kept as-is: {retained} turn{'s' if retained != 1 else ''}",
                id="console-summarize-preview-retained",
                markup=False,
            )
            yield Static(
                (
                    f"Context: ~{preview.before_tokens:,} → "
                    f"~{preview.after_tokens:,} tokens "
                    f"(summary cap {preview.output_cap:,})"
                ),
                id="console-summarize-preview-tokens",
                markup=False,
            )
            yield Static(
                "No model call happens until you confirm.",
                id="console-summarize-preview-note",
                markup=False,
            )
            yield Input(
                placeholder="Focus the summary on… (optional)",
                id="console-summarize-preview-focus",
                # TASK-26018 AC#4: bounded at the widget; the controller
                # re-sanitizes (collapse/cap/marker-refusal) regardless.
                max_length=MAX_SUMMARY_FOCUS_CHARS,
            )
            with Horizontal(id="console-summarize-preview-actions"):
                yield Button("Cancel", id="console-summarize-preview-cancel")
                yield Button(
                    "Summarize",
                    id="console-summarize-preview-confirm",
                    variant="primary",
                )

    @on(Button.Pressed, "#console-summarize-preview-cancel")
    async def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="visible")

    async def _perform_safe_cancel(self, *, source: str) -> None:
        del source
        self.dismiss_safe_once(None)

    @on(Input.Submitted, "#console-summarize-preview-focus")
    def _submit_focus(self, event: Input.Submitted) -> None:
        """Enter in the focus field confirms -- typing then Enter is the
        natural gesture; requiring a mouse trip to Summarize is not."""
        event.stop()
        self.dismiss(event.value or "")

    @on(Button.Pressed, "#console-summarize-preview-confirm")
    def _confirm(self, event: Button.Pressed) -> None:
        event.stop()
        try:
            topic = self.query_one("#console-summarize-preview-focus", Input).value
        except Exception:  # noqa: BLE001 -- a missing input never blocks commit
            topic = ""
        self.dismiss(topic or "")
