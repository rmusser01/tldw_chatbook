"""Console selection feedback comment modal."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static

from tldw_chatbook.Widgets.Console.console_selection_menu import (
    ConsoleSelectionFeedbackRequested,
)
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin

_ACTION_HEADERS = {
    ConsoleSelectionFeedbackRequested.ACTION_REQUEST_CHANGES: (
        "Request changes — leave a comment"
    ),
    ConsoleSelectionFeedbackRequested.ACTION_LGM: "LGTM — leave a comment",
    ConsoleSelectionFeedbackRequested.ACTION_COMMENT: "Comment on selection",
}
_DEFAULT_HEADER = "Comment on selection"

# Display cap for the read-only quote preview. The quote itself is already
# capped by ``cap_quote`` before it leaves the transcript; this keeps the
# small modal clip-safe no matter what arrives.
PREVIEW_QUOTE_CAP = 600
PREVIEW_TRUNCATION_MARKER = "… [truncated]"


def _preview_text(quote: str) -> str:
    if len(quote) <= PREVIEW_QUOTE_CAP:
        return quote
    return quote[: PREVIEW_QUOTE_CAP - len(PREVIEW_TRUNCATION_MARKER)] + (
        PREVIEW_TRUNCATION_MARKER
    )


class ConsoleFeedbackCommentModal(SafeModalDismissMixin, ModalScreen[str | None]):
    """Collect an optional comment for a selection feedback action.

    Dismisses the stripped comment on Submit/Enter — ``""`` when the input
    is empty or whitespace-only, meaning "submit WITHOUT a comment" (the
    comment is optional, spec §3; feedback still flows). ``None`` is
    cancel-only: Cancel/Escape/backdrop via ``SafeModalDismissMixin``
    abandon the feedback entirely.
    """

    BUNDLED_CSS = """
    ConsoleFeedbackCommentModal {
        align: center middle;
    }

    #console-feedback-comment-modal {
        width: 56;
        height: auto;
        border: tall gray;
        background: black;
        padding: 1 2;
    }

    #console-feedback-comment-quote {
        height: 5;
        border: round gray;
        padding: 0 1;
        margin: 1 0;
        color: $text-muted;
    }

    #console-feedback-comment-input {
        width: 100%;
        margin: 0 0 1 0;
    }

    #console-feedback-comment-actions {
        height: 3;
        min-height: 3;
        margin: 1 0 0 0;
        align-horizontal: right;
    }

    #console-feedback-comment-cancel,
    #console-feedback-comment-submit {
        width: 10;
        min-width: 10;
        height: 3;
        min-height: 3;
    }
    """

    SAFE_MODAL_CONTENT = "#console-feedback-comment-modal"
    BINDINGS = [("escape", "request_safe_cancel", "Cancel")]

    def __init__(self, *, action: str, quote: str) -> None:
        super().__init__()
        self._action = action
        self._quote = quote

    def compose(self) -> ComposeResult:
        with Vertical(id="console-feedback-comment-modal"):
            yield Static(
                _ACTION_HEADERS.get(self._action, _DEFAULT_HEADER),
                id="console-feedback-comment-header",
            )
            yield Static(
                _preview_text(self._quote),
                id="console-feedback-comment-quote",
                markup=False,
            )
            yield Input(
                value="",
                id="console-feedback-comment-input",
                placeholder="Optional comment",
            )
            with Horizontal(id="console-feedback-comment-actions"):
                yield Button("Cancel", id="console-feedback-comment-cancel")
                yield Button(
                    "Submit",
                    id="console-feedback-comment-submit",
                    variant="primary",
                )

    def on_mount(self) -> None:
        self.query_one("#console-feedback-comment-input", Input).focus()

    @on(Button.Pressed, "#console-feedback-comment-cancel")
    async def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        await self.request_safe_cancel(source="button")

    @on(Button.Pressed, "#console-feedback-comment-submit")
    def _submit_button(self, event: Button.Pressed) -> None:
        event.stop()
        self._submit()

    @on(Input.Submitted, "#console-feedback-comment-input")
    def _submit_input(self, event: Input.Submitted) -> None:
        event.stop()
        self._submit()

    def _submit(self) -> None:
        comment = self.query_one("#console-feedback-comment-input", Input).value
        # "" (not None): submitting without a comment still sends the
        # feedback; None stays reserved for Cancel/Escape/backdrop.
        self.dismiss(comment.strip())
