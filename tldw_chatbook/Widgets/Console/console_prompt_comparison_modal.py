"""Read-only before/after review for an automatically improved draft."""

from __future__ import annotations

from typing import Literal

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static, TextArea

from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin

PromptComparisonResult = Literal["keep", "restore"]


class ConsolePromptComparisonModal(
    SafeModalDismissMixin, ModalScreen[PromptComparisonResult | None]
):
    """Compare the original and improved drafts without editing either copy."""

    BUNDLED_SCREEN_CSS = """
    ConsolePromptComparisonModal {
        align: center middle;
    }

    #console-prompt-comparison-modal {
        width: 90%;
        max-width: 100;
        min-width: 40;
        height: 88%;
        max-height: 40;
        min-height: 16;
        border: tall gray;
        background: black;
        padding: 1 2;
    }

    .console-prompt-comparison-label {
        width: 100%;
        height: 1;
        color: gray;
        text-style: bold;
    }

    #console-prompt-comparison-before,
    #console-prompt-comparison-after {
        width: 100%;
        height: 1fr;
        min-height: 4;
        border: solid gray;
        background: black;
    }

    #console-prompt-comparison-actions {
        width: 100%;
        height: 3;
        min-height: 3;
        margin: 1 0 0 0;
        align-horizontal: right;
    }

    #console-prompt-comparison-restore {
        width: 18;
        min-width: 18;
    }

    #console-prompt-comparison-keep {
        width: 22;
        min-width: 22;
    }
    """

    SAFE_MODAL_CONTENT = "#console-prompt-comparison-modal"
    BINDINGS = [("escape", "request_safe_cancel", "Close")]

    def __init__(self, *, before: str, after: str) -> None:
        """Initialize the comparison with immutable before/after text.

        Args:
            before: Draft text captured before automatic improvement.
            after: Draft text produced by automatic improvement.
        """
        super().__init__()
        self._before = before
        self._after = after

    def compose(self) -> ComposeResult:
        """Compose the read-only comparison and its decision actions.

        Returns:
            ComposeResult: Child widgets for the modal.
        """
        with Vertical(id="console-prompt-comparison-modal"):
            yield Static("Review prompt changes", classes="console-modal-header")
            yield Static(
                "Compare the automatic replacement with your original draft.",
                markup=False,
            )
            yield Static(
                "Original draft",
                classes="console-prompt-comparison-label",
                markup=False,
            )
            yield TextArea(
                self._before,
                read_only=True,
                id="console-prompt-comparison-before",
            )
            yield Static(
                "Improved draft",
                classes="console-prompt-comparison-label",
                markup=False,
            )
            yield TextArea(
                self._after,
                read_only=True,
                id="console-prompt-comparison-after",
            )
            with Horizontal(id="console-prompt-comparison-actions"):
                yield Button(
                    "Restore original",
                    id="console-prompt-comparison-restore",
                )
                yield Button(
                    "Keep improved draft",
                    id="console-prompt-comparison-keep",
                    variant="primary",
                )

    def on_mount(self) -> None:
        """Focus the non-destructive keep action after mounting.

        Returns:
            None: The modal is focused in place.
        """
        super().on_mount()
        self.query_one("#console-prompt-comparison-keep", Button).focus()

    @on(Button.Pressed, "#console-prompt-comparison-keep")
    def _keep(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss("keep")

    @on(Button.Pressed, "#console-prompt-comparison-restore")
    def _restore(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss("restore")
