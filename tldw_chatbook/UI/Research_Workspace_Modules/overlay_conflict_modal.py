"""Explicit recovery choices for device-only Research overlay conflicts."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from ...Widgets.modal_dismissal import SafeModalDismissMixin


class ResearchOverlayConflictModal(SafeModalDismissMixin, ModalScreen[str | None]):
    """Offer non-destructive CAS conflict recovery without silent overwrite."""

    BINDINGS = [Binding("escape", "request_safe_cancel", "Cancel", show=False)]
    SAFE_MODAL_CONTENT = "#research-overlay-conflict-dialog"

    def compose(self) -> ComposeResult:
        with Vertical(id="research-overlay-conflict-dialog"):
            yield Static("Device overlay changed", classes="dialog-title")
            yield Static(
                "Another screen or process saved this workspace's device-only "
                "layout. Your local draft is retained until you choose a recovery.",
                id="research-overlay-conflict-reason",
                markup=False,
            )
            yield Button("Reload saved overlay", id="research-overlay-conflict-reload")
            yield Button("Export device overlay", id="research-overlay-conflict-export")
            yield Button(
                "Fork / copy device layout", id="research-overlay-conflict-fork"
            )
            yield Static(
                "Overwrite is unavailable because the overlay owner exposes no "
                "confirmed force-write contract.",
                id="research-overlay-conflict-overwrite-unavailable",
                markup=False,
            )
            yield Button("Keep local draft", id="research-overlay-conflict-cancel")

    @on(Button.Pressed)
    def choose(self, event: Button.Pressed) -> None:
        action = str(event.button.id or "").removeprefix("research-overlay-conflict-")
        if action in {"reload", "export", "fork"}:
            self.dismiss(action)
        elif action == "cancel":
            self.dismiss(None)

    async def _perform_safe_cancel(self, *, source: str) -> None:
        del source
        self.dismiss_safe_once(None)
