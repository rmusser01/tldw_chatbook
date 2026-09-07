"""Quick Note optimistic-conflict and switch-recovery dialogs."""

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from ...Widgets.modal_dismissal import SafeModalDismissMixin


class ResearchNoteConflictModal(SafeModalDismissMixin, ModalScreen[str | None]):
    BINDINGS = [Binding("escape", "request_safe_cancel", "Cancel", show=False)]
    SAFE_MODAL_CONTENT = "#research-note-conflict-dialog"

    def __init__(self) -> None:
        super().__init__(id="research-note-conflict-modal")

    def compose(self) -> ComposeResult:
        with Vertical(id="research-note-conflict-dialog"):
            yield Static("Quick Note changed", classes="dialog-title")
            yield Static(
                "The canonical owner has a newer version. Reload it, or preserve "
                "this draft as a new note.",
                markup=False,
            )
            yield Button("Reload", id="research-note-conflict-reload")
            yield Button("Copy as new", id="research-note-conflict-copy")
            yield Button("Cancel", id="research-note-conflict-cancel")

    @on(Button.Pressed)
    def choose(self, event: Button.Pressed) -> None:
        choice = str(event.button.id or "").removeprefix("research-note-conflict-")
        self.dismiss(choice if choice in {"reload", "copy"} else None)

    async def _perform_safe_cancel(self, *, source: str) -> None:
        del source
        self.dismiss_safe_once(None)


class ResearchNoteSwitchRecoveryModal(SafeModalDismissMixin, ModalScreen[str | None]):
    BINDINGS = [Binding("escape", "request_safe_cancel", "Cancel", show=False)]
    SAFE_MODAL_CONTENT = "#research-note-switch-dialog"

    def __init__(self) -> None:
        super().__init__(id="research-note-switch-recovery-modal")

    def compose(self) -> ComposeResult:
        with Vertical(id="research-note-switch-dialog"):
            yield Static("Quick Note was not saved", classes="dialog-title")
            yield Static(
                "The workspace or editor switch is paused. Retry the exact captured "
                "owner, discard these editor changes, or cancel the switch.",
                markup=False,
            )
            yield Button("Retry", id="research-note-switch-retry")
            yield Button("Discard editor changes", id="research-note-switch-discard")
            yield Button("Cancel", id="research-note-switch-cancel")

    @on(Button.Pressed)
    def choose(self, event: Button.Pressed) -> None:
        choice = str(event.button.id or "").removeprefix("research-note-switch-")
        self.dismiss(choice if choice in {"retry", "discard"} else None)

    async def _perform_safe_cancel(self, *, source: str) -> None:
        del source
        self.dismiss_safe_once(None)
