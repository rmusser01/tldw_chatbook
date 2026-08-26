"""Intent-only controls for copying a selected local GGUF into managed storage."""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import Button, Static

from tldw_chatbook.UI.Screens.model_browser_state import format_mib


class LocalGGUFImportRequested(Message):
    """Request consent for importing one explicitly selected GGUF."""

    def __init__(self, path: Path) -> None:
        """Create an import request that preserves the exact selected path.

        Args:
            path: Explicit local GGUF selected by the user.
        """
        super().__init__()
        self.path = path


class LocalGGUFImportControls(Widget):
    """Render an intent-only Import action for an unmanaged GGUF row."""

    DEFAULT_CSS = """
    LocalGGUFImportControls {
        height: 3;
    }

    LocalGGUFImportControls Button {
        width: auto;
    }
    """

    def __init__(self, path: Path, *, pending: bool = False) -> None:
        """Create one local-GGUF import action.

        Args:
            path: User-selected local GGUF path.
            pending: Whether another import currently owns this action.
        """
        self.path = path
        self.pending = pending
        super().__init__()

    def compose(self) -> ComposeResult:
        """Compose the stable import action."""
        yield Button("Import…", classes="model-import", disabled=self.pending)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Post exact import intent without performing file work.

        Args:
            event: Textual button event for the row's Import action.
        """
        event.stop()
        if not self.pending:
            self.post_message(LocalGGUFImportRequested(self.path))


class LocalGGUFImportConsentModal(ModalScreen[bool]):
    """Return consent for one local managed copy without performing I/O."""

    DEFAULT_CSS = """
    LocalGGUFImportConsentModal {
        align: center middle;
    }

    LocalGGUFImportConsentModal .local-gguf-import-modal {
        width: 76;
        height: 90%;
        max-height: 90%;
        border: tall $accent;
        background: $surface;
        padding: 1 2;
    }

    LocalGGUFImportConsentModal .local-gguf-import-facts {
        height: 1fr;
        overflow-x: hidden;
    }

    LocalGGUFImportConsentModal .local-gguf-import-facts Static {
        height: auto;
        text-wrap: wrap;
    }

    LocalGGUFImportConsentModal .model-install-actions {
        height: 3;
        margin-top: 1;
        align-horizontal: right;
    }
    """

    BINDINGS = [("escape", "cancel", "Close")]

    def __init__(self, source: Path, size_bytes: int) -> None:
        """Build a consent prompt from caller-supplied local-file facts.

        Args:
            source: Explicitly selected local GGUF path; transient modal-only state.
            size_bytes: Caller-measured source size shown before consent.
        """
        self.source = source
        self.size_bytes = size_bytes
        super().__init__()

    def compose(self) -> ComposeResult:
        """Compose the local-copy consent facts and decision controls."""
        with Vertical(classes="local-gguf-import-modal"):
            with VerticalScroll(classes="local-gguf-import-facts"):
                yield Static(self.source.name, markup=False)
                yield Static(str(self.source), markup=False)
                yield Static(format_mib(self.size_bytes), markup=False)
                yield Static(
                    "Chatbook will create a managed copy. The original stays in place.",
                    markup=False,
                )
                yield Static(
                    "License and runtime compatibility are not verified.",
                    markup=False,
                )
            with Horizontal(classes="model-install-actions"):
                yield Button("Cancel", id="local-gguf-import-cancel", variant="default")
                yield Button(
                    "Import",
                    id="local-gguf-import-confirm",
                    variant="primary",
                )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Dismiss with the decision represented by the pressed control.

        Args:
            event: Textual button event for Import or Cancel.
        """
        if event.button.id == "local-gguf-import-confirm":
            self.dismiss(True)
        elif event.button.id == "local-gguf-import-cancel":
            self.dismiss(False)

    def action_cancel(self) -> None:
        """Dismiss the modal without consent."""
        self.dismiss(False)
