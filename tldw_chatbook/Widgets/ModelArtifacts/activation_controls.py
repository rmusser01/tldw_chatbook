"""Intent-only activation and deletion controls for installed models."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button

from tldw_chatbook.Model_Artifacts.service import ArtifactRef


class ActivationRequested(Message):
    """Request activation of one exact installed model reference."""

    def __init__(self, reference: ArtifactRef) -> None:
        super().__init__()
        self.reference = reference


class DeletionRequested(Message):
    """Request deletion of one exact installed model reference."""

    def __init__(self, reference: ArtifactRef) -> None:
        super().__init__()
        self.reference = reference


class RepairRequested(Message):
    """Request an explicit managed-store reconciliation."""


class ModelActivationControls(Widget):
    """Post lifecycle intents without calling the model service directly."""

    DEFAULT_CSS = """
    ModelActivationControls {
        height: 3;
    }

    ModelActivationControls Button {
        width: auto;
        margin-right: 1;
    }
    """

    def __init__(
        self,
        reference: ArtifactRef,
        *,
        active: bool,
        ready: bool,
        pending: bool = False,
    ) -> None:
        """Create controls for one exact installed reference.

        Args:
            reference: Exact managed-model identity.
            active: Whether this reference is already selected.
            ready: Whether verification/readiness allows activation.
            pending: Whether another lifecycle operation is running.
        """
        self.reference = reference
        self.active = active
        self.ready = ready
        self.pending = pending
        super().__init__()

    def compose(self) -> ComposeResult:
        """Compose the activation and deletion buttons."""
        with Horizontal():
            yield Button(
                "Active" if self.active else "Activate",
                classes="model-activate",
                variant="primary",
                disabled=self.pending or self.active or not self.ready,
            )
            yield Button(
                "Delete now",
                classes="model-delete",
                variant="error",
                disabled=self.pending,
            )

    def set_pending(self, pending: bool) -> None:
        """Disable or restore lifecycle controls while work is pending.

        Args:
            pending: Whether activation/deletion is pending.
        """
        self.pending = pending
        activate = self.query_one(".model-activate", Button)
        delete = self.query_one(".model-delete", Button)
        activate.disabled = pending or self.active or not self.ready
        delete.disabled = pending

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Post the exact lifecycle intent represented by a button."""
        event.stop()
        if self.pending:
            return
        if event.button.has_class("model-activate") and self.ready and not self.active:
            self.post_message(ActivationRequested(self.reference))
        elif event.button.has_class("model-delete"):
            self.post_message(DeletionRequested(self.reference))
