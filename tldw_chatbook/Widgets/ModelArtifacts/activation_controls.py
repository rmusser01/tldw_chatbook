"""Intent-only activation and deletion controls for installed models."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Static

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
        height: auto;
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
        allow_activation: bool | None = None,
        disabled_reason: str | None = None,
    ) -> None:
        """Create controls for one exact installed reference.

        Args:
            reference: Exact managed-model identity.
            active: Whether this reference is already selected.
            ready: Whether a readiness record already exists. Missing readiness
                does not prevent an eligible root from being activated.
            pending: Whether another lifecycle operation is running.
            allow_activation: Explicit activation eligibility. When omitted,
                readiness determines eligibility for compatibility with
                recovery surfaces.
        """
        self.reference = reference
        self.active = active
        self.ready = ready
        self.pending = pending
        self.allow_activation = allow_activation
        self.disabled_reason = disabled_reason
        self._activation_eligible = (
            ready if allow_activation is None else allow_activation
        )
        super().__init__()

    def compose(self) -> ComposeResult:
        """Compose the activation and deletion buttons."""
        with Horizontal():
            if self.allow_activation is not False:
                yield Button(
                    "Active" if self.active else "Activate",
                    classes="model-activate",
                    variant="primary",
                    disabled=(
                        self.pending or self.active or not self._activation_eligible
                    ),
                )
            yield Button(
                "Delete…",
                classes="model-delete",
                variant="error",
                disabled=self.pending,
            )
        if self.pending and self.disabled_reason is not None:
            yield Static(
                self.disabled_reason,
                classes="model-disabled-reason",
                markup=False,
            )

    def set_pending(self, pending: bool) -> None:
        """Disable or restore lifecycle controls while work is pending.

        Args:
            pending: Whether activation/deletion is pending.
        """
        self.pending = pending
        delete = self.query_one(".model-delete", Button)
        for activate in self.query(".model-activate"):
            activate.disabled = pending or self.active or not self._activation_eligible
        delete.disabled = pending

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Post the exact lifecycle intent represented by a button."""
        event.stop()
        if self.pending:
            return
        if (
            self._activation_eligible
            and event.button.has_class("model-activate")
            and not self.active
        ):
            self.post_message(ActivationRequested(self.reference))
        elif event.button.has_class("model-delete"):
            self.post_message(DeletionRequested(self.reference))
