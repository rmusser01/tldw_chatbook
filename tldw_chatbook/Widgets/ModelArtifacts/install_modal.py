"""Consent-only modal for a managed-model acquisition plan."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox

from .plan_panel import ModelPlanPanel

if TYPE_CHECKING:
    from tldw_chatbook.Model_Artifacts.acquisition import PreflightReport


class ModelInstallModal(ModalScreen[bool]):
    """Return consent for a plan while leaving all work to the host screen."""

    DEFAULT_CSS = """
    ModelInstallModal {
        align: center middle;
    }

    ModelInstallModal .model-install-modal {
        width: 76;
        height: 90%;
        max-height: 90%;
        border: tall $accent;
        background: $surface;
        padding: 1 2;
    }

    ModelInstallModal .model-plan-panel {
        height: 1fr;
        overflow-y: auto;
    }

    ModelInstallModal .model-install-actions {
        height: 3;
        margin-top: 1;
        align-horizontal: right;
    }
    """

    BINDINGS = [("escape", "cancel", "Close")]

    def __init__(
        self,
        report: PreflightReport,
        *,
        model_label: str,
        container_id: str = "model-install-modal",
        confirm_id: str = "model-install-confirm",
        cancel_id: str = "model-install-cancel",
        required_acknowledgment: str | None = None,
        selected_file_details: tuple[tuple[str, int, str, str], ...] = (),
    ) -> None:
        """Build a consent prompt from an immutable preflight report.

        Args:
            report: The preflight plan to display and gate.
            model_label: User-visible model name.
            container_id: Container id for the consuming surface.
            confirm_id: Confirm button id for the consuming surface.
            cancel_id: Cancel button id for the consuming surface.
            required_acknowledgment: Optional text that must be acknowledged
                before installing.
            selected_file_details: Optional selected-only upstream file values
                to show before confirmation.
        """
        self.report = report
        self.model_label = model_label
        self.container_id = container_id
        self.confirm_id = confirm_id
        self.cancel_id = cancel_id
        self.required_acknowledgment = required_acknowledgment
        self.selected_file_details = selected_file_details
        self._acknowledged = required_acknowledgment is None
        super().__init__()

    @property
    def ungrantable(self) -> bool:
        """Return whether the report cannot produce valid consent."""
        return (
            bool(self.report.gating_errors)
            or not self.report.sufficient_space
            or not self._acknowledged
        )

    def compose(self) -> ComposeResult:
        """Compose the plan and decision controls."""
        with Vertical(id=self.container_id, classes="model-install-modal"):
            yield ModelPlanPanel(
                self.report,
                model_label=self.model_label,
                selected_file_details=self.selected_file_details,
            )
            if self.required_acknowledgment is not None:
                yield Checkbox(self.required_acknowledgment)
            with Horizontal(classes="model-install-actions"):
                yield Button("Cancel", id=self.cancel_id, variant="default")
                yield Button(
                    "Install",
                    id=self.confirm_id,
                    variant="primary",
                    disabled=self.ungrantable,
                )

    @on(Checkbox.Changed)
    def _acknowledgment_changed(self, event: Checkbox.Changed) -> None:
        """Update consent availability after an optional acknowledgment."""
        self._acknowledged = event.value
        self.query_one(f"#{self.confirm_id}", Button).disabled = self.ungrantable

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Dismiss with the decision represented by the pressed control."""
        if event.button.id == self.confirm_id:
            self.dismiss(True)
        elif event.button.id == self.cancel_id:
            self.dismiss(False)

    def action_cancel(self) -> None:
        """Dismiss the modal without consent."""
        self.dismiss(False)
