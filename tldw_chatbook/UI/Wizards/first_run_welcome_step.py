"""The first-run wizard's Welcome step.

Moved whole out of ``FirstRunSetupWizard.py`` (TASK-34100.1), so its ``@on``
handlers and workers stay registered on the class and fixes to this step have
room under the size ratchet. ``FirstRunSetupWizard`` still exports the class.
Patch this module, not the wizard, to replace what the step calls.
"""

from __future__ import annotations

from typing import (
    Any,
    Dict,
)

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import (
    Button,
    RadioButton,
    Static,
)

from tldw_chatbook.UI.Wizards import first_run_setup_state as wizard_state
from tldw_chatbook.UI.Wizards.first_run_setup_widgets import (
    SetupRadioButton,
    SetupRadioSet,
    SetupStep,
)


class WelcomeStep(SetupStep):
    """Track choice: Quick / Full / Skip."""

    def compose_step(self) -> ComposeResult:
        with Vertical(classes="setup-welcome"):
            yield Static("Welcome to tldw chatbook", classes="setup-title")
            # TASK-21149 (UAT W-3/W-2): say what the app IS before asking
            # jargon questions, and give each path an honest time estimate.
            yield Static(
                "Chat with cloud or local AI models, keep notes, and work "
                "with your own documents — all in your terminal.",
                classes="setup-subtitle",
            )
            # task-31820: don't promise "every step can be skipped with
            # Next" -- the Provider step refuses Next for a keyed provider
            # until a key is supplied. Name the out that always works.
            yield Static(
                "Quick takes about 2 minutes; Full about 10. Everything can "
                "be changed later in Settings, and most steps can be "
                "skipped with Next — Esc exits setup.",
                classes="setup-subtitle",
            )
            with SetupRadioSet(id="setup-track-choice", classes="setup-choice-list"):
                # TASK-2154.9 (FR-02): name the steps the tracker will show
                # (Welcome is this one; Provider, Model and Summary follow)
                # so the "Step 1 of 4" count is not a surprise after picking
                # what read as a two-item "provider & model" track.
                yield SetupRadioButton(
                    "Quick setup — provider, model, voice, protection (recommended)",
                    value=True,
                    id="setup-track-quick",
                )
                yield SetupRadioButton(
                    "Full setup — configure everything", id="setup-track-full"
                )
            yield Button("Restore a backup", id="setup-backup-restore")
            yield Static("", classes="setup-step-error")

    @on(Button.Pressed, "#setup-backup-restore")
    def open_backup_restore(self, event: Button.Pressed) -> None:
        """Keep setup choices intact while the app owns the recovery view."""
        event.stop()
        self.app.action_backup_restore()

    def get_step_data(self) -> Dict[str, Any]:
        return {"track": self.chosen_track()}

    def busy_label(self) -> str:
        """What a slow Next from Welcome is doing (TASK-34100.1)."""
        full = self.chosen_track() == wizard_state.TRACK_FULL
        return f"Preparing the {'Full' if full else 'Quick'} setup…"

    def chosen_track(self) -> str:
        try:
            full = self.query_one("#setup-track-full", RadioButton).value
        except Exception:
            full = False
        return wizard_state.TRACK_FULL if full else wizard_state.TRACK_QUICK
