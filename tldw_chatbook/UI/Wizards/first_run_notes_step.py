"""The first-run wizard's Notes sync step.

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

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from tldw_chatbook.UI.Wizards.first_run_setup_widgets import SetupStep


class NotesSyncStep(SetupStep):
    """Explain where reviewed lasting folder sync is configured."""

    def compose_step(self) -> ComposeResult:
        with Vertical(classes="setup-notes"):
            yield Static("Notes folder sync", classes="setup-title")
            yield Static(
                "After setup, use Library → Notes → Add from files… to review a folder before activating sync.",
                classes="setup-subtitle",
            )
            # TASK-21140 (UAT G-3): reassurance, not an error — the error
            # class painted this calm sentence bold red-on-maroon.
            yield Static(
                "Nothing is activated during first-run setup.",
                classes="setup-step-note",
            )

    async def commit(self) -> tuple[bool, str]:
        return True, ""

    def get_step_data(self) -> Dict[str, Any]:
        return {}
