"""Evaluations screen implementation."""

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Vertical

from ..Navigation.base_app_screen import BaseAppScreen
from ..Workbench.workbench_state import WorkbenchHeaderState
from ..Workbench.workbench_widgets import DestinationHeader
from .lab_mode_strip import LabModeStrip

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class EvalsScreen(BaseAppScreen):
    """Evals destination seat hosting the evaluation workbench in the shell.

    The evaluation hub used to be pushed as a separate Textual `Screen` inside
    a `Container`, which is not a supported way to mount a `Screen` and left
    the body empty in the real app shell (confirmed by before/after capture).
    This is a stub -- header and mode strip only, empty workbench panel -- so
    the destination stays reachable while the real three-pane workbench is
    built on top of it.
    """

    def __init__(self, app_instance: "TldwCli", **kwargs):
        super().__init__(app_instance, "evals", **kwargs)

    def compose_content(self) -> ComposeResult:
        """Compose the Evals seat: identity header plus an empty workbench panel."""
        yield DestinationHeader(
            WorkbenchHeaderState(
                title="Evals",
                subtitle="Run and review evaluation jobs.",
                status="ready",
            ),
            id="evals-destination-header",
        )
        yield LabModeStrip(active_route="evals", id="lab-mode-strip")
        yield Vertical(id="evals-workbench", classes="ds-panel destination-workbench")

    def save_state(self):
        """Save evals screen state."""
        return super().save_state()

    def restore_state(self, state):
        """Restore evals screen state."""
        super().restore_state(state)
