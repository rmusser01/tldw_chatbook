"""Evaluations screen implementation."""

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.binding import Binding

from ..Evals.evals_window_v3 import EvalsWindowV3
from ..Evals.navigation import EvalNavigationScreen, NavigateToEvalScreen
from ..Evals.navigation.eval_nav_screen import evals_workflows_chip_label
from ..Navigation.base_app_screen import BaseAppScreen
from ..Workbench.workbench_state import WorkbenchHeaderState
from ..Workbench.workbench_widgets import DestinationHeader
from .lab_mode_strip import LabModeStrip

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class EvalsScreen(BaseAppScreen):
    """Evals destination seat hosting the evaluation workbench in the shell.

    The evaluation hub used to be pushed as a separate screen on mount, which
    hid the shell chrome, left its card navigation unhandled, and stranded
    users on a permanent "Loading Evaluation Lab..." placeholder when they
    pressed Escape. The workbench now renders inline: the destination nav,
    status line, and footer stay visible, card navigation works through
    EvalsWindowV3, and Escape walks the workbench's own back stack.
    """

    BINDINGS = [
        Binding("escape", "evals_back", "Back", show=False),
        Binding("1", "evals_open('quick_test')", "Quick Test", show=False),
        Binding("4", "evals_open('results')", "Results", show=False),
        Binding("5", "evals_open('tasks')", "Tasks", show=False),
        # 2/3/6 (comparison, batch_eval, models) are planned workflows: bound
        # but dead keys violate ADR-031 rule 4, so they are not bound at all.
    ]

    def __init__(self, app_instance: "TldwCli", **kwargs):
        super().__init__(app_instance, "evals", **kwargs)

    def compose_content(self) -> ComposeResult:
        """Compose the Evals seat: identity header plus the inline workbench."""
        yield DestinationHeader(
            WorkbenchHeaderState(
                title="Evals",
                subtitle="Run and review evaluation jobs.",
                status="ready",
                status_label=evals_workflows_chip_label(),
            ),
            id="evals-destination-header",
        )
        yield LabModeStrip(active_route="evals", id="lab-mode-strip")
        yield EvalsWindowV3(self.app_instance, id="evals-window")

    #: Footer hint context (registered on mount; matches BINDINGS, ADR-031).
    EVALS_SHORTCUTS: tuple[tuple[str, str], ...] = (
        ("1/4/5", "open workflow"),
        ("esc", "back"),
    )

    def on_mount(self) -> None:
        super().on_mount()
        self.register_footer_shortcuts(
            source="evals", shortcuts=self.EVALS_SHORTCUTS
        )

    def action_evals_back(self) -> None:
        """Walk the evaluation workbench back stack, if it has one."""
        window = self.query_one(EvalsWindowV3)
        if window.screen_stack:
            window.go_back()

    def action_evals_open(self, screen_id: str) -> None:
        """Open an evaluation workflow by number shortcut from the hub.

        Only active while the navigation hub is current, mirroring the "Press
        [n]" hints on its cards. Text inputs consume digit keys first, so
        forms inside workflows are unaffected.
        """
        window = self.query_one(EvalsWindowV3)
        if isinstance(window.current_screen, EvalNavigationScreen):
            window.handle_navigation(NavigateToEvalScreen(screen_id))

    def save_state(self):
        """Save evals screen state."""
        return super().save_state()

    def restore_state(self, state):
        """Restore evals screen state."""
        super().restore_state(state)
