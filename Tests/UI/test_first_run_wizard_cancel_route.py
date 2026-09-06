"""First-run wizard cancellation routing.

UAT (2026-08-31, TASK-31813 triage): Esc-exiting the boot-offered setup
wizard left the user on Home -- the screen the wizard was pushed over --
because a cancelled result carries no route and the handler returned early.
Cancellation must land on the Console workbench; cancelling a Settings /
command-palette RE-RUN must leave the current screen alone.
"""

from __future__ import annotations

from types import SimpleNamespace

from tldw_chatbook.app import TldwCli
from tldw_chatbook.Constants import TAB_CHAT, TAB_HOME, TAB_SETTINGS


class CancelHarness:
    """Duck-typed app recording the navigation the cancel branch performs."""

    def __init__(self, *, deferred_focus: bool = False, current_tab: str = TAB_HOME):
        self.calls: list[str] = []
        self.posted: list[object] = []
        self.focus_mode = False
        self._deferred_focus_request = deferred_focus
        self.current_tab = current_tab
        self.app_config = {"first_run": {"setup_completed": False}}

    def post_message(self, message) -> None:
        self.posted.append(message)
        name = getattr(message, "screen_name", None)
        self.calls.append(f"post:{name}" if name else "post:?")

    def run_worker(self, work, **_kwargs) -> None:
        self.calls.append("worker")

    def _schedule_startup_model_catalog_refresh(self, **_kwargs) -> None:
        self.calls.append("catalog")

    async def handle_screen_navigation(self, event) -> None:
        self.calls.append(f"navigate:{event.screen_name}")


def _cancel_result() -> None:
    return None


def test_cancelled_boot_wizard_routes_to_the_console() -> None:
    app = CancelHarness()

    TldwCli._continue_first_run_wizard_result(app, None)

    assert app.calls == [f"post:{TAB_CHAT}"], app.calls


def test_cancelled_boot_wizard_applies_deferred_focus_request() -> None:
    app = CancelHarness(deferred_focus=True)

    TldwCli._continue_first_run_wizard_result(app, None)

    assert app.focus_mode is True
    assert app._deferred_focus_request is False
    assert app.calls == [f"post:{TAB_CHAT}"], app.calls


def test_cancelled_rerun_leaves_the_current_screen_alone() -> None:
    app = CancelHarness(current_tab=TAB_SETTINGS)

    TldwCli._continue_first_run_wizard_result(
        app, None, cancel_to_console=False
    )

    assert app.calls == [], app.calls
    assert app.posted == []


def test_completed_exit_route_still_navigates_regression_pin() -> None:
    app = CancelHarness()

    TldwCli._continue_first_run_wizard_result(
        app,
        {"completed": True, "exit_route": TAB_HOME, "exit_context": None},
    )

    # The completed path still routes through the navigation worker (the
    # harness records the worker; the real worker navigates to Home).
    assert app.calls == ["worker"], app.calls


def test_interview_wrapper_forwards_cancel_flag() -> None:
    """The public entry must forward cancel_to_console to the continuation."""
    app = CancelHarness()

    TldwCli._handle_first_run_wizard_result(
        app, None, cancel_to_console=False
    )

    assert app.calls == [], app.calls

    TldwCli._handle_first_run_wizard_result(app, None)

    assert app.calls == [f"post:{TAB_CHAT}"], app.calls


def test_empty_registered_workspace_keeps_its_tree_node() -> None:
    """UAT triage pin: the tree derives nodes from the registry, not rows.

    A registered workspace with zero conversations must keep its node (with
    the empty status), so the tree never silently drops a workspace the
    user created.
    """
    from tldw_chatbook.Workspaces.workspace_tree_state import (
        build_workspace_tree_state,
    )

    projection = build_workspace_tree_state(
        workspaces=(("ws-empty", "Empty Workspace"),),
        rows=(),
    )

    assert len(projection) == 1
    assert projection[0].workspace_id == "ws-empty"
    assert projection[0].conversations == ()
