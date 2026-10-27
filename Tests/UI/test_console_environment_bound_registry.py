"""TASK-33620.15.1.6: the Environment panel's worker root reads go through the real registry.

The panel now reads the workspace root on workers -- the automatic refresh's
read, and the stale-scope guard's re-check after each gather -- instead of on
the UI loop. Those reads reach the real Change Review consent service and the
workspace registry database from a worker thread, where a failure is silent:
``capture_change_review_admission`` turns it into "no roots", and a bound
workspace would read as unbound or never land at all. This binds a folder to
a named workspace with Change Review consent through the real registry and
consent service, and checks the panel lands that folder with no consent read
on the UI loop.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

import tldw_chatbook.Workspaces.environment_status as env_mod
from Tests.UI.test_console_environment_root_reads import _environment_entry
from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_environment_state import (
    EnvSourceAvailability,
    GitEnvState,
    PrEnvState,
    TasksEnvState,
)

# Writes a named workspace, a folder binding and consent into the registry:
# a private profile keeps them out of every other test's database.
pytestmark = pytest.mark.ui

_WORKSPACE = "ws-environment-bound"


def _canned_gatherers(monkeypatch) -> None:
    monkeypatch.setattr(
        env_mod,
        "gather_git_env",
        lambda path, *, previous=None: GitEnvState(
            availability=EnvSourceAvailability.OK, root=str(path), branch="main"
        ),
    )
    monkeypatch.setattr(
        env_mod,
        "gather_pr_env",
        lambda path, branch, *, previous=None: PrEnvState(
            availability=EnvSourceAvailability.NOT_APPLICABLE
        ),
    )
    monkeypatch.setattr(
        env_mod.BacklogTaskScanner,
        "scan",
        lambda self, workspace_root, branch: TasksEnvState(),
    )


async def _until(pilot, predicate, what: str) -> None:
    for _ in range(250):
        if predicate():
            return
        await pilot.pause(0.02)
    raise AssertionError(f"timed out waiting for {what}")


@private_profile_test
async def test_a_bound_workspace_lands_through_the_real_consent_service(
    request, monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("TLDW_CHANGE_REVIEW_ENABLED", "1")
    _canned_gatherers(monkeypatch)
    app = _build_test_app()
    _configure_native_ready_console(app)
    registry = app.workspace_registry_service
    consent = app.change_review_consent_service
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    registry.create_workspace(workspace_id=_WORKSPACE, name="Environment bound")
    registry.add_folder_binding(_WORKSPACE, root)
    registry.set_change_review_enabled(_WORKSPACE, True)
    registry.set_active_workspace(_WORKSPACE)
    # Root preparation (the shadow history repo) is not what this checks.
    monkeypatch.setattr(consent, "_initialize_root", lambda _path: None)

    main = threading.main_thread().ident
    reads: list[tuple[str, tuple]] = []
    admit_turn = consent.admit_turn

    def recording_admit_turn(workspace_id):
        entry = _environment_entry()
        admission = admit_turn(workspace_id)
        if entry is not None:  # asked on account of the Environment panel
            thread = "ui" if threading.get_ident() == main else "worker"
            reads.append((thread, tuple(admission.ready_roots)))
        return admission

    monkeypatch.setattr(consent, "admit_turn", recording_admit_turn)

    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 45)) as pilot:
        screen = host.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        await pilot.pause(0.2)
        chat = screen._ensure_console_chat_controller()
        session_id = chat.store.active_session_id
        assert chat.store.session_workspace_id(session_id) == _WORKSPACE, (
            "harness precondition: the Console session is in the bound workspace"
        )
        # The first admission schedules root preparation; wait until ready.
        await _until(
            pilot,
            lambda: consent.admit_turn(_WORKSPACE).ready_roots == (str(root),),
            "the consent service to admit the bound folder",
        )
        controller = screen._console_environment
        reads.clear()

        await screen.action_toggle_console_inspector_rail()
        await _until(
            pilot,
            lambda: (
                controller._landed_root == str(root)
                and controller.snapshot.pr.availability
                is EnvSourceAvailability.NOT_APPLICABLE
            ),
            "the bound folder's local and net landings",
        )

        # No UI-loop read: the rail open's root read and the guard's
        # re-checks of the local and net landings all ran on workers, through
        # the real registry, and saw the bound folder.
        assert [roots for thread, roots in reads if thread == "ui"] == []
        worker = [roots for thread, roots in reads if thread == "worker"]
        assert worker == [(str(root),)] * 3, reads
        assert controller.snapshot.git.root == str(root)
        assert controller.snapshot.git.availability is EnvSourceAvailability.OK
