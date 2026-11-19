"""TASK-33620.15.1.6 review: consent revoked in the hop before a landing still drops it.

The Environment panel's gather worker re-reads the workspace root after
gathering, then hands its landing to the UI loop. Consent revoked in that
hop reaches neither read; the landing must still be dropped -- on the real
consent service's admission revision, compared on the UI loop without a
database read -- with no ``gh`` fetch for the folder, and the next poll
lands UNBOUND. This drives the real registry and ``ChangeReviewConsentService``
(the controller-level and mounted-fake versions live in
``test_console_environment_controller.py`` and
``test_console_environment_root_reads.py``).
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

import tldw_chatbook.Workspaces.environment_status as env_mod
from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_environment_bound_registry import (
    _WORKSPACE,
    _canned_gatherers,
    _until,
)
from Tests.UI.test_console_environment_root_reads import _environment_entry
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_environment_state import (
    EnvSourceAvailability,
    PrEnvState,
)

# Writes a named workspace, a folder binding and consent into the registry:
# a private profile keeps them out of every other test's database.
pytestmark = pytest.mark.ui


@private_profile_test
async def test_consent_revoked_after_a_worker_re_check_drops_the_landing(
    request, monkeypatch, tmp_path: Path
):
    """AC#2, the hop, through the real consent service and registry.

    Change Review consent is revoked on the UI loop while the bound folder's
    gather worker has already re-checked the root and is handing its landing
    over. The real ``ChangeReviewConsentService.toggle`` moves the admission
    revision, so the landing is dropped -- no UI-loop read, no ``gh`` fetch
    for the folder -- and the next poll lands UNBOUND.
    """
    monkeypatch.setenv("TLDW_CHANGE_REVIEW_ENABLED", "1")
    _canned_gatherers(monkeypatch)
    gh_roots: list[str] = []

    def gather_pr(path, branch, *, previous=None):
        gh_roots.append(str(path))
        return PrEnvState(availability=EnvSourceAvailability.NOT_APPLICABLE)

    monkeypatch.setattr(env_mod, "gather_pr_env", gather_pr)
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
    monkeypatch.setattr(consent, "_initialize_root", lambda _path: None)

    main = threading.main_thread().ident
    ui_reads: list[str | None] = []
    admit_turn = consent.admit_turn

    def recording_admit_turn(workspace_id):
        entry = _environment_entry()
        if entry is not None and threading.get_ident() == main:
            ui_reads.append(entry)
        return admit_turn(workspace_id)

    monkeypatch.setattr(consent, "admit_turn", recording_admit_turn)

    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 45)) as pilot:
        screen = host.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        await pilot.pause(0.2)
        await _until(
            pilot,
            lambda: admit_turn(_WORKSPACE).ready_roots == (str(root),),
            "the consent service to admit the bound folder",
        )
        controller = screen._console_environment
        landings: list = []
        land = controller._land

        def recording_land(scope_root, *args):
            landings.append(scope_root)
            return land(scope_root, *args)

        controller._land = recording_land
        in_hop = threading.Event()
        revoked = threading.Event()
        marshal = controller._marshal_to_ui

        def hold_the_landing(fn, *args):
            # Runs on the gather worker, after its re-check saw the folder.
            if fn is recording_land and not revoked.is_set():
                in_hop.set()
                revoked.wait(5)
            return marshal(fn, *args)

        controller._marshal_to_ui = hold_the_landing

        await screen.action_toggle_console_inspector_rail()
        await _until(pilot, in_hop.is_set, "the folder's landing to reach the hop")
        consent.toggle(
            _WORKSPACE,
            expected=registry.read_change_review_consent(_WORKSPACE),
            enabled=False,
        )
        revoked.set()
        await _until(pilot, lambda: str(root) in landings, "the folder's landing")
        await pilot.pause()

        assert controller._has_landed is False
        assert gh_roots == [], "no gh fetch for a folder consent no longer admits"

        screen._poll_console_environment()
        await _until(
            pilot,
            lambda: controller.snapshot.git.availability
            is EnvSourceAvailability.UNBOUND,
            "the UNBOUND landing after consent was revoked",
        )
        assert ui_reads == [], ui_reads
