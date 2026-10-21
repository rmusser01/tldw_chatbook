"""TASK-33620.15.1.3: an Environment refresh never builds a turn configuration on the UI loop.

Measured live (isolated profile, Anthropic claude-haiku-4-5, 235x52) on
origin/dev 2d071d6d71: ``ConsoleEnvironmentController.request_refresh`` held
the UI loop 1.18-1.36 s when the Inspect rail opened and 0.17-0.20 s at each
run completion, and every 10 s poll tick held it 0.2-0.5 s. Sampled stacks put
95% of it in ONE call: the panel's workspace-root accessor
(``ChatScreen._console_environment_root`` -> ``_review_selection_workspace_
roots``) asked ``resolve_turn_execution_context`` for a WHOLE turn
configuration snapshot -- RAG profile depth through storage admission, the MCP
definition maximum, the skill catalog, project authority, ... -- and kept only
``workspace_roots``, which is one consent read (1-12 ms). The controller asks
its accessor on every request, poll tick and landing, so one rail open paid it
twice and one poll tick three times.

The root accessor now reads only the consent-admitted roots. These tests pin
that through the REAL mounted Console: the rail-open, run-completion and poll
paths reach the panel's landing for the consent-admitted root, and the full
turn-configuration builder is never entered on account of the Environment
refresh. The builder is spied on the live controller instance, so the check is
"was the slow part on this path at all", not a wall-clock threshold.
"""

from __future__ import annotations

import sys
import threading
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

import tldw_chatbook.Workspaces.environment_status as env_mod
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_environment_state import (
    EnvSourceAvailability,
    GitEnvState,
    PrEnvState,
    TasksEnvState,
)

# Drives the real ChatScreen, controller, store and app: keep the
# collection-time profile (the per-test redirect fails config admission).
pytestmark = pytest.mark.bootstrap_profile

_ENVIRONMENT_MODULE = "Console_Modules/environment.py"


def _admitting(root: str) -> SimpleNamespace:
    """A Change Review consent service that admits exactly ``root``."""
    return SimpleNamespace(
        admit_turn=lambda _workspace_id: SimpleNamespace(
            ready_roots=(root,),
            ready_aliases=("alias-one",),
            skipped_roots=(),
        )
    )


@asynccontextmanager
async def _console_with_admitted_root(root: str):
    """Mount the send-ready Console with ``root`` consent-admitted."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.change_review_consent_service = _admitting(root)
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 45)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await pilot.pause(0.2)
        yield pilot, console


def _canned_gatherers(monkeypatch, root: str) -> None:
    """Keep git/``gh``/backlog off the test: the landing is what matters."""
    monkeypatch.setattr(
        env_mod,
        "gather_git_env",
        lambda path, *, previous=None: GitEnvState(
            availability=EnvSourceAvailability.OK, root=root, branch="main"
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


def _spy_turn_configuration_builds(controller) -> list[str]:
    """Record every full turn-configuration build the Environment panel causes.

    Wraps the live controller instance's builder entry point
    (``resolve_turn_execution_context`` delegates to it). A build counts when
    the Environment controller's module is on the calling stack -- i.e. it was
    made on account of an Environment refresh, poll or landing -- and the
    record names the thread it ran on.
    """
    original = controller.resolve_turn_configuration_snapshot
    main = threading.main_thread().ident
    builds: list[str] = []

    def spy(session_id, *args, **kwargs):
        frame = sys._getframe(1)
        on_account_of_environment = False
        while frame is not None:
            if frame.f_code.co_filename.replace("\\", "/").endswith(
                _ENVIRONMENT_MODULE
            ):
                on_account_of_environment = True
                break
            frame = frame.f_back
        del frame
        if on_account_of_environment:
            builds.append("ui-loop" if threading.get_ident() == main else "worker")
        return original(session_id, *args, **kwargs)

    controller.resolve_turn_configuration_snapshot = spy
    return builds


async def _wait_for_landing(pilot, screen, root: str) -> None:
    controller = screen._console_environment
    for _ in range(100):
        if controller._has_landed and controller._landed_root == root:
            return
        await pilot.pause(0.02)
    raise AssertionError(
        "the Environment panel never landed the consent-admitted root "
        f"{root!r} (landed={controller._has_landed}, "
        f"root={controller._landed_root!r}, git={controller.snapshot.git!r})"
    )


@pytest.mark.asyncio
async def test_rail_open_completion_and_poll_never_build_a_turn_configuration(
    monkeypatch, tmp_path
):
    """AC#3: the slow part of an Environment refresh is gone from the UI loop.

    Base (origin/dev 2d071d6d71) fails this on the first assertion: opening
    the rail alone builds the whole turn configuration twice on the UI loop
    (``request_refresh`` asks the root, and so does the landing), the
    completion nudge does it again, and the poll tick three times.
    """
    root = str(tmp_path)
    _canned_gatherers(monkeypatch, root)
    async with _console_with_admitted_root(root) as (pilot, screen):
        chat_controller = screen._ensure_console_chat_controller()
        assert chat_controller.store.active_session_id, (
            "harness precondition: the Console has an active session, so the "
            "root accessor reaches the controller rather than UNKNOWN_ROOT"
        )
        builds = _spy_turn_configuration_builds(chat_controller)

        # Rail open, through the real accelerator (notify_rail_opened).
        await screen.action_toggle_console_inspector_rail()
        await pilot.pause()
        await _wait_for_landing(pilot, screen, root)
        # Run completion: the coalesced fleet sync's Environment nudge.
        screen._run_coalesced_console_agent_fleet_sync()
        await pilot.pause()
        # The 10 s poll tick.
        screen._poll_console_environment()
        await pilot.pause()
        await _wait_for_landing(pilot, screen, root)

        assert builds == [], (
            "an Environment refresh built a full turn configuration "
            f"{len(builds)} time(s) ({builds}) just to read its workspace roots"
        )
        # Positive control: the panel is reporting on the admitted root, read
        # through the same accessor Change Review uses.
        assert screen._console_environment_root() == root
        assert screen._console_environment.snapshot.git.root == root


@pytest.mark.asyncio
async def test_the_narrow_root_read_equals_the_full_snapshots_roots(tmp_path):
    """Displayed content is unchanged: same roots as the full turn snapshot."""
    root = str(tmp_path)
    async with _console_with_admitted_root(root) as (pilot, screen):
        chat_controller = screen._ensure_console_chat_controller()
        active = chat_controller.store.active_session_id
        full = chat_controller.resolve_turn_execution_context(active).workspace_roots

        assert full == (root,)
        assert screen._review_selection._console_change_review_workspace_roots() == full
        assert screen._console_environment_root() == root
