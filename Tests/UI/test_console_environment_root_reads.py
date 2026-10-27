"""TASK-33620.15.1.6: Environment refreshes read Change Review consent off the UI loop.

Measured live on 2026-10-22 (isolated profiles, Anthropic claude-haiku-4-5,
235x52, host load 30-43): every Environment refresh over 50 ms was two
Change Review consent reads -- SELECTs on the workspace registry database --
and nothing else, e.g. 64.5 + 47.8 ms of a 113.1 ms Inspect-rail open.
``request_refresh`` read the root, ``_land``'s stale-scope guard read it
again, and ``poll_tick`` read it once more before delegating.

With one read left on the loop a rail open still held it 104.5 ms and a
poll tick 146 ms at load 40-47, so the automatic refreshes now read the root
on a worker. These drive the REAL mounted Console (screen, controller, store,
app) and count the consent reads the Environment controller makes, by thread
and by the controller entry point on the calling stack: none on the UI loop
for the rail open, run completion and poll tick, one for an explicit Refresh
press, none in a landing, and the guard's re-check on the gather worker. The
stale-scope guard still drops a landing whose root changed while it was
being gathered.
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


def _environment_entry() -> str | None:
    """The outermost Environment-controller function on the calling stack."""
    frame = sys._getframe(2)
    entry = None
    try:
        while frame is not None:
            code = frame.f_code
            if code.co_filename.replace("\\", "/").endswith(_ENVIRONMENT_MODULE):
                entry = code.co_name
            frame = frame.f_back
        return entry
    finally:
        del frame


class _CountingConsent:
    """A Change Review consent service that records who asked, and where."""

    def __init__(self, *roots: str) -> None:
        self.roots = tuple(roots)
        self.reads: list[tuple[str, str | None]] = []
        self._main = threading.main_thread().ident

    def admit_turn(self, _workspace_id):
        thread = "ui" if threading.get_ident() == self._main else "worker"
        self.reads.append((thread, _environment_entry()))
        return SimpleNamespace(
            ready_roots=self.roots,
            ready_aliases=tuple(f"alias-{index}" for index, _ in enumerate(self.roots)),
            skipped_roots=(),
        )

    def environment_reads(self, thread: str, entry: str | None = None) -> int:
        return sum(
            1
            for read_thread, read_entry in self.reads
            if read_thread == thread
            and read_entry is not None
            and (entry is None or read_entry == entry)
        )


@asynccontextmanager
async def _console(consent: _CountingConsent):
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.change_review_consent_service = consent
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 45)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await pilot.pause(0.2)
        chat_controller = console._ensure_console_chat_controller()
        assert chat_controller.store.active_session_id, (
            "harness precondition: an active session, so the root accessor "
            "reaches the consent service rather than UNKNOWN_ROOT"
        )
        yield pilot, console


def _canned_gatherers(
    monkeypatch,
    *,
    gate: threading.Event | None = None,
    started: threading.Event | None = None,
) -> None:
    """Keep git/``gh``/backlog off the test; optionally hold the git gather."""

    def gather_git(path, *, previous=None):
        if started is not None:
            started.set()
        if gate is not None:
            gate.wait(5)
        return GitEnvState(
            availability=EnvSourceAvailability.OK, root=str(path), branch="main"
        )

    monkeypatch.setattr(env_mod, "gather_git_env", gather_git)
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


def _record_landings(controller) -> list:
    """Record the scope of every landing the controller processes."""
    landings: list = []
    land = controller._land

    def recording_land(scope_root, *args):
        landings.append(scope_root)
        return land(scope_root, *args)

    controller._land = recording_land
    return landings


async def _until(pilot, predicate, what: str) -> None:
    for _ in range(150):
        if predicate():
            return
        await pilot.pause(0.02)
    raise AssertionError(f"timed out waiting for {what}")


async def _open_rail(pilot, screen) -> None:
    await screen.action_toggle_console_inspector_rail()
    await pilot.pause()
    assert screen._console_environment._rail_open_accessor(), (
        "harness precondition: the Inspect rail is open"
    )


@pytest.mark.asyncio
async def test_an_unbound_panel_s_refreshes_read_consent_off_the_ui_loop(monkeypatch):
    """AC#1, the measured path: no folder bound.

    Base (origin/dev 3be493413a): the rail open and the run-completion nudge
    read consent twice on the UI loop, and the poll tick three times.
    """
    consent = _CountingConsent()  # admits nothing: the workspace is unbound
    _canned_gatherers(monkeypatch)
    async with _console(consent) as (pilot, screen):
        controller = screen._console_environment
        await _open_rail(pilot, screen)
        await _until(
            pilot,
            lambda: (
                controller.snapshot.git.availability is EnvSourceAvailability.UNBOUND
            ),
            "the UNBOUND landing",
        )
        per_refresh = {"rail open": consent.environment_reads("ui")}

        for name, refresh in (
            ("run completion", screen._run_coalesced_console_agent_fleet_sync),
            ("poll tick", screen._poll_console_environment),
            (
                "explicit refresh",
                lambda: controller.request_refresh(include_net=True, force_net=True),
            ),
        ):
            before_ui = consent.environment_reads("ui")
            before_worker = consent.environment_reads("worker")
            refresh()
            per_refresh[name] = consent.environment_reads("ui") - before_ui
            if name != "explicit refresh":
                await _until(
                    pilot,
                    lambda: consent.environment_reads("worker") > before_worker,
                    f"the {name}'s off-loop root read",
                )
            await pilot.pause()

        assert per_refresh == {
            "rail open": 0,
            "run completion": 0,
            "poll tick": 0,
            "explicit refresh": 1,
        }
        assert consent.environment_reads("worker") >= 3
        assert controller.snapshot.git.availability is EnvSourceAvailability.UNBOUND


@pytest.mark.asyncio
async def test_a_bound_panel_s_refreshes_and_landings_read_no_consent_on_the_ui_loop(
    monkeypatch, tmp_path
):
    """AC#1, bound: no UI-loop read to dispatch or land; all on the workers.

    Base: the dispatch and the local and net landings each read consent on
    the UI loop, and no worker read it.
    """
    root = str(tmp_path)
    consent = _CountingConsent(root, f"{root}/second")
    _canned_gatherers(monkeypatch)
    async with _console(consent) as (pilot, screen):
        controller = screen._console_environment
        landings = _record_landings(controller)
        await _open_rail(pilot, screen)
        await _until(
            pilot,
            lambda: (
                landings.count(root) >= 2
                and controller.snapshot.pr.availability
                is EnvSourceAvailability.NOT_APPLICABLE
            ),
            "the local and net landings for the admitted root",
        )
        landed = len(landings)
        screen._poll_console_environment()
        await _until(pilot, lambda: len(landings) > landed, "the poll's landing")

        assert consent.environment_reads("ui") == 0, (
            f"the Environment panel read consent on the UI loop: {consent.reads}"
        )
        # Rail open: root, local and net re-checks; poll: root and local.
        assert consent.environment_reads("worker") >= 5
        assert controller._landed_root == root
        assert screen._console_environment.snapshot.git.root == root


@pytest.mark.asyncio
async def test_a_landing_whose_root_changed_while_gathering_never_lands(
    monkeypatch, tmp_path
):
    """AC#2: the stale-scope guard still drops it, then the new root lands."""
    old_root = str(tmp_path / "old")
    new_root = str(tmp_path / "new")
    consent = _CountingConsent(old_root)
    gate = threading.Event()
    started = threading.Event()
    _canned_gatherers(monkeypatch, gate=gate, started=started)
    async with _console(consent) as (pilot, screen):
        controller = screen._console_environment
        painted: list[str] = []
        land_snapshot = controller._on_snapshot

        def record(snapshot):
            painted.append(snapshot.git.root)
            land_snapshot(snapshot)

        controller._on_snapshot = record
        landings = _record_landings(controller)

        await _open_rail(pilot, screen)
        await _until(pilot, started.is_set, "the old root's gather to start")
        # The old root's gather is held in its worker; consent now admits
        # another folder (a workspace switch, or a rebind).
        consent.roots = (new_root,)
        gate.set()
        await _until(pilot, lambda: old_root in landings, "the old root's landing")
        await pilot.pause()
        assert old_root not in painted
        assert controller._landed_root != old_root

        screen._poll_console_environment()
        await _until(
            pilot, lambda: controller._landed_root == new_root, "the new root's landing"
        )
        assert old_root not in painted
        assert painted[-1] == new_root
