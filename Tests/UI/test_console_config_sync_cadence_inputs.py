"""Which inputs make the Console poll re-run its checked config projections.

TASK-33620.15.1 (owner decision 2026-10-09). The 0.2 s poll runs the
config-dependent projections (rail state, settings summary, control bar) only
when one of their inputs changed since the last projection, when an explicit
sync asked, or once a second during a run. Each input here is changed alone
and must cause exactly one projection on the next poll tick, and none on the
tick after it. The mounted counterpart, over the real ChatScreen poll, is
``test_console_config_sync_cadence.py``.

The screen is a double that runs the two hooks ``ChatScreen._sync_native_
console_chat_ui`` runs, so these tests read the decision alone, fast.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from tldw_chatbook import config
from tldw_chatbook.Backup_Recovery import storage_admission
from tldw_chatbook.Chat.console_chat_models import ConsoleRunState, ConsoleRunStatus
from tldw_chatbook.UI.Console_Modules import config_sync

# Tests/UI's autouse fixtures read the config through participant admission.
pytestmark = pytest.mark.bootstrap_profile


class _Controller:
    def __init__(self) -> None:
        self.states: dict[str, ConsoleRunState] = {}
        self.pending = 0

    def run_states(self) -> dict[str, ConsoleRunState]:
        return dict(self.states)

    def pending_round_count(self, session_id: str) -> int:
        return self.pending


class _Screen:
    """Runs the sync's two cadence hooks around a counted projection."""

    def __init__(self) -> None:
        self._console_chat_store = SimpleNamespace(active_session_id="s1")
        self._console_chat_controller = _Controller()
        self.active = False
        self.projections = 0
        self.result: bool | Exception = True

    def _console_run_active(self) -> bool:
        return self.active

    def _sync_console_rail_and_controls(self) -> bool:
        self.projections += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    async def _sync_native_console_chat_ui(self) -> None:
        config_sync.begin_console_sync(self)
        await asyncio.sleep(0)
        config_sync.project_console_tick(self)


class _Clock:
    def __init__(self) -> None:
        self.now = 50.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch) -> _Clock:
    fake = _Clock()
    monkeypatch.setattr(config_sync, "_clock", fake)
    return fake


def _tick(screen: _Screen) -> int:
    """One poll tick; returns how many projections it ran."""
    before = screen.projections
    asyncio.run(config_sync.run_periodic_console_sync(screen._sync_native_console_chat_ui))
    return screen.projections - before


def _explicit(screen: _Screen) -> int:
    before = screen.projections
    asyncio.run(screen._sync_native_console_chat_ui())
    return screen.projections - before


def _settled(clock) -> _Screen:
    screen = _Screen()
    assert _explicit(screen) == 1
    assert [_tick(screen) for _ in range(3)] == [0, 0, 0]
    return screen


def _change_config(monkeypatch, screen):
    monkeypatch.setattr(config, "_CONFIG_CACHE", {"reinstalled": True})


def _change_generation(monkeypatch, screen):
    monkeypatch.setattr(config, "_CONFIG_GENERATION", config._CONFIG_GENERATION + 1)


def _change_settings(monkeypatch, screen):
    monkeypatch.setattr(config, "_SETTINGS_CACHE", {"reinstalled": True})


def _change_persistence_error(monkeypatch, screen):
    monkeypatch.setattr(config, "_CONFIG_PERSISTENCE_ERROR", OSError("write failed"))


def _change_selector(monkeypatch, screen):
    monkeypatch.setenv("TLDW_CONFIG_PATH", "/nonexistent/other-profile/config.toml")


def _local_pause(monkeypatch, screen):
    monkeypatch.setattr(storage_admission, "_pause", object())


def _switch_chat(monkeypatch, screen):
    screen._console_chat_store.active_session_id = "s2"


def _run_state(monkeypatch, screen):
    screen._console_chat_controller.states["s1"] = ConsoleRunState(
        ConsoleRunStatus.COMPLETED
    )


def _background_run_state(monkeypatch, screen):
    screen._console_chat_controller.states["s9"] = ConsoleRunState(
        ConsoleRunStatus.FAILED, "background chat failed"
    )


def _run_activity(monkeypatch, screen):
    screen.active = True


def _approval(monkeypatch, screen):
    screen._console_chat_controller.pending = 1


@pytest.mark.parametrize(
    "change",
    [
        _change_config,
        _change_generation,
        _change_settings,
        _change_persistence_error,
        _change_selector,
        _local_pause,
        _switch_chat,
        _run_state,
        _background_run_state,
        _run_activity,
        _approval,
    ],
)
def test_each_input_change_is_one_projection_on_the_next_tick(
    monkeypatch, clock, change
):
    screen = _settled(clock)
    change(monkeypatch, screen)
    assert _tick(screen) == 1, f"{change.__name__}: the next tick did not project"
    assert _tick(screen) == 0, f"{change.__name__}: the tick after it projected"


def test_an_explicit_sync_always_projects_and_settles_the_next_tick(clock):
    screen = _settled(clock)
    assert _explicit(screen) == 1
    assert _explicit(screen) == 1
    assert _tick(screen) == 0


def test_a_sync_refused_while_another_runs_is_owed_to_the_next_tick(clock):
    """An explicit request that could not run is served by the next tick."""
    screen = _settled(clock)
    config_sync.begin_console_sync(screen)  # Its sync returned early (busy).
    assert _tick(screen) == 1
    assert _tick(screen) == 0


def test_the_poll_mark_does_not_reach_work_its_sync_starts(clock):
    """A sync the poll's sync starts (a worker, a replay) is an explicit one."""
    screen = _settled(clock)
    started: list[asyncio.Task] = []
    projections_before = screen.projections

    async def spawning_sync() -> None:
        config_sync.begin_console_sync(screen)
        started.append(asyncio.ensure_future(_Screen._sync_native_console_chat_ui(screen)))
        config_sync.project_console_tick(screen)

    async def run() -> None:
        screen._sync_native_console_chat_ui = spawning_sync
        await config_sync.run_periodic_console_sync(screen._sync_native_console_chat_ui)
        await started[0]

    asyncio.run(run())
    assert screen.projections - projections_before == 1  # Only the spawned one.


def test_the_backstop_runs_once_a_second_during_a_run_only(clock):
    screen = _settled(clock)
    screen.active = True
    assert _tick(screen) == 1  # A run started: a run-activity change.
    clock.now += 0.75
    assert _tick(screen) == 0
    clock.now += 0.25
    assert _tick(screen) == 1
    assert _tick(screen) == 0
    clock.now += 1.0
    assert _tick(screen) == 1
    # A background chat's run keeps the backstop too.
    screen.active = False
    screen._console_chat_controller.states["s2"] = ConsoleRunState(
        ConsoleRunStatus.STREAMING
    )
    assert _tick(screen) == 1  # The change itself.
    clock.now += 1.0
    assert _tick(screen) == 1
    # Outside any run there is no backstop.
    screen._console_chat_controller.states["s2"] = ConsoleRunState(
        ConsoleRunStatus.COMPLETED
    )
    assert _tick(screen) == 1
    clock.now += 60.0
    assert _tick(screen) == 0


@pytest.mark.parametrize("outcome", [False, RuntimeError("render failed")])
def test_a_deferred_or_failed_projection_is_retried_on_the_next_tick(clock, outcome):
    """Nothing is recorded unless the projection rendered."""
    screen = _settled(clock)
    screen.result = outcome
    screen._console_chat_store.active_session_id = "s2"
    if isinstance(outcome, Exception):
        with pytest.raises(RuntimeError):
            _tick(screen)
        with pytest.raises(RuntimeError):
            _tick(screen)
    else:
        assert _tick(screen) == 1
        assert _tick(screen) == 1
    screen.result = True
    assert _tick(screen) == 1
    assert _tick(screen) == 0


def test_a_screen_without_a_chat_controller_still_projects(clock):
    """Bare screens (the Backup_Recovery doubles) keep today's behaviour."""
    screen = _Screen()
    screen._console_chat_store = None
    screen._console_chat_controller = None
    assert _explicit(screen) == 1
    assert _tick(screen) == 0
