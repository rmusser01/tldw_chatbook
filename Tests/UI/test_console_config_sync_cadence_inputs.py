"""Which inputs make the Console poll re-run its checked config projections.

TASK-33620.15.1 (owner decision 2026-10-09). The 0.2 s poll runs the
config-dependent projections (rail state, settings summary, control bar) only
when one of their inputs changed since the last projection, when an explicit
sync asked, or once a second while the poll ticks; a poll stopping on a
skipped tick renders at the stop. Each input here is changed alone
and must cause exactly one projection on the next poll tick, and none on the
tick after it. The mounted counterpart, over the real ChatScreen poll, is
``test_console_config_sync_cadence.py``.

The screen is a double that runs the two hooks ``ChatScreen._sync_native_
console_chat_ui`` runs, so these tests read the decision alone, fast.
"""

from __future__ import annotations

import asyncio
import gc
import weakref
from contextlib import contextmanager
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
        self.rounds: dict[str, dict[str, int]] = {}  # session -> kind -> count
        self.fleet = (0, 0)
        self.unattributed = 0
        self.ttl: tuple[float | None, bool] = (None, False)

    def run_states(self) -> dict[str, ConsoleRunState]:
        return dict(self.states)

    def pending_round_kinds(self, session_id: str) -> frozenset[str]:
        return frozenset(k for k, n in self.rounds.get(session_id, {}).items() if n)

    def pending_round_count(self, session_id: str, *, kind: str = "approval") -> int:
        return self.rounds.get(session_id, {}).get(kind, 0)

    def fleet_summary_counts(self) -> tuple[int, int]:
        return self.fleet

    def unattributed_fleet_tokens(self, session_id: str) -> int:
        return self.unattributed

    def cache_ttl_snapshot(self, session_id: str) -> tuple[float | None, bool]:
        return self.ttl


class _Store:
    def __init__(self) -> None:
        self.active_session_id = "s1"
        self.session_ids = ["s1", "s9"]
        self.display = self.payload = self.settings = 0

    def sessions(self) -> list:
        return [SimpleNamespace(id=session_id) for session_id in self.session_ids]

    def display_projection_revision(self, session_id: str) -> int:
        return self.display

    def payload_revision(self, session_id: str) -> int:
        return self.payload

    def session_settings_revision(self, session_id: str) -> int:
        return self.settings


class _Agent:
    """What the Agent rail shows: its lines, fleet rows and fleet tokens."""

    def __init__(self) -> None:
        self.lines = ("Agent: running · step 1", "· Thinking…", "")
        self.rows: tuple = ()
        self.tokens = 0

    def _console_agent_section_lines(self) -> tuple[str, str, str]:
        return self.lines

    def _console_agent_fleet_rows(self) -> tuple:
        return self.rows

    def _console_agent_fleet_token_total(self) -> int:
        return self.tokens


class _Screen:
    """Runs the sync's two cadence hooks around a counted projection."""

    def __init__(self) -> None:
        self._console_chat_store = _Store()
        self._console_chat_controller = _Controller()
        self._agent = _Agent()
        self.app_instance = SimpleNamespace(
            console_tool_count=0,
            console_mcp_tool_count=None,
            console_mcp_not_connected_count=0,
            console_provider_ready=None,
        )
        self._active_dictionaries_summary = None
        self._active_world_books_summary = None
        self.scopes = 0
        self._workspace = SimpleNamespace(tick_workspace_build_scope=self._scope)
        self.active = False
        self.projections = 0
        self.result: bool | Exception = True
        self.torn_down = False
        self._console_sync_in_progress = False
        self._console_sync_maintenance_paused = False
        self._console_control_bar_replay_whole_sync = False
        self._console_sync_requested = False

    def _console_run_active(self) -> bool:
        return self.active

    @contextmanager
    def _scope(self):
        self.scopes += 1
        try:
            yield
        finally:
            self.scopes -= 1

    def _sync_console_rail_and_controls(self) -> bool:
        self.projections += 1
        self.scoped = self.scopes > 0
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
    screen._console_chat_controller.rounds["s1"] = {"approval": 1}


# Review round (2026-10-10): what the projections show that changes on its
# own during a run, and every pending round on every chat.
def _question_round(monkeypatch, screen):
    screen._console_chat_controller.rounds["s1"] = {"question": 1}


def _background_chat_round(monkeypatch, screen):
    screen._console_chat_controller.rounds["s9"] = {"approval": 1}


def _fleet_counts(monkeypatch, screen):
    screen._console_chat_controller.fleet = (1, 1)


def _agent_step(monkeypatch, screen):
    screen._agent.lines = ("Agent: running · step 2", "· ⚙ web_search", "")


def _fleet_row_progress(monkeypatch, screen):
    screen._agent.rows = (("child-1", "running · 3s", "1,204 tokens"),)


def _fleet_tokens(monkeypatch, screen):
    screen._agent.tokens = 1_204


def _usage_landed(monkeypatch, screen):
    screen._console_chat_store.display += 1


def _payload_changed(monkeypatch, screen):
    screen._console_chat_store.payload += 1


def _session_settings(monkeypatch, screen):
    screen._console_chat_store.settings += 1


def _unattributed_tokens(monkeypatch, screen):
    screen._console_chat_controller.unattributed = 40


def _cache_warmed(monkeypatch, screen):
    import time

    screen._console_chat_controller.ttl = (time.monotonic() + 300, True)


def _tool_count(monkeypatch, screen):
    screen.app_instance.console_tool_count = 7


def _mcp_tool_count(monkeypatch, screen):
    screen.app_instance.console_mcp_tool_count = 3


def _provider_ready(monkeypatch, screen):
    screen.app_instance.console_provider_ready = False


def _connection_evidence(monkeypatch, screen):
    from tldw_chatbook.Chat import provider_test_evidence

    evidence = SimpleNamespace(version=5)
    monkeypatch.setattr(
        provider_test_evidence, "shared_connection_evidence", lambda app: evidence
    )


def _dictionaries_summary(monkeypatch, screen):
    screen._active_dictionaries_summary = object()


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
        _question_round,
        _background_chat_round,
        _fleet_counts,
        _agent_step,
        _fleet_row_progress,
        _fleet_tokens,
        _usage_landed,
        _payload_changed,
        _session_settings,
        _unattributed_tokens,
        _cache_warmed,
        _tool_count,
        _mcp_tool_count,
        _provider_ready,
        _connection_evidence,
        _dictionaries_summary,
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
        started.append(
            asyncio.ensure_future(_Screen._sync_native_console_chat_ui(screen))
        )
        config_sync.project_console_tick(screen)

    async def run() -> None:
        screen._sync_native_console_chat_ui = spawning_sync
        await config_sync.run_periodic_console_sync(screen._sync_native_console_chat_ui)
        await started[0]

    asyncio.run(run())
    assert screen.projections - projections_before == 1  # Only the spawned one.


def test_the_backstop_runs_once_a_second_on_every_poll_tick(clock):
    """Every tick of the poll gets the backstop, not only an active status's.

    The poll ticks only while run work is live, and that outlasts the run
    statuses: a custodied turn, a wake delivery, a change-review publication
    or a finishing run worker keep it ticking after the status went terminal.
    Gating the backstop on the status left that tail rendering nothing that
    is not an input (the cost chip, the Inspector) for as long as it lasted.
    """
    screen = _settled(clock)
    # A finished run whose poll still ticks: no status is active.
    screen._console_chat_controller.states["s1"] = ConsoleRunState(
        ConsoleRunStatus.COMPLETED
    )
    assert _tick(screen) == 1  # The change itself.
    clock.now += 0.75
    assert _tick(screen) == 0
    clock.now += 0.25
    assert _tick(screen) == 1, "no backstop while the poll ticks after a run"
    assert _tick(screen) == 0
    # A run going (viewed or background) keeps the same cadence.
    screen.active = True
    assert _tick(screen) == 1  # A run-activity change.
    clock.now += 1.0
    assert _tick(screen) == 1
    screen.active = False
    screen._console_chat_controller.states["s2"] = ConsoleRunState(
        ConsoleRunStatus.STREAMING
    )
    assert _tick(screen) == 1  # The change itself.
    clock.now += 1.0
    assert _tick(screen) == 1
    assert _tick(screen) == 0


def _stop(screen: _Screen) -> int:
    """The poll stops itself; returns how many projections that ran."""
    before = screen.projections
    config_sync.settle_stopped_poll(screen, lambda s: s.torn_down)
    return screen.projections - before


def test_a_stopping_poll_renders_what_its_last_ticks_skipped(clock):
    """The poll's last tick used to render; a skipped one is rendered at stop.

    After the poll stops nothing renders until some other sync, so whatever
    a skipped tick left unrendered (cost, Inspector, a late settings read)
    would stay stale for as long as the Console sits idle.
    """
    screen = _settled(clock)  # Ends on three skipped ticks.
    assert _stop(screen) == 1
    assert _stop(screen) == 0, "settled twice"
    assert _tick(screen) == 0
    # A tick that rendered leaves nothing owed.
    screen._console_chat_store.active_session_id = "s2"
    assert _tick(screen) == 1
    assert _stop(screen) == 0
    # As does an explicit sync after a skipped tick.
    assert _tick(screen) == 0
    assert _explicit(screen) == 1
    assert _stop(screen) == 0


def test_a_stopping_poll_leaves_an_owed_render_to_the_sync_that_owns_it(clock):
    """A running sync, a pending replay or a pause owns the render; no second one."""
    screen = _settled(clock)
    screen._console_sync_in_progress = True  # It renders when it gets there.
    assert _stop(screen) == 0
    screen._console_sync_in_progress = False
    screen._console_control_bar_replay_whole_sync = True  # The replay renders.
    assert _stop(screen) == 0
    screen._console_control_bar_replay_whole_sync = False
    screen._console_sync_maintenance_paused = True  # The resume replays it.
    assert _stop(screen) == 0
    assert screen._console_sync_requested, "the pause's resume was not asked"
    screen._console_sync_maintenance_paused = False
    screen.torn_down = True
    assert _stop(screen) == 0
    screen.torn_down = False
    assert _stop(screen) == 1  # Still owed: none of those rendered it.


@pytest.mark.parametrize("outcome", [False, RuntimeError("render failed")])
def test_a_deferred_or_failed_stop_render_stays_owed(clock, outcome):
    """A deferral hands the render to the whole-sync replay, as a tick's does."""
    screen = _settled(clock)
    screen.result = outcome
    if isinstance(outcome, Exception):
        with pytest.raises(RuntimeError):
            _stop(screen)
    else:
        assert _stop(screen) == 1
        assert screen._console_control_bar_replay_whole_sync
        screen._console_control_bar_replay_whole_sync = False
    screen.result = True
    assert _stop(screen) == 1
    assert _stop(screen) == 0


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


def test_a_live_elapsed_counter_projects_once_per_displayed_second(clock):
    """A running sub-agent's elapsed text is an input: each new second projects.

    The rows carry the text the rail shows; between two displayed seconds
    nothing projects, as the parent's re-render of identical text showed
    nothing new.
    """
    screen = _settled(clock)
    for second in (1, 2, 3):
        screen._agent.rows = (("child-1", f"running · {second}s", "10 tokens"),)
        assert _tick(screen) == 1, f"elapsed {second}s did not project"
        assert _tick(screen) == 0


def test_the_stop_render_runs_inside_the_tick_workspace_build_scope(clock):
    """Like the tick's own projection, so it shares one workspace build."""
    screen = _settled(clock)
    screen.scoped = None
    assert _stop(screen) == 1
    assert screen.scoped is True


def test_the_cadence_keeps_no_persistence_error_object_alive(clock):
    """A failure (and its traceback) is not held by the last projection's inputs.

    Assigned directly, not through ``monkeypatch``: its undo stack would hold
    the replaced value itself.
    """

    class _WriteFailed(Exception):
        pass

    screen = _settled(clock)
    original = config._CONFIG_PERSISTENCE_ERROR
    try:
        error = _WriteFailed("write failed")
        config._CONFIG_PERSISTENCE_ERROR = error
        assert _tick(screen) == 1
        witness = weakref.ref(error)
        config._CONFIG_PERSISTENCE_ERROR = None
        del error
        gc.collect()
        assert witness() is None, "the cadence kept the persistence error alive"
        assert _tick(screen) == 1  # The error clearing is itself a change.
    finally:
        config._CONFIG_PERSISTENCE_ERROR = original


def test_the_inputs_read_the_public_config_and_pause_accessors(clock, monkeypatch):
    """Renaming a private global must not reach a live tick (review MINOR 3)."""
    screen = _settled(clock)
    monkeypatch.setattr(config, "current_config_identity", lambda: (99, "/x.toml"))
    assert _tick(screen) == 1
    monkeypatch.setattr(storage_admission, "local_pause_active", lambda: True)
    assert _tick(screen) == 1
    assert _tick(screen) == 0
