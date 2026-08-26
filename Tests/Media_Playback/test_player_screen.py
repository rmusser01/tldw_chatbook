"""VideoPlayerScreen bindings, hints, and worker lifecycle."""

from __future__ import annotations

import asyncio
from threading import Event
from types import SimpleNamespace
from typing import Any

import pytest
from loguru import logger
from textual.app import App, ComposeResult
from textual.widgets import Button, Static
from textual.worker import Worker, WorkerState

from tldw_chatbook.Media_Playback.player_pipeline import PlayerProbe, PlayerRun
from tldw_chatbook.UI.Screens.video_player_screen import (
    _HINTS,
    VideoPlayerScreen,
)


PRIVATE_PATH = "/private/PRIVATE-MODAL-PATH-SENTINEL.mp4"
PRIVATE_ERROR = "PRIVATE-MODAL-ERROR-SENTINEL"
PROBE = PlayerProbe(width=1, height=1, duration_seconds=10.0, has_audio=False)


class _ObservedPlayer(VideoPlayerScreen):
    """Modal that records its non-bubbling worker state messages."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.worker_states: list[WorkerState] = []
        self.seen_workers: list[Worker[Any]] = []
        super().__init__(*args, **kwargs)

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        self.worker_states.append(event.state)
        if event.worker not in self.seen_workers:
            self.seen_workers.append(event.worker)


class _PlayerApp(App[None]):
    """Mounted player host with an independent app action and notifications."""

    BINDINGS = [("x", "ping", "Ping")]

    def __init__(self, player: _ObservedPlayer | None = None) -> None:
        super().__init__()
        self.player = player or _ObservedPlayer(PRIVATE_PATH, render_mode="ascii")
        self.action_count = 0
        self.action_seen = Event()
        self.notifications: list[str] = []

    def compose(self) -> ComposeResult:
        yield Button("Host action", id="host-action")

    def on_mount(self) -> None:
        self.push_screen(self.player)

    def action_ping(self) -> None:
        self.action_count += 1
        self.action_seen.set()

    def notify(self, message: Any, *args: Any, **kwargs: Any) -> None:
        self.notifications.append(str(message))
        super().notify(message, *args, **kwargs)


class _Pipeline:
    """Small deterministic PlayerPipeline seam used by mounted tests."""

    instances: list["_Pipeline"] = []

    def __init__(self, source: str, probe: PlayerProbe) -> None:
        self.source = source
        self.probe = probe
        self.runs: list[PlayerRun] = []
        self.stop_calls = 0
        self.stopped = Event()
        type(self).instances.append(self)

    def start(self) -> PlayerRun:
        run = PlayerRun(1, None, 0.0)
        self.runs.append(run)
        return run

    def iter_frames(self, run: PlayerRun):
        run.eof = True
        return iter(())

    def frames_behind(self, run: PlayerRun, pts: float) -> bool:
        return False

    def frame_due(self, run: PlayerRun, pts: float) -> bool:
        return True

    def sync_clock(self, run: PlayerRun) -> float:
        return pts if (pts := run.stats.position_seconds) else 0.0

    def note_rendered(self, run: PlayerRun, pts: float) -> None:
        run.stats.position_seconds = pts

    def note_dropped(self, run: PlayerRun, pts: float) -> None:
        run.stats.dropped_frames += 1

    def pause(self) -> None:
        pass

    def resume(self) -> None:
        pass

    def seek(self, target: float) -> PlayerRun:
        run = PlayerRun(len(self.runs) + 1, None, target)
        self.runs.append(run)
        return run

    def stop(self) -> None:
        self.stop_calls += 1
        self.stopped.set()


@pytest.fixture(autouse=True)
def _patch_modal_seams(monkeypatch: pytest.MonkeyPatch):
    _Pipeline.instances = []
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.video_player_screen.playback_tools_available",
        lambda: (True, ""),
    )
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.video_player_screen.probe_file",
        lambda path: PROBE,
    )
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.video_player_screen.PlayerPipeline",
        _Pipeline,
    )


async def _wait(event: Event, *, timeout: float = 2.0) -> None:
    assert await asyncio.to_thread(event.wait, timeout), (
        "synchronization event timed out"
    )


async def _finish_workers(app: _PlayerApp, pilot: Any) -> None:
    await app.workers.wait_for_complete()
    await pilot.pause()
    assert app.player.seen_workers
    assert WorkerState.ERROR not in app.player.worker_states
    assert all(
        worker.state is not WorkerState.ERROR for worker in app.player.seen_workers
    )


def _assert_sanitized(records: list[str], phase: str) -> None:
    matching = [record for record in records if "component=modal_player" in record]
    assert matching
    assert any(f"phase={phase}" in record for record in matching)
    assert any("error_type=RuntimeError" in record for record in matching)
    joined = "\n".join(records)
    assert PRIVATE_ERROR not in joined
    assert PRIVATE_PATH not in joined
    assert "Traceback" not in joined


def test_bindings_follow_keybinding_conventions():
    forbidden_chords = {
        "ctrl+c",
        "ctrl+v",
        "ctrl+x",
        "ctrl+s",
        "ctrl+d",
        "ctrl+z",
        "ctrl+a",
        "ctrl+r",
        "ctrl+w",
    }
    keys = {binding.key for binding in VideoPlayerScreen.BINDINGS}
    assert keys == {"space", "s", "left", "right", "q", "escape"}
    assert not (keys & forbidden_chords)
    # decision 031: single-letter htop-style for letters.
    assert all(
        len(key) == 1 or key in {"space", "left", "right", "escape"} for key in keys
    )


def test_hints_line_names_only_implemented_actions():
    # AC2: the footer hints may only advertise implemented actions -- every
    # hinted key has a matching action method on the screen.
    hinted = {
        "space": "action_toggle_pause",
        "s": "action_stop_playback",
        "q": "action_close_player",
    }
    for key, method in hinted.items():
        assert key in _HINTS
        assert callable(getattr(VideoPlayerScreen, method))
    assert "←/→" in _HINTS
    assert callable(getattr(VideoPlayerScreen, "action_seek_back"))
    assert callable(getattr(VideoPlayerScreen, "action_seek_fwd"))


def test_compose_branches_per_mode():
    # halfcell / ascii / kitty: plain Static frame area.
    for mode in ("halfcell", "ascii", "kitty"):
        screen = VideoPlayerScreen("/tmp/x.mp4", render_mode=mode)
        children = list(screen.compose())
        assert isinstance(children[0], Static)
        assert len(children) == 3  # frame + status + hints
    # sixel: tries the sixel widget, degrades to Static when unavailable.
    screen = VideoPlayerScreen("/tmp/x.mp4", render_mode="sixel")
    children = list(screen.compose())
    assert len(children) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(("delta", "expected"), [(-5.0, 0.0), (5.0, 8.0)])
async def test_seek_relative_clamps_and_delegates(monkeypatch, delta, expected):
    seeked = Event()
    pump_entered = Event()
    release = Event()

    class RecordingPipeline(_Pipeline):
        def start(self) -> PlayerRun:
            run = super().start()
            run.stats.position_seconds = 3.0
            return run

        def seek(self, target: float) -> PlayerRun:
            self.target = target
            run = super().seek(target)
            seeked.set()
            return run

        def iter_frames(self, run: PlayerRun):
            pump_entered.set()
            assert release.wait(2.0)
            run.eof = True
            if False:
                yield 0.0, b""

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.video_player_screen.PlayerPipeline",
        RecordingPipeline,
    )
    app = _PlayerApp()
    async with app.run_test() as pilot:
        await _wait(pump_entered)
        app.player._seek_relative(delta)
        await _wait(seeked)
        release.set()
        await _finish_workers(app, pilot)
        assert RecordingPipeline.instances[0].target == expected


# -- real Textual worker / bridge regressions -------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_phase", ["probe", "start"])
@pytest.mark.parametrize("close_key", ["q", "escape"])
async def test_blocked_activation_keeps_ui_responsive_and_reaps_stale_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    blocked_phase: str,
    close_key: str,
):
    entered = Event()
    release = Event()
    cleanup_done = Event()

    class BlockedPipeline(_Pipeline):
        def start(self) -> PlayerRun:
            if blocked_phase == "start":
                entered.set()
                assert release.wait(2.0)
            return super().start()

        def stop(self) -> None:
            super().stop()
            cleanup_done.set()

    def blocked_probe(path: str) -> PlayerProbe:
        if blocked_phase == "probe":
            entered.set()
            assert release.wait(2.0)
        return PROBE

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.video_player_screen.probe_file", blocked_probe
    )
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.video_player_screen.PlayerPipeline", BlockedPipeline
    )
    app = _PlayerApp()
    async with app.run_test() as pilot:
        await _wait(entered)
        await pilot.press(close_key)
        await pilot.press("x")
        assert app.action_count == 1
        release.set()
        await _wait(cleanup_done)
        await _finish_workers(app, pilot)

    assert BlockedPipeline.instances
    pipeline = BlockedPipeline.instances[0]
    assert pipeline.stop_calls == 1
    assert pipeline.stopped.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "selector", ["#video-player-frame", "#video-player-status", "#video-player-hints"]
)
async def test_player_cells_are_whole_screen_content_not_backdrop(selector: str):
    app = _PlayerApp()
    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.pause()
        await pilot.click(selector)
        await pilot.pause()

        assert app.screen is app.player
        assert app.player.is_mounted

        await pilot.press("q")
        await _finish_workers(app, pilot)


@pytest.mark.asyncio
async def test_real_pump_renders_frame_and_eof_without_worker_error(monkeypatch):
    rendered = Event()

    class FramePipeline(_Pipeline):
        def iter_frames(self, run: PlayerRun):
            yield 0.0, bytes((255, 255, 255))
            run.eof = True
            rendered.set()

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.video_player_screen.PlayerPipeline", FramePipeline
    )
    app = _PlayerApp()
    async with app.run_test() as pilot:
        await _wait(rendered)
        await _finish_workers(app, pilot)
        frame = app.player.query_one("#video-player-frame", Static)
        status = app.player.query_one("#video-player-status", Static)
        assert str(frame.renderable).strip()
        assert "finished" in str(status.renderable)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_phase", ["timer", "pump"])
async def test_activation_publication_failure_rolls_back_before_worker_cleanup(
    monkeypatch,
    failure_phase: str,
):
    failure_seen = Event()
    timers: list[Any] = []

    class PublicationFailurePlayer(_ObservedPlayer):
        def set_interval(self, *args: Any, **kwargs: Any) -> Any:
            is_status_timer = len(args) > 1 and args[1] == self._refresh_status
            if failure_phase == "timer" and is_status_timer:
                failure_seen.set()
                raise RuntimeError(PRIVATE_ERROR)
            timer = super().set_interval(*args, **kwargs)
            if is_status_timer:
                timers.append(timer)
            return timer

        def _start_pump(self, *args: Any) -> None:
            if failure_phase == "pump":
                failure_seen.set()
                raise RuntimeError(PRIVATE_ERROR)
            super()._start_pump(*args)

    player = PublicationFailurePlayer(PRIVATE_PATH, render_mode="ascii")
    app = _PlayerApp(player)
    records: list[str] = []
    sink = logger.add(lambda message: records.append(str(message)))
    try:
        async with app.run_test() as pilot:
            await _wait(failure_seen)
            await _wait(_Pipeline.instances[0].stopped)
            await _finish_workers(app, pilot)

            assert player._pipeline is None
            assert player._run is None
            assert player._status_timer is None
            assert _Pipeline.instances[0].stop_calls == 1
            assert app.notifications
            assert "system player" in app.notifications[0].lower()
            if timers:
                assert timers[0]._task is None
    finally:
        logger.remove(sink)
    _assert_sanitized(records, "activation")


@pytest.mark.asyncio
async def test_seek_is_nonblocking_single_flight_and_starts_replacement_pump(
    monkeypatch,
):
    first_pump = Event()
    seek_entered = Event()
    release_seek = Event()
    replacement_pump = Event()

    class SeekPipeline(_Pipeline):
        def iter_frames(self, run: PlayerRun):
            if run.generation == 1:
                first_pump.set()
                while not run.eof:
                    if seek_entered.wait(0.05):
                        return
            replacement_pump.set()
            run.eof = True
            if False:
                yield 0.0, b""

        def seek(self, target: float) -> PlayerRun:
            seek_entered.set()
            assert release_seek.wait(2.0)
            self.runs[0].eof = True
            return super().seek(target)

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.video_player_screen.PlayerPipeline", SeekPipeline
    )
    app = _PlayerApp()
    async with app.run_test() as pilot:
        await _wait(first_pump)
        await pilot.press("right")
        await _wait(seek_entered)
        await pilot.press("right")
        app.call_later(app.action_ping)
        await _wait(app.action_seen)
        assert app.action_count == 1
        assert app.player._seek_in_flight
        release_seek.set()
        await _wait(replacement_pump)
        await _finish_workers(app, pilot)

        pipeline = SeekPipeline.instances[0]
        assert len(pipeline.runs) == 2
        assert app.player._run is pipeline.runs[1]
        assert not app.player._seek_in_flight


@pytest.mark.asyncio
async def test_replacement_pump_starts_before_seek_flag_is_cleared(monkeypatch):
    first_pump = Event()
    replacement_pump = Event()
    release_first = Event()

    class OrderingPlayer(_ObservedPlayer):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.launch_flags: list[tuple[int, bool]] = []
            super().__init__(*args, **kwargs)

        def _start_pump(self, token: int, pipeline: _Pipeline, run: PlayerRun) -> None:
            self.launch_flags.append((run.generation, self._seek_in_flight))
            super()._start_pump(token, pipeline, run)

    class OrderingPipeline(_Pipeline):
        def iter_frames(self, run: PlayerRun):
            if run.generation == 1:
                first_pump.set()
                assert release_first.wait(2.0)
            else:
                replacement_pump.set()
            run.eof = True
            if False:
                yield 0.0, b""

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.video_player_screen.PlayerPipeline",
        OrderingPipeline,
    )
    player = OrderingPlayer(PRIVATE_PATH, render_mode="ascii")
    app = _PlayerApp(player)
    async with app.run_test() as pilot:
        await _wait(first_pump)
        player.action_seek_fwd()
        await _wait(replacement_pump)
        await pilot.pause()

        assert player.launch_flags == [(1, False), (2, True)]
        assert not player._seek_in_flight

        release_first.set()
        await _finish_workers(app, pilot)


@pytest.mark.asyncio
async def test_old_frame_eof_and_failure_after_seek_do_not_replace_new_frame(
    monkeypatch,
):
    old_entered = Event()
    release_old = Event()
    new_rendered = Event()

    class RacingPipeline(_Pipeline):
        def iter_frames(self, run: PlayerRun):
            if run.generation == 1:
                old_entered.set()
                assert release_old.wait(2.0)
                raise RuntimeError(PRIVATE_ERROR)
            yield 2.0, bytes((0, 0, 255))
            run.eof = True
            new_rendered.set()

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.video_player_screen.PlayerPipeline", RacingPipeline
    )
    records: list[str] = []
    sink = logger.add(lambda message: records.append(str(message)))
    try:
        app = _PlayerApp()
        async with app.run_test() as pilot:
            await _wait(old_entered)
            app.player.action_seek_fwd()
            await _wait(new_rendered)
            await pilot.pause()
            replacement = app.player._run
            replacement_frame = str(
                app.player.query_one("#video-player-frame", Static).renderable
            )
            old_run = RacingPipeline.instances[0].runs[0]
            pipeline = RacingPipeline.instances[0]
            token = app.player._activation_token - 1
            assert not app.player._render_frame(
                token, pipeline, old_run, bytes((255, 0, 0))
            )
            assert not app.player._finish_run(token, pipeline, old_run)
            assert not app.player._fail_run(token, pipeline, old_run)
            release_old.set()
            await _finish_workers(app, pilot)

            assert app.player._run is replacement
            assert (
                str(app.player.query_one("#video-player-frame", Static).renderable)
                == replacement_frame
            )
            assert not app.notifications
    finally:
        logger.remove(sink)
    _assert_sanitized(records, "pump")


async def _assert_wrong_identity_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
    *,
    wrong_pipeline: bool,
) -> None:
    pump_entered = Event()
    release = Event()

    class HeldPipeline(_Pipeline):
        def iter_frames(self, run: PlayerRun):
            pump_entered.set()
            assert release.wait(2.0)
            run.eof = True
            if False:
                yield 0.0, b""

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.video_player_screen.PlayerPipeline", HeldPipeline
    )
    app = _PlayerApp()
    async with app.run_test() as pilot:
        await _wait(pump_entered)
        player = app.player
        pipeline = player._pipeline
        run = player._run
        assert pipeline is not None and run is not None
        token = player._activation_token
        candidate_pipeline = (
            HeldPipeline(PRIVATE_PATH, PROBE) if wrong_pipeline else pipeline
        )
        candidate_run = run if wrong_pipeline else PlayerRun(run.generation, None, 0.0)

        assert not player._render_frame(
            token,
            candidate_pipeline,
            candidate_run,
            bytes((255, 0, 0)),
        )
        assert not player._finish_run(token, candidate_pipeline, candidate_run)
        assert not player._fail_run(token, candidate_pipeline, candidate_run)
        assert player._pipeline is pipeline
        assert player._run is run
        assert player._status_timer is not None
        assert player.query_one("#video-player-frame", Static).renderable == ""
        assert not player._finished
        assert not app.notifications
        assert pipeline.stop_calls == 0
        assert player.is_attached

        release.set()
        await _finish_workers(app, pilot)


@pytest.mark.asyncio
async def test_same_token_and_run_with_wrong_pipeline_is_ignored(monkeypatch):
    await _assert_wrong_identity_is_ignored(monkeypatch, wrong_pipeline=True)


@pytest.mark.asyncio
async def test_same_token_and_pipeline_with_wrong_run_is_ignored(monkeypatch):
    await _assert_wrong_identity_is_ignored(monkeypatch, wrong_pipeline=False)


@pytest.mark.asyncio
async def test_unmount_ignores_late_frame_eof_and_cleanup_is_app_owned(
    monkeypatch,
):
    pump_entered = Event()
    release_pump = Event()
    stop_entered = Event()
    release_stop = Event()

    class BlockedStopPipeline(_Pipeline):
        def iter_frames(self, run: PlayerRun):
            pump_entered.set()
            assert release_pump.wait(2.0)
            yield 1.0, bytes((0, 255, 0))
            run.eof = True

        def stop(self) -> None:
            self.stop_calls += 1
            stop_entered.set()
            assert release_stop.wait(2.0)
            self.stopped.set()

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.video_player_screen.PlayerPipeline",
        BlockedStopPipeline,
    )
    app = _PlayerApp()
    async with app.run_test() as pilot:
        await _wait(pump_entered)
        await pilot.press("q")
        await _wait(stop_entered)
        await pilot.press("x")
        assert app.action_count == 1
        release_pump.set()
        release_stop.set()
        await _finish_workers(app, pilot)

    assert app.player._run is None
    assert app.player._pipeline is None
    assert not app.player._finished
    assert BlockedStopPipeline.instances[0].stopped.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("callback_phase", "expected_callback"),
    [
        ("frame", "_render_frame"),
        ("eof", "_finish_run"),
        ("failure", "_fail_run"),
    ],
)
async def test_current_pump_bridge_refusal_returns_without_cleanup_or_ui_fallback(
    monkeypatch,
    callback_phase: str,
    expected_callback: str,
):
    pump_entered = Event()
    pump_done = Event()
    release = Event()
    direct_ui = Event()

    class BridgePlayer(_ObservedPlayer):
        def _render_frame(self, *args: Any) -> bool:
            direct_ui.set()
            return super()._render_frame(*args)

        def _finish_run(self, *args: Any) -> bool:
            direct_ui.set()
            return super()._finish_run(*args)

        def _fail_run(self, *args: Any) -> bool:
            direct_ui.set()
            return super()._fail_run(*args)

    class RefusedPipeline(_Pipeline):
        def iter_frames(self, run: PlayerRun):
            try:
                pump_entered.set()
                assert release.wait(2.0)
                if callback_phase == "frame":
                    yield 0.0, bytes((1, 2, 3))
                    return
                if callback_phase == "eof":
                    run.eof = True
                    return
                raise RuntimeError(PRIVATE_ERROR)
            finally:
                pump_done.set()

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.video_player_screen.PlayerPipeline",
        RefusedPipeline,
    )
    player = BridgePlayer(PRIVATE_PATH, render_mode="ascii")
    app = _PlayerApp(player)
    bridge_calls: list[str] = []
    records: list[str] = []
    sink = logger.add(lambda message: records.append(str(message)))
    try:
        async with app.run_test() as pilot:
            await _wait(pump_entered)

            def refuse(callback: Any, *args: Any, **kwargs: Any) -> Any:
                bridge_calls.append(callback.__name__)
                raise RuntimeError(PRIVATE_ERROR)

            monkeypatch.setattr(app, "call_from_thread", refuse)
            release.set()
            await _wait(pump_done)
            await pilot.pause()

            assert bridge_calls == [expected_callback]
            assert not direct_ui.is_set()
            assert RefusedPipeline.instances[0].stop_calls == 0
            assert player._pipeline is RefusedPipeline.instances[0]
            assert player._run is RefusedPipeline.instances[0].runs[0]
    finally:
        logger.remove(sink)
    _assert_sanitized(records, "frame_dispatch")


@pytest.mark.asyncio
async def test_stale_old_run_bridge_refusal_after_seek_does_not_stop_replacement(
    monkeypatch,
):
    old_entered = Event()
    release_old = Event()
    replacement_entered = Event()
    release_replacement = Event()
    bridge_attempted = Event()

    class SeekRacePipeline(_Pipeline):
        def iter_frames(self, run: PlayerRun):
            if run.generation == 1:
                old_entered.set()
                assert release_old.wait(2.0)
                yield 1.0, bytes((255, 0, 0))
                return
            replacement_entered.set()
            assert release_replacement.wait(2.0)
            run.eof = True
            if False:
                yield 0.0, b""

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.video_player_screen.PlayerPipeline",
        SeekRacePipeline,
    )
    app = _PlayerApp()
    bridge_calls: list[str] = []
    async with app.run_test() as pilot:
        await _wait(old_entered)
        app.player.action_seek_fwd()
        await _wait(replacement_entered)
        await pilot.pause()

        pipeline = SeekRacePipeline.instances[0]
        replacement = pipeline.runs[1]
        assert app.player._run is replacement
        original_bridge = app.call_from_thread

        def refuse(callback: Any, *args: Any, **kwargs: Any) -> Any:
            bridge_calls.append(callback.__name__)
            bridge_attempted.set()
            raise RuntimeError(PRIVATE_ERROR)

        monkeypatch.setattr(app, "call_from_thread", refuse)
        release_old.set()
        await _wait(bridge_attempted)
        await pilot.pause()

        assert bridge_calls == ["_render_frame"]
        assert pipeline.stop_calls == 0
        assert app.player._pipeline is pipeline
        assert app.player._run is replacement
        assert not replacement.eof

        monkeypatch.setattr(app, "call_from_thread", original_bridge)
        release_replacement.set()
        await _finish_workers(app, pilot)


@pytest.mark.asyncio
async def test_refused_current_run_cannot_stop_replacement_after_validation_race(
    monkeypatch,
):
    old_entered = Event()
    release_old = Event()
    old_done = Event()
    cleanup_entered = Event()
    race_observed = Event()
    release_cleanup = Event()
    replacement_entered = Event()
    release_replacement = Event()

    class PausedCleanupPlayer(_ObservedPlayer):
        def _cleanup_pipeline(self, pipeline: _Pipeline) -> None:
            cleanup_entered.set()
            race_observed.set()
            assert release_cleanup.wait(2.0)
            super()._cleanup_pipeline(pipeline)

    class ValidationRacePipeline(_Pipeline):
        def iter_frames(self, run: PlayerRun):
            if run.generation == 1:
                try:
                    old_entered.set()
                    assert release_old.wait(2.0)
                    yield 1.0, bytes((255, 0, 0))
                    return
                finally:
                    old_done.set()
                    race_observed.set()
            replacement_entered.set()
            assert release_replacement.wait(2.0)
            run.eof = True
            if False:
                yield 0.0, b""

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.video_player_screen.PlayerPipeline",
        ValidationRacePipeline,
    )
    player = PausedCleanupPlayer(PRIVATE_PATH, render_mode="ascii")
    app = _PlayerApp(player)
    bridge_calls: list[str] = []
    async with app.run_test() as pilot:
        await _wait(old_entered)
        original_bridge = app.call_from_thread

        def refuse(callback: Any, *args: Any, **kwargs: Any) -> Any:
            bridge_calls.append(callback.__name__)
            raise RuntimeError(PRIVATE_ERROR)

        monkeypatch.setattr(app, "call_from_thread", refuse)
        release_old.set()
        await _wait(race_observed)

        monkeypatch.setattr(app, "call_from_thread", original_bridge)
        player.action_seek_fwd()
        await _wait(replacement_entered)
        await pilot.pause()
        replacement = ValidationRacePipeline.instances[0].runs[1]
        assert player._run is replacement

        release_cleanup.set()
        await _wait(old_done)
        await pilot.pause()

        pipeline = ValidationRacePipeline.instances[0]
        assert bridge_calls == ["_render_frame"]
        assert not cleanup_entered.is_set()
        assert pipeline.stop_calls == 0
        assert player._pipeline is pipeline
        assert player._run is replacement
        assert not replacement.eof

        release_replacement.set()
        await _finish_workers(app, pilot)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["start", "iterator", "clock", "stats"])
async def test_activation_and_current_pump_failures_are_contained(
    monkeypatch,
    phase: str,
):
    failure_seen = Event()

    class FailingPipeline(_Pipeline):
        def start(self) -> PlayerRun:
            if phase == "start":
                failure_seen.set()
                raise RuntimeError(PRIVATE_ERROR)
            return super().start()

        def iter_frames(self, run: PlayerRun):
            if phase == "iterator":
                failure_seen.set()
                raise RuntimeError(PRIVATE_ERROR)
            yield 0.0, bytes((1, 2, 3))

        def frames_behind(self, run: PlayerRun, pts: float) -> bool:
            if phase == "clock":
                failure_seen.set()
                raise RuntimeError(PRIVATE_ERROR)
            return False

        def note_rendered(self, run: PlayerRun, pts: float) -> None:
            if phase == "stats":
                failure_seen.set()
                raise RuntimeError(PRIVATE_ERROR)
            super().note_rendered(run, pts)

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.video_player_screen.PlayerPipeline", FailingPipeline
    )
    records: list[str] = []
    sink = logger.add(lambda message: records.append(str(message)))
    try:
        app = _PlayerApp()
        async with app.run_test() as pilot:
            await _wait(failure_seen)
            await _finish_workers(app, pilot)
            await pilot.press("x")
            assert app.action_count == 1
        assert FailingPipeline.instances[0].stopped.is_set()
        assert app.notifications
        user_copy = "\n".join(app.notifications)
        assert "system player" in user_copy.lower()
        assert PRIVATE_ERROR not in user_copy
        assert PRIVATE_PATH not in user_copy
    finally:
        logger.remove(sink)
    _assert_sanitized(records, "activation" if phase == "start" else "pump")


@pytest.mark.asyncio
async def test_render_and_cleanup_failures_are_sanitized(monkeypatch):
    class PRIVATE_MODAL_ERROR_SENTINEL:
        pass

    class RenderCleanupPipeline(_Pipeline):
        def iter_frames(self, run: PlayerRun):
            yield 0.0, PRIVATE_MODAL_ERROR_SENTINEL()

        def stop(self) -> None:
            self.stop_calls += 1
            self.stopped.set()
            raise RuntimeError(PRIVATE_ERROR)

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.video_player_screen.PlayerPipeline",
        RenderCleanupPipeline,
    )
    records: list[str] = []
    sink = logger.add(lambda message: records.append(str(message)))
    try:
        app = _PlayerApp()
        async with app.run_test() as pilot:
            await _finish_workers(app, pilot)
        user_copy = "\n".join(app.notifications)
        assert PRIVATE_ERROR not in user_copy
        assert PRIVATE_PATH not in user_copy
        assert "PRIVATE_MODAL_ERROR_SENTINEL" not in user_copy
    finally:
        logger.remove(sink)
    joined = "\n".join(records)
    assert "component=modal_player" in joined
    assert "phase=render" in joined
    assert "phase=cleanup" in joined
    assert "error_type=TypeError" in joined
    assert "error_type=RuntimeError" in joined
    assert PRIVATE_ERROR not in joined
    assert PRIVATE_PATH not in joined
    assert "PRIVATE_MODAL_ERROR_SENTINEL" not in joined
    assert "Traceback" not in joined


@pytest.mark.asyncio
async def test_bridge_refusal_is_attempted_once_without_worker_ui_fallback(
    monkeypatch,
):
    start_entered = Event()
    release = Event()
    bridge_calls: list[str] = []
    accepted_directly = Event()

    class BridgePlayer(_ObservedPlayer):
        def _accept_activation(self, *args: Any) -> bool:
            accepted_directly.set()
            return super()._accept_activation(*args)

    class HeldPipeline(_Pipeline):
        def start(self) -> PlayerRun:
            start_entered.set()
            assert release.wait(2.0)
            return super().start()

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.video_player_screen.PlayerPipeline", HeldPipeline
    )
    player = BridgePlayer(PRIVATE_PATH, render_mode="ascii")
    app = _PlayerApp(player)
    records: list[str] = []
    sink = logger.add(lambda message: records.append(str(message)))
    try:
        async with app.run_test() as pilot:
            await _wait(start_entered)

            def refuse(callback: Any, *args: Any, **kwargs: Any) -> Any:
                bridge_calls.append(callback.__name__)
                raise RuntimeError(PRIVATE_ERROR)

            monkeypatch.setattr(app, "call_from_thread", refuse)
            release.set()
            await _wait(HeldPipeline.instances[0].stopped)
            await _finish_workers(app, pilot)
        assert bridge_calls == ["_accept_activation"]
        assert not accepted_directly.is_set()
        assert player._pipeline is None
    finally:
        logger.remove(sink)
    _assert_sanitized(records, "frame_dispatch")


# -- stream mode (task-3401.11) -----------------------------------------------------


def test_non_seekable_hints_swap():
    from tldw_chatbook.UI.Screens.video_player_screen import _HINTS, _HINTS_NO_SEEK

    seekable_screen = VideoPlayerScreen("/tmp/x.mp4", seekable=True)
    no_seek_screen = VideoPlayerScreen("https://cdn.example.net/v", seekable=False)
    seekable_hints = list(seekable_screen.compose())[-1]
    no_seek_hints = list(no_seek_screen.compose())[-1]
    assert seekable_hints.renderable == _HINTS
    assert no_seek_hints.renderable == _HINTS_NO_SEEK
    assert "seek unavailable" in _HINTS_NO_SEEK


def test_non_seekable_seek_is_refused_without_touching_pipeline():
    class _FakePipeline:
        def __init__(self):
            self.stats = type("S", (), {"position_seconds": 3.0})()
            self.seeked: list[float] = []

        def seek(self, target):
            self.seeked.append(target)

    notifications: list[str] = []

    class _FakeAppScreen(VideoPlayerScreen):
        @property
        def app(self):  # type: ignore[override]
            return SimpleNamespace(
                notify=lambda message, severity=None: notifications.append(message)
            )

    screen = _FakeAppScreen("https://cdn.example.net/v", seekable=False)
    pipeline = _FakePipeline()
    screen._pipeline = pipeline
    screen._run = PlayerRun(1, None, 3.0)
    screen._seek_relative(5.0)
    assert pipeline.seeked == []  # AC4: disabled, never reaches the pipeline
    assert notifications and "unavailable" in notifications[0]
