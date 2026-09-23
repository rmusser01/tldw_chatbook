"""`UI/Wizards` + `UI/Speech` + `UI/Watchlists_Modules` fixes (tier-2 S21 P2s).

Gate-free: pure functions and one bare-instance coroutine drive. Nothing here
boots the app, so `Backup_Recovery`'s ADR-126 recovery gate never runs.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
from typing import Any

import pytest

from tldw_chatbook.UI.Speech.speech_playback_mixin import SpeechPlaybackMixin
from tldw_chatbook.UI.Watchlists_Modules.table_selection import row_with_id
from tldw_chatbook.UI.Wizards.first_run_setup_state import middle_truncate_path
from tldw_chatbook.Utils.Utils import elide_path_middle

_CONFIG_PATH = (
    "/Users/macbook-dev/Library/Application Support/tldw_chatbook/"
    "config_for_my_work_profile.toml"
)


@pytest.mark.unit
def test_summary_path_elision_keeps_the_whole_filename() -> None:
    """S21 P2: `middle_truncate_path` split at the midpoint and ate the name.

    `elide_path_middle`'s docstring says it exists precisely because a naive
    head/tail split "discards the very name the user just picked" -- and the
    first-run Summary screen, whose entire job is an honest read-back of what
    landed on disk, was using the naive split.
    """
    elided = middle_truncate_path(_CONFIG_PATH, 40)

    assert "config_for_my_work_profile.toml" in elided, (
        f"the filename must survive the elision; got {elided!r}"
    )
    assert len(elided) <= 40
    assert elided == elide_path_middle(_CONFIG_PATH, 40), (
        "must be the shared elider, not a second implementation of it"
    )

    # Contract kept from the re-roll it replaced.
    assert middle_truncate_path("short.toml", 40) == "short.toml"
    assert len(middle_truncate_path(_CONFIG_PATH, 2)) <= 8


@pytest.mark.unit
def test_row_with_id_coerces_ids_the_same_way_for_every_pane() -> None:
    """S21 P2: five byte-identical `select_<X>_by_id` scans, one rule now.

    The `str(row.get("id") or "")` coercion is what this package's "capture
    IDENTITY, never a row INDEX" contract rests on, so it has to be ONE rule.
    """
    rows = [{"id": 7, "name": "seven"}, {"id": None}, {"name": "no id"}]

    assert row_with_id(rows, "7") == {"id": 7, "name": "seven"}
    assert row_with_id(rows, "") == {"id": None}
    assert row_with_id(rows, "missing") is None
    assert row_with_id([], "7") is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "module_attr",
    (
        "select_run_by_id",
        "select_item_by_id",
        "select_rule_by_id",
        "select_source_by_id",
    ),
)
def test_every_pane_selector_routes_through_the_shared_scan(
    module_attr: str,
) -> None:
    """None of the five panes may keep a private copy of the scan."""
    import inspect

    from tldw_chatbook.UI.Watchlists_Modules import (
        article_list,
        items_pane,
        rules_pane,
        runs_pane,
        sources_pane,
    )

    owners = {
        "select_run_by_id": [runs_pane.RunsPane],
        "select_item_by_id": [items_pane.ItemsPane, article_list.ArticleListPane],
        "select_rule_by_id": [rules_pane.RulesPane],
        "select_source_by_id": [sources_pane.SourcesPane],
    }
    for owner in owners[module_attr]:
        source = inspect.getsource(getattr(owner, module_attr))
        assert "row_with_id" in source, f"{owner.__name__}.{module_attr}"
        assert "for candidate in" not in source, (
            f"{owner.__name__}.{module_attr} still carries its own scan"
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_retiring_the_progress_timer_waits_for_it_to_finish() -> None:
    """S21 P2: cancellation-by-sleep is a race, not a barrier.

    `Task.cancel()` only REQUESTS cancellation -- the loop still has to be
    resumed to raise, run its `except CancelledError` branch and its
    "ensure UI is reset on exit" block, which writes the same three widgets a
    freshly started replacement timer writes. Six sites used to cancel and
    then either sleep a hardcoded 50-100 ms or continue immediately.
    """
    host = SpeechPlaybackMixin.__new__(SpeechPlaybackMixin)
    unwound: list[str] = []
    entered = asyncio.Event()

    async def _loop() -> None:
        try:
            entered.set()
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            # Stand in for the real loop's reset block, which repaints
            # #audio-player-transport and #audio-player-status.
            await asyncio.sleep(0)
            unwound.append("reset")
            raise

    host._progress_timer_task = asyncio.create_task(_loop())
    await entered.wait()

    await host._retire_progress_timer()

    assert unwound == ["reset"], (
        "the retire must not return until the old loop's exit block has run; "
        "otherwise a replacement timer starts while it is still writing"
    )
    assert host._progress_timer_task is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_retiring_an_absent_or_finished_timer_is_a_no_op() -> None:
    host = SpeechPlaybackMixin.__new__(SpeechPlaybackMixin)

    host._progress_timer_task = None
    await host._retire_progress_timer()
    assert host._progress_timer_task is None

    async def _done() -> None:
        return None

    task: Any = asyncio.create_task(_done())
    await task
    host._progress_timer_task = task
    await host._retire_progress_timer()
    assert host._progress_timer_task is None


def _playback_mixin_tree() -> ast.Module:
    from tldw_chatbook.UI.Speech import speech_playback_mixin

    return ast.parse(inspect.getsource(speech_playback_mixin))


def _async_functions(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef):
            yield node


def _retire_calls(tree: ast.AST) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_retire_progress_timer"
    ]


def _awaited_retire_linenos(tree: ast.AST) -> list[int]:
    """Line numbers of `await ..._retire_progress_timer(...)` only.

    A BARE `self._retire_progress_timer()` returns a coroutine that is never
    driven: the cancel-and-join never runs at all. That is the exact defect
    Qodo #2813 finding 4 reported the old substring-based guard could not
    see, so the awaited-ness is part of what gets matched here.
    """
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Await)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "_retire_progress_timer"
    ]


def _timer_start_linenos(tree: ast.AST) -> list[int]:
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute) and target.attr == "_progress_timer_task"
            for target in node.targets
        )
        and "_update_progress_timer" in ast.dump(node)
    ]


@pytest.mark.unit
def test_no_site_may_call_the_retire_barrier_without_awaiting_it() -> None:
    """Qodo #2813 finding 4: a bare call is a silently dead barrier.

    `_retire_progress_timer` is a coroutine function. Dropping the `await`
    is a one-character edit that leaves every other guard green while the
    cancel-and-join never executes, so the outgoing loop keeps running and
    overlaps its replacement writing the same three widgets.
    """
    tree = _playback_mixin_tree()
    awaited = {
        id(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Await) and isinstance(node.value, ast.Call)
    }
    bare = [call.lineno for call in _retire_calls(tree) if id(call) not in awaited]

    assert bare == [], (
        f"_retire_progress_timer is called without `await` at line(s) {bare}; "
        "the coroutine is never driven, so the cancel-and-join does not run "
        "and the outgoing timer overlaps its replacement"
    )


@pytest.mark.unit
def test_every_replacement_progress_timer_retires_its_predecessor() -> None:
    """The barrier is only worth having if the start sites actually use it.

    Covers the wiring between the public playback paths and
    `_retire_progress_timer`, which the helper's own unit tests cannot see.
    Matching is on an `ast.Await` of the barrier earlier in the SAME
    enclosing coroutine -- not a substring of the preceding statements,
    which Qodo #2813 finding 4 showed would accept an unawaited call.
    """
    tree = _playback_mixin_tree()
    starts = 0
    for func in _async_functions(tree):
        retires = _awaited_retire_linenos(func)
        for lineno in _timer_start_linenos(func):
            starts += 1
            assert any(retire < lineno for retire in retires), (
                f"a progress timer is started at line {lineno} in "
                f"{func.name} with no `await self._retire_progress_timer()` "
                "before it in the same coroutine; the outgoing loop's exit "
                "block repaints the same three widgets the new one writes"
            )
    assert starts == 2, f"expected both start sites, found {starts}"

    # And no site may go back to cancelling it by hand.
    cancels = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "cancel"
        and "_progress_timer_task" in ast.dump(node.func)
    ]
    assert cancels == [], "cancel-and-hope replaced the awaited barrier"


@pytest.mark.unit
def test_replacement_playback_retires_the_old_timer_before_it_leases() -> None:
    """Qodo #2813 finding 2, as a source-order invariant.

    The progress loop releases whatever sits in `_active_playback_release`
    the moment it observes an idle or finished player. So the outgoing timer
    has to be DEAD before the replacement lease is installed -- otherwise it
    can free the new lease, and on the PCM path securely delete the WAV copy
    that is playing. (The behavioural drive below is the real pin; this one
    names the ordering so a reader of the source sees why it is that way.)
    """
    tree = _playback_mixin_tree()
    play = next(f for f in _async_functions(tree) if f.name == "_play_audio_async")
    retires = _awaited_retire_linenos(play)
    assert retires, "_play_audio_async no longer retires the outgoing timer"
    barrier = min(retires)

    leases = [
        node.lineno
        for node in ast.walk(play)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and target.attr == "_active_playback_release"
            for target in node.targets
        )
    ]
    assert leases, "_play_audio_async no longer installs a playback lease"
    for lease in leases:
        assert barrier < lease, (
            f"the replacement lease is installed at line {lease}, before the "
            f"outgoing progress timer is retired at line {barrier}; that timer "
            "releases whatever it finds in _active_playback_release when it "
            "sees an idle player, so it can free the NEW artifact"
        )
    for start in _timer_start_linenos(play):
        assert barrier < start, (
            f"a replacement timer starts at line {start} before the barrier "
            f"at line {barrier}"
        )


class _FakeAudioPlayer:
    """The player behaviours the playback paths depend on, and nothing else.

    `stop()` deliberately does NOT flip the reported state: a real backend's
    state machine lags the stop call, and that lag is precisely the window
    the outgoing progress loop used to keep running in.
    """

    def __init__(self, state: Any) -> None:
        self.state = state
        self.on_stop: Any = None
        self.on_play: Any = None
        self.observed: list[Any] = []

    async def get_state(self) -> Any:
        await asyncio.sleep(0)
        return self.state

    async def stop(self) -> bool:
        if self.on_stop is not None:
            self.observed.append(self.on_stop())
        return True

    async def play(self, path: Any) -> bool:
        from tldw_chatbook.TTS.audio_player import PlaybackState

        if self.on_play is not None:
            self.observed.append(self.on_play())
        # A real window for anything still running to interleave in.
        await asyncio.sleep(0.05)
        self.state = PlaybackState.PLAYING
        return True

    async def get_position(self) -> float:
        return 0.0

    async def get_duration(self) -> float:
        return 10.0

    async def is_playing(self) -> bool:
        return True


def _playback_host(player: _FakeAudioPlayer) -> Any:
    """A bare `SpeechPlaybackMixin` wired to `player`, with only the widget
    lookups stubbed -- the playback coroutines, the retire barrier and the
    progress loop are all the real ones."""
    from unittest.mock import MagicMock

    host = SpeechPlaybackMixin.__new__(SpeechPlaybackMixin)
    host.app = MagicMock()
    host.app.audio_player = player
    host._progress_timer_task = None
    host._active_playback_release = None
    host._play_worker_task = None
    host._result_transition_operation_id = None
    host.query_one = MagicMock(return_value=MagicMock())
    host._sync_idle_transport_actions = MagicMock()
    host._sync_active_transport_actions = MagicMock()
    host._current_result_status_copy = MagicMock(return_value="Ready")
    return host


async def _start_real_progress_loop(host: Any) -> Any:
    """Install the REAL progress loop as the outgoing timer and let it reach
    its first sleep, so it is genuinely mid-flight."""
    task = asyncio.create_task(host._update_progress_timer())
    host._progress_timer_task = task
    await asyncio.sleep(0.02)
    assert not task.done(), "the outgoing progress loop exited before the test began"
    return task


@pytest.mark.unit
@pytest.mark.asyncio
async def test_new_playback_never_starts_while_the_old_timer_can_still_release(
    tmp_path: Any,
) -> None:
    """Qodo #2813 finding 2, driven through the real `_play_audio_async`.

    This is also the integration coverage finding 6 asked for: the public
    playback coroutine, the real progress loop and the real barrier wired
    together, rather than the private helper poked on its own.

    The outgoing loop releases whatever sits in `_active_playback_release`
    the moment it observes an idle or finished player. It used to stay alive
    across the PCM copy and the entire `play()` await, i.e. across the point
    where the REPLACEMENT lease is installed -- so a late wake-up could free
    the new lease, and on the PCM path securely delete the WAV copy that is
    playing.
    """
    from tldw_chatbook.TTS.audio_player import PlaybackState

    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"RIFF....WAVE")

    player = _FakeAudioPlayer(PlaybackState.PLAYING)
    host = _playback_host(player)

    old_released: list[str] = []
    host._active_playback_release = lambda: old_released.append("old")
    old_timer = await _start_real_progress_loop(host)

    player.on_play = lambda: (host._progress_timer_task, old_timer.done())

    new_released: list[str] = []
    await host._play_audio_async(audio, lambda: new_released.append("new"))

    assert player.observed, "play() was never reached"
    timer_field, old_timer_done = player.observed[0]
    assert timer_field is None and old_timer_done, (
        "the outgoing progress timer was still alive when the replacement "
        "playback started (_progress_timer_task="
        f"{timer_field!r}, old timer done={old_timer_done}); it releases "
        "whatever is in _active_playback_release when it sees an idle player, "
        "so it can free the NEW lease"
    )
    assert old_released == ["old"], "the outgoing artifact was not released once"
    assert new_released == [], "the new playback lease was released mid-play"
    assert host._active_playback_release is not None, "the new lease was dropped"
    assert host._progress_timer_task is not None, "no replacement timer started"

    await host._retire_progress_timer()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stopping_playback_unwinds_the_timer_before_touching_the_player() -> None:
    """Teardown half of finding 6, through the real `_stop_audio_async`.

    The loop's own exit block repaints `#audio-player-transport` and
    `#audio-player-status`, so it has to be fully unwound before the stop
    path repaints them -- otherwise the transport comes straight back.
    """
    from tldw_chatbook.TTS.audio_player import PlaybackState

    player = _FakeAudioPlayer(PlaybackState.PLAYING)
    host = _playback_host(player)
    old_timer = await _start_real_progress_loop(host)
    player.on_stop = lambda: old_timer.done()

    assert await host._stop_audio_async() is True

    assert player.observed == [True], (
        "stop() reached the player with the progress loop still unwinding; "
        "its exit block repaints the transport the stop path is about to hide"
    )
    assert host._progress_timer_task is None
