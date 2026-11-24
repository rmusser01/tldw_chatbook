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
    assert starts == 1, "the run starts one timer spanning pause and resume"

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
def test_progress_timer_polls_only_the_captured_run_without_artifact_release() -> None:
    tree = _playback_mixin_tree()
    timer = next(
        f for f in _async_functions(tree) if f.name == "_update_progress_timer"
    )
    assert [arg.arg for arg in timer.args.args] == ["self", "attempt"]
    calls = {ast.unparse(n.func) for n in ast.walk(timer) if isinstance(n, ast.Call)}
    assert {"run.get_state", "run.get_position", "run.get_duration"} <= calls
    assert not any("release" in call for call in calls)


@pytest.mark.asyncio
async def test_replaced_progress_timer_cannot_repaint_or_release_successor():
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from tldw_chatbook.TTS.audio_player import PlaybackState

    entered, finish = asyncio.Event(), asyncio.Event()

    async def old_state():
        entered.set()
        await finish.wait()
        return PlaybackState.FINISHED

    old = SimpleNamespace(run=SimpleNamespace(handed_off=False, get_state=old_state))
    host = SpeechPlaybackMixin.__new__(SpeechPlaybackMixin)
    host.init_playback_state()
    host.is_attached = True
    host.query_one = MagicMock()
    host._file_output.latest = old
    timer = asyncio.create_task(host._update_progress_timer(old))
    await asyncio.wait_for(entered.wait(), 2)
    host._file_output.latest = object()
    finish.set()
    await timer
    host.query_one.assert_not_called()
