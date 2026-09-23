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


def _statement_blocks(tree: ast.AST):
    """Every statement list in the tree, so 'next statement' is meaningful."""
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if isinstance(block, list) and block and isinstance(block[0], ast.stmt):
                yield block


@pytest.mark.unit
def test_the_new_release_is_installed_with_no_await_before_the_old_timer_dies():
    """Nothing may suspend between installing the lease and retiring the timer.

    Qodo review of #2813 reported this as a live bug: the old progress timer
    releases whatever currently sits in `_active_playback_release` when it
    sees an idle or finished player, so if it resumed *after* the replacement
    callback was installed it would securely delete the new PCM copy mid-play.

    It cannot resume there today -- `_play_audio_async` installs the new
    callback and calls `_retire_progress_timer` with only a plain assignment
    between them, and `cancel()` is synchronous, so the old loop's next
    resumption raises `CancelledError` and takes the `except` branch instead
    of the release branch. The safety is entirely that adjacency, which is one
    stray `await` away from being untrue. This pins it.
    """
    for block in _statement_blocks(_playback_mixin_tree()):
        for index, statement in enumerate(block):
            if not (
                isinstance(statement, ast.Assign)
                and any(
                    isinstance(target, ast.Attribute)
                    and target.attr == "_active_playback_release"
                    for target in statement.targets
                )
            ):
                continue
            for follower in block[index + 1 :]:
                dumped = ast.dump(follower)
                if "Await" not in dumped:
                    continue
                assert "_retire_progress_timer" in dumped, (
                    "an await now separates installing _active_playback_release "
                    "from retiring the old progress timer; the old timer can "
                    "reach its release branch in that window and free the NEW "
                    f"artifact (line {follower.lineno})"
                )
                break


@pytest.mark.unit
def test_every_replacement_progress_timer_retires_its_predecessor():
    """The barrier is only worth having if the start sites actually use it.

    Covers the wiring between the public playback paths and
    `_retire_progress_timer`, which the helper's own unit tests cannot see.
    """
    tree = _playback_mixin_tree()
    starts = 0
    for block in _statement_blocks(tree):
        for index, statement in enumerate(block):
            if not (
                isinstance(statement, ast.Assign)
                and any(
                    isinstance(target, ast.Attribute)
                    and target.attr == "_progress_timer_task"
                    for target in statement.targets
                )
                and "_update_progress_timer" in ast.dump(statement)
            ):
                continue
            starts += 1
            preceding = ast.dump(ast.Module(body=block[:index], type_ignores=[]))
            assert "_retire_progress_timer" in preceding, (
                f"a progress timer is started at line {statement.lineno} without "
                "first awaiting _retire_progress_timer; the outgoing loop's exit "
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
