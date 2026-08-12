"""TASK-15455: tail-first windowed mount + scrollback hydration.

Session resume used to mount EVERY persisted row (one awaited ``mount()`` per
row, one Textual Markdown widget per assistant message) before the height
watermarks could trim anything. The transcript now mounts only the newest
window on a fresh load and hydrates older messages when the reader scrolls
back.

The first half of this module is the PIN section: invariants that existed
before windowing and must survive it, exercised on a history large enough
(60+ messages) that the window path is actually in play — the pre-existing
suites top out at 30 messages, so none of them would have noticed.
"""

import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
    ConsoleVariantSet,
)
from tldw_chatbook.Widgets.Console.console_transcript import (
    DEFAULT_TRANSCRIPT_HYDRATE_MESSAGES,
    DEFAULT_TRANSCRIPT_WINDOW_LINES,
    DEFAULT_TRANSCRIPT_WINDOW_MESSAGES,
    ConsoleTranscript,
    get_console_transcript_window,
)


class WindowHarness(App):
    """Small viewport so a handful of messages overflow it."""

    CSS = "ConsoleTranscript { height: 10; }"

    def __init__(
        self,
        *,
        low: int = 12000,
        high: int = 20000,
        window_messages: int | None = None,
        window_lines: int | None = None,
        hydrate_messages: int | None = None,
    ) -> None:
        super().__init__()
        chat_defaults: dict[str, object] = {
            "prune_high_watermark": high,
            "prune_low_watermark": low,
        }
        if window_messages is not None:
            chat_defaults["transcript_window_messages"] = window_messages
        if window_lines is not None:
            chat_defaults["transcript_window_lines"] = window_lines
        if hydrate_messages is not None:
            chat_defaults["transcript_hydrate_messages"] = hydrate_messages
        self.app_config = {"chat_defaults": chat_defaults}

    def compose(self) -> ComposeResult:
        yield ConsoleTranscript(id="console-native-transcript")


def _msg(
    i: int | str,
    role: ConsoleMessageRole = ConsoleMessageRole.ASSISTANT,
    *,
    status: str = "completed",
    lines: int = 4,
) -> ConsoleChatMessage:
    return ConsoleChatMessage(
        role=role,
        content="\n".join(f"line {i}.{j}" for j in range(lines)),
        id=f"m{i}",
        status=status,
    )


def _messages(n: int, *, lines: int = 4) -> list[ConsoleChatMessage]:
    return [
        _msg(
            i,
            ConsoleMessageRole.USER if i % 2 == 0 else ConsoleMessageRole.ASSISTANT,
            lines=lines,
        )
        for i in range(n)
    ]


def _mounted_message_ids(transcript: ConsoleTranscript) -> list[str]:
    # Class query, not a type query: assistant rows may render as
    # ConsoleMarkdownMessage (TASK-1990) while other roles stay plain.
    return [
        widget.message_id for widget in transcript.query(".console-transcript-message")
    ]


async def _wait_for(pilot, predicate, *, attempts: int = 50) -> bool:
    for _ in range(attempts):
        await pilot.pause()
        if predicate():
            return True
    return False


async def _settle(pilot, *, times: int = 8) -> None:
    for _ in range(times):
        await pilot.pause()


# ---------------------------------------------------------------------------
# PIN section — behavior that predates windowing and must survive it.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pin_large_history_load_leaves_reader_at_the_tail():
    """A 60-message load lands anchored at the newest content."""
    app = WindowHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(_messages(60))
        await transcript.refresh_messages()
        await _settle(pilot)

        assert transcript._is_following_tail(), "load must keep tail-follow engaged"
        assert transcript.max_scroll_y > 0, "fixture must overflow the viewport"
        assert transcript.scroll_y == transcript.max_scroll_y
        mounted = _mounted_message_ids(transcript)
        assert mounted, "the newest rows must be mounted"
        assert mounted[-1] == "m59", "the newest message must be the last row"


@pytest.mark.asyncio
async def test_pin_selection_and_action_row_on_a_mounted_row():
    """Clicking/selecting a mounted row shows its contextual action row."""
    app = WindowHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(_messages(60))
        await transcript.refresh_messages()
        await _settle(pilot)

        target = _mounted_message_ids(transcript)[-2]
        transcript.select_message(target)
        assert await _wait_for(
            pilot,
            lambda: bool(transcript.query(".console-transcript-action-row")),
        ), "selecting a mounted row must mount its action row"
        assert transcript.selected_message_id == target
        assert target in _mounted_message_ids(transcript)


@pytest.mark.asyncio
async def test_pin_prune_watermarks_bound_mounted_height():
    """The height watermarks still bound the mounted transcript."""
    app = WindowHarness(low=25, high=40)
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(_messages(60))
        await transcript.refresh_messages()
        await _settle(pilot)

        assert transcript.virtual_size.height <= 40, (
            "mounted height must settle at or under the high watermark"
        )
        assert transcript._is_following_tail()
        assert len(transcript._messages) == 60, "the store snapshot is untouched"


@pytest.mark.asyncio
async def test_pin_branch_swap_keeps_shared_prefix_rows():
    """A branch/variant swap replaces the suffix without rebuilding the prefix."""
    app = WindowHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        history = _messages(60)
        transcript.set_messages(history)
        await transcript.refresh_messages()
        await _settle(pilot)

        kept_key = f"message:{history[-3].id}"
        builds_before = transcript.row_build_counts().get(kept_key)
        assert builds_before is not None, "the pinned row must be mounted"

        swapped = list(history[:-1]) + [
            _msg("branch-b", ConsoleMessageRole.ASSISTANT)
        ]
        transcript.set_messages(swapped)
        await transcript.refresh_messages()
        await _settle(pilot)

        mounted = _mounted_message_ids(transcript)
        assert mounted[-1] == "mbranch-b", "the swapped-in branch tail must mount"
        assert history[-1].id not in mounted, "the replaced tail must be gone"
        assert transcript.row_build_counts().get(kept_key) == builds_before, (
            "a branch swap must not rebuild the shared prefix rows"
        )


@pytest.mark.asyncio
async def test_pin_variant_navigation_on_a_mounted_row():
    """`>`/`<` variant navigation still swaps the rendered variant."""
    app = WindowHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        history = _messages(60)
        history[-1] = ConsoleChatMessage(
            role=ConsoleMessageRole.ASSISTANT,
            content="variant one",
            id="mvariants",
            variants=ConsoleVariantSet.from_contents(
                turn_id="turn-variants",
                contents=["variant one", "variant two"],
                selected_index=0,
            ),
        )
        transcript.set_messages(history)
        await transcript.refresh_messages()
        await _settle(pilot)

        transcript.select_next_variant("mvariants")
        await _settle(pilot)
        assert "variant two" in transcript.to_plain_text(width=60)


@pytest.mark.asyncio
async def test_pin_session_switch_replaces_every_row():
    """Switching sessions drops the old rows and renders the new tail."""
    app = WindowHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(_messages(60))
        await transcript.refresh_messages()
        await _settle(pilot)

        other = [
            _msg(f"other{i}", ConsoleMessageRole.ASSISTANT) for i in range(60)
        ]
        transcript.set_messages(other)
        await transcript.refresh_messages()
        await _settle(pilot)

        mounted = _mounted_message_ids(transcript)
        assert mounted, "the new session must render rows"
        assert all(message_id.startswith("mother") for message_id in mounted), (
            "no row from the previous session may survive a switch"
        )
        assert mounted[-1] == "mother59"
        assert transcript._is_following_tail()


@pytest.mark.asyncio
async def test_pin_jump_pill_and_jump_to_latest_after_scrolling_up():
    """Detaching shows the pill; jump_to_latest re-anchors and hides it."""
    app = WindowHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(_messages(60))
        await transcript.refresh_messages()
        await _settle(pilot)

        transcript.release_anchor()
        transcript.scroll_to(y=0, animate=False)
        await _settle(pilot)
        transcript.sync_jump_indicator("streaming")
        pill = transcript.query_one("#console-transcript-jump-pill")
        assert pill.display is True, "a detached reader mid-run sees the pill"

        transcript.jump_to_latest()
        await _settle(pilot)
        assert pill.display is False
        assert transcript._is_following_tail()
        assert transcript.scroll_y == transcript.max_scroll_y


@pytest.mark.asyncio
async def test_pin_exports_render_full_history_regardless_of_mounting():
    """`to_plain_text` renders every message, mounted or not."""
    app = WindowHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(_messages(60))
        await transcript.refresh_messages()
        await _settle(pilot)

        plain = transcript.to_plain_text(width=40)
        assert "line 0.0" in plain
        assert "line 59.3" in plain


# ---------------------------------------------------------------------------
# Batched DOM churn (reconciler).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_new_rows_mount_in_one_batched_call():
    """Contiguous new rows are mounted with a single `mount()` call."""
    app = WindowHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        calls: list[int] = []
        original_mount = transcript.mount

        def _counting_mount(*widgets, **kwargs):
            calls.append(len(widgets))
            return original_mount(*widgets, **kwargs)

        transcript.mount = _counting_mount  # type: ignore[method-assign]
        transcript.set_messages(_messages(20))
        await transcript.refresh_messages()
        await _settle(pilot)
        transcript.mount = original_mount  # type: ignore[method-assign]

        mounted_rows = sum(calls)
        assert mounted_rows > 20, "the load must mount at least one row per message"
        assert len(calls) <= 2, (
            "a fresh load must mount its rows in one batch "
            f"(saw {len(calls)} mount calls for {mounted_rows} rows)"
        )


@pytest.mark.asyncio
async def test_removed_rows_are_pruned_in_one_batched_call():
    """A session switch removes the old rows with one `remove_children()`."""
    app = WindowHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(_messages(20))
        await transcript.refresh_messages()
        await _settle(pilot)

        removals: list[int] = []
        original_remove_children = transcript.remove_children

        def _counting_remove_children(selector="*"):
            widgets = list(selector) if not isinstance(selector, str) else []
            removals.append(len(widgets))
            return original_remove_children(widgets or selector)

        transcript.remove_children = _counting_remove_children  # type: ignore[method-assign]
        transcript.set_messages(
            [_msg(f"other{i}", ConsoleMessageRole.ASSISTANT) for i in range(4)]
        )
        await transcript.refresh_messages()
        await _settle(pilot)
        transcript.remove_children = original_remove_children  # type: ignore[method-assign]

        assert removals, "the switch must remove the previous session's rows"
        assert len(removals) == 1, (
            f"stale rows must be removed in one batch (saw {len(removals)} calls)"
        )
        assert removals[0] > 20, "every stale row belongs to that one batch"
        assert _mounted_message_ids(transcript) == [f"mother{i}" for i in range(4)]


# ---------------------------------------------------------------------------
# Tail-first window + scroll-back hydration.
# ---------------------------------------------------------------------------


def test_window_settings_defaults_and_kill_switch():
    """Config resolution mirrors the watermark helper, including the escape."""
    assert get_console_transcript_window(None) == (
        DEFAULT_TRANSCRIPT_WINDOW_MESSAGES,
        DEFAULT_TRANSCRIPT_WINDOW_LINES,
        DEFAULT_TRANSCRIPT_HYDRATE_MESSAGES,
    )
    assert get_console_transcript_window({"chat_defaults": "nope"}) == (
        DEFAULT_TRANSCRIPT_WINDOW_MESSAGES,
        DEFAULT_TRANSCRIPT_WINDOW_LINES,
        DEFAULT_TRANSCRIPT_HYDRATE_MESSAGES,
    )
    assert get_console_transcript_window(
        {"chat_defaults": {"transcript_window_messages": "x"}}
    ) == (
        DEFAULT_TRANSCRIPT_WINDOW_MESSAGES,
        DEFAULT_TRANSCRIPT_WINDOW_LINES,
        DEFAULT_TRANSCRIPT_HYDRATE_MESSAGES,
    )
    # The kill switch is preserved verbatim; the other two are floored at 1.
    assert get_console_transcript_window(
        {
            "chat_defaults": {
                "transcript_window_messages": 0,
                "transcript_window_lines": 0,
                "transcript_hydrate_messages": -5,
            }
        }
    ) == (0, 1, 1)


@pytest.mark.asyncio
async def test_load_mounts_only_the_tail_window(): 
    """AC#1: a 500-message load mounts the newest window, not the history."""
    app = WindowHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(_messages(500))
        await transcript.refresh_messages()
        await _settle(pilot)

        mounted = _mounted_message_ids(transcript)
        assert len(mounted) <= DEFAULT_TRANSCRIPT_WINDOW_MESSAGES, (
            f"the load window must bound mounted rows (mounted {len(mounted)})"
        )
        assert mounted[-1] == "m499", "the newest message must be mounted"
        assert mounted == [f"m{i}" for i in range(500 - len(mounted), 500)], (
            "the window must be the contiguous newest suffix, in order"
        )
        assert len(transcript._messages) == 500, "the store snapshot is untouched"
        assert transcript._is_following_tail()


@pytest.mark.asyncio
async def test_window_lines_budget_caps_long_messages():
    """A few very long messages hit the line budget before the message cap."""
    app = WindowHarness(window_messages=40, window_lines=60)
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(_messages(60, lines=20))
        await transcript.refresh_messages()
        await _settle(pilot)

        mounted = _mounted_message_ids(transcript)
        assert 3 <= len(mounted) <= 5, (
            f"~23 estimated rows per message must cap the window (got {len(mounted)})"
        )


@pytest.mark.asyncio
async def test_kill_switch_mounts_every_row():
    """`transcript_window_messages = 0` restores the pre-task behavior."""
    app = WindowHarness(window_messages=0)
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(_messages(120))
        await transcript.refresh_messages()
        await _settle(pilot)

        assert transcript.unhydrated_message_ids() == frozenset()
        assert len(_mounted_message_ids(transcript)) == 120


@pytest.mark.asyncio
async def test_scrolling_to_the_top_hydrates_older_messages_in_order():
    """AC#2: scroll-back mounts the next older chunk, in transcript order."""
    app = WindowHarness(window_messages=10, hydrate_messages=5)
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(_messages(60))
        await transcript.refresh_messages()
        await _settle(pilot)

        mounted_before = _mounted_message_ids(transcript)
        assert mounted_before == [f"m{i}" for i in range(50, 60)]

        transcript.release_anchor()
        transcript.scroll_to(y=0, animate=False)
        assert await _wait_for(
            pilot,
            lambda: len(_mounted_message_ids(transcript)) > len(mounted_before),
        ), "scrolling to the top never hydrated older messages"

        mounted_after = _mounted_message_ids(transcript)
        assert mounted_after[-1] == "m59", "the tail must stay mounted"
        assert mounted_after == [
            f"m{i}" for i in range(60 - len(mounted_after), 60)
        ], "hydrated rows must extend the window as an ordered suffix"
        assert set(mounted_before) <= set(mounted_after)
        # Rows, not just ids: the DOM order matches the message order.
        assert [
            widget.message_id
            for widget in transcript.query(".console-transcript-message")
        ] == mounted_after


@pytest.mark.asyncio
async def test_hydration_keeps_the_reader_on_the_same_content():
    """Rows added ABOVE the reader shift the offset, not the content."""
    app = WindowHarness(window_messages=12, hydrate_messages=6)
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(_messages(60))
        await transcript.refresh_messages()
        await _settle(pilot)
        mounted_before = _mounted_message_ids(transcript)
        assert mounted_before == [f"m{i}" for i in range(48, 60)], (
            "the window must be in play for this test to mean anything"
        )
        height_before = transcript.virtual_size.height

        def _top_visible_message_id() -> str | None:
            region = transcript.content_region
            viewport_top = region.y
            viewport_bottom = region.y + region.height
            for widget in transcript.query(".console-transcript-message"):
                row = widget.region
                if row.y + row.height > viewport_top and row.y < viewport_bottom:
                    return widget.message_id
            return None

        # Park at the very top: the row at the viewport top is then the oldest
        # mounted one, so the expected post-hydration answer is known WITHOUT
        # having to catch the pre-hydration frame.
        transcript.release_anchor()
        transcript.scroll_to(y=0, animate=False)
        assert await _wait_for(
            pilot, lambda: len(_mounted_message_ids(transcript)) > 12
        ), "hydration never fired"
        await _settle(pilot)

        assert not transcript._is_following_tail(), (
            "hydration must not re-attach a detached reader"
        )
        assert _top_visible_message_id() == mounted_before[0], (
            "hydration must keep the same row at the viewport top"
        )
        added = transcript.virtual_size.height - height_before
        assert added > 0
        assert transcript.scroll_y == added, (
            "the offset must shift by exactly the height mounted above the reader"
        )


@pytest.mark.asyncio
async def test_window_shorter_than_the_viewport_fills_until_scrollable():
    """A window that cannot scroll would strand history: fill it instead."""
    app = WindowHarness(window_messages=3, hydrate_messages=3)
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(_messages(60, lines=1))
        await transcript.refresh_messages()
        await _settle(pilot, times=20)

        assert transcript.max_scroll_y > 0, (
            "the transcript must end up scrollable so older rows are reachable"
        )
        mounted = _mounted_message_ids(transcript)
        assert len(mounted) > 3, "the 3-message window must have filled"
        assert transcript.unhydrated_message_ids(), (
            "the fill must stop once scrollable, not mount the whole history"
        )
        assert mounted == [f"m{i}" for i in range(60 - len(mounted), 60)]
        assert transcript._is_following_tail(), "the fill must not detach the reader"
        assert transcript.scroll_y == transcript.max_scroll_y


@pytest.mark.asyncio
async def test_selecting_an_unhydrated_message_hydrates_it_first():
    """A jump target outside the window is hydrated before it is selected."""
    app = WindowHarness(window_messages=10, hydrate_messages=5)
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(_messages(60))
        await transcript.refresh_messages()
        await _settle(pilot)

        target = "m20"
        assert target in transcript.unhydrated_message_ids()
        assert target not in _mounted_message_ids(transcript)

        transcript.select_message(target)
        assert await _wait_for(
            pilot, lambda: target in _mounted_message_ids(transcript)
        ), "the jump target was never hydrated"
        assert transcript.selected_message_id == target
        assert target not in transcript.unhydrated_message_ids()
        mounted = _mounted_message_ids(transcript)
        assert mounted == [f"m{i}" for i in range(20, 60)], (
            "hydrating a target must mount it and everything after it"
        )
        assert await _wait_for(
            pilot, lambda: bool(transcript.query(".console-transcript-action-row"))
        ), "the hydrated target must show its action row"


@pytest.mark.asyncio
async def test_keyboard_selection_never_lands_on_an_unhydrated_row():
    """`k` from the oldest mounted row stays inside the mounted window."""
    app = WindowHarness(window_messages=10, hydrate_messages=5)
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(_messages(60))
        await transcript.refresh_messages()
        await _settle(pilot)

        oldest_mounted = _mounted_message_ids(transcript)[0]
        assert oldest_mounted != "m0", "older history must be windowed out"
        assert transcript.unhydrated_message_ids()
        transcript.select_message(oldest_mounted)
        await _settle(pilot)
        transcript.action_select_previous()
        await _settle(pilot)

        assert transcript.selected_message_id == oldest_mounted, (
            "selection must clamp at the oldest MOUNTED row"
        )
        assert transcript.selected_message_id not in transcript.unhydrated_message_ids()


@pytest.mark.asyncio
async def test_hydration_stops_at_the_low_watermark():
    """AC#3: hydration refuses to run once the mounted height reaches the mark."""
    app = WindowHarness(low=60, high=80, window_messages=6, hydrate_messages=2)
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(_messages(60))
        await transcript.refresh_messages()
        await _settle(pilot)

        transcript.release_anchor()
        for _ in range(30):
            transcript.scroll_to(y=0, animate=False)
            await pilot.pause()
        await _settle(pilot)

        assert transcript.unhydrated_message_ids(), (
            "hydration must stop while history remains, not mount everything"
        )
        assert transcript.virtual_size.height <= 80, (
            "hydration must not push the transcript past the high watermark"
        )


async def _hydrated_over_the_high_mark(pilot, app) -> ConsoleTranscript:
    """Load windowed, hydrate scrollback, then arm watermarks it exceeds.

    Pruning starts disabled (``high = 0``) so the window and the hydration are
    observable first; arming the marks afterwards is the pruning suite's own
    idiom for putting a mounted transcript over the line deterministically.
    """
    transcript = app.query_one(ConsoleTranscript)
    transcript.set_messages(_messages(40))
    await transcript.refresh_messages()
    await _settle(pilot)
    assert len(_mounted_message_ids(transcript)) == 10, "the window must be in play"

    # Hydrate without selecting: a selected message is protected from the
    # watermark walk on its own (TASK-1365), which would mask what this
    # fixture is here to exercise.
    assert transcript.ensure_message_hydrated("m15")
    assert await _wait_for(pilot, lambda: "m15" in _mounted_message_ids(transcript)), (
        "the scrollback the reader asked for was never hydrated"
    )
    # Detach WITHOUT parking at the top, so arming the marks below is the only
    # thing that changes -- no extra hydration in between.
    transcript.release_anchor()
    transcript.scroll_to(y=transcript.max_scroll_y / 2, animate=False)
    await _settle(pilot)
    assert transcript.virtual_size.height > 40, "the fixture must exceed the marks"

    app.app_config["chat_defaults"]["prune_high_watermark"] = 40
    app.app_config["chat_defaults"]["prune_low_watermark"] = 25
    await transcript.refresh_messages()
    await _settle(pilot)
    return transcript


@pytest.mark.asyncio
async def test_pruning_does_not_take_back_hydrated_scrollback_while_detached():
    """Hydrated rows survive the watermark walk while the reader reads them."""
    app = WindowHarness(low=25, high=0, window_messages=10, hydrate_messages=5)
    async with app.run_test() as pilot:
        transcript = await _hydrated_over_the_high_mark(pilot, app)

        mounted = _mounted_message_ids(transcript)
        assert "m15" in mounted, (
            "the watermark walk must not delete hydrated scrollback under a "
            "detached reader"
        )
        assert mounted == [f"m{i}" for i in range(15, 40)]
        assert not transcript._pruned_message_ids

        # AC#3: the bound is restored the moment the reader follows the tail
        # again -- protection is scoped to the detached reader, not permanent.
        transcript.jump_to_latest()
        assert await _wait_for(
            pilot, lambda: transcript.virtual_size.height <= 40, attempts=60
        ), (
            "returning to the tail must let the watermarks trim the hydrated "
            f"head (height {transcript.virtual_size.height})"
        )
        assert transcript._pruned_message_ids, "the trim must be recorded"
        assert _mounted_message_ids(transcript)[-1] == "m39"


@pytest.mark.asyncio
async def test_hydrate_prune_cannot_oscillate():
    """The invariant: no message is hydrated, pruned, and hydrated again.

    The reader is pinned at the top for many frames with the transcript over
    the high watermark -- the only state in which hydration and pruning could
    chase each other. Oscillation shows up either as an id leaving the mounted
    set after it arrived, or as a mounted set that never settles.
    """
    app = WindowHarness(low=25, high=0, window_messages=10, hydrate_messages=5)
    async with app.run_test() as pilot:
        transcript = await _hydrated_over_the_high_mark(pilot, app)

        seen: set[str] = set(_mounted_message_ids(transcript))
        samples: list[tuple[str, ...]] = []
        for _ in range(40):
            transcript.scroll_to(y=0, animate=False)
            await pilot.pause()
            mounted = tuple(_mounted_message_ids(transcript))
            samples.append(mounted)
            dropped = seen - set(mounted)
            assert not dropped, (
                f"mounted rows must never be taken back while scrolled up: {dropped}"
            )
            seen |= set(mounted)

        assert transcript.virtual_size.height > 40, (
            "the fixture must stay over the high mark -- the state where "
            "pruning could undo a hydration"
        )
        assert samples[-5:] == [samples[-1]] * 5, (
            "the mounted set must reach a fixed point, not keep churning"
        )
        assert transcript.unhydrated_message_ids(), (
            "hydration must stay refused above the low mark, not creep forward"
        )


@pytest.mark.asyncio
async def test_session_switch_reestablishes_the_window():
    """Hydrated scrollback does not follow the reader into another session."""
    app = WindowHarness(window_messages=10, hydrate_messages=5)
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(_messages(60))
        await transcript.refresh_messages()
        await _settle(pilot)

        transcript.select_message("m10")
        assert await _wait_for(
            pilot, lambda: "m10" in _mounted_message_ids(transcript)
        )

        other = [_msg(f"other{i}", ConsoleMessageRole.ASSISTANT) for i in range(60)]
        transcript.set_messages(other)
        await transcript.refresh_messages()
        await _settle(pilot)

        mounted = _mounted_message_ids(transcript)
        assert len(mounted) <= 10, "the new session gets a fresh tail window"
        assert mounted[-1] == "mother59"
        assert transcript.unhydrated_message_ids() == frozenset(
            f"mother{i}" for i in range(60 - len(mounted))
        )


@pytest.mark.asyncio
async def test_streaming_ticks_do_not_reestablish_the_window():
    """An overlapping ingest (the 0.2s tick, a send) keeps hydrated scrollback."""
    app = WindowHarness(window_messages=10, hydrate_messages=5)
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        history = _messages(60)
        transcript.set_messages(history)
        await transcript.refresh_messages()
        await _settle(pilot)

        transcript.select_message("m30")
        assert await _wait_for(
            pilot, lambda: "m30" in _mounted_message_ids(transcript)
        )
        hydrated = set(_mounted_message_ids(transcript))

        # The streaming tick re-sets the same list; then a send appends.
        transcript.set_messages(list(history))
        await transcript.refresh_messages()
        await _settle(pilot)
        assert set(_mounted_message_ids(transcript)) >= hydrated

        transcript.note_follow_intent()
        transcript.set_messages(history + [_msg(60, ConsoleMessageRole.USER)])
        await transcript.refresh_messages()
        await _settle(pilot)

        mounted = set(_mounted_message_ids(transcript))
        assert hydrated <= mounted, "a send must not re-window the reader's scrollback"
        assert "m60" in mounted
        assert transcript.unhydrated_message_ids() == frozenset(
            f"m{i}" for i in range(30)
        ), "the untouched head of the window must stay windowed out"


@pytest.mark.asyncio
async def test_hydration_step_is_sized_to_the_watermark_headroom():
    """A hydration step must not hand the watermark walk the rows it mounted.

    Measured before the step budget existed: a 3-message window under a 20/40
    configuration hydrated its full 10-message chunk, crossed the high mark,
    and had 9 of those 10 messages pruned one tick later — mount work spent to
    produce rows that were immediately (and permanently) discarded.
    """
    app = WindowHarness(low=20, high=40, window_messages=3, hydrate_messages=10)
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(_messages(60))
        await transcript.refresh_messages()
        await _settle(pilot, times=20)

        assert transcript._pruned_message_ids == set(), (
            "hydration must stay inside the watermark headroom instead of "
            "mounting rows the prune walk then throws away"
        )
        mounted = _mounted_message_ids(transcript)
        assert 3 <= len(mounted) <= 5, f"expected a minimal step, mounted {mounted}"
        assert transcript.virtual_size.height <= 40


@pytest.mark.asyncio
async def test_jump_to_latest_releases_protection_without_a_scroll_change():
    """The jump pill must reclaim scrollback even when it moves nothing.

    A reader detached at the very bottom (release without scrolling) gives
    `scroll_end` nothing to do, so the scroll watcher never fires — the jump
    itself has to drop the protection latch or the watermarks stay blocked for
    the rest of the session.
    """
    app = WindowHarness(low=25, high=0, window_messages=10, hydrate_messages=5)
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(_messages(40))
        await transcript.refresh_messages()
        await _settle(pilot)

        assert transcript.ensure_message_hydrated("m15")
        await transcript.refresh_messages()
        await _settle(pilot)
        transcript.release_anchor()  # detached, still parked at the bottom
        await _settle(pilot)
        scroll_at_bottom = transcript.scroll_y

        app.app_config["chat_defaults"]["prune_high_watermark"] = 40
        app.app_config["chat_defaults"]["prune_low_watermark"] = 25
        await transcript.refresh_messages()
        await _settle(pilot)
        assert transcript.virtual_size.height > 40, "protection must hold here"

        transcript.jump_to_latest()
        assert transcript.scroll_y == scroll_at_bottom, (
            "this test only means something while the jump moves nothing"
        )
        assert await _wait_for(
            pilot, lambda: transcript.virtual_size.height <= 40, attempts=60
        ), (
            "the jump must release protection so the watermarks can trim "
            f"(height {transcript.virtual_size.height})"
        )
