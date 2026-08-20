"""TASK-1365: height-based pruning of the native Console transcript.

Long Console sessions mount one widget tree per message with no bound. When
the transcript's virtual height crosses ``[chat_defaults] prune_high_watermark``
the oldest message rows are dropped from the VIEW until the remainder fits
under ``prune_low_watermark`` — the store keeps the full history, the
streaming row is never pruned, and the scroll position is preserved (or
re-anchored when following the tail). Pruning is a persistent view window:
refresh/reconcile must never resurrect pruned rows.
"""

import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Widgets.Console.console_transcript import (
    DEFAULT_PRUNE_HIGH_WATERMARK,
    DEFAULT_PRUNE_LOW_WATERMARK,
    ConsoleTranscript,
    get_console_prune_watermarks,
)


class PruneHarness(App):
    """Small viewport so a handful of messages overflow it."""

    CSS = "ConsoleTranscript { height: 10; }"

    def __init__(self, *, low: int = 25, high: int = 40) -> None:
        super().__init__()
        self.app_config = {
            "chat_defaults": {
                "prune_high_watermark": high,
                "prune_low_watermark": low,
            }
        }

    def compose(self) -> ComposeResult:
        yield ConsoleTranscript(id="console-native-transcript")


def _msg(
    i: int | str,
    role: ConsoleMessageRole = ConsoleMessageRole.ASSISTANT,
    *,
    status: str = "completed",
) -> ConsoleChatMessage:
    return ConsoleChatMessage(
        role=role,
        content="\n".join(f"line {i}.{j}" for j in range(4)),
        id=f"m{i}",
        status=status,
    )


def _messages(n: int) -> list[ConsoleChatMessage]:
    return [
        _msg(i, ConsoleMessageRole.USER if i % 2 == 0 else ConsoleMessageRole.ASSISTANT)
        for i in range(n)
    ]


def _mounted_message_ids(transcript: ConsoleTranscript) -> list[str]:
    # Class query, not a type query: assistant rows may render as
    # ConsoleMarkdownMessage (TASK-1990) while other roles stay plain.
    return [
        widget.message_id
        for widget in transcript.query(".console-transcript-message")
    ]


async def _wait_for(pilot, predicate, *, attempts: int = 50) -> bool:
    for _ in range(attempts):
        await pilot.pause()
        if predicate():
            return True
    return False


def test_get_console_prune_watermarks_defaults_and_clamps():
    assert get_console_prune_watermarks(None) == (
        DEFAULT_PRUNE_LOW_WATERMARK,
        DEFAULT_PRUNE_HIGH_WATERMARK,
    )
    assert get_console_prune_watermarks({}) == (
        DEFAULT_PRUNE_LOW_WATERMARK,
        DEFAULT_PRUNE_HIGH_WATERMARK,
    )
    # low is clamped to <= high.
    assert get_console_prune_watermarks(
        {"chat_defaults": {"prune_high_watermark": 10, "prune_low_watermark": 99}}
    ) == (10, 10)
    # Invalid values fall back to the defaults.
    assert get_console_prune_watermarks(
        {"chat_defaults": {"prune_high_watermark": "x", "prune_low_watermark": None}}
    ) == (DEFAULT_PRUNE_LOW_WATERMARK, DEFAULT_PRUNE_HIGH_WATERMARK)
    # A non-dict chat_defaults section is ignored.
    assert get_console_prune_watermarks({"chat_defaults": "nope"}) == (
        DEFAULT_PRUNE_LOW_WATERMARK,
        DEFAULT_PRUNE_HIGH_WATERMARK,
    )


@pytest.mark.asyncio
async def test_pruning_drops_oldest_rows_over_high_watermark():
    """Virtual height over the high mark drops oldest rows down to the low mark."""
    # Seed with pruning disabled so the pre-prune state is observable, then
    # arm the watermarks and refresh to trigger the prune.
    app = PruneHarness(low=25, high=0)
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(_messages(14))
        await transcript.refresh_messages()
        await pilot.pause()

        assert len(_mounted_message_ids(transcript)) == 14, (
            "seed must mount every row before pruning runs"
        )
        assert transcript.virtual_size.height > 40, "seed must cross the high mark"

        app.app_config["chat_defaults"]["prune_high_watermark"] = 40
        await transcript.refresh_messages()

        assert await _wait_for(
            pilot, lambda: len(_mounted_message_ids(transcript)) < 14
        ), "pruning never fired"

        mounted = _mounted_message_ids(transcript)
        assert "m0" not in mounted
        assert mounted, "pruning must keep the newest rows"
        assert mounted[-1] == "m13"
        assert len(mounted) == 14 - len(transcript._pruned_message_ids)
        # TASK-15453: explicit order, not just first/last presence -- the
        # surviving rows must render in the same relative order as the
        # (unpruned) message store.
        expected_order = [
            f"m{i}" for i in range(14) if f"m{i}" not in transcript._pruned_message_ids
        ]
        assert mounted == expected_order
        # Pruned down to (at most) the high watermark, erring on keeping rows.
        assert transcript.virtual_size.height <= 40
        # The store snapshot the widget was handed is untouched.
        assert len(transcript._messages) == 14


@pytest.mark.asyncio
async def test_pruning_is_view_only_store_keeps_full_history():
    """messages_for_session still returns everything after pruning (AC4)."""
    store = ConsoleChatStore()
    session = store.create_session()
    for message in _messages(14):
        store.append_message(
            session.id,
            role=message.role,
            content=message.content,
        )

    app = PruneHarness(low=25, high=40)
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(store.messages_for_session(session.id))
        await transcript.refresh_messages()

        assert await _wait_for(
            pilot, lambda: bool(transcript._pruned_message_ids)
        ), "pruning never fired"

        remaining = store.messages_for_session(session.id)
        assert len(remaining) == 14
        # Exports render the full history, pruned rows included.
        plain = transcript.to_plain_text(width=40)
        assert "line 0.0" in plain
        assert "line 13.3" in plain


@pytest.mark.asyncio
async def test_pruning_preserves_scroll_position_when_scrolled_up():
    """A scrolled-up reader keeps the same content in view across a prune."""
    app = PruneHarness(low=40, high=60)
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(_messages(30))
        await transcript.refresh_messages()
        assert await _wait_for(
            pilot, lambda: bool(transcript._pruned_message_ids)
        ), "initial prune never fired"

        def _top_visible_message_id() -> str | None:
            region = transcript.content_region
            viewport_top = region.y
            viewport_bottom = region.y + region.height
            for widget in transcript.query(".console-transcript-message"):
                row = widget.region
                if row.y + row.height > viewport_top and row.y < viewport_bottom:
                    return widget.message_id
            return None

        # Detach and read the newest content (a small later prune drops the
        # oldest groups, so these rows are guaranteed to survive it).
        transcript.release_anchor()
        transcript.scroll_to(y=transcript.max_scroll_y, animate=False)
        await pilot.pause()
        top_before = _top_visible_message_id()
        assert top_before is not None
        assert top_before not in transcript._pruned_message_ids
        pruned_count = len(transcript._pruned_message_ids)

        # Grow the transcript past the high mark again (assistant growth, like
        # streaming, must not yank the reader either).
        history = list(transcript._messages) + [
            _msg(100 + i, ConsoleMessageRole.ASSISTANT) for i in range(5)
        ]
        transcript.set_messages(history)
        await transcript.refresh_messages()
        assert await _wait_for(
            pilot, lambda: len(transcript._pruned_message_ids) > pruned_count
        ), "second prune never fired"
        # Let the deferred scroll restoration land.
        for _ in range(5):
            await pilot.pause()

        assert not transcript._is_following_tail(), (
            "pruning must not re-attach a detached reader"
        )
        assert _top_visible_message_id() == top_before, (
            "pruning while scrolled up must keep the same row at the viewport top"
        )


@pytest.mark.asyncio
async def test_pruning_reanchors_when_following_tail():
    """Following the tail: the view stays pinned to the bottom after a prune."""
    app = PruneHarness(low=25, high=40)
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(_messages(14))
        await transcript.refresh_messages()

        assert await _wait_for(
            pilot, lambda: bool(transcript._pruned_message_ids)
        ), "pruning never fired"
        assert await _wait_for(
            pilot,
            lambda: transcript.scroll_y == transcript.max_scroll_y
            and transcript._is_following_tail(),
        ), "tail-follow was not restored after pruning"


@pytest.mark.asyncio
async def test_streaming_row_is_never_pruned():
    """The in-progress streaming row survives even an aggressive prune."""
    app = PruneHarness(low=1, high=5)
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        history = _messages(10) + [
            _msg("stream", ConsoleMessageRole.ASSISTANT, status="streaming")
        ]
        transcript.set_messages(history)
        await transcript.refresh_messages()

        assert await _wait_for(
            pilot, lambda: bool(transcript._pruned_message_ids)
        ), "pruning never fired"

        assert "mstream" not in transcript._pruned_message_ids
        assert "mstream" in _mounted_message_ids(transcript)


@pytest.mark.asyncio
async def test_pruning_disabled_when_high_watermark_nonpositive():
    """prune_high_watermark <= 0 disables pruning entirely."""
    app = PruneHarness(low=1, high=0)
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages(_messages(12))
        await transcript.refresh_messages()
        # Give the (disabled) check a chance to run.
        for _ in range(5):
            await pilot.pause()

        assert transcript._pruned_message_ids == set()
        assert len(_mounted_message_ids(transcript)) == 12


@pytest.mark.asyncio
async def test_refresh_and_recompose_do_not_resurrect_pruned_rows():
    """Pruning is a persistent view window, not a one-shot row removal."""
    app = PruneHarness(low=25, high=40)
    async with app.run_test() as pilot:
        transcript = app.query_one(ConsoleTranscript)
        history = _messages(14)
        transcript.set_messages(history)
        await transcript.refresh_messages()

        assert await _wait_for(
            pilot, lambda: bool(transcript._pruned_message_ids)
        ), "pruning never fired"
        pruned = set(transcript._pruned_message_ids)
        mounted_after_prune = _mounted_message_ids(transcript)

        # Same-messages refresh (the streaming tick re-sets the list).
        transcript.set_messages(list(history))
        await transcript.refresh_messages()
        assert await _wait_for(
            pilot, lambda: _mounted_message_ids(transcript) == mounted_after_prune
        )
        assert not (set(_mounted_message_ids(transcript)) & pruned)

        # A full recompose derives rows through the same pruned window.
        transcript.refresh(recompose=True)
        assert await _wait_for(
            pilot, lambda: _mounted_message_ids(transcript) == mounted_after_prune
        )
        assert not (set(_mounted_message_ids(transcript)) & pruned)
