"""TASK-294 (P5 minor): direct coverage for `ChatTaskCards.sync_state`.

The batch branch was exercised only through the full ChatScreen integration
path; nothing drove the wrapper directly, so a regression in its own
routing (approval payload -> card, display gating) could hide behind the
bigger harness's noise.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from textual.app import ComposeResult

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp

import tldw_chatbook.UI.Screens.chat_screen as chat_screen_module
from tldw_chatbook.UI.Screens.chat_screen_state import TaskResumeState
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import ChatApprovalCard
from tldw_chatbook.Widgets.Chat_Widgets.chat_task_cards import ChatTaskCards
from tldw_chatbook.Widgets.Chat_Widgets.watchlists_operation_card import (
    WatchlistsOperationCard,
)


class _CardsHarness(ConsolidatedCSSApp):
    def compose(self) -> ComposeResult:
        yield ChatTaskCards(id="chat-task-cards")


@pytest.mark.asyncio
async def test_sync_state_routes_an_approval_batch_to_the_card():
    """A pending_approval payload mounts visible rows; clearing hides all."""
    app = _CardsHarness()
    async with app.run_test() as pilot:
        await pilot.pause()
        cards = app.query_one(ChatTaskCards)
        assert cards.display is False, "precondition: hidden until pending"

        cards.sync_state(
            TaskResumeState(
                pending_approval={
                    "calls": [
                        {
                            "llm_name": "read_file",
                            "server_label": "Built-in",
                            "tool_name": "read_file",
                            "arguments": {"path": "a.md"},
                            "call_id": "c1",
                        },
                        {
                            "llm_name": "read_file",
                            "server_label": "Built-in",
                            "tool_name": "read_file",
                            "arguments": {"path": "b.md"},
                            "call_id": "c2",
                        },
                    ],
                    "timeout_seconds": 45.0,
                    "round_id": "round-x",
                }
            )
        )
        await pilot.pause()
        await pilot.pause()

        assert cards.display is True
        card = app.query_one(ChatApprovalCard)
        rows = list(card.query(".approval-row"))
        assert len(rows) == 2, "one row per call id (TASK-1861 keying)"
        assert card._batch_round_id == "round-x", (
            "the round id must ride through, or a decision resolves the "
            "wrong round"
        )

        cards.sync_state(TaskResumeState())
        await pilot.pause()
        assert cards.display is False, "cleared state must hide the surface"


@pytest.mark.asyncio
async def test_sync_state_mounts_only_canonical_watchlists_receipts():
    app = _CardsHarness()
    async with app.run_test() as pilot:
        cards = app.query_one(ChatTaskCards)
        state = TaskResumeState(
            followed_watchlists_operations=(
                "local:watchlist_run:7",
                "local:briefing:9",
            )
        )

        cards.sync_state(
            state,
            operation_rows={
                "local:watchlist_run:7": {
                    "id": "local:watchlist_run:7",
                    "status_detail": "running",
                    "destination": "runs",
                },
                "local:briefing:9": {
                    "id": "local:briefing:9",
                    "status_detail": "empty",
                    "destination": "artifacts",
                },
            },
        )
        await pilot.pause()
        await pilot.pause()

        assert cards.display is True
        assert {card.operation_id for card in cards.query(WatchlistsOperationCard)} == {
            "local:watchlist_run:7",
            "local:briefing:9",
        }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operation_id", "operation", "expected_call", "expected_followed"),
    [
        (
            "local:watchlist_run:7",
            {"source": {"id": "local:subscription:4"}},
            ("check", 4),
            ("local:watchlist_run:17",),
        ),
        (
            "local:briefing:9",
            {"collection": {"id": "local:watchlist:5"}},
            ("briefing", 5),
            ("local:briefing:19",),
        ),
    ],
)
async def test_receipt_retry_consumes_tool_shaped_canonical_entity_ids(
    operation_id, operation, expected_call, expected_followed
):
    calls: list[tuple[str, int]] = []

    class Coordinator:
        async def accept_checks(self, source_ids):
            calls.append(("check", source_ids[0]))
            return [{"run_id": 17}]

        async def accept_briefing(self, watchlist_id):
            calls.append(("briefing", watchlist_id))
            return {"id": 19}

    screen = ChatScreen.__new__(ChatScreen)
    screen.app_instance = SimpleNamespace(
        watchlists_operation_coordinator=Coordinator()
    )
    screen._watchlists_operation_rows = {operation_id: operation}
    followed: list[tuple[str, ...]] = []
    screen._follow_console_watchlists_operations = followed.append

    await ChatScreen._retry_console_watchlists_operation(screen, operation_id)

    assert calls == [expected_call]
    assert followed == [expected_followed]


@pytest.mark.asyncio
async def test_poll_keeps_following_an_active_receipt_through_a_missing_refresh(
    monkeypatch,
):
    operation_id = "local:watchlist_run:7"

    def unavailable_refresh(_ids):
        raise RuntimeError("temporary local read failure")

    screen = SimpleNamespace(
        is_mounted=True,
        _task_resume_state=TaskResumeState(
            followed_watchlists_operations=(operation_id,)
        ),
        _watchlists_operation_rows={
            operation_id: {"id": operation_id, "status_detail": "running"}
        },
        _read_console_watchlists_operation_rows=unavailable_refresh,
        sync_task_resume_state=lambda: None,
    )
    sleeps: list[float] = []

    async def stop_after_sleep(seconds):
        sleeps.append(seconds)
        screen.is_mounted = False

    monkeypatch.setattr(chat_screen_module.asyncio, "sleep", stop_after_sleep)

    await ChatScreen._poll_console_watchlists_operations(screen)

    assert sleeps == [2.0]


def test_stop_following_removes_controller_and_screen_state_without_cancelling():
    operation_id = "local:watchlist_run:7"

    class Controller:
        def __init__(self):
            self.followed = [operation_id]
            self.cancel_calls = 0

        def unfollow_watchlists_operation(self, receipt_id):
            self.followed.remove(receipt_id)
            return True

        def cancel(self):
            self.cancel_calls += 1

    controller = Controller()
    task_state = TaskResumeState(
        followed_watchlists_operations=(operation_id,)
    )
    screen = SimpleNamespace(
        _console_chat_controller=controller,
        _task_resume_state=task_state,
        _watchlists_operation_rows={operation_id: {"id": operation_id}},
    )

    def set_task_resume_state(state):
        screen._task_resume_state = state

    screen.set_task_resume_state = set_task_resume_state

    ChatScreen.on_watchlists_operation_stop_following(
        screen,
        WatchlistsOperationCard.StopFollowingRequested(operation_id),
    )

    assert controller.followed == []
    assert controller.cancel_calls == 0
    assert screen._task_resume_state.followed_watchlists_operations == ()
    assert screen._watchlists_operation_rows == {}


def test_briefing_receipt_inspect_posts_exact_artifact_context():
    operation_id = "local:briefing:9"
    posted = []
    screen = SimpleNamespace(post_message=posted.append)

    ChatScreen.on_watchlists_operation_inspect(
        screen,
        WatchlistsOperationCard.InspectRequested(operation_id, "artifacts"),
    )

    assert len(posted) == 1
    assert isinstance(posted[0], NavigateToScreen)
    assert posted[0].screen_context == {
        "section": "artifacts",
        "backend": "local",
        "briefing_id": operation_id,
    }


def test_run_receipt_inspect_preserves_exact_runs_context():
    operation_id = "local:watchlist_run:7"
    posted = []
    screen = SimpleNamespace(post_message=posted.append)

    ChatScreen.on_watchlists_operation_inspect(
        screen,
        WatchlistsOperationCard.InspectRequested(operation_id, "runs"),
    )

    assert posted[0].screen_context == {
        "section": "runs",
        "backend": "local",
        "run_id": operation_id,
    }
