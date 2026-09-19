"""Queued approval gestures belong to the batch the user actually saw."""

from copy import deepcopy

import pytest
from textual.widgets import Button

from Tests.UI.test_console_mcp_approval import _CardHarnessApp, _sample_calls
from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import ChatApprovalCard


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["approve-all", "deny-all", "submit"])
@pytest.mark.parametrize(
    "transition", ["round", "calls", "round_trip", "clear", "finishing"]
)
async def test_queued_toolbar_press_cannot_act_on_a_changed_batch(action, transition):
    app = _CardHarnessApp()
    async with app.run_test() as pilot:
        card = app.query_one(ChatApprovalCard)
        calls = _sample_calls()
        card.set_batch(calls, timeout_seconds=0, round_id="original")
        await pilot.pause()
        button = card.query_one(f"#approval-{action}", Button)
        # Button.press publishes asynchronously. Replace the batch before the
        # real public message bubbles to the card, without fabricating an event.
        button.press()
        if transition == "clear":
            card.set_batch([], timeout_seconds=0)
        elif transition == "finishing":
            card.set_batch(
                calls, timeout_seconds=0, round_id="original", phase="finishing"
            )
        elif transition == "calls":
            changed = deepcopy(calls)
            changed[0]["arguments"] = {"query": "replacement target"}
            card.set_batch(changed, timeout_seconds=0, round_id="original")
        else:
            card.set_batch(calls, timeout_seconds=0, round_id="replacement")
            if transition == "round_trip":
                card.set_batch(calls, timeout_seconds=0, round_id="original")
        for select in card._batch_selects:
            select.value = "deny" if action == "approve-all" else "approve_once"
        expected = [select.value for select in card._batch_selects]
        await pilot.pause()
        assert app.decided == [], "the old Submit decided a different batch"
        assert [select.value for select in card._batch_selects] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["submit", "fast-approve", "fast-deny"])
async def test_queued_repeated_submit_decides_the_batch_only_once(action):
    app = _CardHarnessApp()
    async with app.run_test() as pilot:
        card = app.query_one(ChatApprovalCard)
        card.set_batch([_sample_calls()[0]], timeout_seconds=0, round_id="one")
        await pilot.pause()
        selector = (
            "#approval-submit" if action == "submit" else f".approval-row-{action}"
        )
        button = card.query_one(selector, Button)
        button.press()
        button.press()
        await pilot.pause()
        assert len(app.decided) == 1
        assert app.decided_round_ids == ["one"]


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["approve-all", "deny-all", "submit"])
async def test_unchanged_resync_keeps_a_queued_toolbar_action_valid(action):
    app = _CardHarnessApp()
    async with app.run_test() as pilot:
        card = app.query_one(ChatApprovalCard)
        calls = _sample_calls()
        card.set_batch(calls, timeout_seconds=0, round_id="same")
        await pilot.pause()
        for select in card._batch_selects:
            select.value = "deny" if action == "approve-all" else "approve_once"
        card.query_one(f"#approval-{action}", Button).press()
        card.set_batch(calls, timeout_seconds=0, round_id="same")
        await pilot.pause()
        if action == "submit":
            assert len(app.decided) == 1 and app.decided_round_ids == ["same"]
        else:
            expected = "deny" if action == "deny-all" else "approve_once"
            assert all(select.value == expected for select in card._batch_selects)
            assert not app.decided


@pytest.mark.asyncio
async def test_submitted_round_stays_inert_but_a_fresh_round_can_submit():
    app = _CardHarnessApp()
    async with app.run_test() as pilot:
        card = app.query_one(ChatApprovalCard)
        calls = [_sample_calls()[0]]
        card.set_batch(calls, timeout_seconds=0, round_id="first")
        await pilot.pause()
        card.query_one("#approval-submit", Button).press()
        await pilot.pause()
        assert app.decided_round_ids == ["first"]
        card.set_batch(calls, timeout_seconds=0, round_id="first")
        assert all(s.disabled for s in card._batch_selects)
        for selector in (
            "#approval-submit",
            "#approval-approve-all",
            "#approval-deny-all",
        ):
            button = card.query_one(selector, Button)
            assert button.disabled
            button.press()
        await pilot.pause()
        assert app.decided_round_ids == ["first"]
        card.set_batch(calls, timeout_seconds=0, round_id="second")
        await pilot.pause()
        card.query_one("#approval-submit", Button).press()
        await pilot.pause()
        assert app.decided_round_ids == ["first", "second"]
