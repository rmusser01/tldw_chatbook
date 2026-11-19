"""Queued approval gestures belong to the batch the user actually saw."""

from copy import deepcopy
import time
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
from textual import events
from textual.widgets import Button

from Tests.UI.test_console_mcp_approval import _CardHarnessApp, _sample_calls
from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import (
    ApprovalActionButton,
    ChatApprovalCard,
)


async def _wait_for_actionable_card(card, pilot):
    """Wait for actual mounted facts and enabled, laid-out decision controls."""
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        await pilot.pause()
        buttons = [button for button in card.query(Button) if button.display]
        headers = list(card.query(".approval-row-header"))
        if (
            card._information_ready
            and headers
            and all(
                header.region.width > 0 and header.region.height > 0
                for header in headers
            )
            and buttons
            and all(
                not button.disabled
                and button.region.width > 0
                and button.region.height > 0
                for button in buttons
            )
        ):
            return
    raise AssertionError(
        "Approval facts and decision controls did not become actionable"
    )


def test_published_press_keeps_its_generation_after_button_changes(monkeypatch):
    published = []
    monkeypatch.setattr(
        Button,
        "post_message",
        lambda button, message: published.append(message) or True,
    )
    button = ApprovalActionButton("Submit", generation=7)

    assert button.post_message(Button.Pressed(button))
    button.batch_generation = 8

    assert len(published) == 1
    event = published[0]
    assert isinstance(event, ApprovalActionButton.Pressed)
    assert event.button is button and event.batch_generation == 7
    assert event.handler_name == "on_button_pressed"


@pytest.mark.parametrize("fast_first", [False, True])
def test_mixed_submission_paths_lock_before_publishing_and_decide_once(fast_first):
    select = SimpleNamespace(value="approve_session", disabled=False)
    fast = SimpleNamespace(disabled=False)
    reason = SimpleNamespace(value="Keep this private.", disabled=False)
    toolbar = [SimpleNamespace(disabled=False) for _ in range(3)]
    card = SimpleNamespace(
        _draft=None,
        _information_ready=True,
        _view_retired=False,
        _presentation_state=None,
        is_attached=True,
        _close_details=Mock(),
        _details_buttons=[],
        _batch_is_raw_shell=[False],
        _raw_reviewed=set(),
        _batch_legal_values=[["approve_once", "approve_session", "deny"]],
        _batch_submitted=False,
        _batch_phase="pending",
        _batch_names=["call-1"],
        _batch_round_id="round-1",
        _batch_selects=[select],
        _batch_fast_buttons=[fast],
        _batch_reason_inputs=[reason],
        query_one=Mock(
            side_effect=lambda selector, *args: SimpleNamespace(
                update=lambda text: None
            )
            if selector == "#approval-title"
            else toolbar[
                int(selector == "#approval-approve-all")
                + 2 * int(selector == "#approval-deny-all")
            ]
        ),
        ApprovalDecided=ChatApprovalCard.ApprovalDecided,
    )
    card._disable_batch_submit_controls = MethodType(
        ChatApprovalCard._disable_batch_submit_controls, card
    )
    card._denial_reasons = MethodType(ChatApprovalCard._denial_reasons, card)
    card._can_apply_batch = MethodType(ChatApprovalCard._can_apply_batch, card)
    published = []

    def publish(message):
        assert card._batch_submitted
        card._close_details.assert_called_once_with()
        assert select.disabled and fast.disabled and reason.disabled
        assert all(b.disabled for b in toolbar)
        published.append(message)

    card.post_message = publish

    def normal():
        ChatApprovalCard._submit_batch_decisions(card)

    def quick():
        ChatApprovalCard._submit_fast_decision(card, "deny")

    first, second = (quick, normal) if fast_first else (normal, quick)

    first()
    second()
    first()

    assert len(published) == 1
    assert published[0].round_id == "round-1"
    assert published[0].decisions == {
        "call-1": "deny" if fast_first else "approve_session"
    }
    assert published[0].decisions.denial_reasons == (
        {"call-1": "Keep this private."} if fast_first else {}
    )


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
        await _wait_for_actionable_card(card, pilot)
        if action == "submit":
            await pilot.click(card.query_one("#approval-more-options-batch", Button))
            await pilot.pause()
        button = card.query_one(f"#approval-{action}", Button)
        assert button.display and not button.disabled
        assert button.region.width > 0 and button.region.height > 0
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
@pytest.mark.parametrize(
    "timing",
    [
        "release-after-replacement",
        "click-after-replacement",
        "down-before-ready",
        "crossed-buttons",
    ],
)
async def test_toolbar_pointer_gesture_cannot_approve_a_replacement_batch(
    timing, monkeypatch
):
    app = _CardHarnessApp()
    async with app.run_test() as pilot:
        card = app.query_one(ChatApprovalCard)
        calls = _sample_calls()
        card.set_batch(calls, timeout_seconds=0, round_id="original")
        await _wait_for_actionable_card(card, pilot)
        button = card.query_one("#approval-approve-all", Button)
        x, y = button.region.center
        assert app.get_widget_at(x, y)[0] is button

        async def pointer(event_type, mouse_button=1):
            # The App synthesizes Click on release. Pilot.mouse_up forwards
            # directly to the screen and skips that part of the real path.
            await app.on_event(
                event_type(
                    None,
                    x,
                    y,
                    0,
                    0,
                    button=mouse_button,
                    shift=False,
                    meta=False,
                    ctrl=False,
                )
            )

        post_message = Button.post_message
        pending_clicks = []

        def hold_click(widget, message):
            if widget is button and isinstance(message, events.Click):
                pending_clicks.append(message)
                return True
            return post_message(widget, message)

        if timing == "down-before-ready":
            card.set_batch(calls, timeout_seconds=0, round_id="replacement")
            assert button.is_disabled and app.get_widget_at(x, y)[0] is button
            await pointer(events.MouseDown)
        else:
            await pointer(events.MouseDown)
            if timing == "click-after-replacement":
                # Delay only the App's synthesized Click until the new controls
                # are actionable. The production button still receives both inputs.
                with monkeypatch.context() as delayed:
                    delayed.setattr(Button, "post_message", hold_click)
                    await pointer(events.MouseUp)
                assert len(pending_clicks) == 1
            else:
                await pilot.pause()
            card.set_batch(calls, timeout_seconds=0, round_id="replacement")
        await _wait_for_actionable_card(card, pilot)
        assert app.get_widget_at(x, y)[0] is button
        if timing == "crossed-buttons":
            await pointer(events.MouseDown, mouse_button=3)
        if pending_clicks:
            assert post_message(button, pending_clicks.pop())
        else:
            await pointer(events.MouseUp)
        await pilot.pause()
        assert app.decided_round_ids == []

        if timing == "crossed-buttons":
            # The newer secondary press remains valid for its matching release.
            await pointer(events.MouseUp, mouse_button=3)
        else:
            await pointer(events.MouseDown)
            await pointer(events.MouseUp)
        await pilot.pause()
        assert app.decided_round_ids == ["replacement"]
        assert app.decided == [
            {
                "mcp__srv_a__search": "approve_once",
                "mcp__srv_b__write": "approve_once",
            }
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["submit", "fast-approve", "fast-deny"])
async def test_queued_repeated_submit_decides_the_batch_only_once(action):
    app = _CardHarnessApp()
    async with app.run_test() as pilot:
        card = app.query_one(ChatApprovalCard)
        card.set_batch([_sample_calls()[0]], timeout_seconds=0, round_id="one")
        await _wait_for_actionable_card(card, pilot)
        if action == "submit":
            await pilot.click(card.query_one(".approval-more-options", Button))
            await pilot.pause()
        selector = (
            "#approval-submit" if action == "submit" else f".approval-row-{action}"
        )
        button = card.query_one(selector, Button)
        assert card._information_ready and not button.disabled, (
            card._information_ready,
            button.disabled,
            button.region,
        )
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
        await _wait_for_actionable_card(card, pilot)
        if action == "submit":
            card.query_one("#approval-more-options-batch", Button).press()
            await pilot.pause()
        for select in card._batch_selects:
            select.value = "deny" if action == "approve-all" else "approve_once"
        card.query_one(f"#approval-{action}", Button).press()
        card.set_batch(calls, timeout_seconds=0, round_id="same")
        await pilot.pause()
        # Approved Task3: bulk decisions commit immediately for this unchanged
        # snapshot. Changed/replaced generations are rejected by the preceding
        # parameterized test, which remains intact.
        expected = "deny" if action == "deny-all" else "approve_once"
        assert app.decided == [
            {
                "mcp__srv_a__search": expected,
                "mcp__srv_b__write": expected,
            }
        ]
        assert app.decided_round_ids == ["same"]
        assert all(select.disabled for select in card._batch_selects)


@pytest.mark.asyncio
async def test_submitted_round_stays_inert_but_a_fresh_round_can_submit():
    app = _CardHarnessApp()
    async with app.run_test() as pilot:
        card = app.query_one(ChatApprovalCard)
        calls = [_sample_calls()[0]]
        card.set_batch(calls, timeout_seconds=0, round_id="first")
        await _wait_for_actionable_card(card, pilot)
        if not card.has_class("approval-options-open"):
            await pilot.click(card.query_one(".approval-more-options", Button))
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
        await _wait_for_actionable_card(card, pilot)
        if not card.has_class("approval-options-open"):
            await pilot.click(card.query_one(".approval-more-options", Button))
            await pilot.pause()
        card.query_one("#approval-submit", Button).press()
        await pilot.pause()
        assert app.decided_round_ids == ["first", "second"]
