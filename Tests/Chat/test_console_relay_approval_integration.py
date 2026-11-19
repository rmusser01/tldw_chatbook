"""Phone and desktop gestures share the real Console approval authority."""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from types import SimpleNamespace

import pytest

from Tests.Agents import test_mcp_tool_provider as mcp_helpers
from Tests.Chat.test_console_approval_scope_journeys import owner_journey
from tldw_chatbook.Agents.mcp_tool_provider import (
    MCPInvocationPolicy,
    restrict_mcp_invocation,
)
from tldw_chatbook.Agents.run_context import use_run_id, use_tool_call_id
from tldw_chatbook.app_served import _served_relay_decide

running_loop = mcp_helpers.running_loop
pytestmark = pytest.mark.bootstrap_profile


def _phone(controller, round_id, decisions):
    child = SimpleNamespace(console_runtime=SimpleNamespace(chat_controller=controller))
    return _served_relay_decide(child, round_id, decisions)


def _request(controller, session, pending, run_id):
    with use_run_id(run_id):
        try:
            return controller.request_mcp_approvals(pending, session_id=session.id)
        finally:
            controller.store.persistence.db.close()


@pytest.mark.parametrize("winner", ["phone", "desktop"])
def test_competing_phone_and_desktop_gestures_settle_and_dispatch_once(
    tmp_path, running_loop, winner
):
    controller, session, provider, service, transport = owner_journey(
        tmp_path, running_loop
    )
    tool = provider.list_catalog()[0].id
    row = provider.pending_gate_for(tool, {"query": "original"})
    expected = "approve_once" if winner == "phone" else "deny"
    received = []

    def show(payload):
        if payload is None:
            return
        round_id = payload["round_id"]
        if winner == "phone":
            assert _phone(controller, round_id, {tool: "approve_once"}) is True
            controller.resolve_pending_approval({tool: "deny"}, round_id=round_id)
        else:
            controller.resolve_pending_approval({tool: "deny"}, round_id=round_id)
            assert _phone(controller, round_id, {tool: "approve_once"}) is False
        # Competing and repeated gestures arrive before the host can settle.
        assert _phone(controller, round_id, {tool: "always_allow"}) is False
        state = controller._pending_approval_rounds[round_id]
        assert state["event"].is_set() and dict(state["decisions"]) == {tool: expected}
        received.extend(controller.approval_feedback.snapshot(session.id, "relay-run"))

    controller.set_pending_approval = show
    answers = _request(controller, session, [row], "relay-run")
    assert dict(answers) == {tool: expected}
    assert len(received) == 1 and received[0].decision_state == "received"
    assert not controller._pending_approval_rounds
    provider.apply_batch_decisions("relay-run", answers)
    with use_run_id("relay-run"), use_tool_call_id("call"):
        result = provider.invoke(tool, row.arguments)
    assert result.ok is (winner == "phone")
    assert len(transport.execute_calls) == int(winner == "phone")
    assert not service._session_approvals
    fact = controller.approval_feedback.snapshot(session.id, "relay-run")[0]
    assert fact.decision_state == "accepted" and fact.selected_scope == expected


def test_phone_transport_delivery_can_lose_to_actual_host_cancellation(
    tmp_path, running_loop
):
    controller, session, provider, service, transport = owner_journey(
        tmp_path, running_loop
    )
    tool = provider.list_catalog()[0].id
    row = provider.pending_gate_for(tool, {"query": "cancelled"})
    cancelled = threading.Event()
    policy = MCPInvocationPolicy(
        current=lambda: True,
        deadline=time.monotonic() + 10,
        cancel_event=cancelled,
        allow_approval=True,
        wait_scope=lambda _kind: contextlib.nullcontext(),
    )
    receipts = []

    def show(payload):
        if payload is None:
            return
        assert (
            _phone(controller, payload["round_id"], {tool: "approve_session"}) is True
        )
        receipts.extend(controller.approval_feedback.snapshot(session.id, "cancel-run"))
        cancelled.set()

    controller.set_pending_approval = show
    with restrict_mcp_invocation(policy):
        answers = _request(controller, session, [row], "cancel-run")
    assert receipts[0].decision_state == "received"
    assert dict(answers) == {tool: "deny"} and answers.unresolved_keys == {tool}
    provider.apply_batch_decisions("cancel-run", answers)
    with use_run_id("cancel-run"), use_tool_call_id("call"):
        assert not provider.invoke(tool, row.arguments).ok
    assert not transport.execute_calls and not service._session_approvals
    assert (
        controller.approval_feedback.snapshot(session.id, "cancel-run")[
            0
        ].decision_state
        == "cancelled"
    )


def test_stale_phone_and_desktop_rounds_cannot_answer_a_replacement(
    tmp_path, running_loop
):
    controller, session, provider, _service, transport = owner_journey(
        tmp_path, running_loop
    )
    tool = provider.list_catalog()[0].id
    payloads = []

    def show(payload):
        if payload is None:
            return
        payloads.append(payload)
        current_id = payload["round_id"]
        if len(payloads) == 1:
            assert _phone(controller, current_id, {tool: "approve_once"}) is True
            return
        old_id = payloads[0]["round_id"]
        assert _phone(controller, old_id, {tool: "always_allow"}) is False
        controller.resolve_pending_approval({tool: "always_allow"}, round_id=old_id)
        state = controller._pending_approval_rounds[current_id]
        assert not state["event"].is_set() and not state["decisions"]
        assert payload["view"].rows[0].argument_sets == ({"query": "replacement"},)
        controller.resolve_pending_approval({tool: "deny"}, round_id=current_id)

    controller.set_pending_approval = show
    old = provider.pending_gate_for(tool, {"query": "first"})
    first = _request(controller, session, [old], "first-run")
    new = provider.pending_gate_for(tool, {"query": "replacement"})
    second = _request(controller, session, [new], "second-run")
    assert dict(first) == {tool: "approve_once"} and dict(second) == {tool: "deny"}
    assert payloads[0]["presentation_state"] == {"retired": True}
    assert not controller._pending_approval_rounds and not transport.execute_calls
    assert (
        controller.approval_feedback.snapshot(session.id, "second-run")[
            0
        ].selected_scope
        == "deny"
    )


@pytest.mark.asyncio
async def test_phone_resolution_promotes_fifo_without_accepting_the_next_round(
    tmp_path, running_loop
):
    controller, session, provider, _service, _transport = await asyncio.to_thread(
        owner_journey, tmp_path, running_loop
    )
    tool = provider.list_catalog()[0].id
    shown = []
    controller.set_pending_approval = (
        lambda payload: shown.append(payload) if payload else None
    )
    controller.mcp_approval_timeout_seconds = lambda: 10.0
    first = provider.pending_gate_for(tool, {"query": "first"}, "first-call")
    second = provider.pending_gate_for(tool, {"query": "second"}, "second-call")
    tasks = []

    async def wait_until(predicate):
        async with asyncio.timeout(5):
            while not predicate():
                await asyncio.sleep(0.01)

    try:
        tasks.append(
            asyncio.create_task(
                asyncio.to_thread(_request, controller, session, [first], "first-run")
            )
        )
        await wait_until(lambda: len(shown) == 1)
        first_id = shown[0]["round_id"]
        tasks.append(
            asyncio.create_task(
                asyncio.to_thread(_request, controller, session, [second], "second-run")
            )
        )

        def both_registered():
            with controller._approval_state_lock:
                return len(controller._pending_approval_rounds) == 2

        await wait_until(both_registered)
        assert len(shown) == 1
        assert _phone(controller, first_id, {"first-call": "approve_once"}) is True
        assert dict(await asyncio.wait_for(tasks[0], 5)) == {
            "first-call": "approve_once"
        }
        await wait_until(lambda: len(shown) == 2)
        second_id = shown[-1]["round_id"]
        assert second_id != first_id and not tasks[1].done()
        assert _phone(controller, first_id, {"second-call": "always_allow"}) is False
        controller.resolve_pending_approval(
            {"second-call": "always_allow"}, round_id=first_id
        )
        with controller._approval_state_lock:
            state = controller._pending_approval_rounds[second_id]
            assert not state["event"].is_set() and not state["decisions"]
        controller.resolve_pending_approval({"second-call": "deny"}, round_id=second_id)
        assert dict(await asyncio.wait_for(tasks[1], 5)) == {"second-call": "deny"}
        assert not controller._pending_approval_rounds
    finally:
        with controller._approval_state_lock:
            remaining = [
                (key, tuple(state["names"]))
                for key, state in controller._pending_approval_rounds.items()
            ]
        for round_id, names in remaining:
            controller.resolve_pending_approval(
                {name: "deny" for name in names}, round_id=round_id
            )
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 5)


@pytest.mark.asyncio
async def test_phone_settlement_retires_staged_desktop_choices_and_queued_apply(
    tmp_path, running_loop
):
    from textual import on
    from textual.widgets import Button, Select

    from Tests.UI.test_approval_interaction import _PaintedCardHarness
    from Tests.Web_Server.test_served_relay_publisher import FakeRelayClient
    from tldw_chatbook.Web_Server.served_relay_publisher import (
        ServedRelayPublisher,
        wrap_pending_approval_hook,
    )
    from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import (
        ApprovalActionButton,
        ChatApprovalCard,
    )

    controller, session, provider, service, transport = await asyncio.to_thread(
        owner_journey, tmp_path, running_loop
    )
    tool = provider.list_catalog()[0].id

    class RelayCardApp(_PaintedCardHarness):
        def __init__(self):
            super().__init__()
            self.payloads = []

        def show(self, payload):
            card = self.query_one(ChatApprovalCard)
            if payload is None:
                card.set_batch([], timeout_seconds=0)
                return False
            self.payloads.append(payload)
            card.set_batch(
                payload["calls"],
                timeout_seconds=0,
                round_id=payload["round_id"],
                view=payload["view"],
                presentation_revision=payload["presentation_revision"],
                presentation_state=payload["presentation_state"],
            )
            return True

        @on(ChatApprovalCard.ApprovalDecided)
        def resolve_decision(self, event):
            controller.resolve_pending_approval(
                event.decisions, round_id=event.round_id
            )

    app = RelayCardApp()
    client = FakeRelayClient()
    publisher = ServedRelayPublisher(client)
    publisher.attach()
    controller.app = app
    controller.set_pending_approval = wrap_pending_approval_hook(publisher, app.show)
    controller.mcp_approval_timeout_seconds = lambda: 10.0
    tasks = []
    async with app.run_test(size=(120, 40)) as pilot:

        async def wait_for_payload(count):
            async with asyncio.timeout(5):
                while len(app.payloads) < count:
                    await pilot.pause(0.05)
            await pilot.pause()

        try:
            first = provider.pending_gate_for(tool, {"query": "first"})
            tasks.append(
                asyncio.create_task(
                    asyncio.to_thread(
                        _request, controller, session, [first], "staged-run"
                    )
                )
            )
            await wait_for_payload(1)
            card = app.query_one(ChatApprovalCard)
            await pilot.click(card.query_one(".approval-more-options", Button))
            card.query_one(Select).value = "approve_session"
            await pilot.pause()
            assert card._draft.submit_map() == {tool: "approve_session"}
            assert not app.decided and not transport.execute_calls
            old_apply = ApprovalActionButton.Pressed(
                card.query_one("#approval-submit", ApprovalActionButton)
            )
            first_id = app.payloads[0]["round_id"]
            assert _phone(controller, first_id, {tool: "deny"}) is True
            answers = await asyncio.wait_for(tasks[0], 5)
            await pilot.pause()
            assert dict(answers) == {tool: "deny"} and not card.display
            assert app.payloads[0]["presentation_state"] == {"retired": True}
            provider.apply_batch_decisions("staged-run", answers)
            with use_run_id("staged-run"), use_tool_call_id("first-call"):
                result = await asyncio.to_thread(provider.invoke, tool, first.arguments)
            assert not result.ok and not service._session_approvals

            second = provider.pending_gate_for(tool, {"query": "replacement"})
            tasks.append(
                asyncio.create_task(
                    asyncio.to_thread(
                        _request, controller, session, [second], "replacement-run"
                    )
                )
            )
            await wait_for_payload(2)
            second_id = app.payloads[-1]["round_id"]
            assert card._draft.submit_map() == {tool: "approve_once"}
            card.post_message(old_apply)
            await pilot.pause()
            assert not tasks[1].done() and not app.decided
            assert _phone(controller, first_id, {tool: "always_allow"}) is False
            await pilot.click(card.query_one(".approval-row-fast-approve", Button))
            answers = await asyncio.wait_for(tasks[1], 5)
            assert dict(answers) == {tool: "approve_once"}
            assert app.decided_round_ids == [second_id]
            provider.apply_batch_decisions("replacement-run", answers)
            with use_run_id("replacement-run"), use_tool_call_id("second-call"):
                result = await asyncio.to_thread(
                    provider.invoke, tool, second.arguments
                )
            assert result.ok and len(transport.execute_calls) == 1
            assert not service._session_approvals
            await publisher.wait_idle(timeout=2)
            assert [kind for kind, _ in client.frames].count(
                "relay.approval.raised"
            ) == 2
            assert [kind for kind, _ in client.frames].count(
                "relay.approval.cleared"
            ) == 2
        finally:
            for payload in app.payloads:
                controller.resolve_pending_approval(
                    {tool: "deny"}, round_id=payload["round_id"]
                )
            if tasks:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True), 5
                )
            await publisher.aclose()
