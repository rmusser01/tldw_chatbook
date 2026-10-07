"""Painted card gestures through controller settlement and real permission owners."""

import asyncio
import pytest
from textual import on
from textual.widgets import Button

from Tests.Agents import test_mcp_tool_provider as mcp_helpers
from Tests.Chat.test_console_approval_scope_journeys import owner_journey
from Tests.UI.test_approval_interaction import (
    _PaintedCardHarness,
    _set_payload,
    _assert_painted,
)
from tldw_chatbook.Agents.run_context import use_run_id, use_tool_call_id
from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import ChatApprovalCard

running_loop = mcp_helpers.running_loop

pytestmark = pytest.mark.bootstrap_profile


class JourneyApp(_PaintedCardHarness):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.payloads = []

    def show(self, payload):
        if payload:
            self.payloads.append(payload)
            _set_payload(
                self.query_one(ChatApprovalCard), payload, round_id=payload["round_id"]
            )
        else:
            self.query_one(ChatApprovalCard).set_batch([], timeout_seconds=0)

    @on(ChatApprovalCard.ApprovalDecided)
    def _capture_decision(self, event):
        self.controller.resolve_pending_approval(
            event.decisions, round_id=event.round_id
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("action", ["approve", "deny"])
async def test_painted_grouped_batch_releases_once_and_dispatches_only_consent(
    tmp_path,
    running_loop,
    theme,
    action,
):
    controller, session, provider, service, transport = await asyncio.to_thread(
        owner_journey, tmp_path, running_loop
    )
    tool = provider.list_catalog()[0].id
    pending = [
        provider.pending_gate_for(tool, {"query": value})
        for value in ("first", "second")
    ]
    app = JourneyApp(controller)
    app.theme = theme
    controller.app = app
    controller.set_pending_approval = app.show
    # A finite test-only bound protects cleanup; no production deadline changes.
    controller.mcp_approval_timeout_seconds = lambda: 10.0

    def worker():
        with use_run_id("journey"):
            answers = controller.request_mcp_approvals(pending, session_id=session.id)
            provider.apply_batch_decisions("journey", answers)
            results = []
            for index, row in enumerate(pending):
                with use_tool_call_id(str(index)):
                    results.append(provider.invoke(tool, row.arguments))
            controller.store.persistence.db.close()
            return answers, results

    async with app.run_test(size=(80, 24)) as pilot:
        task = asyncio.create_task(asyncio.to_thread(worker))
        try:
            for _ in range(50):
                await pilot.pause(0.05)
                if app.payloads:
                    break
            assert app.payloads
            payload = app.payloads[-1]
            row = payload["view"].rows[0]
            assert row.call_count == 2 and len(row.argument_sets) == 2
            card = app.query_one(ChatApprovalCard)
            card.focus_first_decision()
            await pilot.press("enter")
            assert not app.decided and not transport.execute_calls
            button = card.query_one(f"#approval-{action}-all", Button)
            _assert_painted(app, button)
            painted = " ".join(
                "".join(segment.text for segment in strip)
                for strip in app.screen._compositor.render_strips()
            )
            assert (
                "2" in str(button.label)
                and ("Allow" if action == "approve" else "Deny") in painted
            )
            await pilot.click(button)
            button.press()  # A queued repeat must not settle or dispatch twice.
            answers, results = await asyncio.wait_for(task, timeout=12)
            await pilot.pause()
            assert len(app.decided) == 1
            assert dict(answers) == {
                tool: "approve_once" if action == "approve" else "deny"
            }
            assert [result.ok for result in results] == [action == "approve"] * 2
            assert len(transport.execute_calls) == (2 if action == "approve" else 0)
            assert not card.display
            assert not service._session_approvals
            assert (
                controller.approval_feedback.snapshot(session.id, "journey")[
                    0
                ].decision_state
                == "accepted"
            )
        finally:
            if not task.done():
                for payload in app.payloads:
                    controller.resolve_pending_approval(
                        {tool: "deny"}, round_id=payload["round_id"]
                    )
                await asyncio.wait_for(task, timeout=12)
