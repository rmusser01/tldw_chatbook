"""Feedback projection only paints the current owning session and round."""

import pytest

pytestmark = pytest.mark.bootstrap_profile


def test_projection_uses_latest_snapshot_and_never_replaces_new_card():
    from tldw_chatbook.UI.Console_Modules.approval_feedback import (
        ApprovalFeedbackController,
    )
    from tldw_chatbook.Chat.console_approval_feedback import ApprovalFeedbackStore
    from tldw_chatbook.Agents.approval_observation import ApprovalObservation
    from Tests.Chat.test_console_approval_feedback import captured

    store = ApprovalFeedbackStore()
    store.bind_round(captured())
    painted = []
    refreshed = []
    controller = ApprovalFeedbackController(
        read_snapshot=store.snapshot,
        active_session=lambda: "s",
        card_identity=lambda: ("next", 2),
        paint_card=lambda text: painted.append(text),
        refresh_status=lambda: refreshed.append(True),
    )
    identity = store.context_for_call("run", "c")
    store.publish(ApprovalObservation(identity, "settled", "accepted"))
    controller.refresh("other", "run")
    assert not painted and not refreshed
    controller.refresh("s", "run")
    assert not painted and refreshed == [True]


def test_tool_disclosure_retains_owner_failure_after_card_is_gone():
    from tldw_chatbook.Chat.console_chat_models import ConsoleActivityPresentation
    from tldw_chatbook.Chat.console_approval_feedback import ApprovalFeedback
    from tldw_chatbook.Agents.approval_observation import ApprovalObservationIdentity
    from tldw_chatbook.Widgets.Console.console_assistant_turn import (
        ConsoleActivityDisclosure,
    )

    fact = ApprovalFeedback(
        ApprovalObservationIdentity("s", "run", "r", 1, "c"),
        decision_state="accepted",
        grant_state="failed",
        execution_state="tool_completed",
        terminal_outcome="success",
    )
    disclosure = ConsoleActivityDisclosure(
        "marker",
        "Read",
        "success",
        tool_presentation=ConsoleActivityPresentation(
            "tool", "Read", "success", approval_feedback=fact
        ),
    )
    assert "Allowed this call; permission was not remembered" in str(
        disclosure.feedback_notice.render()
    )


@pytest.mark.asyncio
async def test_submitted_card_paints_latest_fact_and_replacement_rejects_old_feedback():
    from textual.widgets import Button, Static
    from Tests.Chat.test_console_approval_feedback import captured
    from Tests.UI.test_console_mcp_approval import _CardHarnessApp, _sample_calls
    from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import ChatApprovalCard

    app = _CardHarnessApp()
    async with app.run_test(size=(80, 24)) as pilot:
        card = app.query_one(ChatApprovalCard)
        card.set_batch(
            _sample_calls()[:1],
            timeout_seconds=0,
            round_id="first",
            view=captured("first"),
        )
        await pilot.pause()
        original = card.feedback_identity()
        await pilot.click(card.query_one(".approval-row-fast-approve", Button))
        await pilot.pause()
        card.paint_feedback("Decision received")
        assert "Decision received" in str(
            card.query_one("#approval-title", Static).render()
        )
        card.set_batch(
            _sample_calls()[:1],
            timeout_seconds=0,
            round_id="replacement",
            view=captured("replacement", 2),
        )
        await pilot.pause()
        assert card.feedback_identity() != original
        card.paint_feedback("old fact")
        assert "old fact" not in str(card.query_one("#approval-title", Static).render())


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 40), (170, 48)])
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
async def test_full_grant_failure_remains_visible_on_collapsed_tool(size, theme):
    from Tests.UI.test_console_assistant_turn import StyledActivityHarness
    from tldw_chatbook.Widgets.Console.console_assistant_turn import (
        ConsoleActivityDisclosure,
    )
    from tldw_chatbook.Chat.console_chat_models import ConsoleActivityPresentation
    from tldw_chatbook.Chat.console_approval_feedback import ApprovalFeedback
    from tldw_chatbook.Agents.approval_observation import ApprovalObservationIdentity

    fact = ApprovalFeedback(
        ApprovalObservationIdentity("s", "run", "r", 1, "c"),
        decision_state="accepted",
        grant_state="failed",
        execution_state="tool_completed",
        terminal_outcome="success",
    )
    disclosure = ConsoleActivityDisclosure(
        "c",
        "Read",
        "success",
        tool_presentation=ConsoleActivityPresentation(
            "tool", "Read", "success", approval_feedback=fact
        ),
    )
    app = StyledActivityHarness(disclosure)
    app.theme = theme
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        notice = app.query_one(".console-approval-feedback")
        assert notice.display and notice.region.right <= size[0]
        painted = " ".join(
            "".join(segment.text for segment in strip)
            for strip in app.screen._compositor.render_strips()
        )
        assert "Allowed this call; permission was not remembered" in " ".join(
            painted.split()
        )
