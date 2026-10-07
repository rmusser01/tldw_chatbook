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


@pytest.mark.parametrize(
    "status", ["success", "failed", "timed_out", "stopped", "blocked"]
)
def test_terminal_member_disclosure_does_not_retain_group_starting(status):
    from tldw_chatbook.Chat.console_chat_models import ConsoleActivityPresentation
    from tldw_chatbook.Chat.console_approval_feedback import ApprovalFeedback
    from tldw_chatbook.Agents.approval_observation import ApprovalObservationIdentity
    from tldw_chatbook.Widgets.Console.console_assistant_turn import (
        ConsoleActivityDisclosure,
    )

    fact = ApprovalFeedback(
        ApprovalObservationIdentity("s", "run", "r", 1, "group"),
        decision_state="accepted",
        selected_scope="approve_session",
        grant_state="applied",
        applied_scope="approve_session",
        execution_state="dispatch_started",
    )
    presentation = ConsoleActivityPresentation(
        "tool", "Read", status, call_id="member", approval_feedback=fact
    )
    disclosure = ConsoleActivityDisclosure(
        "marker", "Read", status, tool_presentation=presentation
    )
    text = str(disclosure.feedback_notice.render())
    assert "Starting" not in text and "Running" not in text
    assert "Until Chatbook exits" in text


def test_actual_group_projection_keeps_terminal_and_pending_members_distinct():
    from tldw_chatbook.Agents.agent_models import AgentStep
    from tldw_chatbook.Agents.approval_observation import ApprovalObservationIdentity
    from tldw_chatbook.Chat.console_approval_feedback import ApprovalFeedback
    from tldw_chatbook.Chat.console_chat_models import ConsoleActivityPresentation
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_tool_activity import ConsoleToolActivity
    from tldw_chatbook.Widgets.Console.console_assistant_turn import (
        ConsoleActivityDisclosure,
    )
    from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript

    store = ConsoleChatStore()
    session = store.ensure_session()
    activity = ConsoleToolActivity(store, session.id)
    for call_id in ("finished", "pending"):
        activity.observe(
            AgentStep(index=1, kind="tool_proposed", tool_name="read", call_id=call_id),
            1,
        )
    fact = ApprovalFeedback(
        ApprovalObservationIdentity(session.id, "run", "r", 1, "group"),
        decision_state="accepted",
        selected_scope="approve_session",
        grant_state="applied",
        applied_scope="approve_session",
        execution_state="dispatch_started",
        sequence=2,
    )
    activity.feedback((fact,), call_keys=lambda identity: ("finished", "pending"))
    assert activity.complete(
        AgentStep(
            index=2,
            kind="tool_result",
            tool_name="read",
            call_id="finished",
            result="done",
        ),
        ConsoleActivityPresentation("tool", "read", "success"),
        "done",
        None,
        record_trajectory=False,
    )
    transcript = ConsoleTranscript()
    for message in store.messages_for_session(session.id):
        presentation = message.activity_presentation
        disclosure = ConsoleActivityDisclosure(
            message.id,
            presentation.label,
            presentation.status,
            tool_presentation=presentation,
        )
        collapsed = str(disclosure.feedback_notice.render())
        expanded = str(
            transcript._activity_components(message, ()).detail_widgets[0].render()
        )
        assert (
            "Until Chatbook exits" in collapsed and "Until Chatbook exits" in expanded
        )
        for text in (collapsed, expanded):
            assert ("Starting" in text) == (presentation.call_id == "pending")
        assert presentation.approval_feedback == fact
