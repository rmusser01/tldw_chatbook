"""Workspace inbox rows derive from current authority, never ambient selection."""

from types import SimpleNamespace

from tldw_chatbook.Chat.console_activity_receipts import ConsoleActivityReceipt
from tldw_chatbook.Chat.console_chat_models import ConsoleRunState, ConsoleRunStatus
from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession


def receipt(key, session=None, conversation=None):
    return ConsoleActivityReceipt(
        key, "ordinary", key, 1, session, conversation, None, None, "done", "now"
    )


def project(sessions=(), receipts=(), statuses=None, activities=None, members=None):
    from tldw_chatbook.Persona_Buddy.inbox import project_workspace_inbox

    return project_workspace_inbox(
        "workspace-a",
        sessions=sessions,
        receipts=receipts,
        run_states=statuses or {},
        activities=activities or {},
        member_titles=members or {},
    )


def test_inbox_filters_history_and_other_workspaces_and_orders_attention_first():
    waiting = ConsoleChatSession(id="waiting", workspace_id="workspace-a")
    running = ConsoleChatSession(id="running", workspace_id="workspace-a")
    history = ConsoleChatSession(id="history", workspace_id="workspace-a")
    outside = ConsoleChatSession(id="outside", workspace_id="workspace-b")
    entries = project(
        (running, history, outside, waiting),
        statuses={
            key: ConsoleRunState(ConsoleRunStatus.STREAMING)
            for key in ("running", "outside", "waiting")
        },
        activities={"waiting": SimpleNamespace(needs_approval=True)},
    )
    assert [(row.binding.target_id, row.group) for row in entries] == [
        ("waiting", "needs_you"),
        ("running", "running"),
    ]


def test_two_results_in_one_conversation_keep_separate_acknowledgement_ids():
    session = ConsoleChatSession(
        id="live", workspace_id="workspace-a", persisted_conversation_id="saved"
    )
    entries = project(
        (session,), (receipt("old", "live", "saved"), receipt("new", "live", "saved"))
    )
    assert [row.receipt_ids for row in entries] == [("old",), ("new",)]
    assert entries[0].key != entries[1].key


def test_repurposed_live_slot_cannot_claim_old_result():
    session = ConsoleChatSession(
        id="live", workspace_id="workspace-a", persisted_conversation_id="different"
    )
    assert project((session,), (receipt("old", "live", "former"),)) == ()


def test_unloaded_result_requires_current_membership_and_keeps_durable_target():
    entries = project(
        receipts=(
            receipt("result", "old-runtime", "saved"),
            receipt("outside", None, "other"),
        ),
        members={"saved": "Research"},
    )
    assert len(entries) == 1
    assert entries[0].title == "Research"
    assert entries[0].binding.conversation_id == "saved"


def test_current_session_workspace_wins_over_stale_membership():
    session = ConsoleChatSession(
        id="live", workspace_id="workspace-b", persisted_conversation_id="saved"
    )
    assert (
        project(
            (session,),
            (receipt("result", "live", "saved"),),
            members={"saved": "Stale"},
        )
        == ()
    )


def test_paused_queue_needs_attention_without_pretending_to_be_a_question():
    session = ConsoleChatSession(id="live", workspace_id="workspace-a")
    entries = project(
        (session,),
        activities={"live": SimpleNamespace(queue_paused=True, queued_count=2)},
    )
    assert entries[0].group == "needs_you"
    assert "paused" in entries[0].summary.lower()
