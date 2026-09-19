import pytest

from tldw_chatbook.Workspaces.conversation_attention import (
    ConversationAttentionFact as Fact,
)
from tldw_chatbook.Workspaces.conversation_attention import (
    present_conversation_attention,
)


def test_approval_overrides_unread_without_losing_explanation():
    view = present_conversation_attention(
        (Fact("unread", "Unread"), Fact("approval", "Approval required")),
        custom_icon="💡",
    )
    assert view.icon == "✋"
    assert view.label == "Approval required"
    assert view.summary == "Approval required · Unread"
    assert present_conversation_attention((), custom_icon="💡").icon == "💡"


@pytest.mark.parametrize(
    "kind,icon,ascii_icon",
    [
        ("approval", "✋", "[approve]"),
        ("blocked", "⛔", "[blocked]"),
        ("failed", "✗", "[failed]"),
        ("running", "⟳", "[running]"),
        ("paused", "⏸", "[paused]"),
        ("stopped", "⏹", "[stopped]"),
        ("unread", "✉", "[unread]"),
        ("ready", "✓", "[ready]"),
        ("outcome_unknown", "🔔", "[new]"),
    ],
)
def test_representative_vocabulary(kind, icon, ascii_icon):
    facts = (Fact(kind, "State"),)
    assert present_conversation_attention(facts).icon == icon
    assert present_conversation_attention(facts, ascii_mode=True).icon == ascii_icon


def test_priority_is_deterministic_and_duplicate_facts_collapse():
    facts = (
        Fact("unread", "Unread"),
        Fact("failed", "Failed"),
        Fact("unread", "Unread"),
    )
    view = present_conversation_attention(facts)
    assert view == present_conversation_attention(tuple(reversed(facts)))
    assert view.summary == "Failed · Unread"
    assert view.icon == "✗"


def test_ordinary_chat_and_custom_icon_fallbacks():
    assert present_conversation_attention(()).icon == "💬"
    assert present_conversation_attention((), ascii_mode=True).icon == "[chat]"
    assert (
        present_conversation_attention((), custom_icon="💡", ascii_mode=True).icon
        == "[icon]"
    )


def test_hidden_rows_keep_semantic_unread_and_tree_keeps_custom_appearance():
    from tldw_chatbook.Workspaces.conversation_browser_state import (
        ConsoleConversationBrowserInputRow,
        build_console_conversation_browser_state,
    )
    from tldw_chatbook.Workspaces.workspace_tree_state import build_workspace_tree_state

    row = ConsoleConversationBrowserInputRow(
        row_key="c",
        conversation_id="c",
        native_session_id=None,
        title="Reminder",
        scope_type="workspace",
        workspace_id="w",
        workspace_label="Work",
        attention=(Fact("unread", "Unread"),),
        manual_unread=True,
        icon="💡",
        color="#123456",
    )
    child = build_workspace_tree_state(workspaces=(("w", "Work"),), rows=(row,))[
        0
    ].conversations[0]
    assert (child.icon, child.color, child.attention, child.manual_unread) == (
        row.icon,
        row.color,
        row.attention,
        True,
    )
    from dataclasses import replace

    state = build_console_conversation_browser_state(
        rows=(replace(row, scope_type="global"),), active_workspace_id=None
    )
    assert any(
        section.run_marker == "✉"
        or any(group.run_marker == "✉" for group in section.groups)
        for section in state.sections
    )


def test_blocked_controller_state_is_not_reported_as_running():
    from types import SimpleNamespace

    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_chat_models import ConsoleRunStatus

    controller = SimpleNamespace(
        _pending_approvals={},
        _unvisited_outcomes={},
        _live_busy_session_ids=lambda: set(),
        activity_for=lambda _: SimpleNamespace(queue_paused=False),
        run_state_for=lambda _: SimpleNamespace(status=ConsoleRunStatus.BLOCKED),
    )
    assert ConsoleChatController.conversation_attention_for(controller, "s") == (
        Fact("blocked", "Blocked"),
    )


@pytest.mark.parametrize("saved", [False, True])
def test_stuck_receipt_keeps_intervention_priority_over_running_and_unread(saved):
    from types import SimpleNamespace

    from tldw_chatbook.Chat.console_activity_receipts import ConsoleActivityReceipt
    from tldw_chatbook.UI.Console_Modules.workspace import ConsoleWorkspaceController
    from tldw_chatbook.Workspaces.conversation_browser_state import (
        ConsoleConversationBrowserInputRow,
    )

    cid = "conversation" if saved else None
    receipt = ConsoleActivityReceipt(
        "activity",
        "fleet",
        "outcome",
        1,
        "session",
        cid,
        "run",
        None,
        "stuck",
        "2026-09-18T00:00:00Z",
    )
    row = ConsoleConversationBrowserInputRow(
        row_key="row",
        conversation_id=cid,
        native_session_id="session",
        title="Chat",
        scope_type="global",
        workspace_id=None,
        workspace_label="Global",
    )
    owner = SimpleNamespace(
        _manual_unread_for_rows=lambda _: {cid: True},
        _console_chat_store=SimpleNamespace(
            sessions=lambda: [
                SimpleNamespace(id="session", persisted_conversation_id=cid)
            ]
        ),
        _console_chat_controller=SimpleNamespace(
            conversation_attention_for=lambda _: (Fact("running", "Running"),)
        ),
        app_instance=SimpleNamespace(
            console_runtime=SimpleNamespace(
                activity_receipts=SimpleNamespace(unseen_snapshot=lambda: (receipt,))
            )
        ),
        _console_fleet_unseen_ids=lambda: {cid} if saved else set(),
    )
    (projected,) = ConsoleWorkspaceController._conversation_attention_rows(
        owner, (row,)
    )
    view = present_conversation_attention(projected.attention)
    assert view.icon == "⛔"
    assert view.label == "Stuck — needs attention"
    assert "Running" in view.summary and "Unread" in view.summary
