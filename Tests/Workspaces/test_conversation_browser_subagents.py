"""[N Sub-Agents] count is threaded through the pure browser-state builder."""
from tldw_chatbook.Workspaces.conversation_browser_state import (
    ConsoleConversationBrowserInputRow, build_console_conversation_browser_state,
)


def _row(conversation_id, title):
    return ConsoleConversationBrowserInputRow(
        row_key=conversation_id, conversation_id=conversation_id,
        native_session_id=None, title=title, scope_type="global",
        workspace_id=None, workspace_label="", updated_sort="2026-07-13T00:00:00Z")


def _all_rows(state):
    rows = []
    for section in state.sections:
        rows.extend(section.rows)
        for group in section.groups:
            rows.extend(group.rows)
    return rows


def test_subagent_count_attaches_to_matching_conversation_row():
    state = build_console_conversation_browser_state(
        rows=[_row("conv-a", "Alpha"), _row("conv-b", "Beta")],
        active_workspace_id=None,
        subagent_counts={"conv-a": 3},
    )
    by_id = {r.conversation_id: r for r in _all_rows(state)}
    assert by_id["conv-a"].subagent_count == 3
    assert by_id["conv-b"].subagent_count == 0


def test_subagent_counts_default_to_zero_when_absent():
    state = build_console_conversation_browser_state(
        rows=[_row("conv-a", "Alpha")], active_workspace_id=None)
    assert _all_rows(state)[0].subagent_count == 0
