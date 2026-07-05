"""Pure Console session-switcher result contracts."""

from tldw_chatbook.Chat.console_switcher_state import (
    ConsoleSwitcherEntry,
    build_console_switcher_entries,
)
from tldw_chatbook.Workspaces.conversation_browser_state import (
    ConsoleConversationBrowserInputRow,
)


def _row(**overrides) -> ConsoleConversationBrowserInputRow:
    defaults = dict(
        row_key="conv-1",
        conversation_id="conv-1",
        native_session_id=None,
        title="API refactor plan",
        scope_type="workspace",
        workspace_id="ws-1",
        workspace_label="Workspace 1",
        status="workspace-thread",
        updated_label="2m",
        updated_sort="2026-07-04T11:58:00+00:00",
    )
    defaults.update(overrides)
    return ConsoleConversationBrowserInputRow(**defaults)


def test_entries_are_recent_first_with_active_pinned():
    rows = [
        _row(row_key="old", conversation_id="old", title="Old chat",
             updated_sort="2026-06-01T00:00:00+00:00"),
        _row(row_key="new", conversation_id="new", title="New chat",
             updated_sort="2026-07-04T00:00:00+00:00"),
        _row(row_key="active", conversation_id="active", title="Active chat",
             selected=True, updated_sort="2026-05-01T00:00:00+00:00"),
    ]
    titles = [entry.title for entry in build_console_switcher_entries(rows)]
    assert titles == ["Active chat", "New chat", "Old chat"]
    assert build_console_switcher_entries(rows)[0].is_active is True


def test_query_tokens_all_must_match_case_insensitive():
    rows = [
        _row(row_key="a", conversation_id="a", title="Groq testing"),
        _row(row_key="b", conversation_id="b", title="API refactor plan"),
    ]
    hits = build_console_switcher_entries(rows, query="groq test")
    assert [e.title for e in hits] == ["Groq testing"]
    assert build_console_switcher_entries(rows, query="REFACTOR")[0].title == "API refactor plan"
    # Token can match workspace label or status, not just title.
    assert [e.title for e in build_console_switcher_entries(rows, query="workspace 1 api")] == [
        "API refactor plan"
    ]


def test_entries_dedupe_by_row_key_and_cap_at_limit():
    rows = [_row(row_key="dup", conversation_id="dup", title="First wins"),
            _row(row_key="dup", conversation_id="dup", title="Second loses")]
    hits = build_console_switcher_entries(rows)
    assert len(hits) == 1 and hits[0].title == "First wins"
    many = [
        _row(row_key=f"k{i}", conversation_id=f"k{i}", title=f"Chat {i}",
             updated_sort=f"2026-07-04T{i:02d}:00:00+00:00")
        for i in range(30)
    ]
    assert len(build_console_switcher_entries(many, limit=20)) == 20


def test_limit_zero_returns_no_entries():
    assert build_console_switcher_entries([_row()], limit=0) == ()


def test_subtitle_joins_available_parts():
    entry = build_console_switcher_entries([_row()])[0]
    assert entry.subtitle == "Workspace 1 - workspace-thread - 2m"
    bare = build_console_switcher_entries(
        [_row(row_key="x", workspace_label="", status="", updated_label="")]
    )[0]
    assert bare.subtitle == ""
