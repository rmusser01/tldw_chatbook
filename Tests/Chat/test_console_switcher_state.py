"""Pure Console session-switcher result contracts."""

from datetime import datetime, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from tldw_chatbook.Chat.console_switcher_state import (
    ActivityGroup,
    ConsoleSwitcherActivitySignal,
    ConsoleSwitcherEntry,
    SwitcherTargetKind,
    UnavailableSessionNotice,
    _matches,
    build_console_active_results,
    build_console_switcher_entries,
    console_history_section,
    filter_console_active_results,
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


def _receipt(**overrides):
    defaults = dict(
        activity_id="activity-1",
        status="done",
        session_id=None,
        conversation_id="conv-1",
        created_at="2026-07-04T12:00:00+00:00",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _active(rows, *, receipts=(), signals=()):
    return build_console_active_results(
        rows,
        receipts=receipts,
        controller_signals=signals,
        profile_authority="profile-a",
        authority_token="runtime-a",
        now=datetime(2026, 7, 4, 13, tzinfo=timezone.utc),
    )


def test_active_merges_duplicate_conversation_tabs_and_prefers_actionable_target():
    rows = (
        _row(
            row_key="conv-1",
            native_session_id="session-current",
            selected=True,
            title="Shared work",
        ),
        _row(
            row_key="conv-1",
            native_session_id="session-running",
            selected=False,
            title="Shared work duplicate",
            run_marker="[*]",
            updated_sort="2026-07-04T12:30:00+00:00",
        ),
    )

    results = _active(rows)

    assert len(results) == 1
    entry = results[0]
    assert isinstance(entry, ConsoleSwitcherEntry)
    assert entry.stable_result_key == "conversation:profile-a:conv-1"
    assert entry.group is ActivityGroup.WORKING
    assert entry.target is not None
    assert entry.target.kind is SwitcherTargetKind.NATIVE_SESSION
    assert entry.target.session_id == "session-running"
    assert entry.multiplicity == 2


def test_active_reduces_same_target_receipts_and_shell_without_losing_evidence():
    row = _row(native_session_id="session-1", selected=False, run_marker="")
    receipts = (
        _receipt(activity_id="done-1", session_id="session-1", status="done"),
        _receipt(
            activity_id="failed-2",
            session_id="session-1",
            status="failed",
            created_at="2026-07-04T12:05:00+00:00",
        ),
    )

    entry = _active((row,), receipts=receipts)[0]

    assert isinstance(entry, ConsoleSwitcherEntry)
    assert entry.group is ActivityGroup.WAITING_FOR_YOU
    assert entry.activity_state == "failed"
    assert entry.target is not None
    assert [receipt.activity_id for receipt in entry.target.receipts] == [
        "done-1",
        "failed-2",
    ]
    assert entry.multiplicity == 2


def test_active_orders_group_then_existing_star_then_time():
    rows = (
        _row(
            row_key="running-unstarred",
            conversation_id="running-unstarred",
            native_session_id="run-u",
            title="Zulu",
            run_marker="[*]",
            starred=False,
        ),
        _row(
            row_key="running-starred",
            conversation_id="running-starred",
            native_session_id="run-s",
            title="Alpha",
            run_marker="[*]",
            starred=True,
            updated_sort="2026-07-01T00:00:00+00:00",
        ),
        _row(
            row_key="approval",
            conversation_id="approval",
            native_session_id="approval-s",
            title="Needs me",
            run_marker="[!]",
        ),
    )

    results = _active(rows)

    assert [result.group for result in results] == [
        ActivityGroup.WAITING_FOR_YOU,
        ActivityGroup.WORKING,
        ActivityGroup.WORKING,
    ]
    assert results[1].title == "Alpha"


def test_session_only_receipts_aggregate_into_explicit_unavailable_notice():
    receipts = (
        _receipt(
            activity_id="gone-done",
            conversation_id=None,
            session_id="gone-session",
            status="done",
        ),
        _receipt(
            activity_id="gone-stuck",
            conversation_id=None,
            session_id="gone-session",
            status="stuck",
            created_at="2026-07-04T12:30:00+00:00",
        ),
    )

    result = _active((), receipts=receipts)[0]

    assert isinstance(result, UnavailableSessionNotice)
    assert result.group is ActivityGroup.WAITING_FOR_YOU
    assert result.primary_status == "stuck"
    assert result.stable_result_key.endswith(":gone-session")
    assert "+1" in result.subtitle


def test_domain_semantic_search_uses_safe_metadata_and_filters():
    rows = (
        _row(
            row_key="approval",
            conversation_id="approval",
            native_session_id="approval-s",
            title="Release review",
            workspace_label="Platform",
            run_marker="[!]",
        ),
        _row(
            row_key="running",
            conversation_id="running",
            native_session_id="running-s",
            title="Indexer",
            workspace_label="Research",
            run_marker="[*]",
        ),
    )
    results = _active(rows)

    assert [item.title for item in filter_console_active_results(results, "waiting on me")] == [
        "Release review"
    ]
    assert [item.title for item in filter_console_active_results(results, "is:working workspace:research")] == [
        "Indexer"
    ]
    assert filter_console_active_results(results, "is:invented") == ()


def test_controller_signal_is_content_free_and_can_raise_open_shell_priority():
    rows = (_row(native_session_id="session-1", selected=True),)
    signals = (
        ConsoleSwitcherActivitySignal(
            source_key="controller:native:session-1:paused",
            state="paused",
            session_id="session-1",
            occurred_at="2026-07-04T12:30:00+00:00",
        ),
    )

    result = _active(rows, signals=signals)[0]

    assert result.group is ActivityGroup.WAITING_FOR_YOU
    assert result.activity_state == "paused"


def test_history_calendar_sections_obey_local_dates_dst_and_invalid_values():
    zone = ZoneInfo("America/Los_Angeles")
    now = datetime(2026, 3, 9, 0, 30, tzinfo=zone)

    assert console_history_section(
        "2026-03-09T06:45:00+00:00", now=now, local_timezone=zone
    ) == "Yesterday"
    assert console_history_section(
        "2026-03-09T08:45:00+00:00", now=now, local_timezone=zone
    ) == "Today"
    assert console_history_section(
        "2026-03-12T08:45:00+00:00", now=now, local_timezone=zone
    ) == "Today"
    assert console_history_section(
        "2026-03-03T12:00:00+00:00", now=now, local_timezone=zone
    ) == "Previous 7 days"
    assert console_history_section("not-a-time", now=now, local_timezone=zone) == "Older"


def test_entries_are_recent_first_with_active_pinned():
    rows = [
        _row(
            row_key="old",
            conversation_id="old",
            title="Old chat",
            updated_sort="2026-06-01T00:00:00+00:00",
        ),
        _row(
            row_key="new",
            conversation_id="new",
            title="New chat",
            updated_sort="2026-07-04T00:00:00+00:00",
        ),
        _row(
            row_key="active",
            conversation_id="active",
            title="Active chat",
            selected=True,
            updated_sort="2026-05-01T00:00:00+00:00",
        ),
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
    assert (
        build_console_switcher_entries(rows, query="REFACTOR")[0].title
        == "API refactor plan"
    )
    # Token can match workspace label or status, not just title.
    assert [
        e.title for e in build_console_switcher_entries(rows, query="workspace 1 api")
    ] == ["API refactor plan"]


def test_entries_dedupe_by_row_key_and_cap_at_limit():
    rows = [
        _row(row_key="dup", conversation_id="dup", title="First wins"),
        _row(row_key="dup", conversation_id="dup", title="Second loses"),
    ]
    hits = build_console_switcher_entries(rows)
    assert len(hits) == 1 and hits[0].title == "First wins"
    many = [
        _row(
            row_key=f"k{i}",
            conversation_id=f"k{i}",
            title=f"Chat {i}",
            updated_sort=f"2026-07-04T{i:02d}:00:00+00:00",
        )
        for i in range(30)
    ]
    assert len(build_console_switcher_entries(many, limit=20)) == 20


def test_limit_zero_returns_no_entries():
    assert build_console_switcher_entries([_row()], limit=0) == ()


def test_subtitle_joins_available_parts():
    entry = build_console_switcher_entries([_row()])[0]
    # TASK-356: the switcher shows the shared friendly vocabulary, not raw status.
    assert entry.subtitle == "Workspace 1 · saved chat · 2m"
    bare = build_console_switcher_entries(
        [
            _row(
                row_key="x",
                workspace_label="",
                status="",
                updated_label="",
                updated_sort="",
            )
        ]
    )[0]
    assert bare.subtitle == "saved chat"


def test_matcher_tolerates_none_fields_without_raising():
    # Rows aren't validated and are assembled by several builders, so a
    # None in title/workspace_label/status must not raise TypeError when
    # joined for searching -- an empty query (no tokens) should match
    # trivially, and a real query should just fail to match.
    none_row = _row(
        title=None,  # type: ignore[arg-type]
        workspace_label=None,  # type: ignore[arg-type]
        status=None,  # type: ignore[arg-type]
    )
    assert _matches(none_row, []) is True
    assert _matches(none_row, ["anything"]) is False


def test_switcher_status_uses_saved_chat_vocabulary_not_in_progress():
    """TASK-356: the switcher must not label idle saved conversations
    'in-progress' (raw status) — it must use the same friendly vocabulary
    the rail shows ('saved chat'), so the two surfaces don't contradict."""
    entries = build_console_switcher_entries(
        [_row(status="in-progress", updated_label="")]
    )
    subtitle = entries[0].subtitle
    assert "in-progress" not in subtitle
    assert "saved chat" in subtitle.lower()


def test_switcher_status_maps_membership_and_session_states():
    saved = build_console_switcher_entries(
        [_row(row_key="a", status="workspace-thread", updated_label="")]
    )[0].subtitle
    active = build_console_switcher_entries(
        [_row(row_key="b", status="active", updated_label="")]
    )[0].subtitle
    assert "saved chat" in saved.lower()
    assert "saved chat" in active.lower()


def test_switcher_shows_recency_when_updated_label_absent():
    """TASK-356: the rail carries age labels; the switcher must too. When a
    row lacks a precomputed updated_label, derive it from updated_sort so
    recognition works where it matters most."""
    from datetime import datetime, timezone

    now = datetime(2026, 7, 4, 12, 0, 0, tzinfo=timezone.utc)
    entries = build_console_switcher_entries(
        [_row(updated_label="", updated_sort="2026-07-04T11:58:00+00:00")],
        now=now,
    )
    # 2 minutes before `now`.
    assert "2m" in entries[0].subtitle


def test_search_matches_the_friendly_status_label_now_shown():
    """TASK-356 follow-up (Qodo #4): the subtitle now shows the friendly status
    ('saved chat'), so a query for the VISIBLE word must match — searching the
    raw 'in-progress' string would silently return nothing for the exact states
    this change made user-facing. The raw status stays searchable too."""
    rows = [_row(row_key="a", conversation_id="a", status="in-progress")]
    assert [e.row_key for e in build_console_switcher_entries(rows, query="saved")] == [
        "a"
    ]
    # Back-compat: the underlying status token still matches.
    assert [
        e.row_key for e in build_console_switcher_entries(rows, query="in-progress")
    ] == ["a"]


def test_open_agent_entry_projects_fleet_queue_and_current_state():
    """Removing any source fleet field must make the switcher lose its state."""
    entry = build_console_switcher_entries(
        [
            _row(
                native_session_id="session-1",
                conversation_id="conv-1",
                source_kind="native",
                status="active session",
                selected=True,
                run_marker="●",
                queued_count=2,
            )
        ]
    )[0]

    assert entry.section == "open"
    assert entry.state_label == "CURRENT · RUNNING · 2 QUEUED"
    assert entry.openable is True


def test_saved_unavailable_entry_is_not_presented_as_an_open_agent():
    entry = build_console_switcher_entries(
        [
            _row(
                native_session_id=None,
                source_kind="persisted",
                openable=False,
            )
        ]
    )[0]

    assert entry.section == "saved"
    assert entry.state_label == "UNAVAILABLE · saved chat"
    assert entry.openable is False


def test_open_and_saved_sections_remain_contiguous_when_saved_row_is_selected():
    entries = build_console_switcher_entries(
        [
            _row(
                row_key="open",
                native_session_id="session-open",
                source_kind="native",
                selected=False,
            ),
            _row(row_key="saved-current", selected=True),
            _row(row_key="saved-other", selected=False),
        ]
    )

    assert [entry.section for entry in entries] == ["open", "saved", "saved"]


def test_semantic_filters_match_operational_state_without_title_keywords():
    rows = [
        _row(
            row_key="running",
            conversation_id="running",
            native_session_id="session-running",
            source_kind="native",
            title="Release planning",
            run_marker="●",
            queued_count=2,
        ),
        _row(
            row_key="approval",
            conversation_id="approval",
            native_session_id="session-approval",
            source_kind="native",
            title="Budget review",
            run_marker="◆",
        ),
        _row(
            row_key="saved",
            conversation_id="saved",
            native_session_id=None,
            source_kind="persisted",
            title="Migration notes",
            workspace_label="Research Lab",
        ),
    ]

    assert [
        e.row_key for e in build_console_switcher_entries(rows, query="running")
    ] == ["running"]
    assert [
        e.row_key for e in build_console_switcher_entries(rows, query="approval")
    ] == ["approval"]
    assert [
        e.row_key for e in build_console_switcher_entries(rows, query="queued")
    ] == ["running"]
    assert [
        e.row_key for e in build_console_switcher_entries(rows, query="is:saved")
    ] == ["saved"]
    assert [
        e.row_key
        for e in build_console_switcher_entries(rows, query="workspace:research")
    ] == ["saved"]
