"""Tests for the pure unified-row adapter (schedules redesign PR-2, Task 1).

Covers the bucket table (every chip x both primitives x transfer states),
the dual local/server id-space unread resolution (the survey's warning --
a server-mirrored definition's results must still be counted), sort
orders, and search. No Textual, no DB, no asyncio -- `unified_rows.py` is
pure, so these are plain function-call tests.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tldw_chatbook.Scheduling.models import ReminderTask, ScheduleKind
from tldw_chatbook.UI.Screens.scheduling.unified_rows import (
    UnifiedRow,
    build_unified_rows,
    definition_is_armed,
    filter_rows,
    reminder_has_fired,
    sort_rows,
)


def _reminder(**overrides) -> ReminderTask:
    defaults = dict(
        id="rem-1",
        owner_id="local",
        title="Recurring Reminder",
        body="Reminder body text",
        schedule_kind=ScheduleKind.RECURRING,
        cron="0 9 * * *",
        timezone="UTC",
        enabled=True,
        next_run_at=datetime(2026, 9, 5, 9, 0, tzinfo=timezone.utc),
        last_run_at=None,
        transfer_state=None,
    )
    defaults.update(overrides)
    return ReminderTask(**defaults)


def _one_time_reminder(**overrides) -> ReminderTask:
    defaults = dict(
        schedule_kind=ScheduleKind.ONE_TIME,
        run_at=datetime(2026, 9, 5, 9, 0, tzinfo=timezone.utc),
        cron=None,
        timezone=None,
    )
    defaults.update(overrides)
    return _reminder(**defaults)


def _definition(**overrides) -> dict:
    defaults = dict(
        id="def-1",
        server_id=None,
        owner_id="local",
        name="A Definition",
        lifecycle="configured",
        transfer_state=None,
        schedule={"kind": "cron", "cron": "0 9 * * *", "timezone": "UTC"},
        next_run_at="2026-09-05T09:00:00+00:00",
        updated_at="2026-09-01T00:00:00+00:00",
        input={"question": "What changed?"},
    )
    defaults.update(overrides)
    return defaults


def _result(**overrides) -> dict:
    defaults = dict(
        id="res-1",
        definition_id="def-1",
        owner_id="local",
        review_state="unread",
        kind="finding",
    )
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Bucket table -- reminders
# ---------------------------------------------------------------------------


class TestReminderHasFired:
    def test_fired_when_disabled_no_next_run_but_has_run(self):
        task = _reminder(
            enabled=False,
            next_run_at=None,
            last_run_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
        )
        assert reminder_has_fired(task) is True

    def test_not_fired_when_never_run(self):
        task = _reminder(enabled=False, next_run_at=None, last_run_at=None)
        assert reminder_has_fired(task) is False

    def test_not_fired_when_enabled(self):
        task = _reminder(
            enabled=True,
            next_run_at=None,
            last_run_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
        )
        assert reminder_has_fired(task) is False

    def test_not_fired_when_next_run_still_set(self):
        task = _reminder(
            enabled=False,
            next_run_at=datetime(2026, 9, 5, tzinfo=timezone.utc),
            last_run_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
        )
        assert reminder_has_fired(task) is False


@pytest.mark.parametrize(
    "transfer_state", [None, "to_server_pending", "to_server_failed"]
)
def test_enabled_reminder_stays_active_when_not_dormant(transfer_state):
    """spec SS3: Active "includ[es] to_server_pending/to_server_failed...
    they still execute locally" -- these are NOT in DORMANT_TRANSFER_STATES
    and keep arming, same as the definition side (`definition_is_armed`)."""
    task = _reminder(enabled=True, transfer_state=transfer_state)
    rows = build_unified_rows([task], [], [])
    assert rows[0].bucket == "active"


@pytest.mark.parametrize("transfer_state", ["to_server_sent", "from_server_pending"])
def test_enabled_reminder_parks_under_paused_when_dormant(transfer_state):
    """Review round 1, finding 1: `_reminder_bucket` must mirror
    `_definition_bucket`'s dormant fallback (see
    `test_configured_definition_parks_under_paused_when_dormant` below) --
    `PriorityQueue.load()` excludes a dormant transfer_state for BOTH
    primitives (`scheduler/queue.py:96-108`), and
    `list_reminder_tasks(armable_only=True)`/`reminders_due_before`
    already apply the same exclusion on the reminder side. An enabled
    reminder sitting out a dormant transfer is not "armed" -- it does not
    still execute locally -- so it parks under Paused, not Active;
    `transfer_state` still carries the raw dormant value for the badge."""
    task = _reminder(enabled=True, transfer_state=transfer_state)
    rows = build_unified_rows([task], [], [])
    assert rows[0].bucket == "paused"
    assert rows[0].transfer_state == transfer_state


def _bucket_for(kind: str, transfer_state: str | None) -> str:
    """Build one armed (enabled/configured) row of ``kind`` and return its bucket."""
    if kind == "reminder":
        rows = build_unified_rows(
            [_reminder(enabled=True, transfer_state=transfer_state)], [], []
        )
    else:
        definition = _definition(lifecycle="configured", transfer_state=transfer_state)
        rows = build_unified_rows([], [definition], [])
    return rows[0].bucket


@pytest.mark.parametrize("kind", ["reminder", "definition"])
@pytest.mark.parametrize(
    "transfer_state", [None, "to_server_pending", "to_server_failed"]
)
def test_both_primitives_stay_active_when_not_dormant(kind, transfer_state):
    """Pins the symmetry itself (review round 1, finding 2): reminders and
    definitions must agree on which transfer states still count as
    "armed" -- neither primitive gets its own rule."""
    assert _bucket_for(kind, transfer_state) == "active"


@pytest.mark.parametrize("kind", ["reminder", "definition"])
@pytest.mark.parametrize("transfer_state", ["to_server_sent", "from_server_pending"])
def test_both_primitives_park_under_paused_when_dormant(kind, transfer_state):
    """Pins the symmetry itself (review round 1, finding 2): a dormant
    transfer state excludes a row from Active on BOTH primitives, not
    just definitions."""
    assert _bucket_for(kind, transfer_state) == "paused"


def test_disabled_unfired_reminder_is_paused():
    task = _reminder(
        enabled=False, next_run_at=datetime(2026, 9, 5, tzinfo=timezone.utc)
    )
    rows = build_unified_rows([task], [], [])
    assert rows[0].bucket == "paused"


def test_fired_reminder_is_completed():
    task = _reminder(
        enabled=False,
        next_run_at=None,
        last_run_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
    )
    rows = build_unified_rows([task], [], [])
    assert rows[0].bucket == "completed"


# ---------------------------------------------------------------------------
# Bucket table -- definitions
# ---------------------------------------------------------------------------


class TestDefinitionIsArmed:
    @pytest.mark.parametrize(
        "transfer_state", [None, "to_server_pending", "to_server_failed"]
    )
    def test_armed_when_configured_and_not_dormant(self, transfer_state):
        assert definition_is_armed(_definition(transfer_state=transfer_state)) is True

    @pytest.mark.parametrize(
        "transfer_state", ["to_server_sent", "from_server_pending"]
    )
    def test_not_armed_when_dormant(self, transfer_state):
        assert definition_is_armed(_definition(transfer_state=transfer_state)) is False

    @pytest.mark.parametrize("lifecycle", ["paused", "archived", "disabled"])
    def test_not_armed_when_not_configured(self, lifecycle):
        assert definition_is_armed(_definition(lifecycle=lifecycle)) is False


@pytest.mark.parametrize(
    "transfer_state", [None, "to_server_pending", "to_server_failed"]
)
def test_configured_definition_stays_active_when_not_dormant(transfer_state):
    definition = _definition(lifecycle="configured", transfer_state=transfer_state)
    rows = build_unified_rows([], [definition], [])
    assert rows[0].bucket == "active"


@pytest.mark.parametrize("transfer_state", ["to_server_sent", "from_server_pending"])
def test_configured_definition_parks_under_paused_when_dormant(transfer_state):
    """Judgment call (documented in unified_rows._definition_bucket): a
    `configured` definition sitting out a dormant transfer is not counted
    Active (mirrors `list_armable_automation_definitions`'s own gate), and
    `UnifiedRow.bucket` has no 4th "in transfer" state -- it is parked
    under Paused so it still appears under the All chip. `transfer_state`
    still carries the raw dormant value for the caller's own badge."""
    definition = _definition(lifecycle="configured", transfer_state=transfer_state)
    rows = build_unified_rows([], [definition], [])
    assert rows[0].bucket == "paused"
    assert rows[0].transfer_state == transfer_state


def test_paused_lifecycle_definition_is_paused():
    rows = build_unified_rows([], [_definition(lifecycle="paused")], [])
    assert rows[0].bucket == "paused"


def test_disabled_lifecycle_definition_is_paused():
    """Plan ruling 2's explicit trailing sentence: disabled-lifecycle
    definitions bucket as Paused."""
    rows = build_unified_rows([], [_definition(lifecycle="disabled")], [])
    assert rows[0].bucket == "paused"


def test_archived_lifecycle_definition_is_completed():
    rows = build_unified_rows([], [_definition(lifecycle="archived")], [])
    assert rows[0].bucket == "completed"


# ---------------------------------------------------------------------------
# Glyphs (spec SS4)
# ---------------------------------------------------------------------------


def test_recurring_active_reminder_glyph_is_circle():
    rows = build_unified_rows([_reminder(schedule_kind=ScheduleKind.RECURRING)], [], [])
    assert rows[0].glyph == "○"


def test_one_time_active_reminder_glyph_is_triangle():
    rows = build_unified_rows([_one_time_reminder()], [], [])
    assert rows[0].glyph == "▶"


def test_paused_reminder_glyph_is_pause_bar():
    rows = build_unified_rows([_reminder(enabled=False)], [], [])
    assert rows[0].glyph == "⏸"


def test_completed_reminder_glyph_is_check():
    rows = build_unified_rows(
        [
            _reminder(
                enabled=False,
                next_run_at=None,
                last_run_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
            )
        ],
        [],
        [],
    )
    assert rows[0].glyph == "✓"


def test_cron_definition_glyph_is_circle():
    rows = build_unified_rows([], [_definition()], [])
    assert rows[0].glyph == "○"


def test_one_time_definition_glyph_is_triangle():
    schedule = {"kind": "one_time", "run_at": "2026-09-05T09:00:00+00:00"}
    rows = build_unified_rows([], [_definition(schedule=schedule)], [])
    assert rows[0].glyph == "▶"


def test_archived_definition_glyph_is_check_not_circle():
    rows = build_unified_rows([], [_definition(lifecycle="archived")], [])
    assert rows[0].glyph == "✓"


# ---------------------------------------------------------------------------
# Unread resolution across both id spaces (plan ruling 3 / survey warning)
# ---------------------------------------------------------------------------


def test_unread_count_resolves_across_local_and_server_id_spaces():
    """The survey's warning IS the test: a server-mirrored definition's
    results must be counted even though ONE of the two results below
    carries the SERVER's id (not the definition's own local `id`)."""
    definition = _definition(id="def-local-1", server_id="srv-77")
    results = [
        _result(definition_id="srv-77", review_state="unread"),  # server id space
        _result(definition_id="def-local-1", review_state="unread"),  # local id space
        _result(definition_id="srv-77", review_state="read"),  # not unread -- excluded
        _result(definition_id="unrelated-id", review_state="unread"),  # matches nothing
    ]
    rows = build_unified_rows([], [definition], results)
    assert rows[0].unread_count == 2


def test_reminder_rows_never_carry_unread_count():
    rows = build_unified_rows([_reminder()], [], [_result(definition_id="rem-1")])
    assert rows[0].unread_count == 0


def test_zero_unread_when_no_matching_results():
    rows = build_unified_rows(
        [], [_definition()], [_result(definition_id="def-1", review_state="read")]
    )
    assert rows[0].unread_count == 0


# ---------------------------------------------------------------------------
# Filter (chip + search)
# ---------------------------------------------------------------------------


def _mixed_rows() -> list[UnifiedRow]:
    return build_unified_rows(
        reminders=[
            _reminder(id="active-rem", title="Ping the server"),
            _reminder(
                id="paused-rem",
                title="Weekly check",
                enabled=False,
                next_run_at=datetime(2026, 9, 5, tzinfo=timezone.utc),
            ),
            _reminder(
                id="done-rem",
                title="Old one-shot",
                enabled=False,
                next_run_at=None,
                last_run_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
            ),
        ],
        definitions=[
            _definition(
                id="active-def", name="Daily digest", input={"question": "Any outages?"}
            ),
            _definition(id="paused-def", name="Held automation", lifecycle="paused"),
            _definition(
                id="archived-def", name="Retired automation", lifecycle="archived"
            ),
        ],
        results=[],
    )


def test_chip_all_excludes_completed():
    rows = filter_rows(_mixed_rows(), chip="all", query="")
    buckets = {row.bucket for row in rows}
    assert buckets == {"active", "paused"}
    assert len(rows) == 4


def test_chip_active_only():
    rows = filter_rows(_mixed_rows(), chip="active", query="")
    assert {row.row_id for row in rows} == {
        "reminder:active-rem",
        "definition:active-def",
    }


def test_chip_paused_only():
    rows = filter_rows(_mixed_rows(), chip="paused", query="")
    assert {row.row_id for row in rows} == {
        "reminder:paused-rem",
        "definition:paused-def",
    }


def test_chip_completed_only():
    rows = filter_rows(_mixed_rows(), chip="completed", query="")
    assert {row.row_id for row in rows} == {
        "reminder:done-rem",
        "definition:archived-def",
    }


def test_search_matches_title_case_insensitively():
    rows = filter_rows(_mixed_rows(), chip="all", query="DIGEST")
    assert {row.row_id for row in rows} == {"definition:active-def"}


def test_search_matches_question_body_not_just_title():
    rows = filter_rows(_mixed_rows(), chip="all", query="outages")
    assert {row.row_id for row in rows} == {"definition:active-def"}


def test_blank_search_matches_everything_in_chip():
    rows = filter_rows(_mixed_rows(), chip="all", query="   ")
    assert len(rows) == 4


def test_search_with_no_match_returns_empty():
    rows = filter_rows(_mixed_rows(), chip="all", query="nonexistent-xyz")
    assert rows == []


# ---------------------------------------------------------------------------
# Sort (plan ruling 5)
# ---------------------------------------------------------------------------


def test_active_sorts_by_next_run_ascending_none_last():
    soon = _reminder(id="soon", next_run_at=datetime(2026, 9, 1, tzinfo=timezone.utc))
    later = _reminder(
        id="later", next_run_at=datetime(2026, 9, 10, tzinfo=timezone.utc)
    )
    unscheduled = _reminder(
        id="unscheduled", next_run_at=None, cron="0 9 * * *", timezone="UTC"
    )
    rows = build_unified_rows([later, unscheduled, soon], [], [])
    ordered = sort_rows(rows, "active")
    assert [row.row_id for row in ordered] == [
        "reminder:soon",
        "reminder:later",
        "reminder:unscheduled",
    ]


def test_paused_sorts_by_recency_descending_none_last():
    recent = _reminder(
        id="recent",
        enabled=False,
        next_run_at=datetime(2026, 9, 5, tzinfo=timezone.utc),
        last_run_at=datetime(2026, 9, 2, tzinfo=timezone.utc),
    )
    older = _reminder(
        id="older",
        enabled=False,
        next_run_at=datetime(2026, 9, 5, tzinfo=timezone.utc),
        last_run_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
    )
    never_run = _reminder(
        id="never-run",
        enabled=False,
        next_run_at=datetime(2026, 9, 5, tzinfo=timezone.utc),
        last_run_at=None,
    )
    rows = build_unified_rows([older, never_run, recent], [], [])
    ordered = sort_rows(rows, "paused")
    assert [row.row_id for row in ordered] == [
        "reminder:recent",
        "reminder:older",
        "reminder:never-run",
    ]


def test_completed_sorts_definitions_by_updated_at_descending():
    newer = _definition(
        id="newer", lifecycle="archived", updated_at="2026-09-05T00:00:00+00:00"
    )
    older = _definition(
        id="older", lifecycle="archived", updated_at="2026-09-01T00:00:00+00:00"
    )
    rows = build_unified_rows([], [newer, older], [])
    ordered = sort_rows(rows, "completed")
    assert [row.row_id for row in ordered] == ["definition:newer", "definition:older"]


def test_all_chip_puts_active_rows_first_then_paused_by_recency():
    active_far = _reminder(
        id="active-far", next_run_at=datetime(2026, 9, 20, tzinfo=timezone.utc)
    )
    active_near = _reminder(
        id="active-near", next_run_at=datetime(2026, 9, 1, tzinfo=timezone.utc)
    )
    paused_recent = _reminder(
        id="paused-recent",
        enabled=False,
        next_run_at=datetime(2026, 9, 5, tzinfo=timezone.utc),
        last_run_at=datetime(2026, 9, 3, tzinfo=timezone.utc),
    )
    rows = build_unified_rows([active_far, paused_recent, active_near], [], [])
    ordered = sort_rows(rows, "all")
    assert [row.row_id for row in ordered] == [
        "reminder:active-near",
        "reminder:active-far",
        "reminder:paused-recent",
    ]


# ---------------------------------------------------------------------------
# Row shape sanity
# ---------------------------------------------------------------------------


def test_row_id_is_namespaced_by_kind():
    rows = build_unified_rows(
        [_reminder(id="shared-id")], [_definition(id="shared-id")], []
    )
    ids = {row.row_id for row in rows}
    assert ids == {"reminder:shared-id", "definition:shared-id"}


def test_owner_label_this_device_for_local():
    rows = build_unified_rows([_reminder(owner_id="local")], [], [])
    assert rows[0].owner_label == "This device"


def test_owner_label_server_id_for_server_scoped():
    rows = build_unified_rows([_reminder(owner_id="server:srv-9")], [], [])
    assert rows[0].owner_label == "srv-9"
