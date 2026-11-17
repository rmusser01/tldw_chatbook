"""TASK-41310: the v86->v87 restore step may touch exactly its own objects."""

from __future__ import annotations

import sqlite3

import pytest

from tldw_chatbook.Backup_Recovery.sqlite_validation import _Restrictions

TABLE = "console_trace_child_segments"
OK, DENY = sqlite3.SQLITE_OK, sqlite3.SQLITE_DENY


def _sandbox(child_step=True):
    restrictions = _Restrictions(sqlite3.connect(":memory:"))
    restrictions.migrating = True
    restrictions.migration_owner = "db.chachanotes.primary"
    restrictions.child_segments_migration = child_step
    return restrictions


@pytest.mark.parametrize(
    ("action", "first", "second"),
    [
        (sqlite3.SQLITE_CREATE_TABLE, TABLE, None),
        (sqlite3.SQLITE_CREATE_TRIGGER, f"{TABLE}_insert_guard", TABLE),
        (
            sqlite3.SQLITE_DROP_TRIGGER,
            "console_trace_owners_empty_root_guard",
            "console_trace_owners",
        ),
        (
            sqlite3.SQLITE_CREATE_TRIGGER,
            "console_trace_segments_parent_owner_guard",
            "console_trace_segments",
        ),
    ],
)
def test_the_child_segments_step_may_touch_its_own_objects(action, first, second):
    assert _sandbox().authorize(action, first, second, "main", None) == OK
    # Outside the step the same statement is refused.
    assert _sandbox(False).authorize(action, first, second, "main", None) == DENY


@pytest.mark.parametrize(
    ("action", "first", "second"),
    [
        (sqlite3.SQLITE_DROP_TRIGGER, "console_trace_events_no_update", "console_trace_events"),
        # A listed trigger name on the wrong table is refused too.
        (sqlite3.SQLITE_DROP_TRIGGER, "console_trace_calls_owner_lineage", "messages"),
        (sqlite3.SQLITE_CREATE_TRIGGER, "console_trace_unlisted_guard", "console_trace_calls"),
        (sqlite3.SQLITE_CREATE_TABLE, "console_trace_other_table", None),
        (sqlite3.SQLITE_DROP_TABLE, TABLE, None),
    ],
)
def test_the_child_segments_step_refuses_anything_unlisted(action, first, second):
    assert _sandbox().authorize(action, first, second, "main", None) == DENY
