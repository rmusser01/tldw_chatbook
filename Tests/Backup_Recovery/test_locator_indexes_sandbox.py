"""TASK-42517: the v87->v88 restore step may create exactly its nine indexes."""

from __future__ import annotations

import sqlite3

import pytest

from tldw_chatbook.Backup_Recovery.sqlite_validation import _Restrictions
from tldw_chatbook.DB.recovery_core_schema import CHACHANOTES_V88_LOCATOR_INDEXES

OK, DENY = sqlite3.SQLITE_OK, sqlite3.SQLITE_DENY
HOOK_INDEX = "idx_hook_continuation_receipts_assistant_message"


def _sandbox(locator_step=True):
    restrictions = _Restrictions(sqlite3.connect(":memory:"))
    restrictions.migrating = True
    restrictions.migration_owner = "db.chachanotes.primary"
    restrictions.locator_indexes = CHACHANOTES_V88_LOCATOR_INDEXES if locator_step else {}
    return restrictions


def test_the_gate_lists_the_nine_installed_indexes():
    assert len(CHACHANOTES_V88_LOCATOR_INDEXES) == 9
    assert HOOK_INDEX in CHACHANOTES_V88_LOCATOR_INDEXES


@pytest.mark.parametrize(("name", "table"), sorted(CHACHANOTES_V88_LOCATOR_INDEXES.items()))
def test_the_locator_step_may_create_each_of_its_indexes(name, table):
    action = sqlite3.SQLITE_CREATE_INDEX
    assert _sandbox().authorize(action, name, table, "main", None) == OK
    # Outside the step the same statement is refused.
    assert _sandbox(False).authorize(action, name, table, "main", None) == DENY


@pytest.mark.parametrize(
    ("action", "first", "second", "database", "source"),
    [
        # A listed name on a table that is not its own.
        (sqlite3.SQLITE_CREATE_INDEX, HOOK_INDEX, "messages", "main", None),
        (
            sqlite3.SQLITE_CREATE_INDEX,
            HOOK_INDEX,
            "console_machine_followup_receipts",
            "main",
            None,
        ),
        # An unlisted name on a locator table.
        (
            sqlite3.SQLITE_CREATE_INDEX,
            "idx_hook_continuation_receipts_chain",
            "console_hook_continuation_receipts",
            "main",
            None,
        ),
        # The listed pair in the temp database, or from a trigger body.
        (
            sqlite3.SQLITE_CREATE_INDEX,
            HOOK_INDEX,
            "console_hook_continuation_receipts",
            "temp",
            None,
        ),
        (
            sqlite3.SQLITE_CREATE_INDEX,
            HOOK_INDEX,
            "console_hook_continuation_receipts",
            "main",
            "console_hook_continuation_receipts_guard",
        ),
        # Dropping a listed index is not creating it.
        (
            sqlite3.SQLITE_DROP_INDEX,
            HOOK_INDEX,
            "console_hook_continuation_receipts",
            "main",
            None,
        ),
    ],
    ids=["other-table", "sibling-table", "unlisted", "temp", "trigger", "drop"],
)
def test_the_locator_step_refuses_anything_else(action, first, second, database, source):
    assert _sandbox().authorize(action, first, second, database, source) == DENY
