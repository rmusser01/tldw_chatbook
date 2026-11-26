"""The restore sandbox's ChaChaNotes index steps.

TASK-42517: the v87->v88 step may create exactly its nine indexes.
TASK-42544: the v88->v89 step may drop exactly the superseded
response-revision index.
"""

from __future__ import annotations

import sqlite3

import pytest

from tldw_chatbook.Backup_Recovery.sqlite_validation import _Restrictions
from tldw_chatbook.DB.recovery_core_schema import (
    CHACHANOTES_V88_LOCATOR_INDEXES,
    CHACHANOTES_V89_DROPPED_INDEXES,
)

OK, DENY = sqlite3.SQLITE_OK, sqlite3.SQLITE_DENY
HOOK_INDEX = "idx_hook_continuation_receipts_assistant_message"


def _sandbox(locator_step=True, drop_step=False):
    restrictions = _Restrictions(sqlite3.connect(":memory:"))
    restrictions.migrating = True
    restrictions.migration_owner = "db.chachanotes.primary"
    restrictions.locator_indexes = CHACHANOTES_V88_LOCATOR_INDEXES if locator_step else {}
    restrictions.dropped_indexes = CHACHANOTES_V89_DROPPED_INDEXES if drop_step else {}
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


DROPPED, LINKS = "idx_console_trace_response_revision", "console_trace_response_links"
# What SQLite's DROP INDEX asks the authorizer for, beside the drop itself.
DROP_BOOKKEEPING = [
    (sqlite3.SQLITE_DELETE, "sqlite_master", None),
    (sqlite3.SQLITE_UPDATE, "sqlite_master", "rootpage"),
]


def test_the_drop_step_lists_only_the_superseded_index():
    assert CHACHANOTES_V89_DROPPED_INDEXES == {DROPPED: LINKS}


@pytest.mark.parametrize(
    ("action", "first", "second"),
    [(sqlite3.SQLITE_DROP_INDEX, DROPPED, LINKS), *DROP_BOOKKEEPING],
    ids=["drop", "catalog-delete", "rootpage"],
)
def test_the_drop_step_may_drop_its_index(action, first, second):
    sandbox = _sandbox(locator_step=False, drop_step=True)
    assert sandbox.authorize(action, first, second, "main", None) == OK
    # Outside the step the same request is refused.
    assert _sandbox(False).authorize(action, first, second, "main", None) == DENY


def test_the_locator_step_may_not_drop_the_superseded_index():
    action = sqlite3.SQLITE_DROP_INDEX
    assert _sandbox().authorize(action, DROPPED, LINKS, "main", None) == DENY


def test_the_real_drop_statement_passes_only_inside_the_step():
    def run(drop_step):
        connection = sqlite3.connect(":memory:")
        connection.execute(f"CREATE TABLE {LINKS}(semantic_revision_id TEXT)")
        connection.execute(f"CREATE INDEX {DROPPED} ON {LINKS}(semantic_revision_id)")
        restrictions = _Restrictions(connection)
        restrictions.migrating = True
        restrictions.migration_owner = "db.chachanotes.primary"
        restrictions.dropped_indexes = CHACHANOTES_V89_DROPPED_INDEXES if drop_step else {}
        connection.execute(f"DROP INDEX {DROPPED}")
        return connection.execute(
            "SELECT count(*) FROM sqlite_master WHERE name = ?", (DROPPED,)
        ).fetchone()[0]

    assert run(True) == 0
    with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
        run(False)


@pytest.mark.parametrize(
    ("action", "first", "second", "database", "source"),
    [
        # Any other index on the same table, the v72 exact index included.
        (sqlite3.SQLITE_DROP_INDEX, "idx_console_trace_response_artifact", LINKS, "main", None),
        (
            sqlite3.SQLITE_DROP_INDEX,
            "idx_console_trace_response_links_semantic_revision_exact",
            LINKS,
            "main",
            None,
        ),
        # Another table's index, and the listed name on another table.
        (
            sqlite3.SQLITE_DROP_INDEX,
            "idx_console_trace_events_semantic_revision_exact",
            "console_trace_events",
            "main",
            None,
        ),
        (sqlite3.SQLITE_DROP_INDEX, DROPPED, "console_trace_events", "main", None),
        # The listed pair in the temp database, or from a trigger body.
        (sqlite3.SQLITE_DROP_INDEX, DROPPED, LINKS, "temp", None),
        (sqlite3.SQLITE_DROP_INDEX, DROPPED, LINKS, "main", "console_trace_response_links_no_delete"),
        # Re-creating it, dropping its table, or rewriting catalog text.
        (sqlite3.SQLITE_CREATE_INDEX, DROPPED, LINKS, "main", None),
        (sqlite3.SQLITE_DROP_TABLE, LINKS, None, "main", None),
        (sqlite3.SQLITE_UPDATE, "sqlite_master", "sql", "main", None),
        # The bookkeeping admissions are for sqlite_master only, not rows.
        (sqlite3.SQLITE_DELETE, LINKS, None, "main", None),
        (sqlite3.SQLITE_DELETE, "messages", None, "main", None),
        (sqlite3.SQLITE_UPDATE, LINKS, "rootpage", "main", None),
    ],
    ids=[
        "same-table",
        "exact-index",
        "other-table-index",
        "other-table",
        "temp",
        "trigger",
        "create",
        "drop-table",
        "catalog-sql",
        "delete-links-rows",
        "delete-messages-rows",
        "update-links-rootpage",
    ],
)
def test_the_drop_step_refuses_anything_else(action, first, second, database, source):
    sandbox = _sandbox(locator_step=False, drop_step=True)
    assert sandbox.authorize(action, first, second, database, source) == DENY
