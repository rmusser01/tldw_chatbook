"""Fresh native schema comparisons retain their original SQL and byte bounds."""

import sqlite3

import pytest

from tldw_chatbook.Backup_Recovery import mobile_preview_budget as accounting
from tldw_chatbook.Backup_Recovery import sqlite_validation as validation
from tldw_chatbook.Backup_Recovery.storage_admission import (
    _CaptureLease,
    _preview_reads,
)
from tldw_chatbook.Chat.console_trace_read_budget import (
    _TraceReadBudget,
    _TraceReadLimit,
)
from tldw_chatbook.DB.private_sqlite import connect_private_sqlite


class _ObservedCursor(sqlite3.Cursor):
    def execute(self, sql, parameters=()):
        self.queries.append((sql, tuple(parameters)))
        if sql.startswith("SELECT rowid FROM sqlite_schema"):
            before = self.connection.before_rowids
            self.connection.before_rowids = None
            if before is not None:
                self.fetchall()
                before()
        return super().execute(sql, parameters)


class _ObservedConnection(sqlite3.Connection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cursors = []
        self.before_rowids = None

    def cursor(self):
        cursor = super().cursor(factory=_ObservedCursor)
        cursor.queries = []
        self.cursors.append(cursor)
        return cursor


@pytest.fixture
def native_reference():
    # Same genuine preview issuer/reference lease as sqlite_validation._reference.
    budget = _TraceReadBudget()
    with _preview_reads(_read_budget=budget):
        lease = _CaptureLease(accounting.current())
        connection = connect_private_sqlite(
            "recovery.validation_schema", ":memory:", factory=_ObservedConnection
        )
        lease.attach(connection)
        try:
            yield connection, budget
        finally:
            lease.retire()
        assert lease.resource_closed and lease not in accounting.current().resources
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")


def _seed(connection):
    for statement in (
        "CREATE TABLE zebra(value TEXT)",
        "CREATE VIEW middle AS SELECT value FROM zebra",
        "CREATE INDEX alpha_index ON zebra(value)",
        "CREATE TABLE alpha(value TEXT UNIQUE)",
        (
            "CREATE TRIGGER omega AFTER INSERT ON zebra "
            "BEGIN UPDATE zebra SET value=NEW.value; END"
        ),
    ):
        connection.execute(statement)
    # This is the ordinary installed validator's native SQL-text oracle.
    return tuple(
        row[0]
        for row in connection.execute(
            "SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL ORDER BY type,name"
        )
    )


def _closed(cursor):
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        cursor.fetchone()


def test_large_fresh_catalog_matches_with_original_native_sql_step_budget(
    native_reference,
):
    connection, _ = native_reference
    for index in range(1000):
        connection.execute(f"CREATE TABLE native_{index:04d}(value INTEGER)")
    expected = tuple(
        row[0]
        for row in connection.execute(
            "SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL ORDER BY type,name"
        )
    )
    # Original _Restrictions owns the unchanged 5M-step/30s native progress guard.
    restrictions = validation._restrict_connection(connection)
    schemas = ((7, expected),)
    assert accounting.matching_schemas(connection, schemas) == schemas
    assert restrictions.steps > 0 and not restrictions.expired()
    _closed(connection.cursors[-1])


def test_native_order_null_rows_versions_and_fresh_comparison(native_reference):
    connection, _ = native_reference
    actual = _seed(connection)
    ordered = connection.execute(
        "SELECT rowid FROM sqlite_schema WHERE sql IS NOT NULL ORDER BY type,name"
    ).fetchall()
    assert ordered != sorted(ordered)
    assert (
        connection.execute(
            "SELECT count(*) FROM sqlite_schema WHERE sql IS NULL"
        ).fetchone()[0]
        > 0
    )
    first_wrong = (actual[0] + " /* other version */", *actual[1:])
    last_wrong = (*actual[:-1], actual[-1] + " /* other version */")
    schemas = ((1, first_wrong), (2, actual), (3, actual), (4, last_wrong))
    assert accounting.matching_schemas(connection, schemas) == schemas[1:3]
    cursor = connection.cursors[-1]
    projections = [row for row in cursor.queries if row[0].startswith("SELECT rowid")]
    assert len(projections) == 1 and projections[0][1] == (len(actual) + 1,)
    assert all("OFFSET" not in sql for sql, _ in cursor.queries)
    _closed(cursor)
    connection.execute("DROP VIEW middle")
    connection.execute("CREATE VIEW middle AS SELECT value FROM alpha")
    assert accounting.matching_schemas(connection, schemas) == ()
    _closed(connection.cursors[-1])


def test_matching_never_combines_parts_of_different_versions(native_reference):
    connection, _ = native_reference
    actual = _seed(connection)
    schemas = (
        (1, (*actual[:-1], actual[-1] + " /* other version */")),
        (2, (actual[0] + " /* other version */", *actual[1:])),
    )
    assert accounting.matching_schemas(connection, schemas) == ()
    _closed(connection.cursors[-1])


@pytest.mark.parametrize("no_schema", [False, True])
def test_no_initial_candidates_skip_projection_and_keep_final_charge(
    native_reference, no_schema
):
    connection, budget = native_reference
    actual = _seed(connection)
    schemas = () if no_schema else ((1, actual[:-1]),)
    remaining = budget._remaining
    assert accounting.matching_schemas(connection, schemas) == ()
    cursor = connection.cursors[-1]
    assert len(cursor.queries) == 1
    assert budget._remaining == remaining - (2048 + 64 * len(schemas)) - 128
    _closed(cursor)


def test_empty_native_catalog_preserves_all_empty_versions(native_reference):
    connection, _ = native_reference
    schemas = ((1, ()), (2, ()), (3, ("CREATE TABLE absent(value)",)))
    assert accounting.matching_schemas(connection, schemas) == schemas[:2]
    _closed(connection.cursors[-1])


def test_rowid_allocation_reserves_before_native_query(native_reference):
    connection, budget = native_reference
    actual = _seed(connection)
    schemas = ((1, actual),)
    charge = 512 + 128 * (len(actual) + 1)
    budget.reserve(budget._remaining - (2048 + 64 * len(schemas)) - charge + 1)
    with pytest.raises(_TraceReadLimit, match="mobile_trace_input_limit"):
        accounting.matching_schemas(connection, schemas)
    cursor = connection.cursors[-1]
    assert len(cursor.queries) == 1
    assert budget._remaining == charge - 1
    _closed(cursor)


@pytest.mark.parametrize("change", ["grow", "shrink"])
def test_native_catalog_cardinality_drift_is_bounded_and_refused(
    native_reference, change
):
    connection, _ = native_reference
    actual = _seed(connection)

    def change_catalog():
        if change == "grow":
            connection.execute("CREATE VIEW extra_one AS SELECT 1")
            connection.execute("CREATE VIEW extra_two AS SELECT 2")
        else:
            connection.execute("DROP VIEW middle")

    connection.before_rowids = change_catalog
    with pytest.raises(ValueError, match="^sqlite_resource_limit$"):
        accounting.matching_schemas(connection, ((1, actual),))
    cursor = connection.cursors[-1]
    projection = [row for row in cursor.queries if row[0].startswith("SELECT rowid")]
    assert len(projection) == 1 and projection[0][1] == (len(actual) + 1,)
    _closed(cursor)
