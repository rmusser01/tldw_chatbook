"""Native passive-history components; fixtures do not authorize consolidation."""

import json
from contextlib import contextmanager
from dataclasses import replace

import pytest

from Tests.DB.test_console_command_receipts import receipt

pytestmark = pytest.mark.bootstrap_profile
OPERATION = "component-operation"
SOURCE = "b" * 64


@contextmanager
def native_database(path):
    from tldw_chatbook.Chat.console_execution_owner import ConsoleExecutionOwner
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    owner = ConsoleExecutionOwner.acquire(path)
    database = None
    try:
        database = AgentRunsDB(path, execution_owner=owner, reconcile_on_init=False)
        yield database
    finally:
        if database is not None:
            database.close()
        owner.close()


@pytest.fixture
def database(tmp_path):
    with native_database(tmp_path / "runs.db") as database:
        yield database


def historical(database, command_id="historical", *, target_run=False, padding=0):
    from tldw_chatbook.Backup_Recovery.mobile_runs_preflight import _OperationalHistory
    from tldw_chatbook.Chat.console_remote_models import ConsoleTarget

    item = receipt(command_id, updated_at="2026-10-25T00:00:00Z")
    associations = ()
    if target_run:
        mapped = database.create_run(
            conversation_id="conversation", agent_kind="primary"
        )
        item = replace(
            item,
            target=ConsoleTarget(
                server_epoch="server",
                console_id="console",
                session_id="session",
                incarnation_id="incarnation",
                run_id="original-run",
            ),
            native_id="opaque-native",
        )
        associations = (("run", "original-run", mapped),)
    database.command_receipts.claim(item)
    with database.transaction() as connection:
        connection.execute(
            "UPDATE console_command_receipts SET owner_epoch=? WHERE command_id=?",
            ('[ "server", "console" ]' + " " * padding, command_id),
        )
        row = connection.execute(
            "SELECT * FROM console_command_receipts WHERE command_id=?", (command_id,)
        ).fetchone()
        values = tuple(dict(row).items())
        connection.execute(
            "DELETE FROM console_command_receipts WHERE command_id=?", (command_id,)
        )
    return _OperationalHistory(
        "console_command_receipts", (command_id,), values, associations
    )


def retain(database, rows):
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget
    from tldw_chatbook.DB.console_command_history import _retain_command_history

    with database.transaction(immediate=True) as connection:
        _retain_command_history(
            database,
            connection,
            OPERATION,
            SOURCE,
            tuple(rows),
            _read_budget=_TraceReadBudget(),
        )


def test_passive_history_roundtrip_and_native_read_only_reopen(database):
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    source = historical(database, target_run=True)
    retain(database, [source])
    page = database.command_receipts.list_imported_history(OPERATION, SOURCE)
    assert page.items[0].original_fields == source.values
    assert page.items[0].run_associations == (
        (source.associations[0][1], source.associations[0][2]),
    )
    assert page.next_cursor is None
    assert database.command_receipts.get("historical") is None
    with database.connection() as connection:
        assert [
            row[0]
            for row in connection.execute(
                "SELECT batch_key FROM mobile_consolidation_batches ORDER BY batch_key"
            )
        ] == ["history.command", "history.command:historical"]
    reader = AgentRunsDB.open_read_only(database.db_path)
    try:
        assert reader.command_receipts.list_imported_history(OPERATION, SOURCE) == page
    finally:
        reader.close()


def test_native_transaction_rollback_hides_passive_history(database):
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget
    from tldw_chatbook.DB.console_command_history import _retain_command_history

    source = historical(database)
    with (
        pytest.raises(RuntimeError, match="crash"),
        database.transaction(immediate=True) as connection,
    ):
        _retain_command_history(
            database,
            connection,
            OPERATION,
            SOURCE,
            (source,),
            _read_budget=_TraceReadBudget(),
        )
        raise RuntimeError("crash")
    with pytest.raises(ValueError, match="command_history"):
        database.command_receipts.list_imported_history(OPERATION, SOURCE)
    retain(database, [source])
    assert (
        len(database.command_receipts.list_imported_history(OPERATION, SOURCE).items)
        == 1
    )


def test_passive_history_pages_bind_source_and_do_not_enter_live_controls(database):
    rows = [historical(database, f"history-{i}") for i in range(3)]
    retain(database, rows)
    repo = database.command_receipts
    page = repo.list_imported_history(OPERATION, SOURCE, limit=2)
    assert len(page.items) == 2 and page.next_cursor is not None
    following = repo.list_imported_history(
        OPERATION, SOURCE, cursor=page.next_cursor, limit=2
    )
    assert len(following.items) == 1 and following.next_cursor is None
    with pytest.raises(ValueError, match="command_history"):
        repo.list_imported_history(OPERATION, "c" * 64, cursor=page.next_cursor)
    repo.invalidate_prior_owner('["new","console"]')
    assert repo.prune_invalid(receipt().owner_epoch, 1000) == 0
    assert repo.list_unresolved().items == ()
    assert repo.list_imported_history(OPERATION, SOURCE, limit=2) == page
    assert repo.claim(receipt("history-0")).inserted
    assert repo.list_imported_history(OPERATION, SOURCE, limit=2) == page


@pytest.mark.parametrize(
    "damage",
    [
        "count",
        "source",
        "unknown",
        "duplicate",
        "command-key",
        "association",
        "empty-key",
        "master-version",
    ],
)
def test_corrupt_passive_receipts_refuse_without_active_fallback(database, damage):
    retain(database, [historical(database)])
    with database.transaction() as connection:
        if damage == "count":
            connection.execute(
                "DELETE FROM mobile_consolidation_batches WHERE batch_key='history.command:historical'"
            )
        elif damage == "source":
            connection.execute(
                "UPDATE mobile_consolidation_batches SET source_fingerprint=? WHERE batch_key='history.command:historical'",
                ("c" * 64,),
            )
        elif damage == "empty-key":
            connection.execute(
                "UPDATE mobile_consolidation_batches SET batch_key='history.command:' WHERE batch_key='history.command:historical'"
            )
        else:
            key = (
                "history.command"
                if damage == "master-version"
                else "history.command:historical"
            )
            raw = connection.execute(
                "SELECT mapping_json FROM mobile_consolidation_batches WHERE batch_key=?",
                (key,),
            ).fetchone()[0]
            value = json.loads(raw)
            if damage == "unknown":
                value["extra"] = "no"
            elif damage == "command-key":
                value["record"]["command_id"] = "other"
            elif damage == "association":
                value["run_associations"] = [["invented", "invented"]]
            elif damage == "master-version":
                value["version"] = True
            raw = json.dumps(value)
            if damage == "duplicate":
                raw = raw[:-1] + ', "version": 1}'
            connection.execute(
                "UPDATE mobile_consolidation_batches SET mapping_json=? WHERE batch_key=?",
                (raw, key),
            )
    with pytest.raises(ValueError, match="command_history"):
        database.command_receipts.list_imported_history(OPERATION, SOURCE)


def test_retention_requires_native_current_transaction_and_refuses_unreadable_before_writes(
    database,
):
    import sqlite3

    from tldw_chatbook.Chat.console_trace_read_budget import (
        _TraceReadBudget,
        _TraceReadLimit,
    )
    from tldw_chatbook.DB.console_command_history import _retain_command_history

    row = historical(database)
    foreign = sqlite3.connect(":memory:")
    try:
        with (
            database.transaction() as connection,
            pytest.raises(ValueError, match="owner_transaction"),
        ):
            _retain_command_history(
                database,
                foreign,
                OPERATION,
                SOURCE,
                (row,),
                _read_budget=_TraceReadBudget(),
            )
        with (
            database.connection() as connection,
            pytest.raises(ValueError, match="owner_transaction"),
        ):
            _retain_command_history(
                database,
                connection,
                OPERATION,
                SOURCE,
                (row,),
                _read_budget=_TraceReadBudget(),
            )
    finally:
        foreign.close()
    enormous = replace(
        row,
        original=("enormous",),
        values=tuple(
            (
                key,
                value + " " * 2_000_000
                if key == "owner_epoch"
                else "enormous"
                if key == "command_id"
                else value,
            )
            for key, value in row.values
        ),
    )
    with pytest.raises(_TraceReadLimit):
        retain(database, [row, enormous])
    with database.connection() as connection:
        assert (
            connection.execute(
                "SELECT count(*) FROM mobile_consolidation_batches"
            ).fetchone()[0]
            == 0
        )


def test_sql_byte_bounds_apply_to_child_and_manifest_before_text_reads(database):
    retain(database, [historical(database)])
    for key in ("history.command:historical", "history.command"):
        with database.transaction() as connection:
            original = connection.execute(
                "SELECT mapping_json FROM mobile_consolidation_batches WHERE batch_key=?",
                (key,),
            ).fetchone()[0]
            connection.execute(
                "UPDATE mobile_consolidation_batches SET mapping_json=? WHERE batch_key=?",
                ("é" * (8 * 1024**2 + 1), key),
            )
        reads = []
        with database.connection() as connection:
            connection.set_trace_callback(
                lambda statement, reads=reads: reads.append(
                    statement.startswith("SELECT mapping_json")
                )
            )
            try:
                with pytest.raises(ValueError, match="record_unreadable"):
                    database.command_receipts.list_imported_history(OPERATION, SOURCE)
            finally:
                connection.set_trace_callback(None)
        assert sum(reads) == (1 if key.endswith(":historical") else 0)
        with database.transaction() as connection:
            connection.execute(
                "UPDATE mobile_consolidation_batches SET mapping_json=? WHERE batch_key=?",
                (original, key),
            )


def test_real_live_capacity_remains_4096_with_passive_history(database):
    from tldw_chatbook.DB.console_command_receipts import (
        MAX_COMMAND_RECEIPTS,
        ConsoleReceiptCapacity,
    )

    retain(database, [historical(database)])
    repo = database.command_receipts
    repo.claim(receipt("active-0000"))
    with database.transaction() as connection:
        raw = tuple(
            connection.execute("SELECT * FROM console_command_receipts").fetchone()
        )
        connection.executemany(
            "INSERT INTO console_command_receipts VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            ((f"active-{i:04}", *raw[1:]) for i in range(1, MAX_COMMAND_RECEIPTS)),
        )
    with pytest.raises(ConsoleReceiptCapacity):
        repo.claim(receipt("over-capacity"))
    assert len(repo.list_imported_history(OPERATION, SOURCE).items) == 1
    repo.invalidate_prior_owner('["new","console"]')
    assert repo.list_unresolved(limit=100).items[0].state == "outcome_unknown"
    assert repo.prune_invalid(receipt().owner_epoch, 1000) == 0
    assert (
        dict(repo.list_imported_history(OPERATION, SOURCE).items[0].original_fields)[
            "state"
        ]
        == "pending"
    )


def test_representation_page_returns_fewer_complete_records_then_progresses(database):
    # Stored fixtures are intentionally qualified independently of import readiness.
    source = historical(database)
    record = dict(source.values)
    record["owner_epoch"] += " " * 220_000
    with database.transaction() as connection:
        connection.execute(
            "INSERT INTO mobile_consolidation_batches VALUES (?,?,?,?,?)",
            (
                OPERATION,
                SOURCE,
                "history.command",
                json.dumps(
                    {"version": 1, "kind": "command_history", "record_count": 8}
                ),
                "fixture",
            ),
        )
        for i in range(8):
            record["command_id"] = f"p{i}"
            raw = json.dumps({"version": 1, "record": record, "run_associations": []})
            connection.execute(
                "INSERT INTO mobile_consolidation_batches VALUES (?,?,?,?,?)",
                (OPERATION, SOURCE, f"history.command:p{i}", raw, "fixture"),
            )
    page = database.command_receipts.list_imported_history(OPERATION, SOURCE)
    assert 0 < len(page.items) < 8 and page.next_cursor is not None
    found = list(page.items)
    while page.next_cursor is not None:
        page = database.command_receipts.list_imported_history(
            OPERATION, SOURCE, cursor=page.next_cursor
        )
        assert page.items
        found.extend(page.items)
    assert [dict(item.original_fields)["command_id"] for item in found] == [
        f"p{i}" for i in range(8)
    ]


def test_only_strict_empty_manifest_means_retained_empty_history(database):
    with pytest.raises(ValueError, match="manifest_required"):
        database.command_receipts.list_imported_history(OPERATION, SOURCE)
    retain(database, [])
    page = database.command_receipts.list_imported_history(OPERATION, SOURCE)
    assert page.items == () and page.next_cursor is None
    with pytest.raises(ValueError, match="manifest_required"):
        database.command_receipts.list_imported_history("another", SOURCE)


def test_duplicate_rows_refuse_before_any_retained_record(database):
    row = historical(database)
    with pytest.raises(ValueError, match="duplicate_record"):
        retain(database, [row, row])
    with database.connection() as connection:
        assert (
            connection.execute(
                "SELECT count(*) FROM mobile_consolidation_batches"
            ).fetchone()[0]
            == 0
        )


@pytest.mark.parametrize("limit", [0, 101, True])
def test_history_page_rejects_invalid_limit_before_read(database, limit):
    with pytest.raises(ValueError, match="page_invalid"):
        database.command_receipts.list_imported_history(OPERATION, SOURCE, limit=limit)


def test_single_stored_record_over_representation_allowance_refuses_without_body_read(
    database,
):
    retain(database, [historical(database)])
    with database.transaction() as connection:
        raw = connection.execute(
            "SELECT mapping_json FROM mobile_consolidation_batches WHERE batch_key='history.command:historical'"
        ).fetchone()[0]
        connection.execute(
            "UPDATE mobile_consolidation_batches SET mapping_json=? WHERE batch_key='history.command:historical'",
            (raw + " " * 4_200_000,),
        )
    observed = []
    with database.connection() as connection:
        connection.set_trace_callback(
            lambda statement: observed.append(
                statement.startswith("SELECT mapping_json")
            )
        )
        try:
            with pytest.raises(ValueError, match="record_unreadable"):
                database.command_receipts.list_imported_history(OPERATION, SOURCE)
        finally:
            connection.set_trace_callback(None)
    assert sum(observed) == 1  # Only the bounded manifest was materialized.


def test_reader_cannot_present_uncommitted_owner_rows_as_retained_history(database):
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget
    from tldw_chatbook.DB.console_command_history import _retain_command_history

    row = historical(database)
    with database.transaction() as connection:
        _retain_command_history(
            database,
            connection,
            OPERATION,
            SOURCE,
            (row,),
            _read_budget=_TraceReadBudget(),
        )
        with pytest.raises(ValueError, match="command_history_uncommitted"):
            database.command_receipts.list_imported_history(OPERATION, SOURCE)
    assert (
        len(database.command_receipts.list_imported_history(OPERATION, SOURCE).items)
        == 1
    )


def test_page_keeps_native_snapshot_while_other_connection_commits(database):
    from concurrent.futures import ThreadPoolExecutor

    retain(database, [historical(database)])
    committed = []

    def remove_component_fixture():
        try:
            with database.transaction() as connection:
                connection.execute(
                    "DELETE FROM mobile_consolidation_batches WHERE batch_key='history.command:historical'"
                )
                connection.execute(
                    "UPDATE mobile_consolidation_batches SET mapping_json=? WHERE batch_key='history.command'",
                    (
                        json.dumps(
                            {"version": 1, "kind": "command_history", "record_count": 0}
                        ),
                    ),
                )
        finally:
            database.close()

    with (
        ThreadPoolExecutor(max_workers=1) as executor,
        database.connection() as connection,
    ):

        def observe(statement):
            if statement.startswith("SELECT mapping_json") and not committed:
                committed.append(executor.submit(remove_component_fixture))
                committed[0].result()

        connection.set_trace_callback(observe)
        try:
            page = database.command_receipts.list_imported_history(OPERATION, SOURCE)
        finally:
            connection.set_trace_callback(None)
    assert committed and committed[0].exception() is None
    assert len(page.items) == 1
    assert (
        database.command_receipts.list_imported_history(OPERATION, SOURCE).items == ()
    )


def test_native_owner_and_sqlite_handles_retire_before_successor(tmp_path):
    import sqlite3

    from tldw_chatbook.Chat.console_execution_owner import ConsoleExecutionOwner

    path = tmp_path / "retirement.db"
    with native_database(path) as database:
        retain(database, [historical(database)])
        with database.connection() as connection:
            assert connection.execute("SELECT 1").fetchone()[0] == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")
    successor = ConsoleExecutionOwner.acquire(path)
    successor.close()


def test_passive_history_source_count_has_no_live_4096_quota(database):
    source = historical(database)
    original = dict(source.values)
    count = 4097
    with database.transaction() as connection:
        connection.execute(
            "INSERT INTO mobile_consolidation_batches VALUES (?,?,?,?,?)",
            (
                OPERATION,
                SOURCE,
                "history.command",
                json.dumps(
                    {"version": 1, "kind": "command_history", "record_count": count}
                ),
                "fixture",
            ),
        )
        for number in range(count):
            original["command_id"] = f"history-{number:04}"
            connection.execute(
                "INSERT INTO mobile_consolidation_batches VALUES (?,?,?,?,?)",
                (
                    OPERATION,
                    SOURCE,
                    "history.command:" + original["command_id"],
                    json.dumps(
                        {"version": 1, "record": original, "run_associations": []}
                    ),
                    "fixture",
                ),
            )
    seen = 0
    cursor = None
    while True:
        page = database.command_receipts.list_imported_history(
            OPERATION, SOURCE, cursor=cursor, limit=100
        )
        assert page.items
        seen += len(page.items)
        cursor = page.next_cursor
        if cursor is None:
            break
    assert seen == count
    assert database.command_receipts.list_unresolved().items == ()
