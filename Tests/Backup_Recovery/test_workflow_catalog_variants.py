"""Exact installed catalog variants across constructor and recovery readers."""

import sqlite3
from contextlib import closing
from hashlib import sha256
from threading import Event
from unittest.mock import patch

import pytest

from Tests.Backup_Recovery.test_agent_runs_recovery_schema import owner
from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.parametrize(
    "analyzed,degraded", [(True, False), (False, True), (True, True)]
)
def test_installed_workflow_variants_across_readers(tmp_path, analyzed, degraded):
    path = tmp_path / "runs.db"
    original = AgentRunsDB._create_console_activity_receipts_schema

    def install_receipts(db, conn):
        original(db, conn)
        if degraded:
            raise sqlite3.OperationalError("injected failure after receipt DDL")

    with patch.object(
        AgentRunsDB, "_create_console_activity_receipts_schema", install_receipts
    ):
        with closing(AgentRunsDB(path)) as db:
            run_id = db.create_run(conversation_id="private-chat", agent_kind="primary")
            assert db.receipt_capability_available is not degraded
            if analyzed:
                with db.connection() as conn:
                    conn.execute("ANALYZE")
        before = sha256(path.read_bytes()).digest()
        assert owner().validate(path) == ()
        assert validate_candidate(owner(), path, Event(), migrate=False) == ()
        assert sha256(path.read_bytes()).digest() == before
        with closing(AgentRunsDB(path)) as db:
            assert db.get_run(run_id)["conversation_id"] == "private-chat"
            assert db.receipt_capability_available is not degraded
    with closing(AgentRunsDB(path)) as db:
        assert db.receipt_capability_available
        assert db.get_run(run_id)["conversation_id"] == "private-chat"
        with db.connection() as conn:
            assert tuple(
                r[0]
                for r in conn.execute(
                    "SELECT version FROM schema_version ORDER BY version"
                )
            ) == tuple(range(4, 31))


@pytest.mark.parametrize("damage", ["schema", "domain"])
def test_operational_validation_pins_one_wal_view(tmp_path, monkeypatch, damage):
    from Tests.Backup_Recovery.test_workflow_recovery_schema import saved_result
    from tldw_chatbook.Backup_Recovery import sqlite_validation

    path = tmp_path / "live.db"
    with closing(AgentRunsDB(path)) as db, db.connection() as writer:
        saved_result(db, tmp_path)
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        main_digest = sha256(path.read_bytes()).digest()
        mode = path.stat().st_mode
        writer.execute("ANALYZE")
        assert sha256(path.read_bytes()).digest() == main_digest
        assert owner().validate(path) == ()
        original = sqlite_validation._metadata
        observed = []

        def concurrent_commit(connection, catalog, restrictions=None):
            if restrictions is not None and not observed:
                observed.append(connection.in_transaction)
                assert any(row[1] == "sqlite_stat1" for row in catalog)
                writer.execute(
                    "CREATE TABLE extra(value)"
                    if damage == "schema"
                    else "UPDATE workflow_payload_reservations SET bytes=1"
                )
            return original(connection, catalog, restrictions)

        monkeypatch.setattr(sqlite_validation, "_metadata", concurrent_commit)
        assert owner().validate(path) == ()
        assert observed == [True]
        assert owner().validate(path)
        assert (
            sha256(path.read_bytes()).digest() == main_digest
            and path.stat().st_mode == mode
        )


@pytest.mark.parametrize("analyzed", [False, True])
@pytest.mark.parametrize(
    "damage",
    [
        "other_lineage",
        "installed_stamp",
        "missing_stamp",
        "partial",
        "extra",
        "wrong_version",
    ],
)
def test_exact_receipt_and_stamp_refusal_precedes_repair(tmp_path, damage, analyzed):
    from Tests.Backup_Recovery.test_workflow_dev_integration import install
    from tldw_chatbook.DB.recovery_workflow_profiles import CANONICAL_SCHEMAS

    path = tmp_path / "runs.db"
    if damage == "other_lineage":
        schema = next(schema for v, schema in CANONICAL_SCHEMAS if v == 30)
        install(path, 30, schema)
    else:
        AgentRunsDB(path).close()
    with closing(sqlite3.connect(path)) as conn:
        if damage == "other_lineage":
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executemany(
                "INSERT OR IGNORE INTO schema_version VALUES (?)",
                ((v,) for v in range(4, 31)),
            )
        if damage == "partial":
            conn.execute("DROP INDEX idx_console_activity_receipts_unseen")
        elif damage == "extra":
            conn.execute("CREATE TABLE extra(x)")
        elif damage == "wrong_version":
            conn.execute("DELETE FROM schema_version WHERE version=30")
        else:
            conn.execute("DROP TABLE console_activity_receipts")
            if damage != "installed_stamp":
                conn.execute("DELETE FROM schema_version WHERE version=15")
            if damage == "missing_stamp":
                conn.execute("DELETE FROM schema_version WHERE version=9")
        if analyzed:
            conn.execute("ANALYZE")
        conn.commit()
    before = sha256(path.read_bytes()).digest()
    assert owner().validate(path)
    assert validate_candidate(owner(), path, Event(), migrate=False)
    with pytest.raises(ValueError, match="unsupported workflow schema"):
        AgentRunsDB(path)
    assert sha256(path.read_bytes()).digest() == before


def test_full_catalog_generation_matches_each_installed_profile():
    from pathlib import Path

    from scripts.freeze_workflow_catalogs import originals, render
    from tldw_chatbook.DB import recovery_workflow_catalog_data as data
    from tldw_chatbook.DB.recovery_workflow_profiles import (
        CANONICAL_SCHEMAS,
        REVIEW_SCHEMAS,
    )

    reviews, canonical, _ = originals()
    assert reviews == REVIEW_SCHEMAS and canonical == CANONICAL_SCHEMAS
    assert [sum(number == v for number, _ in canonical) for v in range(26, 31)] == [
        13
    ] * 5
    assert render() == Path(data.__file__).read_text()


def test_workflow_statistics_permission_is_statement_and_catalog_local():
    from tldw_chatbook.Backup_Recovery.sqlite_validation import _Restrictions

    with closing(sqlite3.connect(":memory:")) as conn:
        restrictions = _Restrictions(conn)
        restrictions.migrating = True
        restrictions.migration_owner = "db.agent_runs"
        restrictions.workflow_statistics = ("sqlite_stat1",)

        def check(action, table, column=None, database="main", source=None):
            return restrictions.authorize(action, table, column, database, source)

        assert check(sqlite3.SQLITE_DELETE, "sqlite_stat1") == sqlite3.SQLITE_DENY
        restrictions.dropping_workflow_phase = True
        assert check(sqlite3.SQLITE_DELETE, "sqlite_stat1") == sqlite3.SQLITE_OK
        for args in (
            (sqlite3.SQLITE_DELETE, "sqlite_stat4"),
            (sqlite3.SQLITE_DELETE, "sqlite_sequence"),
            (sqlite3.SQLITE_UPDATE, "sqlite_stat1", "stat"),
            (sqlite3.SQLITE_INSERT, "sqlite_stat1"),
            (sqlite3.SQLITE_ANALYZE, "agent_runs"),
            (sqlite3.SQLITE_DELETE, "sqlite_stat1", None, "temp"),
            (sqlite3.SQLITE_DELETE, "sqlite_stat1", None, "main", "some_trigger"),
        ):
            assert check(*args) == sqlite3.SQLITE_DENY
        restrictions.dropping_workflow_phase = False
        assert check(sqlite3.SQLITE_DELETE, "sqlite_stat1") == sqlite3.SQLITE_DENY


@pytest.mark.parametrize("source_version", [12, 15, 17])
@pytest.mark.parametrize("damage", [None, "stamp", "additional_stamp", "receipts"])
def test_historical_outcomes_require_complete_identity(
    tmp_path, source_version, damage
):
    from Tests.DB.fixtures.agent_runs_history import install_historical

    path = tmp_path / "historical.db"
    with closing(sqlite3.connect(path)) as conn:
        install_historical(conn, source_version)
    AgentRunsDB(path).close()
    with closing(sqlite3.connect(path)) as conn:
        if damage == "stamp":
            conn.execute("DELETE FROM schema_version WHERE version < 30")
        elif damage == "additional_stamp":
            conn.execute("INSERT INTO schema_version VALUES (3)")
        elif damage == "receipts":
            conn.execute("DROP TABLE console_activity_receipts")
            conn.execute("DELETE FROM schema_version WHERE version=15")
        conn.execute("ANALYZE")
        conn.commit()
    before = sha256(path.read_bytes()).digest()
    assert bool(owner().validate(path)) == bool(damage)
    assert bool(validate_candidate(owner(), path, Event(), migrate=False)) == bool(
        damage
    )
    assert sha256(path.read_bytes()).digest() == before
    if damage:
        with pytest.raises(ValueError, match="unsupported workflow schema"):
            AgentRunsDB(path)
        assert sha256(path.read_bytes()).digest() == before
    else:
        AgentRunsDB(path).close()


def test_capture_uses_original_cancellation_during_post_copy_check(
    tmp_path, monkeypatch
):
    from Tests.Backup_Recovery.test_core_owners import application_authority
    from tldw_chatbook.Backup_Recovery import sqlite_validation
    from tldw_chatbook.Backup_Recovery.models import StorageItem

    source = tmp_path / "agent.db"
    AgentRunsDB(source).close()
    source.chmod(0o600)
    adapter = owner()
    item = StorageItem(
        adapter.owner_id, "profile:fixture:db.agent_runs", source, "included", ()
    )
    cancel = Event()
    original = sqlite_validation._check
    seen = []

    def cancel_check(connection, installed, policy, restrictions):
        seen.append(restrictions.cancel is cancel)
        cancel.set()
        return original(connection, installed, policy, restrictions)

    monkeypatch.setattr(sqlite_validation, "_check", cancel_check)
    authority = application_authority(tmp_path, source, monkeypatch)
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    with (
        authority.maintenance(("core", "bootstrap.unbound"), 1) as session,
        session.capture_scope((source,), stage),
        pytest.raises(ValueError, match="cancelled"),
    ):
        adapter.capture(item, stage / "capture.db", cancel)
    assert seen == [True]


def test_refused_delete_journal_profile_preserves_schema_stamps_and_history(tmp_path):
    from Tests.Backup_Recovery.test_workflow_dev_integration import install
    from tldw_chatbook.DB.recovery_workflow_profiles import CANONICAL_SCHEMAS

    path = install(
        tmp_path / "delete.db", 30, next(s for v, s in CANONICAL_SCHEMAS if v == 30)
    )

    def state():
        with closing(sqlite3.connect(path)) as conn:
            return [
                conn.execute(query).fetchall()
                for query in (
                    "SELECT * FROM sqlite_schema ORDER BY type,name",
                    "SELECT * FROM schema_version",
                    "SELECT * FROM agent_runs",
                )
            ]

    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone() == ("delete",)
        conn.execute("CREATE TABLE unexpected(value)")
    before = sha256(path.read_bytes()).digest()
    original = state()
    assert owner().validate(path) and validate_candidate(
        owner(), path, Event(), migrate=False
    )
    assert sha256(path.read_bytes()).digest() == before
    with pytest.raises(ValueError, match="unsupported workflow schema catalog"):
        AgentRunsDB(path)
    assert state() == original
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone() == ("wal",)


def test_operational_preview_and_held_source_keep_registered_file_authority(
    tmp_path, monkeypatch
):
    from Tests.Backup_Recovery.test_core_owners import application_authority
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    path = tmp_path / "runs.db"
    with closing(AgentRunsDB(path)) as db, db.connection() as conn:
        conn.execute("ANALYZE")
    path.chmod(0o640)
    outside = tmp_path / "outside.db"
    outside.write_bytes(path.read_bytes())
    original = (sha256(path.read_bytes()).digest(), path.stat().st_mode)
    actual = {}
    for name in ("_capture_sqlite_target", "_preview_sqlite_target"):
        method = getattr(storage, name)

        def observe(*args, name=name, method=method):
            target = method(*args)
            if args[-1] == path and target is not None:
                actual[name] = target
            return target

        monkeypatch.setattr(storage, name, observe)
    with storage._preview_reads():
        assert owner().validate(path) == ()
    authority = application_authority(tmp_path, path, monkeypatch)
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    with (
        authority.maintenance(("core", "bootstrap.unbound"), 1) as session,
        session.capture_scope((path,), stage),
    ):
        assert owner().validate(path) == ()
        with pytest.raises(bootstrap.RecoveryRequired, match="outside_scope"):
            owner().validate(outside)
    assert set(actual) == {"_capture_sqlite_target", "_preview_sqlite_target"}
    assert all(target != path for target in actual.values())
    assert (sha256(path.read_bytes()).digest(), path.stat().st_mode) == original


def test_analyzed_profile_respects_original_resource_budget(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery import sqlite_validation as validation

    path = tmp_path / "analyzed.db"
    with closing(AgentRunsDB(path)) as db, db.connection() as conn:
        conn.execute("ANALYZE")
    before = sha256(path.read_bytes()).digest()
    monkeypatch.setattr(validation, "_STEP_BUDGET", 1)
    assert owner().validate(path) == ("sqlite_resource_limit",)
    assert validate_candidate(owner(), path, Event(), migrate=False) == (
        "sqlite_resource_limit",
    )
    assert sha256(path.read_bytes()).digest() == before


def test_populated_analyzed_v26_rebuild_matches_installed_migrations(tmp_path):
    import shutil

    from Tests.Backup_Recovery.test_workflow_dev_integration import install
    from Tests.DB.test_workflow_runs import planned
    from tldw_chatbook.Backup_Recovery.sqlite_validation import _catalog
    from tldw_chatbook.DB.recovery_workflow_profiles import CANONICAL_SCHEMAS

    path = install(
        tmp_path / "v26.db", 26, next(s for v, s in CANONICAL_SCHEMAS if v == 26)
    )
    # Seed the genuine v26 catalog with compatible rows produced by the current store.
    with closing(AgentRunsDB(tmp_path / "current-source.db")) as source:
        snap = planned(source, tmp_path)
        with source.connection() as original, closing(sqlite3.connect(path)) as conn:
            for table in (
                "automatic_work_chains",
                "workflow_runs",
                "workflow_plan_revisions",
                "workflow_phases",
                "workflow_tasks",
            ):
                columns = [
                    row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')
                ]
                names = ",".join(f'"{column}"' for column in columns)
                rows = original.execute(f'SELECT {names} FROM "{table}"').fetchall()
                assert columns and rows, table
                placeholders = ",".join("?" for _ in columns)
                conn.executemany(
                    f'INSERT INTO "{table}" ({names}) VALUES ({placeholders})', rows
                )
            conn.execute(
                "INSERT INTO workflow_phase_attempts(id,workflow_id,plan_revision,phase_id,ordinal,status) VALUES ('phase-attempt',?,1,'phase-1',1,'prepared')",
                (snap.id,),
            )
            assert conn.execute("SELECT version FROM schema_version").fetchall() == [
                (26,)
            ]
            assert not conn.execute(
                "SELECT 1 FROM sqlite_schema WHERE name='workflow_change_windows'"
            ).fetchone()
            conn.execute("ANALYZE")
            conn.commit()
    trusted = tmp_path / "trusted.db"
    shutil.copyfile(path, trusted)
    with closing(sqlite3.connect(trusted)) as reference:
        stats = [
            r[1]
            for r in _catalog(reference)
            if r[1] in ("sqlite_stat1", "sqlite_stat4")
        ]
        for table in stats:
            assert reference.execute(
                f"SELECT 1 FROM {table} WHERE tbl='workflow_phase_attempts'"
            ).fetchone() == (1,)
            assert reference.execute(
                f"SELECT 1 FROM {table} WHERE tbl='agent_runs'"
            ).fetchone() == (1,)
        for start, _, statements in owner().schema_policy().migration_steps:
            if start >= 26:
                for statement in statements:
                    reference.execute(statement)
        reference.commit()
        assert validate_candidate(owner(), path, Event(), migrate=True) == ()
        with closing(sqlite3.connect(path)) as migrated:
            assert _catalog(migrated) == _catalog(reference)
            for kind, name, _, _ in _catalog(reference):
                if kind == "table":
                    query = f'SELECT * FROM "{name}" ORDER BY rowid'
                    equal = (
                        migrated.execute(query).fetchall()
                        == reference.execute(query).fetchall()
                    )
                    assert equal, name


def test_operational_validator_authenticates_installed_policy(tmp_path):
    from dataclasses import replace

    path = tmp_path / "runs.db"
    AgentRunsDB(path).close()
    assert replace(owner(), schemas=()).validate(path) == ("unsupported_schema_policy",)


def test_local_preflight_pins_the_catalog_before_a_concurrent_receipt_retry(
    tmp_path, monkeypatch
):
    from tldw_chatbook.DB import recovery_workflow_profiles as profiles
    from tldw_chatbook.DB.recovery_workflow_schema import (
        upgrade_legacy_workflow_profile,
    )

    path = tmp_path / "receipt-retry.db"
    with (
        patch.object(
            AgentRunsDB,
            "_create_console_activity_receipts_schema",
            side_effect=sqlite3.OperationalError("optional receipt savepoint failure"),
        ),
        closing(AgentRunsDB(path)) as db,
    ):
        assert not db.receipt_capability_available

    with closing(sqlite3.connect(path, isolation_level=None)) as reader:
        assert reader.execute("PRAGMA journal_mode").fetchone() == ("wal",)
        upgrade_legacy_workflow_profile(reader)
        original = profiles.identify
        retried = False

        def retry_after_catalog(connection, catalog):
            nonlocal retried
            if connection is reader and not retried:
                retried = True
                with closing(AgentRunsDB(path)) as writer:
                    assert writer.receipt_capability_available
            return original(connection, catalog)

        monkeypatch.setattr(profiles, "identify", retry_after_catalog)
        upgrade_legacy_workflow_profile(reader)
        assert retried and not reader.in_transaction
        assert reader.execute(
            "SELECT version FROM schema_version WHERE version=15"
        ).fetchone() == (15,)
        upgrade_legacy_workflow_profile(reader)


@pytest.mark.parametrize("damaged", [False, True])
def test_local_preflight_leaves_borrowed_transaction_to_caller(tmp_path, damaged):
    from tldw_chatbook.DB.recovery_workflow_schema import (
        upgrade_legacy_workflow_profile,
    )

    path = tmp_path / "borrowed.db"
    with closing(AgentRunsDB(path)) as db:
        run_id = db.create_run(conversation_id="original-chat", agent_kind="primary")
    with (
        closing(sqlite3.connect(path)) as reader,
        closing(sqlite3.connect(path)) as observer,
    ):
        reader.execute("BEGIN")
        reader.execute(
            "UPDATE agent_runs SET conversation_id='pending' WHERE id=?", (run_id,)
        )
        if damaged:
            reader.execute("CREATE TABLE unexpected(value)")
            with pytest.raises(ValueError, match="unsupported workflow schema"):
                upgrade_legacy_workflow_profile(reader)
        else:
            upgrade_legacy_workflow_profile(reader)
        assert reader.in_transaction
        query = "SELECT conversation_id FROM agent_runs WHERE id=?"
        assert reader.execute(query, (run_id,)).fetchone() == ("pending",)
        assert observer.execute(query, (run_id,)).fetchone() == ("original-chat",)
        reader.rollback()
        assert reader.execute(query, (run_id,)).fetchone() == ("original-chat",)
