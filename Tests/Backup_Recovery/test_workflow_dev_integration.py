"""Current dev review v25 and exact pre-release workflow catalogs coexist."""

import sqlite3
from contextlib import closing
from threading import Event

import pytest

from Tests.Backup_Recovery.test_agent_runs_recovery_schema import owner
from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.DB.recovery_workflow_legacy_schema import LEGACY_WORKFLOW_SCHEMAS

pytestmark = pytest.mark.bootstrap_profile


def install(path, version, catalog):
    with closing(sqlite3.connect(path)) as conn:
        for sql in sorted(catalog, key=lambda s: not s.startswith("CREATE TABLE")):
            if not sql.startswith("CREATE TABLE sqlite_sequence"):
                conn.execute(sql)
        conn.execute("INSERT INTO schema_version VALUES (?)", (version,))
        conn.execute(
            "INSERT INTO agent_runs(id,conversation_id,agent_kind,status,result,created_at,updated_at) VALUES ('retained','chat','primary','done','private result','then','then')"
        )
        conn.commit()
    return path


def catalog(conn):
    return tuple(
        row[0]
        for row in conn.execute(
            "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type,name"
        )
    )


@pytest.mark.parametrize("version,schema", LEGACY_WORKFLOW_SCHEMAS)
@pytest.mark.parametrize("restore", [False, True])
def test_exact_legacy_catalog_upgrades_and_cold_reopens(
    tmp_path, version, schema, restore
):
    path = install(tmp_path / "runs.db", version, schema)
    before = path.read_bytes()
    assert validate_candidate(owner(), path, Event(), migrate=False) == ()
    assert path.read_bytes() == before
    if restore:
        assert validate_candidate(owner(), path, Event(), migrate=True) == ()
    else:
        AgentRunsDB(path).close()
    AgentRunsDB(path).close()
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("SELECT MAX(version) FROM schema_version").fetchone() == (
            30,
        )
        assert conn.execute(
            "SELECT 1 FROM schema_version WHERE version=?", (version,)
        ).fetchone() == (1,)
        assert conn.execute(
            "SELECT result FROM agent_runs WHERE id='retained'"
        ).fetchone() == ("private result",)
        assert "review_boundary_id" in {
            row[1] for row in conn.execute("PRAGMA table_info(automatic_wake_attempts)")
        }
        assert (
            conn.execute("SELECT * FROM automatic_review_boundary_claims").fetchall()
            == []
        )
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    assert validate_candidate(owner(), path, Event(), migrate=False) == ()


def test_shipped_review_v25_has_no_workflow_rows_to_validate(tmp_path):
    from tldw_chatbook.DB.recovery_operations import _AGENT_RUNS_SCHEMA

    schema = next(
        s
        for v, s in _AGENT_RUNS_SCHEMA
        if v == 25
        and any("CREATE TABLE automatic_review_boundary_claims " in q for q in s)
    )
    path = install(tmp_path / "review.db", 25, schema)
    assert validate_candidate(owner(), path, Event(), migrate=False) == ()
    assert validate_candidate(owner(), path, Event(), migrate=True) == ()
    assert validate_candidate(owner(), path, Event(), migrate=False) == ()


@pytest.mark.parametrize("analyzed", [False, True])
@pytest.mark.parametrize("damage", ["extra_column", "wrong_stamp", "review_hybrid", "altered_trigger"])
def test_unrecognized_local_catalog_refused_before_constructor_ddl(tmp_path, damage, analyzed):
    version, schema = LEGACY_WORKFLOW_SCHEMAS[0]
    path = install(tmp_path / "hybrid.db", version, schema)
    with closing(sqlite3.connect(path)) as conn:
        if damage == "extra_column":
            conn.execute("ALTER TABLE workflow_runs ADD COLUMN impostor TEXT")
        elif damage == "wrong_stamp":
            conn.execute("UPDATE schema_version SET version=29")
        elif damage == "altered_trigger":
            conn.execute("DROP TRIGGER workflow_result_immutable")
            conn.execute("CREATE TRIGGER workflow_result_immutable BEFORE UPDATE ON workflow_results BEGIN SELECT 1; END")
        else:
            conn.execute(
                "ALTER TABLE automatic_wake_attempts ADD COLUMN review_boundary_id TEXT"
            )
        if analyzed:
            conn.execute("ANALYZE")
        conn.commit()
        before = tuple(conn.iterdump())
    raw = path.read_bytes()
    assert owner().validate(path)
    assert validate_candidate(owner(), path, Event(), migrate=False)
    bytes_preserved = path.read_bytes() == raw
    assert bytes_preserved
    with pytest.raises(ValueError, match="unsupported.*workflow"):
        AgentRunsDB(path)
    with closing(sqlite3.connect(path)) as conn:
        dump_preserved = tuple(conn.iterdump()) == before
        assert dump_preserved
    assert validate_candidate(owner(), path, Event(), migrate=False)


@pytest.mark.parametrize("restore", [False, True])
def test_review_v25_populated_claim_and_fk_survive_admission_rebuild(tmp_path, restore):
    from tldw_chatbook.DB.recovery_operations import _AGENT_RUNS_SCHEMA

    schema = next(
        s
        for v, s in _AGENT_RUNS_SCHEMA
        if v == 25
        and any("CREATE TABLE automatic_review_boundary_claims " in q for q in s)
    )
    path = install(tmp_path / "review.db", 25, schema)
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(
            "INSERT INTO automatic_work_chains(id,conversation_id,root_submission_id,limits_json,created_at,last_observed_at) VALUES ('chain','chat','root','{}',1,1)"
        )
        conn.execute(
            "INSERT INTO automatic_work_reservations(id,chain_id,owner_id,kind,amount,state,created_at,updated_at) VALUES ('charge','chain','owner','generation',1,'uncertain',1,1)"
        )
        conn.execute(
            "INSERT INTO automatic_wake_attempts(id,chain_id,conversation_id,session_id,owner_id,generation_reservation_id,run_ids_json,state,created_at,review_boundary_id) VALUES ('wake','chain','chat','session','owner','charge','[]','review_required',1,'boundary')"
        )
        conn.execute(
            "INSERT INTO automatic_review_boundary_claims VALUES ('boundary','wake','owner','chain','review','slot','slot-attempt',1,2,'grant','target','team','override','[]','[]')"
        )
        before = tuple(conn.execute("SELECT * FROM automatic_review_boundary_claims"))
        conn.commit()
    if restore:
        assert validate_candidate(owner(), path, Event(), migrate=True) == ()
    else:
        AgentRunsDB(path).close()
    with closing(sqlite3.connect(path)) as conn:
        assert (
            tuple(conn.execute("SELECT * FROM automatic_review_boundary_claims"))
            == before
        )
        assert conn.execute(
            "SELECT review_boundary_id FROM automatic_wake_attempts"
        ).fetchone() == ("boundary",)
        assert conn.execute(
            "SELECT state,amount FROM automatic_work_reservations"
        ).fetchone() == ("uncertain", 1)
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


@pytest.mark.parametrize("stage", [25, 29])
def test_staging_bridge_cancellation_rolls_back_catalog_stamp_and_rows(
    tmp_path, monkeypatch, stage
):
    from tldw_chatbook.Backup_Recovery import sqlite_validation as validation

    schema = next(s for v, s in LEGACY_WORKFLOW_SCHEMAS if v == stage)
    path = install(tmp_path / "cancel.db", stage, schema)
    before = path.read_bytes()
    cancel = Event()
    native = validation._Restrictions.authorize

    def interrupted(self, action, first, second, database, source):
        if (
            self.migrating
            and action == sqlite3.SQLITE_CREATE_TABLE
            and first == "automatic_review_boundary_claims"
        ):
            cancel.set()
        return native(self, action, first, second, database, source)

    monkeypatch.setattr(validation._Restrictions, "authorize", interrupted)
    assert validate_candidate(owner(), path, cancel, migrate=True) == ("cancelled",)
    assert path.read_bytes() == before
    assert validate_candidate(owner(), path, Event(), migrate=False) == ()


def test_constructor_bridge_failure_rolls_back_after_column_addition(
    tmp_path, monkeypatch
):
    from tldw_chatbook.DB import recovery_workflow_schema as workflow

    version, schema = LEGACY_WORKFLOW_SCHEMAS[0]
    path = install(tmp_path / "rollback.db", version, schema)
    with closing(sqlite3.connect(path)) as conn:
        before = tuple(conn.iterdump())
    native = workflow.bridge_legacy_workflow

    def fail_after_bridge(conn, version):
        native(conn, version)
        raise ValueError("injected bridge failure")

    monkeypatch.setattr(workflow, "bridge_legacy_workflow", fail_after_bridge)
    with pytest.raises(ValueError, match="injected bridge failure"):
        AgentRunsDB(path)
    with closing(sqlite3.connect(path)) as conn:
        assert tuple(conn.iterdump()) == before


@pytest.mark.parametrize("restore", [False, True])
def test_populated_local_v29_preserves_private_results_and_uncertain_charges(
    tmp_path, restore
):
    from Tests.Backup_Recovery.test_workflow_recovery_schema import saved_result

    source = tmp_path / "current.db"
    with closing(AgentRunsDB(source)) as db:
        snapshot, task, result = saved_result(db, tmp_path)
        db.set_status(result.run_id, "error", "failed check")
        with db.transaction() as conn:
            conn.execute(
                "INSERT INTO automatic_work_reservations(id,chain_id,owner_id,kind,amount,state,created_at,updated_at) VALUES ('uncertain-charge',?,'old-owner','tokens',7,'uncertain',1,1)",
                (snapshot.chain_id,),
            )
    schema = next(
        schema for version, schema in LEGACY_WORKFLOW_SCHEMAS if version == 29
    )
    target = install(tmp_path / "legacy29.db", 29, schema)
    with (
        closing(sqlite3.connect(source)) as old,
        closing(sqlite3.connect(target)) as new,
    ):
        # Copy actual populated native rows into the frozen pre-release catalog;
        # review artifacts are the sole schema difference at this stage.
        tables = [
            row[0]
            for row in new.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT IN ('schema_version','sqlite_sequence')"
            )
        ]
        triggers = new.execute(
            "SELECT name,sql FROM sqlite_master WHERE type='trigger'"
        ).fetchall()
        for name, _sql in triggers:
            new.execute(f'DROP TRIGGER "{name}"')
        for table in tables:
            columns = [row[1] for row in new.execute(f'PRAGMA table_info("{table}")')]
            names = ",".join('"' + column + '"' for column in columns)
            new.execute(f'DELETE FROM "{table}"')
            rows = old.execute(f'SELECT {names} FROM "{table}"').fetchall()
            new.executemany(
                f'INSERT INTO "{table}" ({names}) VALUES ({",".join("?" for _ in columns)})',
                rows,
            )
        for _name, sql in triggers:
            new.execute(sql)
        new.commit()
        legacy_columns = {
            table: [row[1] for row in new.execute(f'PRAGMA table_info("{table}")')]
            for table in tables
        }
        before = {
            table: new.execute(f'SELECT * FROM "{table}"').fetchall()
            for table in tables
        }
        assert new.execute("PRAGMA foreign_key_check").fetchall() == []
    original = target.read_bytes()
    assert validate_candidate(owner(), target, Event(), migrate=False) == ()
    assert target.read_bytes() == original
    if restore:
        assert validate_candidate(owner(), target, Event(), migrate=True) == ()
    with closing(AgentRunsDB(target, reconcile_on_init=False)) as db:
        assert db.workflow_runs.read_result(snapshot.id, task.id) == result
        assert db.workflow_runs.get(snapshot.id).accounting.uncertain
        with db.connection() as conn:
            for table in tables:
                columns = legacy_columns[table]
                names = ",".join('"' + column + '"' for column in columns)
                assert [
                    tuple(row) for row in conn.execute(f'SELECT {names} FROM "{table}"')
                ] == before[table]
            assert (
                conn.execute(
                    "SELECT COUNT(*) FROM automatic_review_boundary_claims"
                ).fetchone()[0]
                == 0
            )
    with closing(AgentRunsDB(target)) as db:
        assert db.workflow_runs.get(snapshot.id).accounting.uncertain
        assert (
            db.workflow_runs.read_result(snapshot.id, task.id).handoff
            == "required private bytes"
        )


def test_workflow_catalog_owner_has_only_memory_authority(tmp_path):
    from tldw_chatbook.DB.private_sqlite import (
        SQLITE_OWNER_REGISTRY, SQLiteTargetKind, connect_private_sqlite,
    )

    policy = SQLITE_OWNER_REGISTRY["recovery.workflow_catalog"]
    assert policy.allowed_target_kinds == frozenset({SQLiteTargetKind.MEMORY})
    assert not policy.centralized_backup_allowed
    assert not policy.recovery_capture_allowed
    with closing(connect_private_sqlite("recovery.workflow_catalog", ":memory:")) as conn:
        assert conn.execute("SELECT 1").fetchone() == (1,)
    target = tmp_path / "forbidden.sqlite"
    with pytest.raises(ValueError, match="target kind"):
        connect_private_sqlite("recovery.workflow_catalog", target)
    assert not target.exists()
