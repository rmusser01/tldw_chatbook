"""Actual v24 histories upgrade/reopen without changing shipped catalogs."""

import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path

import pytest

pytestmark = pytest.mark.bootstrap_profile


@contextmanager
def _open_workflow_db(path):
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.DB.console_execution_owner import ConsoleExecutionOwner

    with (
        ConsoleExecutionOwner.acquire(path) as execution_owner,
        closing(AgentRunsDB(path, execution_owner=execution_owner)) as db,
    ):
        yield db


def v24(path):
    from tldw_chatbook.DB.recovery_operations import _AGENT_RUNS_SCHEMA

    schema = next(schema for version, schema in _AGENT_RUNS_SCHEMA if version == 24)
    with closing(sqlite3.connect(path)) as conn:
        for sql in sorted(schema, key=lambda s: not s.startswith("CREATE TABLE")):
            if not sql.startswith("CREATE TABLE sqlite_sequence"):
                conn.execute(sql)
        conn.execute("INSERT INTO schema_version(version) VALUES (24)")
        conn.execute(
            "INSERT INTO agent_runs(id,conversation_id,agent_kind,status,result,created_at,updated_at) VALUES ('legacy','saved','primary','completed','preserved result','then','then')"
        )
        conn.commit()
        conn.executescript(
            Path(
                "tldw_chatbook/DB/migrations/agent_runs_v24_to_v25_review_boundaries.sql"
            ).read_text()
        )


@pytest.mark.parametrize("standalone", [False, True])
def test_v24_upgrade_reopen_and_repeat_preserves_history(tmp_path, standalone):
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.DB.recovery_operations import _AGENT_RUNS_SCHEMA

    path = tmp_path / "runs.db"
    v24(path)
    sql = Path(
        "tldw_chatbook/DB/migrations/agent_runs_v25_to_v26_workflow_runs.sql"
    ).read_text()
    if standalone:
        with closing(sqlite3.connect(path)) as conn:
            conn.executescript(sql)
            conn.executescript(sql)
    for _ in range(2):
        with _open_workflow_db(path):
            pass
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("SELECT id,result FROM agent_runs").fetchall() == [
            ("legacy", "preserved result")
        ]
        assert conn.execute("SELECT MAX(version) FROM schema_version").fetchone() == (
            AgentRunsDB._CURRENT_SCHEMA_VERSION,
        )
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        actual = tuple(
            row[0]
            for row in conn.execute(
                "SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL ORDER BY type,name"
            )
        )
        assert actual in [
            schema
            for version, schema in _AGENT_RUNS_SCHEMA
            if version == AgentRunsDB._CURRENT_SCHEMA_VERSION
        ]


def test_standalone_and_constructor_workflow_catalogs_match_exactly(tmp_path):
    paths = [tmp_path / "runtime.db", tmp_path / "standalone.db"]
    for path in paths:
        v24(path)
    with closing(sqlite3.connect(paths[1])) as conn:
        conn.executescript(
            Path(
                "tldw_chatbook/DB/migrations/agent_runs_v25_to_v26_workflow_runs.sql"
            ).read_text()
        )
    with closing(sqlite3.connect(paths[1])) as conn:
        conn.executescript(
            Path(
                "tldw_chatbook/DB/migrations/agent_runs_v26_to_v27_workflow_admission.sql"
            ).read_text()
        )
        conn.executescript(
            Path(
                "tldw_chatbook/DB/migrations/agent_runs_v27_to_v28_workflow_checkpoints.sql"
            ).read_text()
        )
    with closing(sqlite3.connect(paths[1])) as conn:
        conn.executescript(
            Path(
                "tldw_chatbook/DB/migrations/agent_runs_v28_to_v29_workflow_change_windows.sql"
            ).read_text()
        )
    with closing(sqlite3.connect(paths[1])) as conn:
        conn.executescript(
            Path(
                "tldw_chatbook/DB/migrations/agent_runs_v29_to_v30_workflow_repair_input.sql"
            ).read_text()
        )
    with _open_workflow_db(paths[0]):
        pass
    with closing(sqlite3.connect(paths[1])) as conn:
        conn.executescript(
            Path(
                "tldw_chatbook/DB/migrations/agent_runs_v30_to_v31_workflow_recovery_review.sql"
            ).read_text()
        )
        for migration in (
            "agent_runs_v31_to_v32_console_command_receipts.sql",
            "agent_runs_v32_to_v33_mobile_consolidation.sql",
        ):
            conn.executescript(
                (Path("tldw_chatbook/DB/migrations") / migration).read_text()
            )
    catalogs = []
    for path in paths:
        with closing(sqlite3.connect(path)) as conn:
            catalogs.append(
                conn.execute(
                    "SELECT type,name,tbl_name,sql FROM sqlite_schema ORDER BY type,name"
                ).fetchall()
            )
    assert catalogs[0] == catalogs[1]


@pytest.mark.parametrize("standalone", [False, True])
def test_real_v24_goal_and_fleet_rows_survive_workflow_upgrade(tmp_path, standalone):
    import hashlib
    import json
    from dataclasses import asdict

    from Tests.Agents.test_goal_models import request
    from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkLimits

    path = tmp_path / "old-goals.db"
    v24(path)
    goal_request = request()
    payload = goal_request.canonical_json()
    with closing(sqlite3.connect(path)) as conn:
        for chain_id, conversation, submission, limits in (
            (
                "goal-chain",
                "goal-chat",
                "goal:old-goal",
                goal_request.policy.chain_limits(),
            ),
            ("fleet-chain", "fleet-chat", "old-fleet", AutomaticWorkLimits()),
        ):
            conn.execute(
                "INSERT INTO automatic_work_chains(id,conversation_id,root_submission_id,limits_json,created_at,last_observed_at) VALUES (?,?,?,?,1,1)",
                (chain_id, conversation, submission, json.dumps(asdict(limits))),
            )
        conn.execute(
            "INSERT INTO goal_runs(id,launch_id,payload_hash,request_json,conversation_id,chain_id,created_at,updated_at) VALUES ('old-goal','old-launch',?,?,'goal-chat','goal-chain',1,1)",
            (hashlib.sha256(payload.encode()).hexdigest(), payload),
        )
        conn.execute(
            "INSERT INTO automatic_work_reservations(id,chain_id,owner_id,kind,amount,state,created_at,updated_at) VALUES ('old-charge','fleet-chain','owner','tokens',17,'committed',1,1)"
        )
        conn.commit()
        before = {
            table: conn.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
            for table in (
                "agent_runs",
                "goal_runs",
                "automatic_work_chains",
                "automatic_work_reservations",
            )
        }
    if standalone:
        with closing(sqlite3.connect(path)) as conn:
            conn.executescript(
                Path(
                    "tldw_chatbook/DB/migrations/agent_runs_v25_to_v26_workflow_runs.sql"
                ).read_text()
            )
    with _open_workflow_db(path):
        pass
    with closing(sqlite3.connect(path)) as conn:
        assert {
            table: conn.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
            for table in before
        } == before
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


@pytest.mark.parametrize("standalone", [False, True])
def test_v28_repair_metadata_upgrade_preserves_existing_history(tmp_path, standalone):
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.DB.recovery_operations import _AGENT_RUNS_SCHEMA

    path = tmp_path / "v28.db"
    catalog = next(catalog for version, catalog in _AGENT_RUNS_SCHEMA if version == 29)
    with closing(sqlite3.connect(path)) as conn:
        for sql in sorted(catalog, key=lambda sql: not sql.startswith("CREATE TABLE")):
            if not sql.startswith("CREATE TABLE sqlite_sequence"):
                conn.execute(sql)
        conn.executemany(
            "INSERT INTO schema_version(version) VALUES (?)",
            [(version,) for version in range(4, 30)],
        )
        conn.execute(
            "INSERT INTO agent_runs(id,conversation_id,agent_kind,status,result,created_at,updated_at) VALUES ('old','saved','primary','completed','unchanged','then','then')"
        )
        conn.commit()
        if standalone:
            conn.executescript(
                Path(
                    "tldw_chatbook/DB/migrations/agent_runs_v29_to_v30_workflow_repair_input.sql"
                ).read_text()
            )
    for _ in range(2):
        with _open_workflow_db(path):
            pass
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute(
            "SELECT result FROM agent_runs WHERE id='old'"
        ).fetchone() == ("unchanged",)
        assert conn.execute("SELECT MAX(version) FROM schema_version").fetchone() == (
            AgentRunsDB._CURRENT_SCHEMA_VERSION,
        )
        assert "repair_instruction_sha256" in {
            row[1] for row in conn.execute("PRAGMA table_info(workflow_phase_attempts)")
        }
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


@pytest.mark.parametrize("fail_migration", [False, True])
def test_v29_recovery_marker_is_additive_and_failure_rolls_back(
    tmp_path, monkeypatch, fail_migration
):
    from tldw_chatbook.DB import recovery_workflow_schema
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.DB.recovery_operations import _AGENT_RUNS_SCHEMA

    path = tmp_path / "v29.db"
    catalog = next(catalog for version, catalog in _AGENT_RUNS_SCHEMA if version == 30)
    with closing(sqlite3.connect(path)) as conn:
        for sql in sorted(catalog, key=lambda sql: not sql.startswith("CREATE TABLE")):
            if not sql.startswith("CREATE TABLE sqlite_sequence"):
                conn.execute(sql)
        conn.executemany(
            "INSERT INTO schema_version(version) VALUES (?)",
            [(v,) for v in range(4, 31)],
        )
        conn.execute(
            "INSERT INTO agent_runs(id,conversation_id,agent_kind,status,result,created_at,updated_at) VALUES ('old','saved','primary','completed','unchanged','then','then')"
        )
        conn.commit()
    if fail_migration:
        migration = recovery_workflow_schema.WORKFLOW_RECOVERY_REVIEW_MIGRATION
        with monkeypatch.context() as patch:
            patch.setattr(
                recovery_workflow_schema,
                "WORKFLOW_RECOVERY_REVIEW_MIGRATION",
                (*migration[:-1], "invalid installed SQL"),
            )
            with pytest.raises(sqlite3.Error), _open_workflow_db(path):
                pass
        with closing(sqlite3.connect(path)) as conn:
            assert conn.execute(
                "SELECT MAX(version) FROM schema_version"
            ).fetchone() == (30,)
            assert "recovery_reviewed_revision" not in {
                r[1] for r in conn.execute("PRAGMA table_info(workflow_runs)")
            }
    for _ in range(2):
        with _open_workflow_db(path):
            pass
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("SELECT MAX(version) FROM schema_version").fetchone() == (
            AgentRunsDB._CURRENT_SCHEMA_VERSION,
        )
        assert conn.execute(
            "SELECT result FROM agent_runs WHERE id='old'"
        ).fetchone() == ("unchanged",)
        assert "recovery_reviewed_revision" in {
            r[1] for r in conn.execute("PRAGMA table_info(workflow_runs)")
        }
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_shipped_dev25_catalog_keeps_its_review_boundary_meaning():
    from tldw_chatbook.DB.recovery_operations import _AGENT_RUNS_SCHEMA
    from tldw_chatbook.DB.recovery_review_boundary_schema import (
        review_boundary_catalogs,
    )

    expected = {
        variant
        for version, catalog in _AGENT_RUNS_SCHEMA
        if version == 24
        for variant in review_boundary_catalogs(catalog)
    }
    assert {
        catalog for version, catalog in _AGENT_RUNS_SCHEMA if version == 25
    } == expected


@pytest.mark.parametrize("lineage", [0, 1])
@pytest.mark.parametrize("upgrade", ["constructor", "recovery", "standalone"])
def test_shipped_dev25_review_rows_survive_native_workflow_upgrade(
    tmp_path, lineage, upgrade
):
    from threading import Event

    from Tests.Agents.workflow_support import make_workflow_request
    from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.DB.recovery_operations import (
        _AGENT_RUNS_SCHEMA,
        recovery_adapters,
    )
    from tldw_chatbook.DB.recovery_review_boundary_schema import (
        review_boundary_catalogs,
    )

    path = tmp_path / "dev25.db"
    predecessor = next(
        catalog for version, catalog in _AGENT_RUNS_SCHEMA if version == 24
    )
    catalog = review_boundary_catalogs(predecessor)[lineage]
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        for sql in sorted(catalog, key=lambda sql: not sql.startswith("CREATE TABLE")):
            if not sql.startswith("CREATE TABLE sqlite_sequence"):
                conn.execute(sql)
        conn.executemany(
            "INSERT INTO schema_version(version) VALUES (?)",
            [(v,) for v in range(4, 26)],
        )
        conn.execute(
            "INSERT INTO automatic_work_chains(id,conversation_id,root_submission_id,limits_json,created_at,last_observed_at) VALUES ('review-chain','saved','manual','{}',0,0)"
        )
        conn.execute(
            "INSERT INTO automatic_work_reservations(id,chain_id,owner_id,kind,amount,state,created_at,updated_at) VALUES ('review-charge','review-chain','owner','generation',1,'committed',0,0)"
        )
        conn.execute(
            "INSERT INTO automatic_wake_attempts(id,chain_id,conversation_id,session_id,owner_id,generation_reservation_id,run_ids_json,state,created_at,review_boundary_id) VALUES ('review-attempt','review-chain','saved','session','owner','review-charge','[]','completed',0,'boundary')"
        )
        conn.execute(
            "INSERT INTO automatic_review_boundary_claims(boundary_id,attempt_id,owner_id,chain_id,review_id,slot_id,slot_attempt_id,source_epoch,context_revision,grant_id,target_fingerprint,team_fingerprint,override_fingerprint,predecessor_run_ids_json,predecessor_settlement_ids_json) VALUES ('boundary','review-attempt','owner','review-chain','review','slot','slot-attempt',1,1,'grant','target','team','override','[]','[]')"
        )
        tables = (
            "automatic_work_chains",
            "automatic_work_reservations",
            "automatic_wake_attempts",
            "automatic_review_boundary_claims",
        )
        columns = {
            table: ",".join(
                row[1] for row in conn.execute(f"PRAGMA table_info({table})")
            )
            for table in tables
        }
        before = {
            table: conn.execute(
                f"SELECT {columns[table]} FROM {table} ORDER BY rowid"
            ).fetchall()
            for table in tables
        }
        conn.commit()
    path.chmod(0o600)
    owner = next(
        item for item in recovery_adapters() if item.owner_id == "db.agent_runs"
    )
    # The standalone arm stops after the native workflow SQL chain.
    expected_version = (
        31 if upgrade == "standalone" else AgentRunsDB._CURRENT_SCHEMA_VERSION
    )
    if upgrade == "recovery":
        assert validate_candidate(owner, path, Event(), migrate=True) == ()
    elif upgrade == "standalone":
        for start, suffix in (
            (25, "runs"),
            (26, "admission"),
            (27, "checkpoints"),
            (28, "change_windows"),
            (29, "repair_input"),
            (30, "recovery_review"),
        ):
            sql = Path(
                f"tldw_chatbook/DB/migrations/agent_runs_v{start}_to_v{start + 1}_workflow_{suffix}.sql"
            ).read_text()
            with closing(sqlite3.connect(path)) as conn:
                conn.executescript(sql)
    else:
        with _open_workflow_db(path):
            pass
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("SELECT MAX(version) FROM schema_version").fetchone() == (
            expected_version,
        )
        assert {
            table: conn.execute(
                f"SELECT {columns[table]} FROM {table} ORDER BY rowid"
            ).fetchall()
            for table in before
        } == before
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        actual = tuple(
            row[0]
            for row in conn.execute(
                "SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL ORDER BY type,name"
            )
        )
        assert actual in [
            catalog
            for version, catalog in _AGENT_RUNS_SCHEMA
            if version == expected_version
        ]
    with _open_workflow_db(path) as db:
        workflow = db.workflow_runs.create(
            make_workflow_request(tmp_path), launch_id="after-review-upgrade"
        )
        assert workflow.status == "awaiting_plan"
        assert db.workflow_runs.get(workflow.id) == workflow
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    assert validate_candidate(owner, path, Event(), migrate=False) == ()


@pytest.mark.parametrize("version", range(25, 31))
@pytest.mark.parametrize("upgrade", ["constructor", "recovery"])
def test_standalone_legacy_workflow_history_remains_unreviewed_after_integration(
    tmp_path, version, upgrade
):
    from threading import Event

    from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    path, before, owner = _legacy_workflow_fixture(tmp_path, version)
    if upgrade == "recovery":
        assert validate_candidate(owner, path, Event(), migrate=True) == ()
    else:
        with _open_workflow_db(path):
            pass
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("SELECT MAX(version) FROM schema_version").fetchone() == (
            AgentRunsDB._CURRENT_SCHEMA_VERSION,
        )
        assert (
            conn.execute(
                "SELECT id,launch_id,payload_hash,request_json,conversation_id,chain_id,status,control_revision FROM workflow_runs"
            ).fetchall()
            == before
        )
        assert conn.execute(
            "SELECT recovery_reviewed_revision FROM workflow_runs"
        ).fetchone() == (0,)
        assert conn.execute(
            "SELECT COUNT(*) FROM automatic_review_boundary_claims"
        ).fetchone() == (0,)
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert {
            v for (v,) in conn.execute("SELECT version FROM schema_version")
        } >= set(range(4, 32))
    assert validate_candidate(owner, path, Event(), migrate=False) == ()
    with _open_workflow_db(path) as db:
        assert db.workflow_runs.get("legacy").status == "closed"


def _legacy_workflow_fixture(tmp_path, version):
    import hashlib
    import json
    from dataclasses import asdict

    from Tests.Agents.workflow_support import make_workflow_request
    from tldw_chatbook.DB.recovery_operations import (
        _AGENT_RUNS_SCHEMA,
        recovery_adapters,
    )
    from tldw_chatbook.DB import recovery_workflow_schema as workflow_schema

    predecessor = next(catalog for v, catalog in _AGENT_RUNS_SCHEMA if v == 24)
    transforms = (
        workflow_schema.workflow_catalog,
        workflow_schema.workflow_admission_catalog,
        workflow_schema.workflow_checkpoint_catalog,
        workflow_schema.workflow_window_catalog,
        workflow_schema.workflow_repair_input_catalog,
        workflow_schema.workflow_recovery_review_catalog,
    )
    catalog = predecessor
    for transform in transforms[: version - 24]:
        catalog = transform(catalog)
    path = tmp_path / "legacy-workflow.db"
    request = make_workflow_request(tmp_path)
    payload = request.canonical_json()
    with closing(sqlite3.connect(path)) as conn:
        for sql in sorted(catalog, key=lambda sql: not sql.startswith("CREATE TABLE")):
            if not sql.startswith("CREATE TABLE sqlite_sequence"):
                conn.execute(sql)
        conn.executemany(
            "INSERT INTO schema_version(version) VALUES (?)",
            [(v,) for v in range(4, version + 1)],
        )
        conn.execute(
            "INSERT INTO automatic_work_chains(id,conversation_id,root_submission_id,limits_json,created_at,last_observed_at) VALUES ('legacy-chain',?,'workflow:legacy',?,1,1)",
            (request.conversation_id, json.dumps(asdict(request.policy))),
        )
        conn.execute(
            "INSERT INTO workflow_runs(id,launch_id,payload_hash,request_json,conversation_id,chain_id,status,created_at,updated_at) VALUES ('legacy','legacy-launch',?,?,?,'legacy-chain','closed',1,1)",
            (
                hashlib.sha256(payload.encode()).hexdigest(),
                payload,
                request.conversation_id,
            ),
        )
        before = conn.execute(
            "SELECT id,launch_id,payload_hash,request_json,conversation_id,chain_id,status,control_revision FROM workflow_runs"
        ).fetchall()
        conn.commit()
    path.chmod(0o600)
    owner = next(
        item for item in recovery_adapters() if item.owner_id == "db.agent_runs"
    )
    return path, before, owner


def _legacy_sql_state(path):
    with closing(sqlite3.connect(path)) as conn:
        return (
            conn.execute(
                "SELECT type,name,tbl_name,sql FROM sqlite_schema ORDER BY type,name"
            ).fetchall(),
            conn.execute(
                "SELECT version FROM schema_version ORDER BY version"
            ).fetchall(),
            conn.execute("SELECT * FROM workflow_runs ORDER BY rowid").fetchall(),
        )


@pytest.mark.parametrize("version", [25, 30])
@pytest.mark.parametrize("route", ["constructor", "recovery"])
@pytest.mark.parametrize("failure", ["sql", "unknown_catalog"])
def test_legacy_workflow_bridge_refusal_preserves_exact_catalog_and_rows(
    tmp_path, monkeypatch, version, route, failure
):
    from threading import Event

    from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate
    from tldw_chatbook.DB import recovery_operations

    path, _before, owner = _legacy_workflow_fixture(tmp_path, version)
    if failure == "unknown_catalog":
        with closing(sqlite3.connect(path)) as conn:
            conn.execute(
                "CREATE INDEX unexpected_workflow_index ON workflow_runs(status)"
            )
            conn.commit()
    before = _legacy_sql_state(path)
    original = recovery_operations._legacy_workflow_upgrade

    def invalid_after_real_ddl(connection):
        end, statements = original(connection)
        return end, (*statements, "invalid installed SQL")

    with monkeypatch.context() as patch:
        if failure == "sql":
            patch.setattr(
                recovery_operations, "_legacy_workflow_upgrade", invalid_after_real_ddl
            )
        if route == "constructor":
            with pytest.raises(sqlite3.Error), _open_workflow_db(path):
                pass
        else:
            assert validate_candidate(owner, path, Event(), migrate=True) == (
                "unsupported_schema"
                if failure == "unknown_catalog"
                else "sqlite_validation_unavailable",
            )
    assert _legacy_sql_state(path) == before


@pytest.mark.parametrize("version", [25, 30])
def test_legacy_workflow_bridge_cancellation_rolls_back_real_first_ddl(
    tmp_path, monkeypatch, version
):
    from threading import Event

    from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate
    from tldw_chatbook.DB import recovery_operations

    path, _before, owner = _legacy_workflow_fixture(tmp_path, version)
    before = _legacy_sql_state(path)
    cancel = Event()
    original = recovery_operations._legacy_workflow_upgrade
    first_ddl_returned = []

    def interrupted(connection):
        end, statements = original(connection)

        def actual_statements():
            for index, statement in enumerate(statements):
                if index == 1:
                    # The actual first ALTER returned; the same transaction must roll back.
                    assert statements[0].startswith(
                        "ALTER TABLE automatic_wake_attempts ADD COLUMN review_boundary_id"
                    )
                    first_ddl_returned.append(True)
                    cancel.set()
                yield statement

        return end, actual_statements()

    with monkeypatch.context() as patch:
        patch.setattr(recovery_operations, "_legacy_workflow_upgrade", interrupted)
        assert validate_candidate(owner, path, cancel, migrate=True) == ("cancelled",)
    assert first_ddl_returned == [True]
    assert cancel.is_set()
    assert _legacy_sql_state(path) == before


@pytest.mark.parametrize("lineage", ["shipped-v25", "legacy-workflow-v30"])
def test_concurrent_workflow_upgrade_rechecks_after_the_writer_lock(
    tmp_path, monkeypatch, lineage
):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, current_thread

    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.DB.console_execution_owner import ConsoleExecutionOwner
    from tldw_chatbook.DB.recovery_operations import _AGENT_RUNS_SCHEMA

    if lineage == "shipped-v25":
        path = tmp_path / "runs.db"
        v24(path)
    else:
        path, legacy_rows, _ = _legacy_workflow_fixture(tmp_path, 30)
    before_begin = Event()
    release_begin = Event()
    main_thread = current_thread()
    original_configure = AgentRunsDB._configure_connection

    def configure(db, conn):
        conn = original_configure(db, conn)
        if current_thread() is not main_thread:
            full_sync = False

            def pause_before_lock(statement):
                nonlocal full_sync
                sql = statement.strip().upper().rstrip(";")
                if sql == "PRAGMA SYNCHRONOUS=FULL":
                    full_sync = True
                if full_sync and sql == "BEGIN IMMEDIATE" and not before_begin.is_set():
                    before_begin.set()
                    release_begin.wait(timeout=10)

            conn.set_trace_callback(pause_before_lock)
        return conn

    def open_agent_runs(execution_owner):
        AgentRunsDB(
            path, reconcile_on_init=False, execution_owner=execution_owner
        ).close()

    monkeypatch.setattr(AgentRunsDB, "_configure_connection", configure)
    with (
        ConsoleExecutionOwner.acquire(path) as execution_owner,
        ThreadPoolExecutor(max_workers=1) as pool,
    ):
        first = pool.submit(open_agent_runs, execution_owner)
        try:
            assert before_begin.wait(timeout=10)
            open_agent_runs(execution_owner)
        finally:
            release_begin.set()
        first.result(timeout=10)

    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("SELECT MAX(version) FROM schema_version").fetchone() == (
            AgentRunsDB._CURRENT_SCHEMA_VERSION,
        )
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        catalog = tuple(
            row[0]
            for row in conn.execute(
                "SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL ORDER BY type,name"
            )
        )
        assert catalog in [
            installed
            for version, installed in _AGENT_RUNS_SCHEMA
            if version == AgentRunsDB._CURRENT_SCHEMA_VERSION
        ]
        if lineage == "shipped-v25":
            assert conn.execute("SELECT id,result FROM agent_runs").fetchall() == [
                ("legacy", "preserved result")
            ]
        else:
            assert conn.execute(
                "SELECT id,launch_id,payload_hash,request_json,conversation_id,chain_id,"
                "status,control_revision FROM workflow_runs"
            ).fetchall() == legacy_rows
