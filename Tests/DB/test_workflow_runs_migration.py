"""Actual v24 histories upgrade/reopen without changing shipped catalogs."""

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

pytestmark = pytest.mark.bootstrap_profile


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


@pytest.mark.parametrize("standalone", [False, True])
def test_v24_upgrade_reopen_and_repeat_preserves_history(tmp_path, standalone):
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.DB.recovery_operations import _AGENT_RUNS_SCHEMA

    path = tmp_path / "runs.db"
    v24(path)
    sql = Path(
        "tldw_chatbook/DB/migrations/agent_runs_v24_to_v25_workflow_runs.sql"
    ).read_text()
    if standalone:
        with closing(sqlite3.connect(path)) as conn:
            conn.executescript(sql)
            conn.executescript(sql)
    AgentRunsDB(path).close()
    AgentRunsDB(path).close()
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("SELECT id,result FROM agent_runs").fetchall() == [
            ("legacy", "preserved result")
        ]
        assert conn.execute("SELECT MAX(version) FROM schema_version").fetchone() == (
            30,
        )
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        actual = tuple(
            row[0]
            for row in conn.execute(
                "SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL ORDER BY type,name"
            )
        )
        assert actual in [
            schema for version, schema in _AGENT_RUNS_SCHEMA if version == 30
        ]


def test_standalone_and_constructor_workflow_catalogs_match_exactly(tmp_path):
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    paths = [tmp_path / "runtime.db", tmp_path / "standalone.db"]
    for path in paths:
        v24(path)
    with closing(sqlite3.connect(paths[1])) as conn:
        conn.executescript(
            Path(
                "tldw_chatbook/DB/migrations/agent_runs_v24_to_v25_workflow_runs.sql"
            ).read_text()
        )
    with closing(sqlite3.connect(paths[1])) as conn:
        conn.executescript(
            Path(
                "tldw_chatbook/DB/migrations/agent_runs_v25_to_v26_workflow_admission.sql"
            ).read_text()
        )
        conn.executescript(
            Path(
                "tldw_chatbook/DB/migrations/agent_runs_v26_to_v27_workflow_checkpoints.sql"
            ).read_text()
        )
    with closing(sqlite3.connect(paths[1])) as conn:
        conn.executescript(
            Path(
                "tldw_chatbook/DB/migrations/agent_runs_v27_to_v28_workflow_change_windows.sql"
            ).read_text()
        )
    with closing(sqlite3.connect(paths[1])) as conn:
        conn.executescript(
            Path(
                "tldw_chatbook/DB/migrations/agent_runs_v28_to_v29_workflow_repair_input.sql"
            ).read_text()
        )
    AgentRunsDB(paths[0]).close()
    with closing(sqlite3.connect(paths[1])) as conn:
        conn.executescript(
            Path(
                "tldw_chatbook/DB/migrations/agent_runs_v29_to_v30_workflow_recovery_review.sql"
            ).read_text()
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
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

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
                    "tldw_chatbook/DB/migrations/agent_runs_v24_to_v25_workflow_runs.sql"
                ).read_text()
            )
    AgentRunsDB(path).close()
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
    catalog = next(catalog for version, catalog in _AGENT_RUNS_SCHEMA if version == 28)
    with closing(sqlite3.connect(path)) as conn:
        for sql in sorted(catalog, key=lambda sql: not sql.startswith("CREATE TABLE")):
            if not sql.startswith("CREATE TABLE sqlite_sequence"):
                conn.execute(sql)
        conn.executemany(
            "INSERT INTO schema_version(version) VALUES (?)",
            [(version,) for version in range(4, 29)],
        )
        conn.execute(
            "INSERT INTO agent_runs(id,conversation_id,agent_kind,status,result,created_at,updated_at) VALUES ('old','saved','primary','completed','unchanged','then','then')"
        )
        conn.commit()
        if standalone:
            conn.executescript(
                Path(
                    "tldw_chatbook/DB/migrations/agent_runs_v28_to_v29_workflow_repair_input.sql"
                ).read_text()
            )
    AgentRunsDB(path).close()
    AgentRunsDB(path).close()
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute(
            "SELECT result FROM agent_runs WHERE id='old'"
        ).fetchone() == ("unchanged",)
        assert conn.execute("SELECT MAX(version) FROM schema_version").fetchone() == (
            30,
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
    catalog = next(catalog for version, catalog in _AGENT_RUNS_SCHEMA if version == 29)
    with closing(sqlite3.connect(path)) as conn:
        for sql in sorted(catalog, key=lambda sql: not sql.startswith("CREATE TABLE")):
            if not sql.startswith("CREATE TABLE sqlite_sequence"):
                conn.execute(sql)
        conn.executemany(
            "INSERT INTO schema_version(version) VALUES (?)",
            [(v,) for v in range(4, 30)],
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
            with pytest.raises(sqlite3.Error):
                AgentRunsDB(path)
        with closing(sqlite3.connect(path)) as conn:
            assert conn.execute(
                "SELECT MAX(version) FROM schema_version"
            ).fetchone() == (29,)
            assert "recovery_reviewed_revision" not in {
                r[1] for r in conn.execute("PRAGMA table_info(workflow_runs)")
            }
    AgentRunsDB(path).close()
    AgentRunsDB(path).close()
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("SELECT MAX(version) FROM schema_version").fetchone() == (
            30,
        )
        assert conn.execute(
            "SELECT result FROM agent_runs WHERE id='old'"
        ).fetchone() == ("unchanged",)
        assert "recovery_reviewed_revision" in {
            r[1] for r in conn.execute("PRAGMA table_info(workflow_runs)")
        }
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
