"""Frozen wake schema upgrades preserve completion metadata and reject surprises."""

import sqlite3
from threading import Event

import pytest

from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.DB.recovery_operations import _AGENT_RUNS_SCHEMA, recovery_adapters

pytestmark = pytest.mark.bootstrap_profile


def owner():
    return next(
        item for item in recovery_adapters() if item.owner_id == "db.agent_runs"
    )


@pytest.mark.parametrize("version", [18, 21, 22])
def test_frozen_completion_schema_migrates_linearly_to_progress(tmp_path, version):
    path = tmp_path / "installed.db"
    catalog = next(
        catalog for installed, catalog in _AGENT_RUNS_SCHEMA if installed == version
    )
    with sqlite3.connect(path) as conn:
        for sql in sorted(
            catalog, key=lambda value: not value.startswith("CREATE TABLE")
        ):
            if not sql.startswith("CREATE TABLE sqlite_sequence"):
                conn.execute(sql)
        conn.execute("INSERT INTO schema_version VALUES (?)", (version,))
        conn.execute(
            "INSERT INTO automatic_work_chains(id,conversation_id,root_submission_id,limits_json,created_at,last_observed_at) VALUES ('chain','chat','manual','{}',0,0)"
        )
        conn.execute(
            "INSERT INTO automatic_work_reservations(id,chain_id,owner_id,kind,amount,state,created_at,updated_at) VALUES ('reserve','chain','owner','generation',1,'committed',0,0)"
        )
        conn.execute(
            "INSERT INTO automatic_wake_attempts(id,chain_id,conversation_id,session_id,owner_id,generation_reservation_id,run_ids_json,state,created_at) VALUES ('completion','chain','chat','native','owner','reserve','[]','completed',0)"
        )
    path.chmod(0o600)
    assert validate_candidate(owner(), path, Event(), migrate=True) == ()
    with sqlite3.connect(path) as conn:
        assert (
            conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] == 23
        )
        assert conn.execute(
            "SELECT state,cause,message_ids_json FROM automatic_wake_attempts"
        ).fetchone() == ("completed", "completion", "[]")
        assert (
            conn.execute(
                "SELECT count(*) FROM automatic_progress_wake_claims"
            ).fetchone()[0]
            == 0
        )


def test_fresh_progress_catalog_matches_and_unknown_table_is_refused(tmp_path):
    path = tmp_path / "fresh.db"
    db = AgentRunsDB(path, client_id="progress-schema")
    db.close()
    assert validate_candidate(owner(), path, Event(), migrate=False) == ()
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE foreign_progress(message_id TEXT)")
    conn.close()
    assert validate_candidate(owner(), path, Event(), migrate=False) == (
        "unsupported_schema",
    )
