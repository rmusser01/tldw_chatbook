"""Exact installed catalog, required payload ownership and inert archive copies."""

import json
import sqlite3
from contextlib import closing
from threading import Event

import pytest

from Tests.Backup_Recovery.test_agent_runs_recovery_schema import owner
from Tests.DB.test_workflow_runs import attempt, planned
from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

pytestmark = pytest.mark.bootstrap_profile


def saved_result(db, tmp_path):
    from tldw_chatbook.Agents.workflow_models import WorkflowTaskResult

    snap = planned(db, tmp_path)
    run = db.create_run(
        conversation_id=snap.request.conversation_id,
        agent_kind="subagent",
        work_chain_id=snap.chain_id,
    )
    a = attempt(db, snap, run)
    db.workflow_runs.reserve_result(snap.id, a.id)
    result = WorkflowTaskResult(
        workflow_id=snap.id,
        attempt_id=a.id,
        run_id=run,
        status="failed",
        summary="failed check",
        handoff="required private bytes",
    )
    db.workflow_runs.publish_result(result)
    return snap, a, result


def test_workflow_archive_exact_and_read_only_validation(tmp_path):
    path = tmp_path / "runs.db"
    with closing(AgentRunsDB(path)) as db:
        snap, a, result = saved_result(db, tmp_path)
    before = path.read_bytes()
    assert validate_candidate(owner(), path, Event(), migrate=False) == ()
    assert path.read_bytes() == before
    with closing(AgentRunsDB(path)) as db:
        assert db.workflow_runs.read_result(snap.id, a.id) == result
        assert db.workflow_runs.get(snap.id).accounting.used["generation"] == 0
        assert len(db.list_runs(snap.request.conversation_id)) == 1


@pytest.mark.parametrize(
    "damage",
    [
        "missing_table",
        "extra_column",
        "wrong_stamp",
        "bad_request",
        "bad_plan",
        "foreign_result",
        "foreign_attempt",
        "missing_result",
        "capacity",
    ],
)
def test_workflow_archive_rejects_hybrid_or_cross_owned_rows(tmp_path, damage):
    path = tmp_path / "runs.db"
    with closing(AgentRunsDB(path)) as db:
        snap, a, result = saved_result(db, tmp_path)
    with closing(sqlite3.connect(path)) as conn:
        if damage == "missing_table":
            conn.execute("DROP TABLE workflow_active_intervals")
        elif damage == "extra_column":
            conn.execute("ALTER TABLE workflow_runs ADD COLUMN extra TEXT")
        elif damage == "wrong_stamp":
            conn.execute("DELETE FROM schema_version WHERE version=31")
        elif damage == "bad_request":
            conn.execute("DROP TRIGGER workflow_launch_immutable")
            conn.execute("UPDATE workflow_runs SET request_json='{}'")
        elif damage == "bad_plan":
            conn.execute("DROP TRIGGER workflow_plan_immutable")
            conn.execute("UPDATE workflow_plan_revisions SET payload_json='{}'")
        elif damage == "foreign_result":
            conn.execute("DROP TRIGGER workflow_result_immutable")
            data = result.model_dump(mode="json")
            data["run_id"] = "foreign"
            conn.execute(
                "UPDATE workflow_results SET payload_json=?", (json.dumps(data),)
            )
        elif damage == "foreign_attempt":
            conn.execute(
                "INSERT INTO agent_runs(id,conversation_id,agent_kind,status,created_at,updated_at) VALUES ('legacy-foreign','other-chat','primary','done','then','then')"
            )
            conn.execute("UPDATE workflow_task_attempts SET run_id='legacy-foreign'")
        elif damage == "missing_result":
            conn.execute("DELETE FROM workflow_results")
        else:
            conn.execute("UPDATE workflow_payload_reservations SET bytes=1")
        if damage in {"bad_request", "bad_plan", "foreign_result"}:
            from tldw_chatbook.DB.recovery_workflow_schema import WORKFLOW_OBJECTS

            trigger = {
                "bad_request": "workflow_launch_immutable",
                "bad_plan": "workflow_plan_immutable",
                "foreign_result": "workflow_result_immutable",
            }[damage]
            conn.execute(
                next(
                    sql
                    for sql in WORKFLOW_OBJECTS
                    if sql.startswith("CREATE TRIGGER " + trigger)
                )
            )
        conn.commit()
    before = path.read_bytes()
    issues = validate_candidate(owner(), path, Event(), migrate=False)
    assert issues
    if damage not in {"missing_table", "extra_column", "wrong_stamp"}:
        assert issues[0].startswith("invalid_workflow"), issues
    assert path.read_bytes() == before


def test_staging_v24_migration_uses_installed_allowlist(tmp_path):
    from Tests.DB.test_workflow_runs_migration import v24

    path = tmp_path / "legacy.db"
    v24(path)
    assert validate_candidate(owner(), path, Event(), migrate=True) == ()
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("SELECT MAX(version) FROM schema_version").fetchone() == (
            31,
        )
        assert conn.execute(
            "SELECT result FROM agent_runs WHERE id='legacy'"
        ).fetchone() == ("preserved result",)


def test_settled_removed_payload_tombstones_remain_valid_archive(tmp_path):
    path = tmp_path / "removed.db"
    with closing(AgentRunsDB(path)) as db:
        snap, a, result = saved_result(db, tmp_path)
        db.workflow_runs.record_retirement(
            snap.id, a.id, run_id=result.run_id, drained=True
        )
        stopped = db.workflow_runs.set_status(
            snap.id, status="stopped", expected_control_revision=2
        )
        db.workflow_runs.remove_payloads(
            snap.id, expected_control_revision=stopped.control_revision
        )
        removed = db.workflow_runs.get(snap.id)
        assert removed.status == "removed" and removed.results == ()
    assert validate_candidate(owner(), path, Event(), migrate=False) == ()


def test_complete_backup_restore_and_rebackup_preserve_private_workflow_results(
    tmp_path,
):
    """Use the existing real complete-profile route with an actual workflow payload."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    from Tests.Backup_Recovery.test_home_citation_retirement import _run
    from Tests.Backup_Recovery.test_temporary_media_capture import (
        _PUBLIC,
        _REOPEN,
        _RESTORE,
    )

    setup = """
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from Tests.Backup_Recovery.test_workflow_recovery_schema import saved_result
    runs_path = Path(app.chachanotes_db.db_path).parent / 'agent_runs.db'
    workflow_db = AgentRunsDB(runs_path)
    try:
        workflow, workflow_attempt, workflow_result = saved_result(workflow_db, home)
        (home / 'workflow.json').write_text(json.dumps({'workflow_id': workflow.id, 'attempt_id': workflow_attempt.id, 'run_id': workflow_result.run_id, 'chain_id': workflow.chain_id}))
    finally:
        workflow_db.close()
"""
    # Both scripts are shared native backup fixtures. Only add workflow witnesses.
    public = _PUBLIC.replace(
        "    images = ImageGenerationService()",
        setup + "    images = ImageGenerationService()",
    )
    public = public.replace(
        "        members = [f for f in manifest['files']",
        "        assert any(f['owner_id'] == 'db.agent_runs' for f in manifest['files'])\n        members = [f for f in manifest['files']",
    )
    restore = _RESTORE.replace(
        "source.rename(source.with_name('source-home-removed'))",
        "(home / 'workflow.json').write_bytes((source / 'workflow.json').read_bytes())\nsource.rename(source.with_name('source-home-removed'))",
    )
    witness = """
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    facts = json.loads((home / 'workflow.json').read_text())
    workflow_db = AgentRunsDB(data / 'agent_runs.db')
    try:
        snapshot = workflow_db.workflow_runs.get(facts['workflow_id'])
        result = workflow_db.workflow_runs.read_result(snapshot.id, facts['attempt_id'])
        assert result.handoff == 'required private bytes' and result.status == 'failed'
        assert snapshot.chain_id == facts['chain_id']
        assert snapshot.accounting.used['generation'] == 0
        assert len(workflow_db.list_runs(snapshot.request.conversation_id)) == 1
    finally:
        workflow_db.close()
"""
    reopen = _REOPEN.replace(
        "    videos = app.generated_video_store",
        witness + "    videos = app.generated_video_store",
    )
    reopen = reopen.replace(
        "        retained = [f for f in manifest['files']",
        "        assert any(f['owner_id'] == 'db.agent_runs' for f in manifest['files'])\n        retained = [f for f in manifest['files']",
    )
    original = tmp_path / "original"
    original.mkdir()
    _run(original, "temporary", "complete", script=public)
    restored = tmp_path / "restored"
    restored.mkdir()
    _run(restored, str(original / "home"), "isolated", script=restore)
    environment = os.environ.copy()
    environment.update(
        HOME=str(restored / "home"),
        USERPROFILE=str(restored / "home"),
        XDG_CONFIG_HOME=str(restored / "config"),
        XDG_DATA_HOME=str(restored / "data"),
        TLDW_CONFIG_PATH=str(restored / "config" / "config.toml"),
        TLDW_TEST_MODE="1",
        TLDW_DISABLE_CONFIG_WATCH="1",
    )
    result = subprocess.run(
        [sys.executable, "-c", reopen],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    (tmp_path / "workflow-rebackup.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stderr[-6000:] + result.stdout[-1000:]
    assert (restored / "home" / "second.tldw-backup.zip").is_file()


@pytest.mark.parametrize(
    "damage",
    [
        "lost_completed_pair",
        "lost_failed_pair",
        "lost_uncertain_pair",
        "active_tombstone",
        "removed_undrained",
        "removed_reserved",
        "removed_missing_tombstone",
        "removed_uncertain_chain",
    ],
)
def test_archive_rejects_attempt_required_history_loss_without_rewrite(
    tmp_path, damage
):
    path = tmp_path / "required-history.db"
    with closing(AgentRunsDB(path)) as db:
        snap, a, _ = saved_result(db, tmp_path)
        with db.connection() as conn:
            conn.execute("DELETE FROM workflow_results WHERE attempt_id=?", (a.id,))
            if damage.startswith("lost_") or damage == "removed_missing_tombstone":
                conn.execute(
                    "DELETE FROM workflow_payload_reservations WHERE attempt_id=?",
                    (a.id,),
                )
            else:
                conn.execute(
                    "UPDATE workflow_payload_reservations SET state=? WHERE attempt_id=?",
                    ("reserved" if damage == "removed_reserved" else "removed", a.id),
                )
            if damage.startswith("lost_"):
                conn.execute(
                    "UPDATE workflow_task_attempts SET status=? WHERE id=?",
                    (damage.split("_")[1], a.id),
                )
            if damage.startswith("removed_"):
                conn.execute(
                    "UPDATE workflow_runs SET status='removed' WHERE id=?", (snap.id,)
                )
                if damage != "removed_undrained":
                    conn.execute(
                        "UPDATE workflow_task_attempts SET retirement='drained' WHERE id=?",
                        (a.id,),
                    )
            if damage == "removed_uncertain_chain":
                conn.execute(
                    "UPDATE automatic_work_chains SET status='review_required' WHERE id=?",
                    (snap.chain_id,),
                )
            assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    before = path.read_bytes()
    assert validate_candidate(owner(), path, Event(), migrate=False) == (
        "invalid_workflow_required_result",
    )
    assert path.read_bytes() == before


@pytest.mark.parametrize("reserved", [False, True])
def test_prepared_no_result_archive_remains_valid(tmp_path, reserved):
    path = tmp_path / "prepared.db"
    with closing(AgentRunsDB(path)) as db:
        snap = planned(db, tmp_path)
        a = attempt(db, snap)
        if reserved:
            db.workflow_runs.reserve_result(snap.id, a.id)
    before = path.read_bytes()
    assert validate_candidate(owner(), path, Event(), migrate=False) == ()
    assert path.read_bytes() == before


def test_archive_refuses_disagreeing_attempt_and_result_outcomes(tmp_path):
    path = tmp_path / "outcome-mismatch.db"
    with closing(AgentRunsDB(path)) as db:
        _, a, _ = saved_result(db, tmp_path)
        with db.connection() as conn:
            conn.execute(
                "UPDATE workflow_task_attempts SET status='completed' WHERE id=?",
                (a.id,),
            )
            assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    before = path.read_bytes()
    assert validate_candidate(owner(), path, Event(), migrate=False) == (
        "invalid_workflow_result_ownership",
    )
    assert path.read_bytes() == before


def test_repair_migration_authorizer_keeps_unrelated_actions_denied():
    from tldw_chatbook.Backup_Recovery.sqlite_validation import _Restrictions

    with closing(sqlite3.connect(":memory:")) as conn:
        conn.execute("CREATE TABLE workflow_phase_attempts(id TEXT)")
        conn.execute("CREATE TABLE unrelated(id TEXT)")
        restrictions = _Restrictions(conn, Event())
        restrictions.migrating = True
        restrictions.migration_owner = "db.agent_runs"
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("PRAGMA quick_check(unrelated)")
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute(
                "CREATE TRIGGER workflow_repair_input_immutable BEFORE UPDATE ON unrelated BEGIN SELECT 1; END"
            )
        restrictions.migrating = False
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute(
                "CREATE TRIGGER workflow_repair_input_immutable BEFORE UPDATE ON workflow_phase_attempts BEGIN SELECT 1; END"
            )


@pytest.mark.parametrize("legacy", [False, True])
def test_workflow_statistics_preserve_exact_schema_validation_and_upgrade(
    tmp_path, legacy
):
    path = tmp_path / "analyzed.db"
    if legacy:
        from Tests.DB.test_workflow_runs_migration import v24

        v24(path)
    else:
        with closing(AgentRunsDB(path)) as db:
            saved_result(db, tmp_path)
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("ANALYZE")
        conn.commit()
        assert conn.execute(
            "SELECT 1 FROM sqlite_schema WHERE name='sqlite_stat1'"
        ).fetchone()
    before = path.read_bytes()
    assert validate_candidate(owner(), path, Event(), migrate=False) == ()
    assert path.read_bytes() == before
    assert owner().validate(path) == ()
    assert validate_candidate(owner(), path, Event(), migrate=True) == ()
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("SELECT MAX(version) FROM schema_version").fetchone() == (
            31,
        )
        conn.execute("PRAGMA writable_schema=ON")
        conn.execute(
            "UPDATE sqlite_schema SET sql='CREATE TABLE sqlite_stat1(tbl,idx,stat,unapproved)' WHERE name='sqlite_stat1'"
        )
        conn.commit()
    before_damage = path.read_bytes()
    assert validate_candidate(owner(), path, Event(), migrate=False)
    assert path.read_bytes() == before_damage
