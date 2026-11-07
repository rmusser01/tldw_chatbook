"""Exact installed catalog, required payload ownership and inert archive copies."""

import json
import sqlite3
import subprocess
from contextlib import closing
from threading import Event

import pytest

from Tests.Agents.test_workflow_child_launch import (
    workflow_host as workflow_host,  # noqa: PLC0414
)
from Tests.Agents.test_workflow_task_scope import (
    task_host as task_host,  # noqa: PLC0414
)
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
            conn.execute("DELETE FROM schema_version WHERE version=30")
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
            30,
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


def test_removed_archive_requires_fully_retired_allowance_family(tmp_path):
    path = tmp_path / "removed-held.db"
    with closing(AgentRunsDB(path)) as db:
        snap = planned(db, tmp_path)
        stopped = db.workflow_runs.set_status(
            snap.id, status="stopped", expected_control_revision=snap.control_revision
        )
        db.workflow_runs.remove_payloads(
            snap.id, expected_control_revision=stopped.control_revision
        )
        with db.transaction() as conn:
            conn.execute(
                "INSERT INTO automatic_work_reservations VALUES ('held-tokens',?,'native-owner','tokens',7,'committed',NULL,1,1)",
                (snap.chain_id,),
            )
    assert validate_candidate(owner(), path, Event(), migrate=False) == (
        "invalid_workflow_required_result",
    )


def _planner_canary():
    return "synthetic-" + "planner-index-" + "test9"


def _planner_witness(connection):
    """Compare private samples across engines without disclosing them."""
    import hashlib
    names = [row[0] for row in connection.execute(
        "SELECT name FROM sqlite_schema WHERE name IN ('sqlite_stat1','sqlite_stat4') ORDER BY name"
    )]
    digest = hashlib.sha256()
    for name in names:
        for row in connection.execute(f"SELECT * FROM {name} ORDER BY rowid"):
            digest.update(repr(tuple(row)).encode())
    return names, digest.hexdigest()


def _private_archive_phase(phase, log, child, marker, *, require_retired=False):
    """Run a child and expose only its phase, exit status and fixed failure reason."""
    failure = None
    try:
        code = child()
    except subprocess.TimeoutExpired as exc:
        failure = f"child timed out after {exc.timeout}s"
    except OSError as exc:
        failure = f"child launch failed (errno {exc.errno})"
    else:
        if code:
            failure = f"child exited {code}"
    contents = log.read_bytes() if log.exists() else b""
    folded = contents.lower()
    if marker in folded or marker.hex().encode() in folded:
        failure = "private marker in output"
    elif failure is None and require_retired and b"retired and reopened" not in contents:
        failure = "retirement witness missing"
    if failure:
        pytest.fail(f"{phase}: {failure}", pytrace=False)


@pytest.mark.parametrize("variant", ["ordinary", "combined"])
def test_complete_backup_restore_and_rebackup_preserve_private_workflow_results(
    tmp_path, variant,
):
    """Use the existing real complete-profile route with an actual workflow payload."""
    _workflow_archive_roundtrip(tmp_path, variant)


def _payload_witness(conn):
    """Compare exact durable bytes and admission at both edges without disclosing payloads."""
    import hashlib

    from tldw_chatbook.Agents.workflow_models import WORKFLOW_BYTES
    from tldw_chatbook.DB.workflow_runs import WorkflowRunsStore

    conn.row_factory = sqlite3.Row
    immutable = (
        "workflow_runs",
        "workflow_plan_revisions",
        "workflow_decisions",
        "workflow_task_attempts",
        "workflow_phase_attempts",
        "workflow_payload_reservations",
        "workflow_results",
        "workflow_checkpoints",
        "workflow_change_windows",
        "workflow_change_window_refs",
        "automatic_work_reservations",
        "change_snapshots",
    )
    from tldw_chatbook.DB.recovery_workflow_schema import workflow_schema_stage

    stage = workflow_schema_stage(conn)
    tables = {
        r[0] for r in conn.execute("SELECT name FROM sqlite_schema WHERE type='table'")
    }
    hashes = {}
    for table in immutable:
        rows = (
            conn.execute(f"SELECT * FROM {table} ORDER BY rowid")
            if table in tables
            else ()
        )
        body = json.dumps(
            [tuple(row) for row in rows],
            ensure_ascii=False,
            default=lambda value: value.hex(),
        )
        hashes[table] = hashlib.sha256(body.encode()).hexdigest()
    workflow_id, status = conn.execute("SELECT id,status FROM workflow_runs").fetchone()
    body_bytes = sum(
        len(row[0].encode())
        for row in conn.execute(
            "SELECT request_json FROM workflow_runs UNION ALL SELECT payload_json FROM workflow_plan_revisions "
            "UNION ALL SELECT payload_json FROM workflow_decisions"
        )
    )
    checkpoint = (
        conn.execute("SELECT payload_json FROM workflow_checkpoints").fetchone()
        if "workflow_checkpoints" in tables
        else None
    )
    if checkpoint:
        body_bytes += len(checkpoint[0].encode())
        body_bytes += len(
            conn.execute("SELECT payload_json FROM workflow_results")
            .fetchone()[0]
            .encode()
        )
    else:
        body_bytes += conn.execute(
            "SELECT COALESCE(SUM(bytes),0) FROM workflow_payload_reservations "
            "WHERE state!='removed'"
        ).fetchone()[0]
    available = WORKFLOW_BYTES - body_bytes
    WorkflowRunsStore._capacity(conn, workflow_id, available, stage=stage)
    with pytest.raises(ValueError, match="workflow_payload_capacity"):
        WorkflowRunsStore._capacity(conn, workflow_id, available + 1, stage=stage)
    return [hashes, status, available]


@pytest.mark.parametrize("remove", [False, True], ids=["retained", "removed"])
def test_complete_archive_of_settled_native_history(tmp_path, workflow_host, remove):
    from Tests.Agents.test_workflow_run_service import _native_result

    host = workflow_host
    service, _, result = _native_result(host)
    host.db.record_change_snapshots_batch(
        run_id=result.run_id,
        records=[
            {"root": str(host.root), "tracking_error": "Native retained snapshot"}
        ],
    )
    saved = service.checkpoint_task(
        result.model_copy(update={"handoff": "résultat 完了"}), drained=True
    )
    assert service._checkpoint(saved.id, result.attempt_id).change_window_refs
    service.settle_execution_phase(host.phase)
    complete = service.review_quality(
        service.prepare_quality_review(saved.id, phase_id="phase-1"), accepted=True
    )
    if remove:
        service.remove_payloads(
            saved.id,
            conversation_id=saved.request.conversation_id,
            expected_control_revision=complete.control_revision,
        )
    seed = tmp_path / "native-seed.db"
    with host.db.connection() as conn, closing(sqlite3.connect(seed)) as copy:
        conn.backup(copy)
        witness = _payload_witness(copy)
    _workflow_archive_roundtrip(tmp_path, "native", seed=seed, expected=witness)


def test_complete_archive_of_pre_checkpoint_history(tmp_path):
    from Tests.DB.test_workflow_evidence_retention import write_historical_workflow_file

    old = tmp_path / "native-legacy.db"
    with closing(AgentRunsDB(tmp_path / "source.db")) as source:
        saved_result(source, tmp_path)
        write_historical_workflow_file(source, old, 26)
    with closing(sqlite3.connect(old)) as conn:
        expected = _payload_witness(conn)
    _workflow_archive_roundtrip(tmp_path, "legacy", seed=old, expected=expected)


def _open_workflow_profile(seed, expected):
    """Create or reopen through original stores, then retire this producer process."""
    import os
    from pathlib import Path

    from Tests.Backup_Recovery.test_capture_service import (
        _populate_required_dependencies,
    )
    from tldw_chatbook.Backup_Recovery.capture_service import preview_capture

    home = Path.home()
    selector = Path(os.environ["TLDW_CONFIG_PATH"])
    options = {"staging_parent": home}
    baseline = preview_capture((selector,), options=options)
    path = next(item.path for item in baseline.items if item.owner == "db.agent_runs")
    if seed:
        from tldw_chatbook.DB.Evals_DB import EvalsDB
        from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
        from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
        from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB

        _populate_required_dependencies(baseline)
        for name, cls in (
            ("db.evals", EvalsDB),
            ("db.library_collections", LibraryCollectionsDB),
            ("db.scheduled_tasks", ScheduledTasksDB),
            ("db.subscriptions", SubscriptionsDB),
        ):
            locator = next(item.path for item in baseline.items if item.owner == name)
            cls(locator).close()
        with (
            closing(AgentRunsDB(path)) as db,
            db.connection() as target,
            closing(sqlite3.connect(seed)) as source,
        ):
            source.backup(target)
    assert owner().validate(path) == ()
    if seed:
        # The actual historical source keeps its original schema until isolated restore.
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as conn:
            assert _payload_witness(conn) == expected
    else:
        with closing(AgentRunsDB(path)) as reopened, reopened.connection() as conn:
            assert _payload_witness(conn) == expected


def _native_archive_scripts(seed, expected):
    setup = """
import asyncio, json, os, sys
from pathlib import Path
from Tests.network_guard import install, blocked_attempts
install()
for name in ('sounddevice', 'pyaudio'):
    sys.modules[name] = None
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
"""
    # Restored profile selection precedes importing any ordinary config consumer.
    select = """
from tldw_chatbook.Backup_Recovery.isolated_restore import select_profile
home = Path.home()
receipt = json.loads((home / 'restored.json').read_text())
select_profile(receipt['profile'], home / 'control')
"""
    create = """
selector = Path(os.environ['TLDW_CONFIG_PATH'])
selector.write_text('[general]\\nusers_name="default_user"\\n')
selector.chmod(0o600)
"""

    def child(initial, source):
        initialize = (
            setup
            + initial
            + (
                "from Tests.Backup_Recovery.test_workflow_recovery_schema import _open_workflow_profile\n"
                f"_open_workflow_profile({source!r}, {expected!r})\nassert not blocked_attempts()"
            )
        )
        capture = (
            setup
            + ("" if source else select)
            + (
                "from Tests.Backup_Recovery.workflow_archive_capture import _capture_workflow_profile\n"
                f"_capture_workflow_profile({source!r}, {expected!r})\nassert not blocked_attempts()\n"
                "print('retired and reopened')"
            )
        )
        return (
            "import subprocess, sys\n"
            f"subprocess.run([sys.executable, '-c', {initialize!r}], check=True, timeout=30)\n"
            + capture
        )

    return child(create, str(seed)), child(select, None)


def _workflow_archive_roundtrip(tmp_path, variant, *, seed=None, expected=None):
    import os
    import sys
    from pathlib import Path

    from Tests.Backup_Recovery.test_home_citation_retirement import _run
    from Tests.Backup_Recovery.test_temporary_media_capture import (
        _PUBLIC,
        _REOPEN,
        _RESTORE,
    )

    degraded = analyzed = variant == "combined"
    inject = """
from unittest.mock import patch
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
patch.object(AgentRunsDB, '_create_console_activity_receipts_schema',
             side_effect=sqlite3.OperationalError('injected optional receipt failure')).start()
""" if degraded else ""
    setup = """
    from Tests.Backup_Recovery.test_workflow_recovery_schema import _planner_witness, _planner_canary
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from Tests.Backup_Recovery.test_workflow_recovery_schema import saved_result
    runs_path = Path(app.chachanotes_db.db_path).parent / 'agent_runs.db'
    workflow_db = AgentRunsDB(runs_path)
    try:
        workflow, workflow_attempt, workflow_result = saved_result(workflow_db, home)
        assert workflow_db.receipt_capability_available is not DEGRADED
        with workflow_db.connection() as connection:
            if ANALYZED:
                connection.execute("INSERT INTO agent_runs(id,conversation_id,agent_kind,status,created_at,updated_at) VALUES ('stats-witness',?,'primary','done','then','then')", (_planner_canary(),))
                connection.execute("ANALYZE")
                if connection.execute("SELECT 1 FROM sqlite_schema WHERE name='sqlite_stat4'").fetchone():
                    assert connection.execute("SELECT 1 FROM sqlite_stat4 WHERE instr(sample,CAST(? AS BLOB))>0", (_planner_canary(),)).fetchone()

            (home / 'planner.json').write_text(json.dumps(_planner_witness(connection)))
        (home / 'workflow.json').write_text(json.dumps({'workflow_id': workflow.id, 'attempt_id': workflow_attempt.id, 'run_id': workflow_result.run_id, 'chain_id': workflow.chain_id}))
    finally:
        workflow_db.close()
"""
    setup = setup.replace("ANALYZED", str(analyzed)).replace("DEGRADED", str(degraded))
    # Both scripts are shared native backup fixtures. Only add workflow witnesses.
    public = _PUBLIC.replace("async def main():", inject + "\nasync def main():").replace(
        "    images = ImageGenerationService()",
        setup + "    images = ImageGenerationService()",
    )
    public = public.replace(
        "        members = [f for f in manifest['files']",
        "        assert any(f['owner_id'] == 'db.agent_runs' for f in manifest['files'])\n        assert any(o['owner_id']=='db.agent_runs' and o['schema_version']==30 for o in manifest['owners'])\n        assert _planner_canary() not in json.dumps(manifest) and _planner_canary().encode().hex() not in json.dumps(manifest).lower()\n        members = [f for f in manifest['files']",
    )
    restore = _RESTORE.replace(
        "source.rename(source.with_name('source-home-removed'))",
        "(home / 'planner.json').write_bytes((source / 'planner.json').read_bytes())\n(home / 'workflow.json').write_bytes((source / 'workflow.json').read_bytes())\nsource.rename(source.with_name('source-home-removed'))",
    )
    witness = """
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from Tests.Backup_Recovery.test_workflow_recovery_schema import _planner_witness, _planner_canary
    facts = json.loads((home / 'workflow.json').read_text())
    workflow_db = AgentRunsDB(data / 'agent_runs.db')
    try:
        assert workflow_db.receipt_capability_available is not DEGRADED
        with workflow_db.connection() as connection:
            assert list(_planner_witness(connection)) == json.loads((home / 'planner.json').read_text())
            assert tuple(r[0] for r in connection.execute("SELECT version FROM schema_version ORDER BY version")) == tuple(v for v in range(4,31) if not DEGRADED or v != 15)
        snapshot = workflow_db.workflow_runs.get(facts['workflow_id'])
        result = workflow_db.workflow_runs.read_result(snapshot.id, facts['attempt_id'])
        assert result.handoff == 'required private bytes' and result.status == 'failed'
        assert snapshot.chain_id == facts['chain_id']
        assert snapshot.accounting.used['generation'] == 0
        assert len(workflow_db.list_runs(snapshot.request.conversation_id)) == 1
    finally:
        workflow_db.close()
"""
    witness = witness.replace("DEGRADED", str(degraded))
    reopen = _REOPEN.replace("async def main():", inject + "\nasync def main():").replace(
        "    videos = app.generated_video_store",
        witness + "    videos = app.generated_video_store",
    )
    reopen = reopen.replace(
        "        retained = [f for f in manifest['files']",
        "        assert any(f['owner_id'] == 'db.agent_runs' for f in manifest['files'])\n        assert any(o['owner_id']=='db.agent_runs' and o['schema_version']==30 for o in manifest['owners'])\n        assert _planner_canary() not in json.dumps(manifest) and _planner_canary().encode().hex() not in json.dumps(manifest).lower()\n        retained = [f for f in manifest['files']",
    )
    if seed is not None:
        public, reopen = _native_archive_scripts(seed, expected)
        restore = _RESTORE
    original = tmp_path / "original"
    original.mkdir()
    marker = _planner_canary().encode()
    _private_archive_phase(
        "producer", original / "restart-output.log",
        lambda: _run(original, "temporary", "complete", script=public, return_status=True),
        marker, require_retired=True,
    )
    assert (original / "home" / "retained-temporary.tldw-backup.zip").is_file()
    restored = tmp_path / "restored"
    restored.mkdir()
    _private_archive_phase(
        "restore", restored / "restart-output.log",
        lambda: _run(restored, str(original / "home"), "isolated", script=restore, return_status=True),
        marker, require_retired=True,
    )
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
    log = tmp_path / "workflow-rebackup.log"

    def rebackup():
        with log.open("w", encoding="utf-8") as output:
            return subprocess.run(
                [sys.executable, "-c", reopen],
                cwd=Path(__file__).resolve().parents[2],
                env=environment,
                stdout=output,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=45,
                check=False,
            ).returncode

    _private_archive_phase("rebackup", log, rebackup, marker)
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


def test_private_archive_failures_keep_samples_out_of_pytest_output(tmp_path):
    import sys
    import uuid

    probe = tmp_path / "test_private_archive_output.py"
    marker = uuid.uuid4().hex.encode()
    adjacent = uuid.uuid4().hex.encode()
    (tmp_path / "marker").write_bytes(marker)
    (tmp_path / "adjacent").write_bytes(adjacent)
    probe.write_text(
        """
import subprocess
from pathlib import Path

import pytest

from Tests.Backup_Recovery.test_workflow_recovery_schema import _private_archive_phase


@pytest.mark.parametrize("outcome", ["raw", "hex", "child_status", "timeout", "missing_witness", "success"])
def test_phase(outcome, tmp_path):
    root = Path(__file__).parent
    marker = (root / "marker").read_bytes()
    adjacent = (root / "adjacent").read_bytes()
    private_log = tmp_path / "child.log"
    leaked = {"raw": marker, "hex": marker.hex().encode()}.get(outcome, b"")
    retirement = b"" if outcome == "missing_witness" else b"retired and reopened"
    private_log.write_bytes(retirement + adjacent + leaked)

    def child():
        if outcome == "timeout":
            raise subprocess.TimeoutExpired([adjacent.decode(), marker.decode()], 45)
        return 3 if outcome == "child_status" else 0

    _private_archive_phase("producer", private_log, child, marker, require_retired=True)
""",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=short", "-p", "no:cacheprovider",
         "--confcutdir", str(tmp_path), "-o", "markers=bootstrap_profile", str(probe)],
        capture_output=True, timeout=45, check=False,
    )
    public = result.stdout + result.stderr
    private_values_hidden = all(value not in public for value in (marker, marker.hex().encode(), adjacent))
    if not private_values_hidden:
        pytest.fail("private archive probe exposed its dummy values", pytrace=False)
    if result.returncode != 1:
        pytest.fail(f"private archive probe exited {result.returncode}", pytrace=False)
    reasons = (b"private marker in output", b"child exited 3", b"child timed out after 45s",
               b"retirement witness missing", b"5 failed, 1 passed")
    expected_failures = all(reason in public for reason in reasons)
    assert expected_failures
