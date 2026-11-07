"""Immutable historical payloads remain byte-stable at the record bound."""

import hashlib
import json

import pytest

from Tests.Agents.workflow_support import plan_data
from tldw_chatbook.Agents.workflow_models import (
    RECORD_BYTES,
    WorkflowPlan,
    WorkflowTaskResult,
)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@pytest.mark.parametrize("kind", ["plan", "result"])
def test_legacy_near_bound_payload_keeps_exact_bytes_and_hash(tmp_path, kind):
    if kind == "plan":
        model = WorkflowPlan
        data = model.model_validate_json(json.dumps(plan_data(tmp_path))).model_dump(
            mode="json"
        )
        task = data["phases"][0]["tasks"][0]
        task.pop("verifiers", None)
        field, key = task, "instructions"
    else:
        model = WorkflowTaskResult
        data = {
            "workflow_id": "w",
            "attempt_id": "a",
            "run_id": "r",
            "status": "completed",
            "summary": "historical model assertion",
            "handoff": "x",
            "checks": [],
        }
        field, key = data, "handoff"
    field[key] += "x" * (RECORD_BYTES - len(_canonical(data).encode()))
    historical = _canonical(data)
    assert len(historical.encode()) == RECORD_BYTES
    parsed = model.model_validate_json(historical)
    assert parsed.canonical_json() == historical
    assert (
        hashlib.sha256(parsed.canonical_json().encode()).digest()
        == hashlib.sha256(historical.encode()).digest()
    )


def write_historical_workflow_file(source, old, version):
    """Copy native histories into the exact installed pre-checkpoint catalog."""
    import sqlite3
    from contextlib import closing

    from tldw_chatbook.DB.recovery_workflow_legacy_schema import LEGACY_WORKFLOW_SCHEMAS

    catalog = next(
        schema for number, schema in LEGACY_WORKFLOW_SCHEMAS if number == version
    )
    with closing(sqlite3.connect(old)) as conn, source.connection() as original:
        for sql in sorted(catalog, key=lambda sql: not sql.startswith("CREATE TABLE")):
            if not sql.startswith("CREATE TABLE sqlite_sequence"):
                conn.execute(sql)
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_schema WHERE type='table' AND name NOT IN ('schema_version','sqlite_sequence')"
            )
        ]
        for table in sorted(tables, key=lambda name: name != "automatic_work_chains"):
            columns = [row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')]
            names = ",".join(f'"{column}"' for column in columns)
            rows = original.execute(f'SELECT {names} FROM "{table}"').fetchall()
            conn.executemany(
                f'INSERT INTO "{table}" ({names}) VALUES ({",".join("?" for _ in columns)})',
                rows,
            )
        conn.execute("INSERT INTO schema_version VALUES (?)", (version,))
        conn.commit()


@pytest.mark.bootstrap_profile
@pytest.mark.parametrize("version", [25, 26])
def test_actual_historical_catalog_near_bound_payloads_upgrade_without_rewriting(
    tmp_path, version
):
    from contextlib import closing

    from Tests.Agents.workflow_support import make_workflow_request
    from Tests.DB.test_workflow_runs import attempt
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.DB.recovery_workflow_schema import validate_workflow_rows

    source = AgentRunsDB(tmp_path / "source.db")
    try:
        snapshot = source.workflow_runs.create(
            make_workflow_request(tmp_path), launch_id="old-start"
        )
        data = WorkflowPlan.model_validate_json(
            json.dumps(plan_data(tmp_path))
        ).model_dump(mode="json")
        task = data["phases"][0]["tasks"][0]
        task["instructions"] += "x" * (RECORD_BYTES - len(_canonical(data).encode()))
        plan = WorkflowPlan.model_validate_json(_canonical(data))
        snapshot = source.workflow_runs.save_plan(
            snapshot.id, plan, expected_control_revision=1
        )
        run = source.create_run(
            conversation_id=snapshot.request.conversation_id,
            agent_kind="subagent",
            work_chain_id=snapshot.chain_id,
        )
        claim = attempt(source, snapshot, run)
        source.workflow_runs.reserve_result(snapshot.id, claim.id)
        result = WorkflowTaskResult(
            workflow_id=snapshot.id,
            attempt_id=claim.id,
            run_id=run,
            status="completed",
            summary="historical",
            handoff="x",
        )
        result = result.model_copy(
            update={
                "handoff": "x"
                * (1 + RECORD_BYTES - len(result.canonical_json().encode()))
            }
        )
        # A completed historical task must retain a genuinely terminal native row on reopen.
        assert source.set_status(run, "done")
        source.workflow_runs.publish_result(result)
        source.workflow_runs.record_retirement(
            snapshot.id, claim.id, run_id=run, drained=True
        )
        old = tmp_path / f"v{version}.db"
        write_historical_workflow_file(source, old, version)
        # A closed task can retain its completed result even on the old checkpointless
        # catalogs. Both readers must still distinguish a done native run from an error.
        import sqlite3
        from threading import Event

        from Tests.Backup_Recovery.test_agent_runs_recovery_schema import owner
        from tldw_chatbook.Agents.workflow_models import WORKFLOW_BYTES
        from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate
        from tldw_chatbook.DB.workflow_runs import WorkflowRunsStore

        for native_status in ("done", "error"):
            candidate = tmp_path / f"closed-native-{native_status}.db"
            with (
                closing(sqlite3.connect(old)) as original,
                closing(sqlite3.connect(candidate)) as passive,
            ):
                original.backup(passive)
                passive.execute(
                    "UPDATE workflow_task_attempts SET status='closed' WHERE id=?",
                    (claim.id,),
                )
                passive.execute(
                    "UPDATE agent_runs SET status=? WHERE id=?", (native_status, run)
                )
                passive.commit()
                if native_status == "done":
                    room = (
                        WORKFLOW_BYTES
                        - len(snapshot.request.canonical_json().encode())
                        - len(plan.canonical_json().encode())
                        - RECORD_BYTES
                    )
                    WorkflowRunsStore._capacity(
                        passive, snapshot.id, room, stage=version
                    )
                    with pytest.raises(ValueError, match="workflow_payload_capacity"):
                        WorkflowRunsStore._capacity(
                            passive, snapshot.id, room + 1, stage=version
                        )
            candidate.chmod(0o600)
            expected = (
                ()
                if native_status == "done"
                else ("invalid_workflow_result_ownership",)
            )
            issues = validate_candidate(owner(), candidate, Event(), migrate=False)
            if native_status == "error":
                with (
                    closing(sqlite3.connect(candidate)) as passive,
                    pytest.raises(
                        ValueError, match="invalid_workflow_result_ownership"
                    ),
                ):
                    WorkflowRunsStore._capacity(passive, "unrelated", 1, stage=version)
            assert issues == expected
        with closing(AgentRunsDB(old)) as upgraded:
            assert (
                upgraded.workflow_runs.get(snapshot.id).plans[0].canonical_json()
                == plan.canonical_json()
            )
            assert (
                upgraded.workflow_runs.read_result(
                    snapshot.id, claim.id
                ).canonical_json()
                == result.canonical_json()
            )
            assert upgraded.workflow_runs.publish_result(result).results == (result,)
            with upgraded.connection() as conn:
                assert validate_workflow_rows(conn) == ()
                assert (
                    conn.execute(
                        "SELECT count(*) FROM workflow_checkpoints"
                    ).fetchone()[0]
                    == 0
                )
                assert (
                    conn.execute(
                        "SELECT payload_hash FROM workflow_plan_revisions"
                    ).fetchone()[0]
                    == hashlib.sha256(plan.canonical_json().encode()).hexdigest()
                )
                assert (
                    conn.execute(
                        "SELECT payload_hash FROM workflow_results"
                    ).fetchone()[0]
                    == hashlib.sha256(result.canonical_json().encode()).hexdigest()
                )
    finally:
        source.close()


def payload_headroom(db, workflow_id, monkeypatch):
    """Probe actual admission at both capacity ceilings, independently."""
    from tldw_chatbook.DB import workflow_runs

    available = []
    for isolated in ("WORKFLOW_BYTES", "TOTAL_BYTES"):
        with monkeypatch.context() as limits:
            limits.setattr(workflow_runs, isolated, 2**40)
            low, high = 0, 2**28
            with db.connection() as conn:
                while low < high:
                    amount = (low + high + 1) // 2
                    try:
                        db.workflow_runs._capacity(conn, workflow_id, amount)
                    except ValueError as exc:
                        assert str(exc) == "workflow_payload_capacity"
                        high = amount - 1
                    else:
                        low = amount
            available.append(low)
    return tuple(available)


@pytest.mark.bootstrap_profile
def test_prepared_recovery_reclaims_unused_payload_capacity_and_keeps_tombstone(
    tmp_path, monkeypatch
):
    from contextlib import closing

    from Tests.DB.test_workflow_runs import planned
    from tldw_chatbook.Agents import run_log
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.DB.recovery_workflow_schema import validate_workflow_rows

    monkeypatch.setattr(
        run_log,
        "_setting",
        lambda key, default: True if key == "workflow_runs_enabled" else default,
    )
    with closing(AgentRunsDB(tmp_path / "unused.db")) as db:
        snapshot = planned(db, tmp_path)
        db.automatic_work.recover(current_owner_id="old")
        db.automatic_work.prepare_workflow_phase(
            snapshot.id,
            revision_id=1,
            phase_id="phase-1",
            owner_id="old",
            session_id="s",
            command_id="prepare",
        )
        before = payload_headroom(db, snapshot.id, monkeypatch)
        used = db.automatic_work.snapshot(snapshot.chain_id).used
        with db.connection() as conn:
            tables = (
                "automatic_work_runtime_owner",
                "automatic_wake_attempts",
                "automatic_work_reservations",
                "workflow_phase_attempts",
                "workflow_task_attempts",
                "workflow_payload_reservations",
            )
            original = {
                table: [tuple(row) for row in conn.execute(f"SELECT * FROM {table}")]
                for table in tables
            }
            conn.execute(
                "CREATE TRIGGER refuse_unused_release BEFORE UPDATE OF state ON workflow_payload_reservations "
                "WHEN NEW.state='removed' BEGIN SELECT RAISE(ABORT, 'release failed'); END"
            )
        with pytest.raises(Exception, match="release failed"):
            db.automatic_work.recover(current_owner_id="new")
        with db.connection() as conn:
            assert original == {
                table: [tuple(row) for row in conn.execute(f"SELECT * FROM {table}")]
                for table in tables
            }
            conn.execute("DROP TRIGGER refuse_unused_release")
        assert payload_headroom(db, snapshot.id, monkeypatch) == before
        db.automatic_work.recover(current_owner_id="new")
        after = payload_headroom(db, snapshot.id, monkeypatch)
        assert tuple(a - b for a, b in zip(after, before, strict=True)) == (
            RECORD_BYTES,
            RECORD_BYTES,
        )
        assert db.workflow_runs.get(snapshot.id).results == ()
        assert db.automatic_work.snapshot(snapshot.chain_id).used == used
        with db.connection() as conn:
            assert tuple(
                conn.execute(
                    "SELECT t.status,t.retirement,t.run_id,r.state FROM workflow_task_attempts t JOIN workflow_payload_reservations r ON r.attempt_id=t.id"
                ).fetchone()
            ) == ("closed", "drained", None, "removed")
            assert validate_workflow_rows(conn) == ()
        db.automatic_work.recover(current_owner_id="new")
        assert payload_headroom(db, snapshot.id, monkeypatch) == after
        assert db.workflow_runs.get(snapshot.id).status == "awaiting_resume"
        unrelated = snapshot.request.model_copy(
            update={"conversation_id": "another-saved-chat"}
        )
        assert (
            db.workflow_runs.create(unrelated, launch_id="after-unused").request
            == unrelated
        )
        import sqlite3
        from threading import Event

        from Tests.Backup_Recovery.test_agent_runs_recovery_schema import owner
        from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate

        staged = tmp_path / "recovered.db"
        with db.connection() as conn, closing(sqlite3.connect(staged)) as candidate:
            conn.backup(candidate)
        staged.chmod(0o600)
        assert validate_candidate(owner(), staged, Event(), migrate=False) == ()


@pytest.mark.bootstrap_profile
@pytest.mark.parametrize("damage", ["prepared-without-slot", "unused-foreign-phase"])
def test_native_claim_requires_owned_phase_and_full_slot_for_unrelated_admission(
    tmp_path, monkeypatch, damage
):
    import sqlite3
    from contextlib import closing
    from threading import Event

    from Tests.Backup_Recovery.test_agent_runs_recovery_schema import owner
    from Tests.DB.test_workflow_runs import planned
    from tldw_chatbook.Agents import run_log
    from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    monkeypatch.setattr(
        run_log,
        "_setting",
        lambda key, default: True if key == "workflow_runs_enabled" else default,
    )
    with closing(AgentRunsDB(tmp_path / "live.db")) as db:
        snap = planned(db, tmp_path)
        db.automatic_work.recover(current_owner_id="first")

        def prepare(row, owner_id):
            return db.automatic_work.prepare_workflow_phase(
                row.id,
                revision_id=1,
                phase_id="phase-1",
                owner_id=owner_id,
                session_id="session",
                command_id="prepare-" + row.id,
            )

        prepare(snap, "first")
        if damage == "unused-foreign-phase":
            db.automatic_work.recover(current_owner_id="next")
            other = planned(db, tmp_path, launch="other", conversation_id="other")
            prepare(other, "next")
        with db.connection() as conn:
            task_id, phase_id, state = conn.execute(
                "SELECT t.id,c.phase_attempt_id,s.state FROM workflow_task_attempts t JOIN workflow_task_claims c ON c.task_attempt_id=t.id JOIN workflow_payload_reservations s ON s.attempt_id=t.id WHERE t.workflow_id=?",
                (snap.id,),
            ).fetchone()
            if damage == "prepared-without-slot":
                assert state == "reserved"
                conn.execute(
                    "DELETE FROM workflow_payload_reservations WHERE attempt_id=?",
                    (task_id,),
                )
            else:
                assert state == "removed"
                foreign = conn.execute(
                    "SELECT id FROM workflow_phase_attempts WHERE workflow_id=?",
                    (other.id,),
                ).fetchone()[0]
                assert foreign != phase_id
                conn.execute(
                    "UPDATE workflow_task_claims SET phase_attempt_id=? WHERE task_attempt_id=?",
                    (foreign, task_id),
                )
            assert not conn.execute("PRAGMA foreign_key_check").fetchall()
            staged = tmp_path / "staged.db"
            with closing(sqlite3.connect(staged)) as copy:
                conn.backup(copy)
        staged.chmod(0o600)
        assert validate_candidate(owner(), staged, Event(), migrate=False) == (
            "invalid_workflow_task_claim",
        )
        with pytest.raises(ValueError, match="invalid_workflow_task_claim"):
            db.workflow_runs.create(
                snap.request.model_copy(update={"conversation_id": "unrelated"}),
                launch_id="unrelated",
            )


@pytest.mark.bootstrap_profile
def test_first_native_phase_coexists_with_restored_result_only_attempt(
    tmp_path, monkeypatch
):
    import sqlite3
    from contextlib import closing
    from threading import Event

    from Tests.Backup_Recovery.test_agent_runs_recovery_schema import owner
    from Tests.Backup_Recovery.test_workflow_recovery_schema import saved_result
    from tldw_chatbook.Agents import run_log
    from tldw_chatbook.Agents.workflow_models import WORKFLOW_BYTES
    from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    monkeypatch.setattr(
        run_log,
        "_setting",
        lambda key, default: True if key == "workflow_runs_enabled" else default,
    )
    old = tmp_path / "old.db"
    with closing(AgentRunsDB(tmp_path / "source.db")) as source:
        snap, prior, result = saved_result(source, tmp_path)
        assert source.set_status(result.run_id, "error")
        source.workflow_runs.record_retirement(
            snap.id, prior.id, run_id=result.run_id, drained=True
        )
        write_historical_workflow_file(source, old, 25)
    with closing(AgentRunsDB(old)) as db:
        assert db.workflow_runs.get(snap.id).results == (result,)
        db.automatic_work.recover(current_owner_id="native")
        db.automatic_work.prepare_workflow_phase(
            snap.id,
            revision_id=1,
            phase_id="phase-1",
            owner_id="native",
            session_id="session",
            command_id="first-native",
        )
    with closing(AgentRunsDB(old)) as reopened:
        with reopened.connection() as conn:
            observed = conn.execute(
                "SELECT t.ordinal,c.task_attempt_id IS NOT NULL,p.bytes,p.state FROM workflow_task_attempts t JOIN workflow_payload_reservations p ON p.attempt_id=t.id LEFT JOIN workflow_task_claims c ON c.task_attempt_id=t.id ORDER BY t.ordinal"
            ).fetchall()
            assert list(map(tuple, observed)) == [
                (1, 0, RECORD_BYTES, "published"),
                (2, 1, RECORD_BYTES, "reserved"),
            ]
        assert reopened.workflow_runs.get(snap.id).results == (result,)
        immutable = len(snap.request.canonical_json().encode()) + len(
            snap.plans[0].canonical_json().encode()
        )
        with reopened.connection() as conn:
            room = WORKFLOW_BYTES - immutable - 2 * RECORD_BYTES
            reopened.workflow_runs._capacity(conn, snap.id, room)
            with pytest.raises(ValueError, match="workflow_payload_capacity"):
                reopened.workflow_runs._capacity(conn, snap.id, room + 1)
        unrelated = snap.request.model_copy(update={"conversation_id": "unrelated"})
        assert (
            reopened.workflow_runs.create(unrelated, launch_id="unrelated").request
            == unrelated
        )
        staged = tmp_path / "staged.db"
        with reopened.connection() as conn, closing(sqlite3.connect(staged)) as copy:
            conn.backup(copy)
        staged.chmod(0o600)
        assert validate_candidate(owner(), staged, Event(), migrate=False) == ()
        # Keeping the original result cannot replace the first native phase's own claim.
        with reopened.connection() as conn:
            assert conn.execute("DELETE FROM workflow_task_claims").rowcount == 1
            assert not conn.execute("PRAGMA foreign_key_check").fetchall()
            with closing(sqlite3.connect(staged)) as copy:
                conn.backup(copy)
        with pytest.raises(ValueError, match="invalid_workflow_task_claim"):
            reopened.workflow_runs.get(snap.id)
        with pytest.raises(ValueError, match="invalid_workflow_task_claim"):
            reopened.workflow_runs.create(
                unrelated.model_copy(update={"conversation_id": "next"}),
                launch_id="next",
            )
        assert validate_candidate(owner(), staged, Event(), migrate=False) == (
            "invalid_workflow_task_claim",
        )
