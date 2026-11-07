"""Required workflow checkpoints never promote model assertions to proof."""

from contextlib import closing
from types import SimpleNamespace

import pytest

from Tests.Agents.test_workflow_child_launch import (
    workflow_host as workflow_host,  # noqa: PLC0414
)
from Tests.Agents.test_workflow_task_scope import invoke
from Tests.Agents.test_workflow_task_scope import (
    task_host as task_host,  # noqa: PLC0414
)
from Tests.DB.test_workflow_runs import attempt, planned
from tldw_chatbook.Agents.workflow_models import WorkflowCheck, WorkflowTaskResult
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

pytestmark = pytest.mark.bootstrap_profile


def test_model_done_and_passed_label_require_original_native_custody(tmp_path):
    from tldw_chatbook.Agents.workflow_run_service import WorkflowRunService

    with closing(AgentRunsDB(tmp_path / "runs.db")) as db:
        snapshot = planned(db, tmp_path)
        run = db.create_run(
            conversation_id=snapshot.request.conversation_id,
            agent_kind="subagent",
            work_chain_id=snapshot.chain_id,
        )
        claim = attempt(db, snapshot, run)
        db.workflow_runs.reserve_result(snapshot.id, claim.id)
        result = WorkflowTaskResult(
            workflow_id=snapshot.id,
            attempt_id=claim.id,
            run_id=run,
            status="completed",
            summary="Model says done",
            handoff="Required private handoff",
            checks=(
                WorkflowCheck(
                    id="unit-tests",
                    status="passed",
                    verifier="unit-tests",
                    arguments=("true",),
                    input_manifest="0" * 64,
                    output_manifest="0" * 64,
                    exit_code=0,
                    reason="model_claim",
                ),
            ),
        )
        # Even passive retirement storage cannot fabricate physical owner proof.
        db.workflow_runs.record_retirement(
            snapshot.id, claim.id, run_id=run, drained=True
        )
        service = WorkflowRunService(db, SimpleNamespace(workspace_registry=None))
        with pytest.raises(ValueError, match="native.*custody|native.*receipt"):
            service.checkpoint_task(result, drained=True)
        assert service.get(snapshot.id).results == ()


@pytest.mark.parametrize("output_size,complete", [(40_000, True), (100_000, False)])
def test_native_verifier_capture_ignores_log_cap_and_honors_own_cap(
    task_host, monkeypatch, output_size, complete
):
    import hashlib
    import shlex

    from tldw_chatbook.Agents.raw_shell_tool_provider import RawShellToolProvider
    from tldw_chatbook.Agents.workflow_run_service import WorkflowRunService
    from tldw_chatbook.Chat.console_raw_cli import RawCliRuntime
    from tldw_chatbook.Tools import raw_cli_executor

    runtime = RawCliRuntime(lambda: True)
    runtime.arm()
    runtime.grant_model_session("session")
    monkeypatch.setattr(raw_cli_executor, "configured_max_record_bytes", lambda: 16)

    def edit(data, root):
        wrapper = root / "verify.sh"
        wrapper.write_text(f"#!/bin/bash\nprintf '%0{output_size}d' 1\n")
        wrapper.chmod(0o700)
        task = data["phases"][0]["tasks"][0]
        task["expected_checks"] = ["unit-tests"]
        task["footprint"].append({"path": "verify.sh", "access": "read"})
        task["verifiers"] = [
            {
                "id": "unit-tests",
                "executor_tool_id": "shell_exec",
                "verifier_path": str(wrapper),
                "verifier_sha256": hashlib.sha256(wrapper.read_bytes()).hexdigest(),
                "input_paths": ["source.py"],
            }
        ]

    host = task_host(
        mode="exclusive",
        tools=("shell_exec",),
        plan_edit=edit,
        configure=lambda registry, root, binding: registry.register_provider(
            RawShellToolProvider(
                runtime=runtime,
                console_session_id="session",
                initial_directory=lambda: root,
            )
        ),
    )
    service = WorkflowRunService(host.db, SimpleNamespace(workspace_registry=None))
    host.controller.app.raw_cli_runtime = runtime
    service.prepare_task_inputs(host.authorization)
    host.bind()
    try:
        unrelated = invoke(host, "shell_exec", {"command": "true", "shell": "bash"})
        assert unrelated.ok, unrelated.error
        assert host.authorization._required_evidence[0].records == []
        result = invoke(
            host,
            "shell_exec",
            {"command": shlex.join((str(host.root / "verify.sh"),)), "shell": "bash"},
        )
        assert result.ok, result.error
        evidence = host.authorization._required_evidence[0].records[0]
        assert evidence.complete is complete
        assert evidence.cleanup_proven
        assert evidence.fresh
        if complete:
            assert len(evidence.stdout) == output_size
        assert evidence.verifier_id == "unit-tests"
    finally:
        runtime.shutdown()


def _native_result(host, *, thread_failure=False, monkeypatch=None):
    from Tests.Agents.test_agent_service import ScriptedChat
    from Tests.Agents.test_workflow_child_launch import (
        _join,
        _launch,
        _prepared_service,
    )
    from tldw_chatbook.Agents.agent_service import _WorkflowChildLaunchFailed
    from tldw_chatbook.Agents.workflow_run_service import WorkflowRunService

    service = WorkflowRunService(host.db, SimpleNamespace(workspace_registry=None))
    messages = service.prepare_task_inputs(host.authorization)
    native, _ = _prepared_service(host, ScriptedChat(["done"]), seed=messages)
    if thread_failure:
        import threading

        original = threading.Thread.start

        def start(thread):
            if thread.name.startswith("fleet-"):
                raise RuntimeError("test thread exhaustion")
            return original(thread)

        monkeypatch.setattr(threading.Thread, "start", start)
        with pytest.raises(_WorkflowChildLaunchFailed) as failed:
            _launch(native, host)
        receipt = failed.value.receipt
    else:
        receipt = _launch(native, host)
        _join(host)
    service.bind_native_receipt(host.authorization, receipt)
    result = WorkflowTaskResult(
        workflow_id=host.authorization.claim.workflow_id,
        attempt_id=receipt.attempt_id,
        run_id=receipt.run_id,
        status="failed" if thread_failure else "completed",
        summary="native work settled",
        handoff="Required private handoff survives logs",
    )
    return service, receipt, result


def test_real_native_checkpoint_is_atomic_idempotent_and_retains_claim_on_failure(
    workflow_host,
):
    import sqlite3

    host = workflow_host
    service, _receipt, result = _native_result(host)
    with host.db.connection() as conn:
        conn.execute(
            "CREATE TRIGGER refuse_checkpoint BEFORE INSERT ON workflow_checkpoints BEGIN SELECT RAISE(ABORT,'checkpoint test refusal'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="test refusal"):
        service.checkpoint_task(result, drained=True)
    with host.db.connection() as conn:
        assert conn.execute("SELECT count(*) FROM workflow_results").fetchone()[0] == 0
        assert (
            conn.execute(
                "SELECT retirement FROM workflow_task_attempts WHERE id=?",
                (result.attempt_id,),
            ).fetchone()[0]
            == "pending"
        )
        conn.execute("DROP TRIGGER refuse_checkpoint")
    with host.activity.prelaunch(footprint=host.authorization.footprint) as blocked:
        assert blocked is None
    first = service.checkpoint_task(result, drained=True)
    assert first.results == (result,)
    assert service.checkpoint_task(result, drained=True) == first
    with pytest.raises(ValueError, match="identity_conflict"):
        service.checkpoint_task(
            result.model_copy(update={"handoff": "changed"}), drained=True
        )
    with host.activity.prelaunch(footprint=host.authorization.footprint) as free:
        assert free is not None


def test_native_settled_pair_charges_exact_stored_utf8_after_reopen(
    workflow_host, tmp_path
):
    from tldw_chatbook.Agents.workflow_models import RECORD_BYTES, WORKFLOW_BYTES

    host = workflow_host
    service, receipt, result = _native_result(host)
    result = result.model_copy(update={"handoff": "Verificación 完了 👍"})
    saved = service.checkpoint_task(result, drained=True)
    with host.db.connection() as conn:
        immutable = sum(
            len(row[0].encode())
            for table, column in (
                ("workflow_runs", "request_json"),
                ("workflow_plan_revisions", "payload_json"),
                ("workflow_decisions", "payload_json"),
            )
            for row in conn.execute(f"SELECT {column} FROM {table}")
        )
        pair = sum(
            len(
                conn.execute(f"SELECT payload_json FROM {table}").fetchone()[0].encode()
            )
            for table in ("workflow_checkpoints", "workflow_results")
        )
        with pytest.raises(ValueError, match="workflow_payload_capacity"):
            host.db.workflow_runs._capacity(
                conn, saved.id, WORKFLOW_BYTES - immutable - pair
            )
    service.settle_execution_phase(host.phase)
    completed = service.review_quality(
        service.prepare_quality_review(saved.id, phase_id="phase-1"), accepted=True
    )
    import sqlite3
    from pathlib import Path
    from threading import Event

    from Tests.Backup_Recovery.test_agent_runs_recovery_schema import owner
    from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate

    unicode_path = tmp_path / "utf16.db"
    utf8_stage = tmp_path / "utf8.db"
    with (
        host.db.connection() as conn,
        closing(sqlite3.connect(unicode_path)) as clone,
        closing(sqlite3.connect(utf8_stage)) as staged,
    ):
        conn.backup(staged)
        path = conn.execute("PRAGMA database_list").fetchone()[2]
        clone.execute("PRAGMA encoding='UTF-16le'")
        clone.executescript("\n".join(conn.iterdump()))
    assert owner().validate(Path(path)) == ()
    for candidate, encoding in ((utf8_stage, "UTF-8"), (unicode_path, "UTF-16le")):
        candidate.chmod(0o600)
        with closing(AgentRunsDB(candidate)) as database, database.connection() as conn:
            assert conn.execute("PRAGMA encoding").fetchone()[0] == encoding
            stored = conn.execute(
                "SELECT SUM(length(CAST(body AS BLOB))) FROM ("
                "SELECT request_json AS body FROM workflow_runs UNION ALL "
                "SELECT payload_json FROM workflow_plan_revisions UNION ALL "
                "SELECT payload_json FROM workflow_decisions)"
            ).fetchone()[0]
            available = WORKFLOW_BYTES - stored - pair
            assert (
                conn.execute(
                    "SELECT bytes FROM workflow_payload_reservations"
                ).fetchone()[0]
                == RECORD_BYTES
            )
            database.workflow_runs._capacity(conn, completed.id, available)
            with pytest.raises(ValueError, match="workflow_payload_capacity"):
                database.workflow_runs._capacity(conn, completed.id, available + 1)
            assert (
                database.workflow_runs.read_result(saved.id, receipt.attempt_id)
                == result
            )

            for mutation, error in (
                (
                    "UPDATE workflow_results SET payload_hash='wrong'",
                    "result_ownership",
                ),
                (
                    "UPDATE workflow_checkpoints SET payload_hash='wrong'",
                    "invalid_workflow_checkpoint",
                ),
                ("DELETE FROM workflow_task_claims", "invalid_workflow_task_claim"),
                (
                    "UPDATE automatic_work_chains SET conversation_id='other'",
                    "result_ownership",
                ),
            ):
                conn.execute("SAVEPOINT corrupt_pair")
                try:
                    conn.execute("DROP TRIGGER workflow_result_immutable")
                    conn.execute("DROP TRIGGER workflow_checkpoint_immutable")
                    conn.execute("DROP TRIGGER automatic_chain_identity_immutable")
                    conn.execute(mutation)
                    with pytest.raises(ValueError, match=error):
                        database.workflow_runs._capacity(conn, completed.id, 0)
                finally:
                    conn.execute("ROLLBACK TO corrupt_pair")
                    conn.execute("RELEASE corrupt_pair")
            for mutation in (
                "DELETE FROM workflow_checkpoints",
                "UPDATE workflow_runs SET status='paused'",
                "UPDATE workflow_phase_attempts SET retirement='unknown'",
                "UPDATE automatic_work_chains SET status='review_required'",
                "UPDATE automatic_work_reservations SET state='uncertain' WHERE kind='child_launch'",
            ):
                conn.execute("SAVEPOINT held_pair")
                try:
                    conn.execute(mutation)
                    database.workflow_runs._capacity(
                        conn, completed.id, WORKFLOW_BYTES - stored - RECORD_BYTES
                    )
                    with pytest.raises(ValueError, match="workflow_payload_capacity"):
                        database.workflow_runs._capacity(
                            conn,
                            completed.id,
                            WORKFLOW_BYTES - stored - RECORD_BYTES + 1,
                        )
                finally:
                    conn.execute("ROLLBACK TO held_pair")
                    conn.execute("RELEASE held_pair")
        assert (
            validate_candidate(owner(), Path(candidate), Event(), migrate=False) == ()
        )


def test_native_removal_frees_only_its_actual_bytes_in_full_portfolio(workflow_host):
    import hashlib
    import json

    from tldw_chatbook.Agents.workflow_models import (
        RECORD_BYTES,
        TOTAL_BYTES,
        WORKFLOW_BYTES,
    )

    host = workflow_host
    db = host.db
    service, _, result = _native_result(host)
    result = result.model_copy(update={"handoff": "résultat" * 500})
    native = service.checkpoint_task(result, drained=True)
    service.settle_execution_phase(host.phase)
    native = service.review_quality(
        service.prepare_quality_review(native.id, phase_id="phase-1"), accepted=True
    )
    owners = [native]
    for index in range(TOTAL_BYTES // WORKFLOW_BYTES):
        snap = planned(
            db,
            host.root,
            launch=f"fixture-{index}",
            conversation_id=f"portfolio-{index}",
        )
        owners.append(snap)
    with db.automatic_work.transaction() as conn:
        pair = conn.execute(
            "SELECT SUM(length(CAST(body AS BLOB))) FROM "
            "(SELECT payload_json body FROM workflow_results UNION ALL SELECT payload_json FROM workflow_checkpoints)"
        ).fetchone()[0]
        native_rows = conn.execute(
            "SELECT request_json FROM workflow_runs WHERE id=? UNION ALL SELECT payload_json FROM workflow_plan_revisions WHERE workflow_id=? UNION ALL SELECT payload_json FROM workflow_decisions WHERE workflow_id=?",
            (native.id, native.id, native.id),
        )
        remaining = (
            TOTAL_BYTES - sum(len(row[0].encode()) for row in native_rows) - pair
        )
        revisions = {native.id: native.control_revision}
        for owner in owners[1:]:
            # Materialize real immutable JSON, hashes and plan/task links up to exactly 4 MiB per owner.
            charged = len(owner.request.canonical_json().encode()) + sum(
                len(item.canonical_json().encode()) for item in owner.plans
            )
            template = owner.plans[-1].canonical_json()
            data = json.loads(template)
            task = data["phases"][0]["tasks"][0]
            base = task["instructions"]
            target = min(WORKFLOW_BYTES, remaining)
            remaining -= target
            room = target - charged
            revision = len(owner.plans)
            while room:
                length = min(RECORD_BYTES, room)
                # Keep the final revision large enough for a valid minimum plan.
                if 0 < room - length < len(template.encode()):
                    length -= len(template.encode())
                task["instructions"] = base + "x" * (length - len(template.encode()))
                payload = json.dumps(
                    data, sort_keys=True, ensure_ascii=False, separators=(",", ":")
                )
                assert len(payload.encode()) == length
                revision += 1
                conn.execute(
                    "INSERT INTO workflow_plan_revisions VALUES (?,?,?,?)",
                    (
                        owner.id,
                        revision,
                        hashlib.sha256(payload.encode()).hexdigest(),
                        payload,
                    ),
                )
                conn.execute(
                    "INSERT INTO workflow_phases SELECT workflow_id,?,phase_id,ordinal FROM workflow_phases WHERE workflow_id=? AND plan_revision=1",
                    (revision, owner.id),
                )
                conn.execute(
                    "INSERT INTO workflow_tasks SELECT workflow_id,?,phase_id,task_id,ordinal FROM workflow_tasks WHERE workflow_id=? AND plan_revision=1",
                    (revision, owner.id),
                )
                room -= length
            added = revision - len(owner.plans)
            conn.execute(
                "UPDATE workflow_runs SET control_revision=control_revision+?,progress_cursor=progress_cursor+? WHERE id=?",
                (added, added, owner.id),
            )
            revisions[owner.id] = owner.control_revision + added
        assert remaining == 0
        trace = []
        conn.set_trace_callback(trace.append)
        try:
            db.workflow_runs._capacity(conn, None, 0)
        finally:
            conn.set_trace_callback(None)
        assert len(trace) == 5
        assert (
            "WITH charges" in trace[0]
            and "k.payload_hash AS checkpoint_hash" in trace[-1]
        )
        from tldw_chatbook.DB.recovery_workflow_schema import validate_workflow_rows

        counts = [0, 0]

        def observe(statement):
            for index, marker in enumerate(
                ("WITH charges", "k.payload_hash AS checkpoint_hash")
            ):
                counts[index] += marker in statement

        conn.set_trace_callback(observe)
        try:
            assert validate_workflow_rows(conn) == ()
        finally:
            conn.set_trace_callback(None)
        assert counts == [1, 1]

    request = native.request.model_copy(
        update={"objective": "x", "conversation_id": "new-portfolio"}
    )
    request = request.model_copy(
        update={"objective": "x" * (pair - len(request.canonical_json().encode()) + 1)}
    )
    assert len(request.canonical_json().encode()) == pair
    with pytest.raises(ValueError, match="workflow_payload_capacity"):
        db.workflow_runs.create(request, launch_id="cannot-fit")
    for owner in owners:
        revision = revisions[owner.id]
        if owner.id != native.id:
            revision = db.workflow_runs.set_status(
                owner.id, status="stopped", expected_control_revision=revision
            ).control_revision
        assert (
            db.workflow_runs.remove_payloads(
                owner.id, expected_control_revision=revision
            ).status
            == "removed"
        )
    admitted = db.workflow_runs.create(request, launch_id="released-by-actual-pair")
    closed = db.workflow_runs.set_status(
        admitted.id,
        status="stopped",
        expected_control_revision=admitted.control_revision,
    )
    db.workflow_runs.remove_payloads(
        admitted.id, expected_control_revision=closed.control_revision
    )
    with pytest.raises(ValueError, match="workflow_payload_capacity"):
        db.workflow_runs.create(request, launch_id="immutable-full")


def test_passive_charging_counts_every_query_across_two_native_roots(
    workflow_host, task_host, tmp_path
):
    import json

    from tldw_chatbook.DB.recovery_workflow_schema import validate_workflow_rows

    db = workflow_host.db
    histories = []
    for index in range(2):
        host = (
            workflow_host if index == 0 else task_host(tools=("fs_read",), shared_db=db)
        )
        if index:
            host.owner.finish_root()
            host.claim.settle(durable=True)
        service, _, result = _native_result(host)
        db.record_change_snapshots_batch(
            run_id=result.run_id,
            records=[
                {"root": str(host.root), "tracking_error": "Retained native reference"}
            ],
        )
        service.checkpoint_task(result, drained=True)
        service.settle_execution_phase(host.phase)
        histories.append(
            service.review_quality(
                service.prepare_quality_review(result.workflow_id, phase_id="phase-1"),
                accepted=True,
            )
        )
    assert len({row.chain_id for row in histories}) == 2

    # A checkpoint can point at B's genuine attempt and independently at A's parent.
    # The installed FK rules admit both IDs. Removing A must not erase B's proof.
    a, b = histories
    with db.connection() as conn:

        def retained():
            return tuple(
                tuple(tuple(row) for row in conn.execute(statement))
                for statement in (
                    "SELECT id,control_revision,status FROM workflow_runs ORDER BY id",
                    "SELECT * FROM workflow_checkpoints ORDER BY attempt_id",
                    "SELECT * FROM workflow_results ORDER BY attempt_id",
                    "SELECT * FROM workflow_payload_reservations ORDER BY attempt_id",
                )
            )

        trigger = conn.execute(
            "SELECT sql FROM sqlite_schema WHERE name='workflow_checkpoint_immutable'"
        ).fetchone()[0]
        conn.execute("DROP TRIGGER workflow_checkpoint_immutable")
        conn.execute(
            "UPDATE workflow_checkpoints SET workflow_id=? WHERE workflow_id=?",
            (a.id, b.id),
        )
        conn.execute(trigger)
        assert not conn.execute("PRAGMA foreign_key_check").fetchall()
        before = retained()
        with pytest.raises(ValueError, match="invalid_workflow_checkpoint"):
            db.workflow_runs.remove_payloads(
                a.id, expected_control_revision=a.control_revision
            )
        assert retained() == before
        conn.execute("DROP TRIGGER workflow_checkpoint_immutable")
        conn.execute(
            "UPDATE workflow_checkpoints SET workflow_id=? WHERE attempt_id IN (SELECT id FROM workflow_task_attempts WHERE workflow_id=?)",
            (b.id, b.id),
        )
        conn.execute(trigger)

    def counts(conn, operation):
        statements = []
        conn.set_trace_callback(statements.append)
        try:
            operation()
        finally:
            conn.set_trace_callback(None)
        return {
            "all": len(statements),
            "charges": sum("WITH charges(" in sql for sql in statements),
            "held_roots": sum("WITH held_members(" in sql for sql in statements),
            "phase_claim_sets": sum(
                "AS phase_attempt,ph.workflow_id" in sql for sql in statements
            ),
            "slot_states": sum("AS claim_owned" in sql for sql in statements),
            "proofs": sum(
                "k.payload_hash AS checkpoint_hash" in sql for sql in statements
            ),
            "checkpoint_refs": sum(
                "SELECT COUNT(*) FROM change_snapshots WHERE run_id=" in sql
                for sql in statements
            ),
            "targeted_root_checks": sum(
                "FROM automatic_work_chains c WHERE (c.id=" in sql
                or "FROM automatic_work_reservations r JOIN automatic_work_chains c ON c.id=r.chain_id WHERE (c.id="
                in sql
                for sql in statements
            ),
        }

    evidence = {}
    for removed in range(3):
        with db.connection() as conn:
            size = counts(conn, lambda: db.workflow_runs._capacity(conn, None, 0))

            def archive():
                assert validate_workflow_rows(conn) == ()

            full = counts(conn, archive)
            evidence[str(removed)] = {"capacity": size, "archive": full}
            assert size == {
                "all": 7 - removed,
                "charges": 1,
                "held_roots": 1,
                "slot_states": 1,
                "phase_claim_sets": 1,
                "proofs": 1,
                "checkpoint_refs": 2 - removed,
                "targeted_root_checks": 0,
            }
            for category in (
                "charges",
                "held_roots",
                "slot_states",
                "phase_claim_sets",
                "proofs",
                "checkpoint_refs",
                "targeted_root_checks",
            ):
                assert full[category] == size[category]
            plan = conn.execute(
                "EXPLAIN QUERY PLAN SELECT COUNT(*) FROM change_snapshots WHERE run_id=? AND id IN (?)",
                ("index", 1),
            ).fetchall()
            assert all("SEARCH" in row[3] for row in plan)
        if removed < 2:
            row = histories[removed]
            db.workflow_runs.remove_payloads(
                row.id, expected_control_revision=row.control_revision
            )
    (tmp_path / "query-counts.json").write_text(json.dumps(evidence))


@pytest.mark.parametrize(
    "damage,reason",
    [
        ("removed-accepted", "invalid_workflow_required_result"),
        ("missing-required", "invalid_workflow_required_result"),
        ("native-error", "invalid_workflow_result_ownership"),
        ("checkpointless-claim", "invalid_workflow_task_claim"),
        ("noncanonical-link", "invalid_workflow_checkpoint"),
        ("canonical-link-to-spaced-result", None),
    ],
)
def test_native_admission_removal_and_restricted_reader_agree_on_slots(
    workflow_host, tmp_path, damage, reason
):
    import hashlib
    import json
    import sqlite3
    from threading import Event

    from Tests.Backup_Recovery.test_agent_runs_recovery_schema import owner
    from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate

    host = workflow_host
    service, _, result = _native_result(host)
    if damage != "removed-accepted":
        service.checkpoint_task(result, drained=True)
        service.settle_execution_phase(host.phase)
        terminal = service.review_quality(
            service.prepare_quality_review(result.workflow_id, phase_id="phase-1"),
            accepted=True,
        )
    with host.db.connection() as conn:
        if damage == "removed-accepted":
            assert (
                conn.execute("SELECT status FROM workflow_task_attempts").fetchone()[0]
                == "accepted"
            )
            conn.execute("UPDATE workflow_payload_reservations SET state='removed'")
        elif damage == "missing-required":
            conn.execute("DELETE FROM workflow_checkpoints")
            conn.execute("DELETE FROM workflow_results")
            conn.execute("UPDATE workflow_payload_reservations SET state='reserved'")
        elif damage == "native-error":
            conn.execute(
                "UPDATE agent_runs SET status='error' WHERE id=?", (result.run_id,)
            )
        elif damage == "checkpointless-claim":
            conn.execute("DELETE FROM workflow_checkpoints")
            conn.execute("DELETE FROM workflow_task_claims")
        else:
            triggers = [
                row[0]
                for row in conn.execute(
                    "SELECT sql FROM sqlite_schema WHERE name IN ('workflow_checkpoint_immutable','workflow_result_immutable')"
                )
            ]
            conn.execute("DROP TRIGGER workflow_checkpoint_immutable")
            conn.execute("DROP TRIGGER workflow_result_immutable")
            data = " " + result.canonical_json()
            digest = hashlib.sha256(data.encode()).hexdigest()
            checkpoint = json.loads(
                conn.execute(
                    "SELECT payload_json FROM workflow_checkpoints"
                ).fetchone()[0]
            )
            if damage == "noncanonical-link":
                checkpoint["result_digest"] = digest
            body = json.dumps(
                checkpoint, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            )
            conn.execute(
                "UPDATE workflow_results SET payload_json=?,payload_hash=?",
                (data, digest),
            )
            conn.execute(
                "UPDATE workflow_checkpoints SET payload_json=?,payload_hash=?",
                (body, hashlib.sha256(body.encode()).hexdigest()),
            )
            for definition in triggers:
                conn.execute(definition)
            if reason is not None:
                with pytest.raises(
                    ValueError, match="native_checkpoint_result_changed"
                ):
                    service._checkpoint(result.workflow_id, result.attempt_id)
            else:
                assert (
                    service._checkpoint(
                        result.workflow_id, result.attempt_id
                    ).result_digest
                    == checkpoint["result_digest"]
                )
                from tldw_chatbook.Agents.workflow_models import WORKFLOW_BYTES

                immutable = conn.execute(
                    "SELECT SUM(length(CAST(body AS BLOB))) FROM (SELECT request_json body FROM workflow_runs UNION ALL SELECT payload_json FROM workflow_plan_revisions UNION ALL SELECT payload_json FROM workflow_decisions)"
                ).fetchone()[0]
                room = (
                    WORKFLOW_BYTES - immutable - len(data.encode()) - len(body.encode())
                )
                host.db.workflow_runs._capacity(conn, terminal.id, room)
                with pytest.raises(ValueError, match="workflow_payload_capacity"):
                    host.db.workflow_runs._capacity(conn, terminal.id, room + 1)
        assert not conn.execute("PRAGMA foreign_key_check").fetchall()
        candidate = tmp_path / "staged.db"
        with closing(sqlite3.connect(candidate)) as staged:
            conn.backup(staged)
    candidate.chmod(0o600)
    observed = {
        "archive": validate_candidate(owner(), candidate, Event(), migrate=False)
    }
    with host.db.connection() as conn:
        before = tuple(
            tuple(row)
            for row in conn.execute(
                "SELECT id,status,control_revision FROM workflow_runs"
            )
        )

    def refused(operation):
        try:
            operation()
        except ValueError as error:
            return error.args
        return ()

    from Tests.Agents.workflow_support import make_workflow_request

    request = make_workflow_request(tmp_path).model_copy(
        update={"conversation_id": "unrelated"}
    )
    observed["admission"] = refused(
        lambda: host.db.workflow_runs.create(request, launch_id="unrelated")
    )
    expected = {
        "archive": (reason,) if reason else (),
        "admission": (
            "workflow_required_result_missing"
            if reason == "invalid_workflow_required_result"
            else reason,
        )
        if reason
        else (),
    }
    if damage != "removed-accepted":
        observed["remove"] = refused(
            lambda: service.remove_payloads(
                terminal.id,
                conversation_id=terminal.request.conversation_id,
                expected_control_revision=terminal.control_revision,
            )
        )
        expected["remove"] = expected["admission"]
    assert observed == expected
    if reason is not None:
        with host.db.connection() as conn:
            assert (
                tuple(
                    tuple(row)
                    for row in conn.execute(
                        "SELECT id,status,control_revision FROM workflow_runs"
                    )
                )
                == before
            )
    else:
        assert service.get(terminal.id).status == "removed"


def test_original_checkpoint_and_phase_can_finish_after_native_stop(workflow_host):
    from tldw_chatbook.Agents.workflow_models import WorkflowDecision

    host = workflow_host
    service, _, result = _native_result(host)
    current = service.get(result.workflow_id)
    decision = WorkflowDecision(
        id="stop-before-checkpoint",
        workflow_id=current.id,
        plan_revision=host.phase.attempt.revision_id,
        phase_id=host.phase.attempt.phase_id,
        control_revision=current.control_revision,
        action="stop",
        reason="user_request",
    )
    stopped = host.phase._control(decision)
    assert stopped.status == "stopped"
    with pytest.raises(ValueError, match="workflow_has_active_evidence"):
        service.remove_payloads(
            stopped.id,
            conversation_id=stopped.request.conversation_id,
            expected_control_revision=stopped.control_revision,
        )
    saved = service.checkpoint_task(result, drained=True)
    assert saved.status == "stopped" and saved.results == (result,)
    final = service.settle_execution_phase(host.phase, outcome="cancelled")
    assert (
        final.status == "stopped" and final.control_revision == stopped.control_revision
    )
    assert final.accounting.used == stopped.accounting.used
    assert (
        service.remove_payloads(
            final.id,
            conversation_id=final.request.conversation_id,
            expected_control_revision=final.control_revision,
        ).status
        == "removed"
    )


def test_real_thread_start_failure_retains_original_receipt_until_checkpoint(
    workflow_host, monkeypatch
):
    host = workflow_host
    service, receipt, result = _native_result(
        host, thread_failure=True, monkeypatch=monkeypatch
    )
    assert receipt.failure == "thread_start"
    assert receipt.owner.run_id == result.run_id
    assert host.authorization._native_launch[0] is receipt
    with host.activity.prelaunch(footprint=host.authorization.footprint) as blocked:
        assert blocked is None
    saved = service.checkpoint_task(result, drained=True)
    assert saved.results == (result,)


def test_last_phase_approval_and_completed_quality_are_distinct(workflow_host):
    host = workflow_host
    service, _receipt, result = _native_result(host)
    approval = service.prepare_decision(result.workflow_id, phase_id="phase-1")
    assert approval.action == "approve" and approval.evidence.ready
    before = service.prepare_quality_review(result.workflow_id, phase_id="phase-1")
    assert before.action == "complete" and not before.evidence.ready
    service.checkpoint_task(result, drained=True)
    host.db.automatic_work.settle_workflow_phase(
        host.phase.attempt.id, owner_id="owner", drained=True
    )
    quality = service.prepare_quality_review(result.workflow_id, phase_id="phase-1")
    assert quality.evidence.ready, quality.reason
    (host.root / "source.py").write_text("external change")
    with pytest.raises(ValueError, match="stale_workflow_decision"):
        service.validate_decision(quality)
    with pytest.raises(ValueError, match="quality_decision_required"):
        service.review_quality(approval, accepted=True)


def test_required_output_needs_native_eof_even_with_zero_exit_and_cleanup(
    tmp_path, monkeypatch
):
    import threading

    from Tests.Tools.test_raw_cli_executor_process import _admit, _request
    from tldw_chatbook.Tools import raw_cli_executor as raw

    original = raw.RawShellExecutor._consume

    class MissingStderrEnd:
        def __init__(self, messages):
            self.messages = messages

        def get(self, **kwargs):
            while True:
                event = self.messages.get(**kwargs)
                if event[:2] != ("stream_end", "stderr"):
                    return event

    def consume(*args):
        values = list(args)
        values[2] = MissingStderrEnd(values[2])
        return original(*values)

    monkeypatch.setattr(raw.RawShellExecutor, "_consume", staticmethod(consume))
    monkeypatch.setattr(raw, "_RAW_POST_EXIT_DRAIN_SECONDS", 0.05)
    capture = raw.RequiredCliOutput()
    result = raw.RawShellExecutor().execute(
        _request(tmp_path, "printf observed"),
        cancel_event=threading.Event(),
        on_event=lambda event: None,
        admit_worker=_admit,
        required_output=capture,
    )
    assert result.exit_code == 0 and result.cleanup_proven
    assert capture.stdout == "observed"
    assert not capture.sealed


def test_file_inputs_are_references_and_changed_seed_or_source_cannot_launch(task_host):
    import json

    from Tests.Agents.test_agent_service import ScriptedChat
    from Tests.Agents.test_workflow_child_launch import _launch, _prepared_service
    from tldw_chatbook.Agents.workflow_run_service import WorkflowRunService

    def edit(data, root):
        data["phases"][0]["tasks"][0]["inputs"] = ["source.py"]
        (root / "source.py").write_text("private literal body")

    host = task_host(plan_edit=edit)
    host.owner.finish_root()
    host.claim.settle(durable=True)
    service = WorkflowRunService(host.db, SimpleNamespace(workspace_registry=None))
    messages = service.prepare_task_inputs(host.authorization)
    assert "private literal body" not in json.dumps(messages)
    assert json.loads(messages[1]["content"])[0]["read_with"] == "fs_read"
    with pytest.raises(ValueError, match="required_task_seed_changed"):
        _prepared_service(
            host,
            ScriptedChat(["done"]),
            seed=messages + ({"role": "user", "content": "unrelated conversation"},),
        )
    native, _ = _prepared_service(host, ScriptedChat(["done"]), seed=messages)
    (host.root / "source.py").write_text("changed after preparation")
    with pytest.raises(ValueError, match="required_inputs_changed"):
        _launch(native, host)
    assert not host.authorization._native_launch


@pytest.mark.parametrize("changed_before_checkpoint", [False, True])
@pytest.mark.parametrize(
    "exit_code,output_size", [(0, 40_000), (7, 40_000), (7, 150_000)]
)
def test_native_verifier_checkpoint_reopens_with_complete_private_output(
    task_host, monkeypatch, changed_before_checkpoint, exit_code, output_size
):
    import hashlib
    import shlex

    from Tests.Agents.test_agent_service import ScriptedChat, fence
    from Tests.Agents.test_workflow_child_launch import (
        _join,
        _launch,
        _prepared_service,
    )
    from tldw_chatbook.Agents.raw_shell_tool_provider import RawShellToolProvider
    from tldw_chatbook.Agents.workflow_run_service import WorkflowRunService
    from tldw_chatbook.Chat.console_raw_cli import RawCliRuntime
    from tldw_chatbook.Tools import raw_cli_executor

    runtime = RawCliRuntime(lambda: True)
    runtime.arm()
    runtime.grant_model_session("session")
    monkeypatch.setattr(raw_cli_executor, "configured_max_record_bytes", lambda: 0)

    def edit(data, root):
        wrapper = root / "verify.sh"
        wrapper.write_text(
            f"#!/bin/bash\nprintf '%0{output_size}d' 1\nexit {exit_code}\n"
        )
        wrapper.chmod(0o700)
        task = data["phases"][0]["tasks"][0]
        task.update(expected_checks=["unit-tests"], inputs=["source.py"])
        task["footprint"].append({"path": "verify.sh", "access": "read"})
        task["verifiers"] = [
            {
                "id": "unit-tests",
                "executor_tool_id": "shell_exec",
                "verifier_path": str(wrapper),
                "verifier_sha256": hashlib.sha256(wrapper.read_bytes()).hexdigest(),
                "input_paths": ["source.py"],
                "arguments": ["approved argument"],
            }
        ]

    host = task_host(
        mode="exclusive",
        tools=("shell_exec",),
        plan_edit=edit,
        configure=lambda registry, root, binding: registry.register_provider(
            RawShellToolProvider(
                runtime=runtime,
                console_session_id="session",
                initial_directory=lambda: root,
            )
        ),
    )
    host.controller.app.raw_cli_runtime = runtime
    host.owner.finish_root()
    host.claim.settle(durable=True)
    service = WorkflowRunService(host.db, SimpleNamespace(workspace_registry=None))
    messages = service.prepare_task_inputs(host.authorization)
    spec = host.authorization.task.verifiers[0]
    native, _ = _prepared_service(
        host,
        ScriptedChat(
            [
                fence(
                    "shell_exec",
                    {
                        "command": shlex.join((spec.verifier_path, *spec.arguments)),
                        "shell": "bash",
                    },
                ),
                "done",
            ]
        ),
        seed=messages,
    )
    try:
        receipt = _launch(native, host)
        _join(host)
        service.bind_native_receipt(host.authorization, receipt)
        result = WorkflowTaskResult(
            workflow_id=host.authorization.claim.workflow_id,
            attempt_id=receipt.attempt_id,
            run_id=receipt.run_id,
            status="completed",
            summary="Executed verifier",
            handoff="Complete required handoff",
        )
        if changed_before_checkpoint:
            (host.root / "source.py").write_text("changed after verifier")
        saved = service.checkpoint_task(result, drained=True)
        assert host.db.get_run(receipt.run_id)["status"] == "done"
        assert service.checkpoint_task(result, drained=True) == saved
        assert saved.results[0].checks[0].reason == (
            "native_evidence_unavailable"
            if changed_before_checkpoint or output_size == 150_000
            else "native_evidence"
        )
        with pytest.raises(ValueError, match="result_identity_conflict"):
            service.checkpoint_task(
                result.model_copy(update={"handoff": "changed"}), drained=True
            )
        if changed_before_checkpoint or output_size == 150_000:
            assert saved.results[0].status == "completed"
        if changed_before_checkpoint:
            assert saved.results[0].checks[0].status == "stale"
            with pytest.raises(ValueError, match="native_check_missing_or_stale"):
                service._fresh_task_result(saved, host.authorization.task)
            return
        if exit_code:
            evidence = service.read_checkpoint(saved.id, result.attempt_id).verifiers[0]
            assert evidence.exit_code == exit_code
            assert saved.results[0].checks[0].status == "failed"
            assert evidence.complete is (output_size == 40_000)
            assert saved.results[0].status == (
                "failed" if evidence.complete else "completed"
            )
            assert evidence.cleanup_proven and evidence.fresh
            return
        assert saved.results[0].checks[0].status == "passed"
        reopened = AgentRunsDB(host.db.db_path_str)
        try:
            history = WorkflowRunService(reopened, _history_persistence(host.root))
            checkpoint = history._checkpoint(saved.id, result.attempt_id)
            evidence = checkpoint.verifiers[0]
            assert evidence.complete and len(evidence.stdout) == 40_000
            assert evidence.verifier_sha256 == spec.verifier_sha256
            assert history.get(saved.id).results[0].handoff == result.handoff
            assert (
                history._fresh_task_result(
                    history.get(saved.id), host.authorization.task
                )
                .checks[0]
                .status
                == "passed"
            )
            (host.root / "verify.sh").write_text("#!/bin/bash\nfalse\n")
            with pytest.raises(ValueError, match="native_check_missing_or_stale"):
                history._fresh_task_result(
                    history.get(saved.id), host.authorization.task
                )
        finally:
            reopened.close()
    finally:
        runtime.shutdown()


def _history_persistence(root):
    # Passive registry metadata only: this object cannot issue native authority.
    binding = SimpleNamespace(
        status=SimpleNamespace(value="ready"),
        workspace_id="w",
        binding_kind=SimpleNamespace(value="local-filesystem"),
        locator=str(root),
        metadata={"access": "rw"},
    )
    return SimpleNamespace(
        workspace_registry=SimpleNamespace(
            get_workspace=lambda identity: SimpleNamespace(archived=False),
            get_runtime_binding=lambda identity: binding,
            list_binding_exclusions=lambda identity: (),
        )
    )


@pytest.mark.parametrize(
    "producer_state", ["completed", "failed", "model_only", "stale"]
)
def test_same_phase_outputs_defer_at_approval_but_require_fresh_native_handoff(
    task_host, producer_state
):
    import copy
    import json

    from Tests.Agents.test_agent_service import ScriptedChat, fence
    from Tests.Agents.test_workflow_child_launch import (
        _join,
        _launch,
        _prepared_service,
    )
    from tldw_chatbook.Agents.workflow_run_service import WorkflowRunService
    from tldw_chatbook.Agents.workflow_task_scope import _prepare_workflow_task

    def edit(data, root):
        task = data["phases"][0]["tasks"][0]
        task["outputs"] = ["artifact.txt"]
        task["footprint"].append({"path": "artifact.txt", "access": "write"})
        consumer = copy.deepcopy(task)
        consumer.update(
            id="task-2",
            instructions="Consume the approved handoff",
            dependencies=["task-1"],
            inputs=["task-1:artifact.txt"],
            outputs=["source.py"],
            footprint=[
                {"path": "source.py", "access": "write"},
                {"path": "artifact.txt", "access": "read"},
            ],
        )
        data["phases"][0]["tasks"].append(consumer)

    host = task_host(plan_edit=edit)
    host.owner.finish_root()
    host.claim.settle(durable=True)
    service = WorkflowRunService(host.db, _history_persistence(host.root))
    workflow_id = host.authorization.claim.workflow_id
    approval = service.prepare_decision(workflow_id, phase_id="phase-1")
    assert approval.evidence.ready and not (host.root / "artifact.txt").exists()
    messages = service.prepare_task_inputs(host.authorization)
    native, _ = _prepared_service(
        host,
        ScriptedChat(
            [
                fence(
                    "fs_write",
                    {
                        "path": "artifact.txt",
                        "content": "native artifact",
                        "expected_absent": True,
                    },
                ),
                "done",
            ]
        ),
        seed=messages,
    )
    receipt = _launch(native, host)
    _join(host)
    service.bind_native_receipt(host.authorization, receipt)
    assert (host.root / "artifact.txt").read_text() == "native artifact"
    result = WorkflowTaskResult(
        workflow_id=workflow_id,
        attempt_id=receipt.attempt_id,
        run_id=receipt.run_id,
        status="failed" if producer_state == "failed" else "completed",
        summary="Producer settled",
        handoff="Complete native producer handoff",
    )
    if producer_state != "model_only":
        service.checkpoint_task(result, drained=True)
    if producer_state == "stale":
        (host.root / "artifact.txt").write_text("external edit")
    if producer_state == "completed":
        consumer = _prepare_workflow_task(
            host.phase, task_id="task-2", registry=host.registry, binding=host.binding
        )
        seed = service.prepare_task_inputs(consumer)
        assert json.loads(seed[1]["content"])[0]["handoff"] == result.handoff
        assert "native artifact" not in json.dumps(seed)
    else:
        with pytest.raises(
            ValueError,
            match="native_checkpoint_missing|task_not_completed|stale_task_output",
        ):
            consumer = _prepare_workflow_task(
                host.phase,
                task_id="task-2",
                registry=host.registry,
                binding=host.binding,
            )
            service.prepare_task_inputs(consumer)
        if producer_state in {"failed", "model_only"}:
            assert "task-2" not in host.phase._task_scopes
            with host.db.connection() as conn:
                assert tuple(
                    conn.execute(
                        "SELECT t.status,c.child_reservation_id,t.run_id FROM workflow_task_attempts t "
                        "JOIN workflow_task_claims c ON c.task_attempt_id=t.id "
                        "WHERE t.workflow_id=? AND t.task_id='task-2'",
                        (workflow_id,),
                    ).fetchone()
                ) == ("prepared", None, None)


@pytest.mark.parametrize("failure", ["no_row", "uncertain", "durable_setup"])
def test_original_failed_launch_receipt_distinguishes_no_row_and_unknown(
    workflow_host, monkeypatch, failure
):
    from Tests.Agents.test_agent_service import ScriptedChat
    from Tests.Agents.test_workflow_child_launch import _launch, _prepared_service
    from Tests.DB.test_workflow_evidence_retention import payload_headroom
    from tldw_chatbook.Agents.agent_service import _WorkflowChildLaunchFailed
    from tldw_chatbook.Agents.workflow_models import RECORD_BYTES
    from tldw_chatbook.Agents.workflow_run_service import WorkflowRunService
    from tldw_chatbook.DB.recovery_workflow_schema import validate_workflow_rows

    host = workflow_host
    service = WorkflowRunService(host.db, _history_persistence(host.root))
    seed = service.prepare_task_inputs(host.authorization)
    native, _ = _prepared_service(host, ScriptedChat(["must not run"]), seed=seed)
    create = host.db.create_run

    def refuse(*args, **kwargs):
        if failure == "durable_setup":
            create(*args, **kwargs)
        raise OSError("create run interrupted")

    monkeypatch.setattr(host.db, "create_run", refuse)
    if failure == "uncertain":
        monkeypatch.setattr(
            host.db,
            "get_run",
            lambda *args: (_ for _ in ()).throw(OSError("database unavailable")),
        )
    workflow_id = host.authorization.claim.workflow_id
    before = payload_headroom(host.db, workflow_id, monkeypatch)
    with pytest.raises(_WorkflowChildLaunchFailed) as failed:
        _launch(native, host)
    after = payload_headroom(host.db, workflow_id, monkeypatch)
    released = RECORD_BYTES if failure == "no_row" else 0
    assert tuple(a - b for a, b in zip(after, before, strict=True)) == (
        released,
        released,
    )
    receipt = failed.value.receipt
    assert (
        receipt.failure
        == {"no_row": "no_run", "uncertain": "unknown", "durable_setup": "setup"}[
            failure
        ]
    )
    service.bind_native_receipt(host.authorization, receipt)
    with pytest.raises(_WorkflowChildLaunchFailed) as repeated:
        _launch(native, host)
    assert repeated.value.receipt is receipt
    assert payload_headroom(host.db, workflow_id, monkeypatch) == after
    if failure == "no_row":
        assert service.get(workflow_id).results == ()
        with host.db.connection() as conn:
            assert validate_workflow_rows(conn) == ()
    with host.activity.prelaunch(footprint=host.authorization.footprint) as claim:
        assert (claim is not None) is (failure == "no_row")
    if failure == "durable_setup":
        assert host.db.get_run(receipt.run_id)["status"] == "error"
        result = WorkflowTaskResult(
            workflow_id=host.authorization.claim.workflow_id,
            attempt_id=receipt.attempt_id,
            run_id=receipt.run_id,
            status="failed",
            summary="No worker started",
            handoff="Setup failure retained",
        )
        assert service.checkpoint_task(result, drained=True).results == (result,)


def test_required_checkpoint_archives_and_protects_change_windows_until_private_removal(
    workflow_host, tmp_path
):
    import sqlite3
    from threading import Event

    from Tests.Backup_Recovery.test_agent_runs_recovery_schema import owner
    from tldw_chatbook.Backup_Recovery.sqlite_validation import validate_candidate

    host = workflow_host
    service, _receipt, result = _native_result(host)
    host.db.record_change_snapshots_batch(
        run_id=result.run_id,
        records=[
            {
                "root": str(host.root),
                "tracking_error": "capture limited by existing consent",
            }
        ],
    )
    assert host.db.delete_change_snapshots_older_than("9999") == 0
    saved = service.checkpoint_task(result, drained=True)
    checkpoint = service._checkpoint(saved.id, result.attempt_id)
    assert len(checkpoint.change_window_refs) == 1
    assert (
        "consent" in checkpoint.change_review_disclosure
        and "rollback" in checkpoint.change_review_disclosure
    )
    assert host.db.delete_change_snapshots_older_than("9999") == 0
    host.db.automatic_work.settle_workflow_phase(
        host.phase.attempt.id, owner_id="owner", drained=True
    )
    review = service.prepare_quality_review(saved.id, phase_id="phase-1")
    completed = service.review_quality(review, accepted=True)
    archive = tmp_path / "required-archive.db"
    with closing(sqlite3.connect(archive)) as copy, host.db.connection() as source:
        source.backup(copy)
    assert validate_candidate(owner(), archive, Event(), migrate=False) == ()
    with closing(AgentRunsDB(archive)) as reopened:
        assert (
            reopened.workflow_runs.read_result(saved.id, result.attempt_id).handoff
            == result.handoff
        )
    used = completed.accounting.used
    removed = host.db.workflow_runs.remove_payloads(
        saved.id, expected_control_revision=completed.control_revision
    )
    assert removed.results == () and removed.accounting.used == used
    with host.db.connection() as conn:
        assert (
            conn.execute("SELECT count(*) FROM workflow_checkpoints").fetchone()[0] == 0
        )
        assert (
            conn.execute("SELECT state FROM workflow_payload_reservations").fetchone()[
                0
            ]
            == "removed"
        )
    assert host.db.delete_change_snapshots_older_than("9999") == 1


def test_forged_receipt_boolean_and_oversize_required_payload_do_not_settle(
    workflow_host,
):
    import dataclasses

    from tldw_chatbook.Agents.workflow_models import RECORD_BYTES

    host = workflow_host
    service, receipt, result = _native_result(host)
    with pytest.raises(ValueError, match="native_receipt_custody_required"):
        service.bind_native_receipt(host.authorization, dataclasses.replace(receipt))
    with pytest.raises(ValueError, match="native_retirement_unproven"):
        service.checkpoint_task(result, drained=False)
    oversized = result.model_copy(
        update={
            "handoff": "x"
            * (
                RECORD_BYTES
                - len(result.canonical_json().encode())
                + len(result.handoff.encode())
            )
        }
    )
    with pytest.raises(ValueError, match="record_capacity"):
        service.checkpoint_task(oversized, drained=True)
    with host.activity.prelaunch(footprint=host.authorization.footprint) as blocked:
        assert blocked is None
    assert service.get(result.workflow_id).results == ()


def test_recovery_close_keeps_unknown_charges_and_claims_and_cannot_be_quality_approved(
    workflow_host,
):
    from tldw_chatbook.Agents.workflow_models import WorkflowExecutionRecovery

    host = workflow_host
    service, _receipt, result = _native_result(host)
    host.db.automatic_work.recover(current_owner_id="new")
    service.project_recovery(host.db.automatic_work.recovery_result)
    before = service.get(result.workflow_id)
    assert before.status == "recovery_required" and before.accounting.uncertain
    review = service.prepare_quality_review(before.id, phase_id="phase-1")
    assert type(review.evidence) is WorkflowExecutionRecovery
    with pytest.raises(ValueError, match="quality_decision_required"):
        service.review_quality(review, accepted=True)
    with pytest.raises(ValueError, match="resume_requires_clean_checkpoint"):
        service.resume(
            before.id,
            expected_control_revision=before.control_revision,
            command_id="no-replay",
        )
    closed = service.resolve_recovery(
        before.id,
        expected_control_revision=before.control_revision,
        decision=WorkflowExecutionRecovery(reason="close_uncertain"),
    )
    assert closed.status == "closed" and closed.accounting.uncertain
    assert closed.accounting.used == before.accounting.used
    assert closed.accounting.reserved == before.accounting.reserved
    with pytest.raises(ValueError, match="active_evidence"):
        host.db.workflow_runs.remove_payloads(
            closed.id, expected_control_revision=closed.control_revision
        )
    with host.activity.prelaunch(footprint=host.authorization.footprint) as blocked:
        assert blocked is None


def test_completed_task_handoff_rejects_later_input_change_without_verifiers(task_host):
    def edit(data, root):
        (root / "input.txt").write_text("approved input")
        task = data["phases"][0]["tasks"][0]
        task["inputs"] = ["input.txt"]
        task["footprint"].append({"path": "input.txt", "access": "read"})

    host = task_host(plan_edit=edit)
    host.owner.finish_root()
    host.claim.settle(durable=True)
    service, _receipt, result = _native_result(host)
    saved = service.checkpoint_task(result, drained=True)
    assert service._fresh_task_result(saved, host.authorization.task) == result
    (host.root / "input.txt").write_text("changed after completion")
    with pytest.raises(ValueError, match="stale_task_input"):
        service._fresh_task_result(saved, host.authorization.task)


def test_partial_quiescent_recovery_reclaims_only_unused_payload_with_real_published_sibling(
    task_host, monkeypatch
):
    import copy

    from Tests.DB.test_workflow_evidence_retention import payload_headroom
    from tldw_chatbook.Agents.workflow_models import RECORD_BYTES
    from tldw_chatbook.DB.recovery_workflow_schema import validate_workflow_rows

    def edit(data, root):
        sibling = copy.deepcopy(data["phases"][0]["tasks"][0])
        sibling["id"] = "task-2"
        data["phases"][0]["tasks"].append(sibling)

    host = task_host(plan_edit=edit)
    host.owner.finish_root()
    host.claim.settle(durable=True)
    service, _receipt, result = _native_result(host)
    service.checkpoint_task(result, drained=True)
    before = payload_headroom(host.db, result.workflow_id, monkeypatch)
    used = service.get(result.workflow_id).accounting.used
    assert host.db.automatic_work.quiesce_workflow_phase(
        host.phase.attempt.id, owner_id="owner", drained=True
    )
    host.db.automatic_work.recover(current_owner_id="new")
    after = payload_headroom(host.db, result.workflow_id, monkeypatch)
    assert tuple(a - b for a, b in zip(after, before, strict=True)) == (
        RECORD_BYTES,
        RECORD_BYTES,
    )
    saved = service.get(result.workflow_id)
    assert saved.results == (result,) and saved.accounting.used == used
    with host.db.connection() as conn:
        assert [
            tuple(row)
            for row in conn.execute(
                "SELECT t.task_id,r.state FROM workflow_task_attempts t JOIN workflow_payload_reservations r ON r.attempt_id=t.id ORDER BY t.task_id"
            )
        ] == [("task-1", "published"), ("task-2", "removed")]
        assert validate_workflow_rows(conn) == ()
    host.db.automatic_work.recover(current_owner_id="new")
    assert payload_headroom(host.db, result.workflow_id, monkeypatch) == after
    assert service.get(result.workflow_id).results == (result,)


def test_native_done_without_approved_verifier_stays_missing_and_blocks_quality_and_next_phase(
    task_host,
):
    import copy
    import hashlib

    from Tests.Agents.test_agent_service import ScriptedChat, fence
    from Tests.Agents.test_workflow_child_launch import (
        _join,
        _launch,
        _prepared_service,
    )
    from tldw_chatbook.Agents.raw_shell_tool_provider import RawShellToolProvider
    from tldw_chatbook.Agents.workflow_run_service import WorkflowRunService
    from tldw_chatbook.Chat.console_raw_cli import RawCliRuntime

    runtime = RawCliRuntime(lambda: True)
    runtime.arm()
    runtime.grant_model_session("session")

    def edit(data, root):
        task = data["phases"][0]["tasks"][0]
        dependent = copy.deepcopy(task)
        dependent.update(
            id="task-2", dependencies=["task-1"], inputs=["task-1:source.py"]
        )
        data["phases"].append({"id": "phase-2", "tasks": [dependent]})
        wrapper = root / "verify.sh"
        wrapper.write_text("#!/bin/bash\ntrue\n")
        wrapper.chmod(0o700)
        task["footprint"].append({"path": "verify.sh", "access": "read"})
        task["expected_checks"] = ["unit-tests"]
        task["verifiers"] = [
            {
                "id": "unit-tests",
                "executor_tool_id": "shell_exec",
                "verifier_path": str(wrapper),
                "verifier_sha256": hashlib.sha256(wrapper.read_bytes()).hexdigest(),
                "input_paths": ["source.py"],
            }
        ]

    host = task_host(
        mode="exclusive",
        tools=("shell_exec",),
        plan_edit=edit,
        configure=lambda registry, root, binding: registry.register_provider(
            RawShellToolProvider(
                runtime=runtime,
                console_session_id="session",
                initial_directory=lambda: root,
            )
        ),
    )
    host.controller.app.raw_cli_runtime = runtime
    host.owner.finish_root()
    host.claim.settle(durable=True)
    service = WorkflowRunService(host.db, _history_persistence(host.root))
    seed = service.prepare_task_inputs(host.authorization)
    native, _ = _prepared_service(
        host,
        ScriptedChat(
            [
                fence("shell_exec", {"command": "true", "shell": "bash"}),
                "done; unit-tests passed",
            ]
        ),
        seed=seed,
    )
    try:
        receipt = _launch(native, host)
        _join(host)
        assert host.db.get_run(receipt.run_id)["status"] == "done"
        service.bind_native_receipt(host.authorization, receipt)
        result = WorkflowTaskResult(
            workflow_id=host.authorization.claim.workflow_id,
            attempt_id=receipt.attempt_id,
            run_id=receipt.run_id,
            status="completed",
            summary="Model says tests passed",
            handoff="Complete handoff",
            checks=(
                WorkflowCheck(
                    id="unit-tests",
                    status="passed",
                    verifier="unit-tests",
                    input_manifest="0" * 64,
                    output_manifest="0" * 64,
                    exit_code=0,
                    reason="model_claim",
                ),
            ),
        )
        saved = service.checkpoint_task(result, drained=True)
        check = saved.results[0].checks[0]
        assert check.status == "missing" and check.exit_code is None
        assert check.reason == "native_evidence_unavailable"
        assert service._checkpoint(saved.id, receipt.attempt_id).verifiers == ()
        host.db.automatic_work.settle_workflow_phase(
            host.phase.attempt.id, owner_id="owner", drained=True
        )
        quality = service.prepare_quality_review(saved.id, phase_id="phase-1")
        dependent = service.prepare_decision(saved.id, phase_id="phase-2")
        for review in (quality, dependent):
            assert (
                not review.evidence.ready
                and review.reason == "native_check_missing_or_stale"
            )
            with pytest.raises(ValueError, match="stale_workflow_decision"):
                service.validate_decision(review)
        with pytest.raises(ValueError, match="stale_workflow_decision"):
            service.review_quality(quality, accepted=True)
    finally:
        runtime.shutdown()


@pytest.mark.parametrize("mode", [
    "unchanged", "external-edit", "removed", "owned-edit", "partial-directory",
    "partial-unchanged", "alias-overlap",
])
def test_checkpoint_keeps_original_read_only_inputs_across_native_return(task_host, mode):
    import copy
    import threading
    from Tests.Agents.test_agent_service import ScriptedChat, fence
    from Tests.Agents.test_workflow_child_launch import _join, _launch, _prepared_service
    from tldw_chatbook.Agents.workflow_run_service import WorkflowRunService

    readonly = "inputs/read.txt" if mode.startswith("partial-") else "input.txt"
    source = "source.py" if mode == "owned-edit" else readonly

    def edit(data, root):
        (root / "inputs").mkdir()
        (root / readonly).write_text("approved input")
        task = data["phases"][0]["tasks"][0]
        if mode == "owned-edit":
            (root / "source.py").write_text("approved input")
            task["inputs"] = ["source.py"]
        elif mode.startswith("partial-"):
            (root / "inputs/owned.txt").write_text("task-owned input")
            task["inputs"] = ["inputs"]
            task["footprint"].extend([
                {"path": "inputs", "access": "read", "subtree": True},
                {"path": "inputs/owned.txt", "access": "write"},
            ])
        else:
            task["inputs"] = [readonly]
            task["footprint"].append({"path": readonly, "access": "read"})
            if mode == "alias-overlap":
                (root / "source.py").write_text("approved input")
                (root / readonly).unlink()
                (root / readonly).hardlink_to(root / "source.py")
        dependent = copy.deepcopy(task)
        dependent.update(id="dependent", dependencies=["task-1"], inputs=["task-1:source.py"])
        dependent["outputs"] = ["outside.py"]
        dependent["footprint"] = [{"path": "source.py", "access": "read"}, {"path": "outside.py", "access": "write"}]
        data["phases"].append({"id": "dependent-phase", "tasks": [dependent]})

    host = task_host(plan_edit=edit)
    host.owner.finish_root()
    host.claim.settle(durable=True)
    service = WorkflowRunService(host.db, SimpleNamespace(workspace_registry=None))
    seed = service.prepare_task_inputs(host.authorization)
    entered, release = threading.Event(), threading.Event()
    replies = [fence("fs_read", {"path": source})]
    if mode == "owned-edit":
        replies.append(fence("fs_write", {"path": "source.py", "content": "intentional task edit"}))
    replies.append("Native handoff based on approved input")
    scripted = ScriptedChat(replies)

    def provider(**kwargs):
        if len(scripted.calls) == 1:
            assert "approved input" in str(kwargs.get("messages", kwargs.get("messages_payload", ())))
            entered.set()
            assert release.wait(15), "original provider was not released"
        return scripted(**kwargs)

    native, _plan = _prepared_service(host, provider, seed=seed)
    receipt = _launch(native, host)
    try:
        assert entered.wait(10), "original native read did not reach provider"
        assert receipt.owner.is_admitted
        if mode in {"external-edit", "partial-directory"}:
            (host.root / readonly).write_text("external changed input")
        elif mode == "removed":
            (host.root / readonly).unlink()
    finally:
        release.set()
        _join(host)
    service.bind_native_receipt(host.authorization, receipt)
    result = WorkflowTaskResult(workflow_id=host.authorization.claim.workflow_id,
        attempt_id=receipt.attempt_id, run_id=receipt.run_id, status="completed",
        summary="Native task returned", handoff="Native handoff based on approved input")
    saved = service.checkpoint_task(result, drained=True)
    checkpoint = service.read_checkpoint(saved.id, result.attempt_id)
    assert saved.results == (result,)  # Native completion remains truthful.
    if mode in {"external-edit", "removed", "partial-directory", "partial-unchanged", "alias-overlap"}:
        assert checkpoint.input_manifest is None
        with pytest.raises(ValueError, match="stale_task_input"):
            service._fresh_task_result(saved, host.authorization.task)
        assert not service.prepare_quality_review(saved.id, phase_id="phase-1").evidence.ready
        assert not service.prepare_decision(saved.id, phase_id="dependent-phase").evidence.ready
    else:
        assert checkpoint.input_manifest is not None
        assert service._fresh_task_result(saved, host.authorization.task) == result
        if mode == "owned-edit":
            assert (host.root / "source.py").read_text() == "intentional task edit"
