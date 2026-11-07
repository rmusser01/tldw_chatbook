"""Review-only exact admission/capacity boundary probe; no provider or runtime effects."""

import pytest

from Tests.DB.test_workflow_attempts import workflow_fixture


pytestmark = pytest.mark.bootstrap_profile


def test_initial_decision_and_results_share_one_capacity_admission(workflow, monkeypatch):
    from tldw_chatbook.Agents.workflow_models import RECORD_BYTES, WorkflowDecision
    from tldw_chatbook.DB import workflow_runs

    db, snapshot, _ = workflow
    with db.connection() as conn:
        retained = sum(
            conn.execute(query).fetchone()[0]
            for query in (
                "SELECT COALESCE(SUM(length(CAST(request_json AS BLOB))),0) FROM workflow_runs",
                "SELECT COALESCE(SUM(length(CAST(payload_json AS BLOB))),0) FROM workflow_plan_revisions",
            )
        )
    monkeypatch.setattr(workflow_runs, "WORKFLOW_BYTES", retained + RECORD_BYTES)
    decision = WorkflowDecision(
        id="capacity-decision", workflow_id=snapshot.id, plan_revision=1,
        control_revision=snapshot.control_revision, phase_id="phase-1",
        action="approve", reason="ready",
    )
    error = None
    try:
        attempt = db.automatic_work._prepare_workflow(
            snapshot.id, revision_id=1, phase_id="phase-1", purpose="execution",
            owner_id="owner", session_id="session", command_id="capacity-command",
            _initial=decision,
        )
    except ValueError as exc:
        error = exc
    with db.connection() as conn:
        rows = {
            table: conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            for table in ("workflow_decisions", "workflow_phase_attempts", "workflow_payload_reservations")
        }
        after_error = None
        try:
            db.workflow_runs._capacity(conn, snapshot.id, 0)
        except ValueError as exc:
            after_error = str(exc)
    print({"retained_before": retained, "ceiling": retained + RECORD_BYTES,
           "decision_bytes": len(decision.canonical_json().encode()),
           "admission_error": str(error) if error else None,
           "post_commit_capacity_error": after_error, "rows": rows})
    assert error is not None and str(error) == "workflow_payload_capacity"
    assert all(count == 0 for count in rows.values())
