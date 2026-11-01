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


@pytest.mark.bootstrap_profile
@pytest.mark.parametrize("version", [25, 26])
def test_actual_historical_catalog_near_bound_payloads_upgrade_without_rewriting(
    tmp_path, version
):
    import sqlite3
    from contextlib import closing

    from Tests.Agents.workflow_support import make_workflow_request
    from Tests.DB.test_workflow_runs import attempt
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.DB.recovery_operations import _AGENT_RUNS_SCHEMA
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
        source.workflow_runs.publish_result(result)
        source.workflow_runs.record_retirement(
            snapshot.id, claim.id, run_id=run, drained=True
        )
        old = tmp_path / f"v{version}.db"
        catalog = next(
            schema for number, schema in _AGENT_RUNS_SCHEMA if number == version
        )
        with closing(sqlite3.connect(old)) as conn, source.connection() as original:
            for sql in sorted(
                catalog, key=lambda sql: not sql.startswith("CREATE TABLE")
            ):
                if not sql.startswith("CREATE TABLE sqlite_sequence"):
                    conn.execute(sql)
            tables = [
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_schema WHERE type='table' AND name NOT IN ('schema_version','sqlite_sequence')"
                )
            ]
            for table in sorted(
                tables, key=lambda name: name != "automatic_work_chains"
            ):
                columns = [
                    row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')
                ]
                names = ",".join(f'"{column}"' for column in columns)
                rows = original.execute(f'SELECT {names} FROM "{table}"').fetchall()
                conn.executemany(
                    f'INSERT INTO "{table}" ({names}) VALUES ({",".join("?" for _ in columns)})',
                    rows,
                )
            conn.execute("INSERT INTO schema_version VALUES (?)", (version,))
            conn.commit()
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
