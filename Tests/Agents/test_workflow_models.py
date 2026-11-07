"""Untrusted workflow plans reject ambiguous or unbounded contracts."""

import json

import pytest

pytestmark = pytest.mark.bootstrap_profile


def test_plan_json_roundtrip_and_duplicate_keys(tmp_path):
    from Tests.Agents.workflow_support import plan_data
    from tldw_chatbook.Agents.workflow_models import WorkflowPlan

    data = plan_data(tmp_path)
    plan = WorkflowPlan.model_validate_json(json.dumps(data))
    assert plan.phases[0].tasks[0].id == "task-1"
    with pytest.raises(ValueError, match="duplicate"):
        WorkflowPlan.model_validate_json('{"request":{},"request":{}}')


@pytest.mark.parametrize("value", [True, False, float("nan"), float("inf"), -1, 2**63])
def test_rejects_nonfinite_boolean_and_out_of_range_limits(tmp_path, value):
    from Tests.Agents.workflow_support import plan_data
    from tldw_chatbook.Agents.workflow_models import WorkflowPlan

    data = plan_data(tmp_path)
    data["request"]["policy"]["generations"] = value
    with pytest.raises(ValueError):
        WorkflowPlan.model_validate_json(json.dumps(data))


@pytest.mark.parametrize(
    "case",
    [
        "phases",
        "tasks",
        "scopes",
        "duplicate_id",
        "forward",
        "cycle",
        "role",
        "target",
        "tool",
        "path",
        "unknown",
        "multibyte",
    ],
)
def test_plan_boundaries(tmp_path, case):
    from copy import deepcopy

    from Tests.Agents.workflow_support import plan_data
    from tldw_chatbook.Agents.workflow_models import WorkflowPlan

    data = plan_data(tmp_path)
    task = data["phases"][0]["tasks"][0]
    if case == "phases":
        data["phases"] = [
            {"id": f"p{i}", "tasks": [{**task, "id": f"t{i}"}]} for i in range(7)
        ]
    elif case == "tasks":
        data["phases"][0]["tasks"] = [{**task, "id": f"t{i}"} for i in range(13)]
    elif case == "scopes":
        task["footprint"] *= 65
    elif case == "duplicate_id":
        data["phases"][0]["tasks"].append(deepcopy(task))
    elif case in {"forward", "cycle"}:
        task["dependencies"] = ["later"]
        data["phases"].append(
            {
                "id": "phase-2",
                "tasks": [
                    {
                        **task,
                        "id": "later",
                        "dependencies": ["task-1"] if case == "cycle" else [],
                    }
                ],
            }
        )
    elif case == "role":
        task["role"] = "unavailable"
    elif case == "target":
        task["target"] = {**task["target"], "model": "unapproved"}
    elif case == "tool":
        task["allowed_tools"].append("run_shell")
    elif case == "path":
        task["footprint"][0]["path"] = "../escape"
    elif case == "unknown":
        task["execute"] = "arbitrary code"
    else:
        task["instructions"] = "é" * 66000
    with pytest.raises(ValueError):
        WorkflowPlan.model_validate_json(json.dumps(data, ensure_ascii=False))


def test_readonly_binding_rejects_write_and_plan_captures_earlier_refs(tmp_path):
    from Tests.Agents.workflow_support import plan_data
    from tldw_chatbook.Agents.workflow_models import WorkflowPlan

    data = plan_data(tmp_path)
    first = data["phases"][0]["tasks"][0]
    data["phases"].append(
        {
            "id": "phase-2",
            "tasks": [
                {
                    **first,
                    "id": "task-2",
                    "dependencies": ["task-1"],
                    "inputs": ["task-1:source.py"],
                }
            ],
        }
    )
    assert len(WorkflowPlan.model_validate_json(json.dumps(data)).phases) == 2
    data["request"]["binding"]["access"] = "ro"
    with pytest.raises(ValueError):
        WorkflowPlan.model_validate_json(json.dumps(data))


@pytest.mark.parametrize(
    "path", ["/absolute", "a/../b", "a//b", "a\\b", "bad\x00name", "a/./b"]
)
def test_footprint_paths_remain_binding_relative_and_canonical(tmp_path, path):
    from Tests.Agents.workflow_support import plan_data
    from tldw_chatbook.Agents.workflow_models import WorkflowPlan

    data = plan_data(tmp_path)
    data["phases"][0]["tasks"][0]["footprint"][0]["path"] = path
    with pytest.raises(ValueError):
        WorkflowPlan.model_validate_json(json.dumps(data))


def test_missing_and_stale_checks_cannot_be_claimed_passed():
    from tldw_chatbook.Agents.workflow_models import WorkflowCheck

    with pytest.raises(ValueError):
        WorkflowCheck(
            id="check", status="passed", verifier="pytest", reason="model said passed"
        )
    failed = WorkflowCheck(
        id="check",
        status="failed",
        verifier="pytest",
        reason="process_exit",
        exit_code=1,
        input_manifest="before",
        output_manifest="after",
    )
    assert failed.status == "failed" and failed.exit_code == 1


@pytest.mark.parametrize(
    "reason, expected, code",
    [
        ("done", "done", "done"),
        ("legacy_failure", "preflight_refused", "legacy_failure"),
    ],
)
def test_goal_iteration_termination_normalization_survives_passive_imports(
    reason, expected, code
):
    from tldw_chatbook.Agents.goal_models import GoalIterationResult

    result = GoalIterationResult(
        goal_id="goal",
        ordinal=1,
        attempt_id=None,
        native_run_id=None,
        outcome=None,
        termination_reason=reason,
    )
    assert result.termination_reason.value == expected
    assert result.reason_code == code


def test_exact_phase_task_and_footprint_ceilings_are_accepted(tmp_path):
    from Tests.Agents.workflow_support import plan_data
    from tldw_chatbook.Agents.workflow_models import WorkflowPlan

    data = plan_data(tmp_path)
    task = data["phases"][0]["tasks"][0]
    task["footprint"] = [
        {"path": f"file-{i}", "access": "read", "subtree": False} for i in range(64)
    ]
    task["footprint"][0] = {"path": "source.py", "access": "write", "subtree": False}
    data["phases"] = [
        {"id": f"p-{i}", "tasks": [{**task, "id": f"t-{i}-{j}"} for j in range(2)]}
        for i in range(6)
    ]
    plan = WorkflowPlan.model_validate_json(json.dumps(data))
    assert len(plan.phases) == 6
    assert sum(len(phase.tasks) for phase in plan.phases) == 12
    assert len(plan.phases[0].tasks[0].footprint) == 64


@pytest.mark.parametrize("field", ["inputs", "outputs"])
def test_task_file_references_cannot_expand_declared_footprint(tmp_path, field):
    from Tests.Agents.workflow_support import plan_data
    from tldw_chatbook.Agents.workflow_models import WorkflowPlan

    data = plan_data(tmp_path)
    data["phases"][0]["tasks"][0][field] = ["outside.py"]
    with pytest.raises(ValueError, match="footprint"):
        WorkflowPlan.model_validate_json(json.dumps(data))
