"""Launch validation catches authority expansion and unbounded private payloads."""

import importlib

import pytest
from pydantic import ValidationError


def models():
    return importlib.import_module("tldw_chatbook.Agents.goal_models")


def request(root="/tmp/goal-fixture", **changes):
    m = models()
    data = {
        "objective": "Repair fixture",
        "criteria": "Validation exits zero",
        "provider": m.GoalProviderRef(
            provider="test",
            model="offline",
            config_ref="local",
            authority_ref="test-instance",
            endpoint_ref="offline://test",
        ),
        "binding": m.GoalBindingRef(
            workspace_id="workspace", binding_id="binding", locator=root, access="rw"
        ),
        "tool_scope": m.GoalToolScope(
            catalog_tools=("local:fs_edit",), runtime_tools=("run_skill_script",)
        ),
    }
    data.update(changes)
    return m.GoalRequest(**data)


@pytest.mark.parametrize("value", [True, -1, None, float("inf"), 2**63, "3"])
def test_policy_rejects_coercion_and_unbounded_limits(value):
    with pytest.raises((ValidationError, ValueError)):
        models().GoalPolicy(iterations=value)


def test_launch_is_deeply_immutable_and_forbids_extra_fields():
    m = models()
    item = request()
    with pytest.raises(ValidationError):
        item.binding.access = "ro"
    with pytest.raises(ValidationError):
        request(approval=True)
    with pytest.raises(ValidationError):
        m.GoalToolScope(runtime_tools=("run_skill_script",), allow_all=True)


@pytest.mark.parametrize("field", ["objective", "criteria"])
def test_objective_and_criteria_limits_count_utf8_bytes(field):
    assert len(getattr(request(**{field: "é" * 4096}), field)) == 4096
    with pytest.raises(ValidationError):
        request(**{field: "é" * 4097})


def test_zero_is_finite_disabled_admission_not_unlimited():
    policy = models().GoalPolicy(iterations=0)
    assert policy.admission_enabled is False
    assert policy.chain_limits().generations == 0
    assert policy.chain_limits().child_launches == 0


def test_verifier_cannot_be_inside_editable_input_binding():
    m = models()
    verifier = m.VerificationSpec(
        id="check",
        executor_tool_id="run_skill_script",
        verifier_path="/tmp/goal-fixture/check.py",
        verifier_sha256="a" * 64,
        arguments=(),
        input_paths=("fixture.txt",),
    )
    with pytest.raises(ValidationError):
        request(verifiers=(verifier,))
    with pytest.raises(ValidationError):
        m.VerificationSpec(
            id="check",
            executor_tool_id="run_skill_script",
            verifier_path="/tmp/check.py",
            verifier_sha256="a" * 64,
            input_paths=("../secret",),
        )


def test_reports_bound_draft_learnings_evidence_and_total_bytes():
    m = models()
    for fields in (
        {"draft": "é" * 16385},
        {"learnings": ("x",) * 9},
        {"evidence_refs": ("x",) * 33},
        {"summary": "é" * 32769},
    ):
        with pytest.raises(ValidationError):
            m.GoalReport(**fields)


def test_unchecked_model_copy_cannot_cross_durable_launch_boundary(tmp_path):
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    db = AgentRunsDB(tmp_path / "runs.db")
    invalid = request().model_copy(update={"objective": "é" * 5000})
    with pytest.raises(ValidationError):
        db.goal_runs.create(invalid, launch_id="invalid")
    with db.connection() as conn:
        assert conn.execute("SELECT count(*) FROM goal_runs").fetchone()[0] == 0
    db.close()
