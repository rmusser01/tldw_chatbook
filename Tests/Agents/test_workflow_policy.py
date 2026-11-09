"""Workflow limits have an independent immutable-origin policy."""

import pytest

pytestmark = pytest.mark.bootstrap_profile


def test_workflow_policy_does_not_inherit_goal_or_wake_settings(monkeypatch):
    from tldw_chatbook.Agents import run_log
    from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkLimits

    monkeypatch.setattr(run_log, "_setting", lambda key, default: default)
    limits = AutomaticWorkLimits.from_settings("workflow_phase")
    assert limits.generations == 8
    assert limits.child_launches == 12
    assert limits.model_calls == 32
    assert limits.budget_tokens == 500000
    assert limits.output_tokens == 8192
    assert limits.wall_seconds == 900
    assert AutomaticWorkLimits.from_settings("goal_iteration").child_launches == 0


@pytest.mark.parametrize(
    "bad", [True, False, float("inf"), float("nan"), -1, 2**63, "nan"]
)
def test_workflow_limits_keep_finite_defaults(monkeypatch, bad):
    from tldw_chatbook.Agents import run_log
    from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkLimits

    monkeypatch.setattr(run_log, "_setting", lambda key, default: bad)
    assert AutomaticWorkLimits.from_settings("workflow_phase") == AutomaticWorkLimits(
        generations=8, child_launches=12
    )


def test_workflow_zero_limits_are_not_replaced_by_defaults(monkeypatch):
    from tldw_chatbook.Agents import run_log
    from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkLimits

    monkeypatch.setattr(run_log, "_setting", lambda key, default: "0")
    limits = AutomaticWorkLimits.from_settings("workflow_phase")
    assert not any(limits.resources().values())
    assert limits.output_tokens == limits.wall_seconds == 0


def test_workflow_origin_cannot_fall_through_to_manual_or_fleet():
    from types import SimpleNamespace

    from tldw_chatbook.Chat.console_automatic_submission import (
        resolve_automatic_authorization,
        submission_scope,
    )
    from tldw_chatbook.Chat.console_chat_models import ConsoleSubmissionOrigin

    origin = ConsoleSubmissionOrigin.WORKFLOW_PHASE
    assert origin.automatic
    with pytest.raises(PermissionError, match="phase coordinator"):
        resolve_automatic_authorization(
            SimpleNamespace(), origin, "session", "stored-id", None
        )
    with pytest.raises(PermissionError, match="phase coordinator"):
        submission_scope(SimpleNamespace(), origin, "session")


def test_phase_authorization_has_no_public_saved_id_constructor():
    from tldw_chatbook.Chat.console_workflow_runs import WorkflowPhaseAuthorization

    with pytest.raises(PermissionError, match="coordinator-internal"):
        WorkflowPhaseAuthorization(None, "stored-id", _key=object())


def test_registry_rechecks_workflow_revocation_before_provider_effect(monkeypatch):
    from types import SimpleNamespace

    from Tests.Agents.test_tool_catalog import FakeCatalogProvider
    from tldw_chatbook.Agents import automatic_work_runtime
    from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused
    from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry

    def refused():
        raise AutomaticWorkRefused("workflow_authority_changed")

    registry = ToolCatalogRegistry()
    provider = FakeCatalogProvider(count=1)
    effects = []
    provider.invoke = lambda *args: effects.append(args)
    registry.register_provider(provider)
    monkeypatch.setattr(
        automatic_work_runtime,
        "current_automatic_work",
        lambda: SimpleNamespace(
            goal=None,
            workflow=SimpleNamespace(attempt=SimpleNamespace(purpose="execution")),
            check=refused,
        ),
    )
    with pytest.raises(RuntimeError, match="workflow_authority_changed"):
        registry.invoke_by_name("t0", {})
    assert effects == []



def test_unregistered_planning_preparation_cannot_run_a_receiver():
    from tldw_chatbook.Agents.run_context import (
        bind_workflow_preparation,
        current_planning_preparation,
    )
    from tldw_chatbook.Agents.workflow_phase import require_returned_planning

    class Unregistered:
        @property
        def phase(self):
            pytest.fail("unregistered preparation invoked a phase descriptor")

    preparation = Unregistered()
    with bind_workflow_preparation(preparation):
        assert current_planning_preparation() is None
    with pytest.raises(ValueError, match="workflow_planning_return_required"):
        require_returned_planning(preparation, object(), RuntimeError())



@pytest.mark.asyncio
async def test_remaining_acceptance_does_not_export_phase_key_or_copy_issuance():
    from copy import copy
    from types import SimpleNamespace

    from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused
    from tldw_chatbook.Agents.workflow_models import WorkflowDecision
    from tldw_chatbook.Agents.workflow_phase import (
        begin_remaining_acceptance,
        require_remaining_unused,
    )

    service = object()
    bridge = object()
    controller = SimpleNamespace(_agent_bridge=bridge)
    runtime = SimpleNamespace(
        chat_controller=controller, agent_bridge=bridge, _workflow_service=service
    )
    controller.app = SimpleNamespace(console_runtime=runtime)
    decision = WorkflowDecision(
        id="decision",
        workflow_id="workflow",
        plan_revision=1,
        control_revision=1,
        phase_id="phase",
        action="approve",
        reason="ready",
    )

    cell = begin_remaining_acceptance(controller, decision, service=service)
    assert not hasattr(cell, "key")
    with pytest.raises(AutomaticWorkRefused, match="original_unused_acceptance_required"):
        require_remaining_unused(copy(cell), service, phase=None)
