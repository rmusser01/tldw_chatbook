"""Injected restrictions at the pure loop's actual preparation/dispatch seams."""

import builtins
from dataclasses import replace

import pytest

from Tests.Agents.test_agent_runtime import CALC, CFG, make_deps
from tldw_chatbook.Agents.agent_models import (
    RUN_DONE,
    SPAWN_TOOL_NAME,
    ModelTurn,
    ToolCall,
    ToolResult,
)
from tldw_chatbook.Agents.agent_runtime import ToolBatchPreparation, run_agent_loop


def test_ordinary_loop_does_not_import_workflow_policy(monkeypatch):
    original_import = builtins.__import__

    def import_without_workflow(name, *args, **kwargs):
        if "workflow_task_scope" in name:
            raise AssertionError("pure loop loaded workflow policy")
        return original_import(name, *args, **kwargs)

    deps = make_deps(
        [
            ModelTurn(tool_calls=(ToolCall("calculator", {}, "a"),)),
            ModelTurn(text="done"),
        ]
    )
    monkeypatch.setattr(builtins, "__import__", import_without_workflow)
    outcome = run_agent_loop(CFG, [], [CALC], deps)
    assert outcome.status == RUN_DONE
    assert any(step.result == "42" for step in outcome.steps)


def test_restriction_refuses_before_instruction_and_hook_preparation():
    effects = []
    deps = make_deps(
        [
            ModelTurn(tool_calls=(ToolCall("calculator", {}, "a"),)),
            ModelTurn(text="done"),
        ],
        invoke=lambda call: effects.append("invoke") or ToolResult(ok=True),
    )

    def refuse(call):
        raise PermissionError("outside task scope")

    deps.validate_tool_call = refuse
    deps.prepare_tool_calls = lambda calls: (
        effects.append("instructions") or ToolBatchPreparation("proceed")
    )
    deps.prepare_hook_call = lambda call: effects.append("hook") or call
    deps.guard_tool_calls = lambda calls: effects.append("guard") or {}
    deps.review_tool_calls = lambda calls: effects.append("approval") or {}
    run_agent_loop(CFG, [], [CALC], deps)
    assert effects == []


def test_restriction_rechecks_hook_mutated_arguments_before_acceptance():
    effects = []
    deps = make_deps(
        [
            ModelTurn(tool_calls=(ToolCall("calculator", {"path": "allowed"}, "a"),)),
            ModelTurn(text="done"),
        ],
        invoke=lambda call: effects.append("invoke") or ToolResult(ok=True),
    )

    def validate(call):
        if call.args["path"] != "allowed":
            raise PermissionError("outside task scope")

    deps.validate_tool_call = validate
    deps.prepare_hook_call = lambda call: replace(call, args={"path": "outside"})
    deps.accept_hook_preparation = lambda call: effects.append("accept")
    deps.review_tool_calls = lambda calls: effects.append("approval") or {}
    run_agent_loop(CFG, [], [CALC], deps)
    assert effects == []


@pytest.mark.parametrize("tool_name", ["calculator", SPAWN_TOOL_NAME, "find_tools"])
def test_restriction_rechecks_after_approval_before_each_dispatch(tool_name):
    current = True
    effects = []
    deps = make_deps(
        [
            ModelTurn(
                tool_calls=(ToolCall(tool_name, {"task": "work", "query": "q"}, "a"),)
            ),
            ModelTurn(text="done"),
        ],
        invoke=lambda call: effects.append("invoke") or ToolResult(ok=True),
        spawn=lambda *args, **kwargs: effects.append("spawn") or ToolResult(ok=True),
    )

    def validate(call):
        if not current:
            raise PermissionError("authority revoked during approval")

    def approve(calls):
        nonlocal current
        current = False
        return {}

    deps.validate_tool_call = validate
    deps.review_tool_calls = approve
    deps.find_tools = lambda query: effects.append("find") or []
    run_agent_loop(CFG, [], [CALC], deps)
    assert not current
    assert effects == []


def test_repeatable_validation_does_not_replay_effectful_batch_guard():
    validated = []
    guarded = []
    invoked = []
    deps = make_deps(
        [
            ModelTurn(tool_calls=(ToolCall("calculator", {}, "a"),)),
            ModelTurn(text="done"),
        ],
        invoke=lambda call: invoked.append(call) or ToolResult(ok=True),
    )
    deps.validate_tool_call = lambda call: validated.append(call)
    deps.guard_tool_calls = lambda calls: guarded.append(calls) or {}
    run_agent_loop(CFG, [], [CALC], deps)
    assert len(validated) > 1
    assert len(guarded) == 1
    assert len(invoked) == 1


def test_final_hook_validation_cannot_widen_task_scope():
    invoked = []
    deps = make_deps(
        [
            ModelTurn(tool_calls=(ToolCall("calculator", {"path": "allowed"}, "a"),)),
            ModelTurn(text="done"),
        ],
        invoke=lambda call: invoked.append(call) or ToolResult(ok=True),
    )

    def validate(call):
        if call.args["path"] != "allowed":
            raise PermissionError("outside task scope")

    deps.validate_tool_call = validate
    deps.validate_hook_dispatch = lambda call: call.args.update(path="outside")
    run_agent_loop(CFG, [], [CALC], deps)
    assert invoked == []


def test_service_injects_registry_bound_validation_for_runtime_tools(
    tmp_path, monkeypatch
):
    from Tests.Agents.test_agent_runtime_review_hook import (
        SVC_CFG,
        ScriptedChat,
        _registry,
        native_call,
    )
    from tldw_chatbook.Agents import workflow_task_scope
    from tldw_chatbook.Agents.agent_service import AgentService
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    registry = _registry()
    seen = []

    def refuse(call, *, registry=None):
        seen.append(registry)
        raise PermissionError("task cannot spawn")

    monkeypatch.setattr(workflow_task_scope, "check_current_task_call", refuse)
    db = AgentRunsDB(tmp_path / "runs.db")
    try:
        service = AgentService(
            db=db,
            registry=registry,
            chat_call=ScriptedChat(
                [
                    {
                        "tool_calls": [
                            native_call(SPAWN_TOOL_NAME, {"task": "work"}, "a")
                        ]
                    },
                    "done",
                ]
            ),
        )
        _, outcome = service.run_turn(
            conversation_id="c",
            messages=[{"role": "user", "content": "go"}],
            config=replace(SVC_CFG, allowed_tools=(SPAWN_TOOL_NAME,)),
            api_endpoint="openai",
            should_cancel=lambda: False,
        )
        assert outcome.status == RUN_DONE
        assert seen and all(item is registry for item in seen)
        assert service.live_subagent_handles() == []
    finally:
        db.close()
