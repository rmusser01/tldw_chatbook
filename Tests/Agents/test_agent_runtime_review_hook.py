# Tests/Agents/test_agent_runtime_review_hook.py
"""Pre-dispatch batch-review hook: LoopDeps.review_tool_calls (P5 Task 4).

The hook is the generic seam the MCP approval flow (Task 6) rides on: it
sees the FULL per-turn tool-call batch once, before any dispatch, and
returns a name -> verdict map ("proceed" or a refusal string). Non-proceed
calls are skipped (never reach deps.invoke_tool) and the verdict string is
fed back to the model as that call's tool result, using the SAME
role/tool_call_id shaping normal results use. The hook must fail OPEN here
(exceptions -> treat as all-"proceed"); MCP-specific fail-closed behavior
lives in the Task 6 closure, not in the generic runtime.
"""
import json

import pytest

from tldw_chatbook.Agents.agent_models import (
    RUN_DONE, SPAWN_TOOL_NAME, STEP_TOOL_CALL, STEP_TOOL_RESULT,
    AgentConfig, ModelTurn, ToolCall, ToolResult, ToolSchema,
)
from tldw_chatbook.Agents.agent_runtime import LoopDeps, run_agent_loop
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.tool_catalog import (
    BuiltinToolProvider, ToolCatalogRegistry,
)
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

CALC = ToolSchema(id="builtin:calculator", name="calculator",
                  description="math", parameters={"type": "object"})

CFG = AgentConfig(model="m", system_prompt="s",
                  allowed_tools=("calculator", "echo", SPAWN_TOOL_NAME))


def fence(name, args):
    return f'```tool_call\n{json.dumps({"name": name, "arguments": args})}\n```'


def make_deps(turns, *, invoke=None, review=None, cancel=None, clock=None):
    """Deps whose call_model pops scripted ModelTurns."""
    script = list(turns)

    def call_model(messages, active_schemas):
        return script.pop(0)

    return LoopDeps(
        call_model=call_model,
        invoke_tool=invoke or (lambda c: ToolResult(ok=True, content="42")),
        spawn=lambda task: ToolResult(ok=True, content="sub done"),
        find_tools=lambda q: [],
        load_schemas=lambda ids: [],
        should_cancel=cancel or (lambda: False),
        clock=clock or (lambda: 0.0),
        review_tool_calls=review,
    )


def _native_turn(calls, text=""):
    raw = [{"id": c.call_id, "type": "function",
           "function": {"name": c.name, "arguments": json.dumps(c.args)}}
          for c in calls]
    return ModelTurn(text=text, tool_calls=tuple(calls),
                     assistant_message={"role": "assistant", "content": text,
                                        "tool_calls": raw})


# --- LoopDeps surface -------------------------------------------------

def test_loop_deps_review_tool_calls_defaults_to_none():
    deps = LoopDeps(
        call_model=lambda m, a: ModelTurn(text="hi"),
        invoke_tool=lambda c: ToolResult(ok=True),
        spawn=lambda t: ToolResult(ok=True),
        find_tools=lambda q: [],
        load_schemas=lambda ids: [],
        should_cancel=lambda: False,
        clock=lambda: 0.0,
    )
    assert deps.review_tool_calls is None


# --- hook receives the full batch before any dispatch ------------------

def test_hook_receives_full_batch_before_any_invoke():
    calls = [ToolCall(name="calculator", args={"v": 1}, call_id="idA"),
             ToolCall(name="echo", args={"v": 2}, call_id="idB")]
    seen_batches = []
    invoked = []

    def review(batch):
        seen_batches.append(list(batch))
        assert invoked == []  # hook runs before ANY dispatch this turn
        return {}  # empty map -> every call defaults to "proceed"

    def invoke(call):
        invoked.append(call.name)
        return ToolResult(ok=True, content=f"ok:{call.name}")

    turns = [_native_turn(calls), ModelTurn(text="done")]
    out = run_agent_loop(CFG, [{"role": "user", "content": "go"}], [CALC],
                         make_deps(turns, invoke=invoke, review=review))
    assert out.status == RUN_DONE
    assert len(seen_batches) == 1
    assert [c.name for c in seen_batches[0]] == ["calculator", "echo"]
    assert invoked == ["calculator", "echo"]


def test_hook_not_called_when_turn_has_no_tool_calls():
    seen = []

    def review(batch):
        seen.append(batch)
        return {}

    out = run_agent_loop(
        CFG, [{"role": "user", "content": "hi"}], [CALC],
        make_deps([ModelTurn(text="Tokyo.")], review=review))
    assert out.status == RUN_DONE and out.final_text == "Tokyo."
    assert seen == []


def test_hook_called_once_per_turn_for_a_fence_single_call():
    seen = []
    fence_text = fence("calculator", {"expression": "6*7"})

    def review(batch):
        seen.append([c.name for c in batch])
        return {}

    deps = make_deps([], review=review)
    deps.call_model = lambda m, a: (
        ModelTurn(text=fence_text) if len(seen) == 0 else ModelTurn(text="ok"))
    out = run_agent_loop(CFG, [{"role": "user", "content": "hi"}], [CALC], deps)
    assert out.status == RUN_DONE
    assert seen == [["calculator"]]


# --- non-proceed verdict: skip dispatch, feed verdict back as the result

def test_non_proceed_verdict_skips_invoke_and_shapes_native_tool_result():
    calls = [ToolCall(name="calculator", args={"v": 1}, call_id="idA"),
             ToolCall(name="echo", args={"v": 2}, call_id="idB")]
    invoked = []
    seen_messages = []
    turns = iter([_native_turn(calls), ModelTurn(text="done")])

    def call_model(messages, active_schemas):
        seen_messages.append([dict(m) for m in messages])
        return next(turns)

    def invoke(call):
        invoked.append(call.name)
        return ToolResult(ok=True, content=f"ok:{call.name}")

    def review(batch):
        return {"calculator": "Blocked: destructive action requires approval"}

    deps = make_deps([], invoke=invoke, review=review)
    deps.call_model = call_model
    out = run_agent_loop(CFG, [{"role": "user", "content": "go"}], [CALC], deps)

    assert out.status == RUN_DONE and out.final_text == "done"
    assert invoked == ["echo"]  # calculator skipped, echo proceeded normally

    second_turn_history = seen_messages[1]
    tool_msgs = [m for m in second_turn_history if m.get("role") == "tool"]
    assert [(m["tool_call_id"], m["content"]) for m in tool_msgs] == [
        ("idA", "Blocked: destructive action requires approval"),
        ("idB", "ok:echo")]

    # No tool_call step for the skipped call (dispatch itself was skipped),
    # but its refusal still lands as a tool_result step for the audit trail.
    tool_call_steps = [s.tool_name for s in out.steps if s.kind == STEP_TOOL_CALL]
    assert tool_call_steps == ["echo"]
    result_steps = {s.tool_name: s.result for s in out.steps
                    if s.kind == STEP_TOOL_RESULT}
    assert result_steps["calculator"] == "Blocked: destructive action requires approval"
    assert result_steps["echo"] == "ok:echo"


def test_non_proceed_verdict_fence_mode_user_role_shaping():
    seen = []
    fence_text = fence("calculator", {"expression": "6*7"})

    def call_model(messages, active_schemas):
        seen.append([dict(m) for m in messages])
        return (ModelTurn(text=fence_text) if len(seen) == 1
                else ModelTurn(text="ok"))

    def review(batch):
        assert [c.name for c in batch] == ["calculator"]
        return {"calculator": "Refused: not allowed right now"}

    invoked = []
    deps = make_deps(
        [], invoke=lambda c: invoked.append(c.name) or ToolResult(
            ok=True, content="42"),
        review=review)
    deps.call_model = call_model
    out = run_agent_loop(CFG, [{"role": "user", "content": "hi"}], [CALC], deps)

    assert out.status == RUN_DONE and out.final_text == "ok"
    assert invoked == []
    history = seen[1]
    assert history[-1] == {
        "role": "user",
        "content": "Tool result for calculator: Refused: not allowed right now"}
    result_step = [s for s in out.steps if s.kind == STEP_TOOL_RESULT][0]
    assert result_step.result == "Refused: not allowed right now"


def test_missing_verdict_for_a_call_name_defaults_to_proceed():
    calls = [ToolCall(name="calculator", args={}, call_id="idA"),
             ToolCall(name="echo", args={}, call_id="idB")]
    invoked = []

    def review(batch):
        return {"calculator": "proceed"}  # echo absent entirely

    def invoke(call):
        invoked.append(call.name)
        return ToolResult(ok=True, content="ok")

    turns = [_native_turn(calls), ModelTurn(text="done")]
    out = run_agent_loop(CFG, [{"role": "user", "content": "go"}], [CALC],
                         make_deps(turns, invoke=invoke, review=review))
    assert out.status == RUN_DONE
    assert invoked == ["calculator", "echo"]


# --- exception policy: fail OPEN for the generic runtime ----------------

def test_hook_exception_is_caught_and_treated_as_proceed_all():
    calls = [ToolCall(name="calculator", args={}, call_id="idA"),
             ToolCall(name="echo", args={}, call_id="idB")]
    invoked = []

    def review(batch):
        raise RuntimeError("boom")

    def invoke(call):
        invoked.append(call.name)
        return ToolResult(ok=True, content="ok")

    turns = [_native_turn(calls), ModelTurn(text="done")]
    out = run_agent_loop(CFG, [{"role": "user", "content": "go"}], [CALC],
                         make_deps(turns, invoke=invoke, review=review))
    assert out.status == RUN_DONE
    assert invoked == ["calculator", "echo"]  # fail-open: exception -> proceed


def test_hook_exception_is_logged(caplog):
    def review(batch):
        raise RuntimeError("boom")

    turns = [_native_turn(
        [ToolCall(name="calculator", args={}, call_id="idA")]),
        ModelTurn(text="done")]
    out = run_agent_loop(CFG, [{"role": "user", "content": "go"}], [CALC],
                         make_deps(turns, review=review))
    assert out.status == RUN_DONE  # never raises out of the loop


# --- hook absent: byte-identical behavior (pinning) ---------------------

def test_hook_none_native_history_convention_unchanged():
    calls = [ToolCall(name="calculator", args={"v": "1"}, call_id="idA"),
             ToolCall(name="echo", args={"v": "2"}, call_id="idB")]
    seen_messages = []
    turns = iter([_native_turn(calls), ModelTurn(text="done")])

    def call_model(messages, active_schemas):
        seen_messages.append([dict(m) for m in messages])
        return next(turns)

    invoked = []

    def invoke(call):
        invoked.append(call)
        return ToolResult(ok=True, content=f"ok:{call.args['v']}")

    deps = make_deps([], invoke=invoke, review=None)
    deps.call_model = call_model
    out = run_agent_loop(CFG, [{"role": "user", "content": "go"}], [CALC], deps)
    assert out.status == RUN_DONE and out.final_text == "done"
    assert [c.call_id for c in invoked] == ["idA", "idB"]
    second_turn_history = seen_messages[1]
    tool_msgs = [m for m in second_turn_history if m.get("role") == "tool"]
    assert [(m["tool_call_id"], m["content"]) for m in tool_msgs] == [
        ("idA", "ok:1"), ("idB", "ok:2")]


def test_hook_none_fence_history_convention_unchanged():
    seen = []
    fence_text = fence("calculator", {"expression": "6*7"})

    def call_model(messages, active_schemas):
        seen.append([dict(m) for m in messages])
        return (ModelTurn(text=fence_text) if len(seen) == 1
                else ModelTurn(text="It is 42."))

    deps = make_deps([], review=None)
    deps.call_model = call_model
    out = run_agent_loop(CFG, [{"role": "user", "content": "hi"}], [CALC], deps)
    assert out.status == RUN_DONE and out.final_text == "It is 42."
    history = seen[1]
    assert history[1] == {"role": "assistant", "content": fence_text}
    assert set(history[1].keys()) == {"role", "content"}
    assert history[2]["role"] == "user"
    assert history[2]["content"].startswith("Tool result for")


# --- AgentService threads review_tool_calls into LoopDeps ---------------

def native_call(name, args, call_id="c1"):
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


class ScriptedChat:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        item = self.replies.pop(0)
        message = item if isinstance(item, dict) else {"content": item}
        return {"choices": [{"message": message}]}


SVC_CFG = AgentConfig(model="test-model", system_prompt="You are helpful.",
                      allowed_tools=("calculator", "get_current_datetime",
                                     SPAWN_TOOL_NAME))


@pytest.fixture()
def db(tmp_path):
    return AgentRunsDB(tmp_path / "runs.db", client_id="test")


def _registry():
    reg = ToolCatalogRegistry()
    reg.register_provider(BuiltinToolProvider())
    return reg


def test_agent_service_threads_review_tool_calls_into_loop_deps(db):
    seen_batches = []

    def review(batch):
        seen_batches.append([c.name for c in batch])
        return {"calculator": "Blocked by policy"}

    chat = ScriptedChat([
        {"content": None, "tool_calls": [
            native_call("calculator", {"expression": "2+2"}, "a"),
            native_call("get_current_datetime", {}, "b")]},
        "done"])
    service = AgentService(db=db, registry=_registry(), chat_call=chat,
                           review_tool_calls=review)
    _run_id, outcome = service.run_turn(
        conversation_id="c", messages=[{"role": "user", "content": "go"}],
        config=SVC_CFG, api_endpoint="openai", should_cancel=lambda: False)

    assert outcome.status == RUN_DONE
    assert seen_batches == [["calculator", "get_current_datetime"]]
    result_steps = {s.tool_name: s.result for s in outcome.steps
                    if s.kind == "tool_result"}
    assert result_steps["calculator"] == "Blocked by policy"
    assert result_steps["get_current_datetime"] != "Blocked by policy"


def test_agent_service_review_tool_calls_defaults_to_none(db):
    """Omitting the ctor arg must not change existing AgentService behavior."""
    chat = ScriptedChat([
        {"content": None, "tool_calls": [
            native_call("calculator", {"expression": "2+2"}, "a")]},
        "4."])
    service = AgentService(db=db, registry=_registry(), chat_call=chat)
    _run_id, outcome = service.run_turn(
        conversation_id="c", messages=[{"role": "user", "content": "2+2?"}],
        config=SVC_CFG, api_endpoint="openai", should_cancel=lambda: False)
    assert outcome.status == RUN_DONE and outcome.final_text == "4."
