# Tests/Agents/test_agent_runtime.py
"""Pure loop tests with deterministic fake callables."""
import json

from tldw_chatbook.Agents.agent_models import (
    LOOP_DETECTION_N, RUN_CANCELLED, RUN_DONE, RUN_STUCK, SPAWN_TOOL_NAME,
    STEP_SPAWN, STEP_TOOL_RESULT, AgentConfig, ModelTurn, RunBudget,
    ToolCall, ToolCatalogEntry, ToolResult, ToolSchema,
)
from tldw_chatbook.Agents.agent_runtime import LoopDeps, run_agent_loop

CALC = ToolSchema(id="builtin:calculator", name="calculator",
                  description="math", parameters={"type": "object"})


def fence(name, args):
    return f'```tool_call\n{json.dumps({"name": name, "arguments": args})}\n```'


def make_deps(turns, *, invoke=None, spawn=None, cancel=None, clock=None):
    """Deps whose call_model pops scripted ModelTurns."""
    script = list(turns)

    def call_model(messages, active_schemas):
        return script.pop(0)

    return LoopDeps(
        call_model=call_model,
        invoke_tool=invoke or (lambda c: ToolResult(ok=True, content="42")),
        spawn=spawn or (lambda task: ToolResult(ok=True, content="sub done")),
        find_tools=lambda q: [ToolCatalogEntry(
            id="builtin:calculator", name="calculator",
            one_line_description="math", source="builtin")],
        load_schemas=lambda ids: [CALC],
        should_cancel=cancel or (lambda: False),
        clock=clock or (lambda: 0.0),
    )


CFG = AgentConfig(model="m", system_prompt="s",
                  allowed_tools=("calculator", SPAWN_TOOL_NAME))


def run(turns, **kw):
    cfg = kw.pop("config", CFG)
    active = kw.pop("active", [CALC])
    return run_agent_loop(cfg, [{"role": "user", "content": "hi"}],
                          active, make_deps(turns, **kw))


def test_plain_answer_no_tools():
    out = run([ModelTurn(text="Tokyo.")])
    assert out.status == RUN_DONE and out.final_text == "Tokyo."
    assert [s.kind for s in out.steps] == ["model"]


def test_fenced_tool_call_then_answer():
    calls = []
    out = run(
        [ModelTurn(text=fence("calculator", {"expression": "6*7"})),
         ModelTurn(text="It is 42.")],
        invoke=lambda c: calls.append(c) or ToolResult(ok=True, content="42"),
    )
    assert out.status == RUN_DONE and out.final_text == "It is 42."
    assert calls[0].name == "calculator"
    kinds = [s.kind for s in out.steps]
    assert kinds == ["model", "tool_call", "tool_result", "model"]


def test_native_tool_calls_take_precedence_over_text():
    out = run(
        [ModelTurn(text="ignored prose",
                   tool_calls=(ToolCall(name="calculator",
                                        args={"expression": "1"}),)),
         ModelTurn(text="done")])
    assert out.status == RUN_DONE
    assert out.steps[1].tool_name == "calculator"


def test_tool_error_is_not_fatal_and_feeds_back():
    seen = []

    def call_model(messages, active_schemas):
        seen.append(list(messages))
        return (ModelTurn(text=fence("calculator", {"expression": "x"}))
                if len(seen) == 1 else ModelTurn(text="recovered"))

    deps = make_deps([], invoke=lambda c: ToolResult(ok=False, error="boom"))
    deps.call_model = call_model
    out = run_agent_loop(CFG, [{"role": "user", "content": "hi"}], [CALC], deps)
    assert out.status == RUN_DONE and out.final_text == "recovered"
    assert "ERROR: boom" in seen[1][-1]["content"]
    result_steps = [s for s in out.steps if s.kind == STEP_TOOL_RESULT]
    assert result_steps and "boom" in result_steps[0].result


def test_spawn_result_and_budget():
    tasks = []
    turns = [ModelTurn(text=fence(SPAWN_TOOL_NAME, {"task": "t1"})),
             ModelTurn(text=fence(SPAWN_TOOL_NAME, {"task": "t2"})),
             ModelTurn(text=fence(SPAWN_TOOL_NAME, {"task": "t3"})),
             ModelTurn(text="all done")]
    # 3 spawn rounds + the final answer = 11 steps; default max_steps=8
    # would trip stuck first, so raise it — this test is about spawn budget.
    out = run(turns, spawn=lambda t: tasks.append(t) or ToolResult(
        ok=True, content=f"did {t}"),
        config=AgentConfig(model="m", system_prompt="s",
                           allowed_tools=("calculator", SPAWN_TOOL_NAME),
                           budget=RunBudget(max_steps=20)))
    assert out.status == RUN_DONE
    assert tasks == ["t1", "t2"]           # max_subagents=2: third refused
    assert out.subagents_spawned == 2
    spawn_steps = [s for s in out.steps if s.kind == STEP_SPAWN]
    assert len(spawn_steps) == 2


def test_disclosure_find_then_load_then_invoke():
    turns = [ModelTurn(text=fence("find_tools", {"query": "math"})),
             ModelTurn(text=fence("load_tools",
                                  {"ids": ["builtin:calculator"]})),
             ModelTurn(text=fence("calculator", {"expression": "6*7"})),
             ModelTurn(text="42.")]
    # find + load + invoke + answer = 10 steps; default max_steps=8 would
    # trip stuck first, so raise it — this test is about disclosure flow.
    out = run(turns, active=[],            # nothing pre-disclosed
              config=AgentConfig(model="m", system_prompt="s",
                                 allowed_tools=("calculator",),
                                 budget=RunBudget(max_steps=16)))
    assert out.status == RUN_DONE and out.final_text == "42."


def test_step_budget_trips_to_stuck():
    endless = [ModelTurn(text=fence("calculator", {"expression": str(i)}))
               for i in range(50)]
    out = run(endless, config=AgentConfig(
        model="m", system_prompt="s", allowed_tools=("calculator",),
        budget=RunBudget(max_steps=4)))
    assert out.status == RUN_STUCK


def test_wall_clock_budget_trips_to_stuck():
    ticker = iter([0.0, 500.0, 1000.0, 1500.0])
    out = run([ModelTurn(text="never reached")],
              clock=lambda: next(ticker))
    assert out.status == RUN_STUCK


def test_identical_consecutive_calls_trip_loop_detection():
    same = ModelTurn(text=fence("calculator", {"expression": "6*7"}))
    out = run([same] * (LOOP_DETECTION_N + 1) + [ModelTurn(text="x")],
              config=AgentConfig(model="m", system_prompt="s",
                                 allowed_tools=("calculator",),
                                 budget=RunBudget(max_steps=50)))
    assert out.status == RUN_STUCK
    assert out.steps[-1].kind == "error"


def test_same_tool_different_args_is_not_stuck():
    turns = [ModelTurn(text=fence("calculator", {"expression": str(i)}))
             for i in range(3)] + [ModelTurn(text="fine")]
    out = run(turns, config=AgentConfig(
        model="m", system_prompt="s", allowed_tools=("calculator",),
        budget=RunBudget(max_steps=50)))
    assert out.status == RUN_DONE


def test_cancel_lands_at_step_boundary():
    flags = iter([False, True])
    out = run([ModelTurn(text=fence("calculator", {"expression": "1"})),
               ModelTurn(text="never")],
              cancel=lambda: next(flags, True))
    assert out.status == RUN_CANCELLED
