# Tests/Agents/test_agent_runtime.py
"""Pure loop tests with deterministic fake callables."""
import json

from tldw_chatbook.Agents.agent_models import (
    LOOP_DETECTION_N, RUN_CANCELLED, RUN_DONE, RUN_STUCK, SPAWN_TOOL_NAME,
    STEP_MODEL, STEP_SPAWN, STEP_TOOL_RESULT, AgentConfig, ModelTurn,
    RunBudget, ToolCall, ToolCatalogEntry, ToolResult, ToolSchema,
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


def test_model_turn_budget_exhaustion_is_distinct_from_step_budget():
    """A run that spends its model turns on tool rounds must stop with the
    model-turn copy while raw steps are still well under max_steps."""
    # budget: max_model_turns=2, max_steps=99 (steps can never bind)
    # script: every model turn returns a fence tool call (never a final
    # answer) -> turn 1 (3 steps), turn 2 (3 steps), then the check fires
    # BEFORE a third provider call.
    turns = [ModelTurn(text=fence("calculator", {"expression": str(i)}))
             for i in range(10)]
    out = run(turns, config=AgentConfig(
        model="m", system_prompt="s", allowed_tools=("calculator",),
        budget=RunBudget(max_steps=99, max_model_turns=2)))
    assert out.status == RUN_STUCK
    assert out.steps[-1].summary == "model-turn budget exhausted"
    assert sum(1 for s in out.steps if s.kind == STEP_MODEL) == 2
    assert len(out.steps) < 99


def test_console_budget_floor_plus_two_extra_rounds_completes():
    """AC #2: the documented 4-turn/10-step discovery floor plus TWO more
    real tool rounds (6 turns / 16 steps) completes under
    CONSOLE_RUN_BUDGET."""
    from tldw_chatbook.Chat.console_agent_bridge import CONSOLE_RUN_BUDGET

    # script: 5 fence tool-call turns then a final answer (6 model turns,
    # 16 steps) with a generous monotonic fake clock -> RUN_DONE, not stuck.
    turns = [ModelTurn(text=fence("calculator", {"expression": str(i)}))
             for i in range(5)] + [ModelTurn(text="final answer")]
    tick = iter(float(i) for i in range(100))
    out = run(turns, config=AgentConfig(
        model="m", system_prompt="s", allowed_tools=("calculator",),
        budget=CONSOLE_RUN_BUDGET), clock=lambda: next(tick))
    assert out.status == RUN_DONE
    assert sum(1 for s in out.steps if s.kind == STEP_MODEL) == 6
    assert len(out.steps) == 16


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


def test_cancel_recognized_after_final_answer_with_no_tool_call():
    # Finding B: a Stop that lands mid a plain final answer (no tool call to
    # dispatch) must not be silently downgraded to "done" -- the loop only
    # re-polls should_cancel at step/tool-call boundaries, and a no-tool-call
    # turn used to return RUN_DONE immediately without one more recheck.
    flags = iter([False, True])
    out = run([ModelTurn(text="Tokyo.")], cancel=lambda: next(flags, True))
    assert out.status == RUN_CANCELLED
    assert out.final_text == "Tokyo."


# --- G1/Q9: load_tools `ids` coercion must never crash and must not
# char-split a bare string id. ---

def test_load_tools_ids_null_does_not_crash_and_reports_no_valid_tools():
    seen_ids = []

    def load_schemas(ids):
        seen_ids.append(ids)
        return []

    deps = make_deps([ModelTurn(text=fence("load_tools", {"ids": None})),
                      ModelTurn(text="done")])
    deps.load_schemas = load_schemas
    out = run_agent_loop(CFG, [{"role": "user", "content": "hi"}], [], deps)
    assert seen_ids == [[]]                    # coerced to empty list, no crash
    assert out.status == RUN_DONE and out.final_text == "done"
    result_steps = [s for s in out.steps if s.kind == STEP_TOOL_RESULT]
    assert result_steps[0].result == "ERROR: No valid tools found to load"


def test_load_tools_ids_as_bare_string_loads_that_one_tool():
    seen_ids = []

    def load_schemas(ids):
        seen_ids.append(ids)
        return [CALC]

    deps = make_deps([ModelTurn(text=fence(
        "load_tools", {"ids": "builtin:calculator"})),
        ModelTurn(text="done")])
    deps.load_schemas = load_schemas
    out = run_agent_loop(CFG, [{"role": "user", "content": "hi"}], [], deps)
    assert seen_ids == [["builtin:calculator"]]  # not char-split
    assert out.status == RUN_DONE


# --- G5: load_tools must distinguish "all ids invalid" from "valid but
# no room". ---

def test_load_tools_all_invalid_ids_reports_no_valid_tools_not_no_room():
    deps = make_deps([ModelTurn(text=fence("load_tools", {"ids": ["nope"]})),
                      ModelTurn(text="done")])
    deps.load_schemas = lambda ids: []
    out = run_agent_loop(CFG, [{"role": "user", "content": "hi"}], [], deps)
    result_steps = [s for s in out.steps if s.kind == STEP_TOOL_RESULT]
    assert result_steps[0].result == "ERROR: No valid tools found to load"


NEW_TOOL = ToolSchema(id="builtin:new_tool", name="new_tool",
                      description="new", parameters={"type": "object"})


def test_load_tools_out_of_room_still_says_no_room():
    """A genuinely NEW tool (not already active) requested while the active
    cap is already full is refused as "no room" -- distinct from
    re-requesting an already-active tool, which reports "already loaded"
    instead (see test_load_tools_already_active_reports_already_loaded_not_no_room,
    Gemini M finding, PR #636 bot review)."""
    turns = [ModelTurn(text=fence("load_tools",
                                  {"ids": ["builtin:new_tool"]})),
             ModelTurn(text="done")]
    deps = make_deps(turns)
    deps.load_schemas = lambda ids: [NEW_TOOL]
    out = run_agent_loop(
        AgentConfig(model="m", system_prompt="s",
                   allowed_tools=("calculator", "new_tool"),
                   budget=RunBudget(max_active_tools=1)),
        [{"role": "user", "content": "hi"}], [CALC], deps)
    result_steps = [s for s in out.steps if s.kind == STEP_TOOL_RESULT]
    assert result_steps[0].result == "no room"


def test_load_tools_already_active_reports_already_loaded_not_no_room():
    """Gemini M finding (PR #636 bot review): re-requesting a tool that's
    already active must not report the same "no room" message as
    genuinely running out of budget -- the model would otherwise think it
    needs to free space when it can simply proceed to call the tool it
    already has."""
    turns = [ModelTurn(text=fence("load_tools",
                                  {"ids": ["builtin:calculator"]})),
             ModelTurn(text="done")]
    out = run(turns, active=[CALC],
              config=AgentConfig(model="m", system_prompt="s",
                                 allowed_tools=("calculator",),
                                 budget=RunBudget(max_active_tools=1)))
    result_steps = [s for s in out.steps if s.kind == STEP_TOOL_RESULT]
    assert result_steps[0].result == "already loaded: calculator"


# --- G4: an empty spawn task must be refused with no budget consumption
# and no STEP_SPAWN. ---

def test_spawn_empty_task_is_refused_without_budget_consumption():
    calls = []
    turns = [ModelTurn(text=fence(SPAWN_TOOL_NAME, {"task": "   "})),
             ModelTurn(text="done")]
    out = run(turns, spawn=lambda t: calls.append(t) or ToolResult(
        ok=True, content="x"))
    assert calls == []
    assert out.subagents_spawned == 0
    assert [s for s in out.steps if s.kind == STEP_SPAWN] == []
    result_steps = [s for s in out.steps if s.kind == STEP_TOOL_RESULT]
    assert "Task description cannot be empty" in result_steps[0].result


# --- Q6: spawn must be refused up front when not in allowed_tools, before
# ever dispatching to deps.spawn. ---

def test_spawn_not_in_allowed_tools_is_refused_before_dispatch():
    calls = []
    turns = [ModelTurn(text=fence(SPAWN_TOOL_NAME, {"task": "do it"})),
             ModelTurn(text="done")]
    cfg = AgentConfig(model="m", system_prompt="s",
                      allowed_tools=("calculator",))   # no spawn permission
    out = run(turns, config=cfg,
              spawn=lambda t: calls.append(t) or ToolResult(
                  ok=True, content="x"))
    assert calls == []
    assert out.subagents_spawned == 0
    assert [s for s in out.steps if s.kind == STEP_SPAWN] == []
    result_steps = [s for s in out.steps if s.kind == STEP_TOOL_RESULT]
    assert "Tool not permitted: spawn_subagent" in result_steps[0].result


# --- task-243 Task 2: native history convention (assistant echo + role=tool
# results), gated on call_id so the fence path stays byte-identical. ---

def _native_turn(calls, text=""):
    raw = [{"id": c.call_id, "type": "function",
           "function": {"name": c.name, "arguments": json.dumps(c.args)}}
          for c in calls]
    return ModelTurn(text=text, tool_calls=tuple(calls),
                     assistant_message={"role": "assistant", "content": text,
                                        "tool_calls": raw})


def test_native_multi_call_batch_dispatches_both_in_one_turn():
    """AC #3: two native calls in one reply -> two tool invocations before
    the next model turn, results paired to call ids as role='tool'."""
    calls = [ToolCall(name="echo", args={"v": "1"}, call_id="idA"),
             ToolCall(name="echo", args={"v": "2"}, call_id="idB")]
    seen_messages = []
    turns = iter([_native_turn(calls), ModelTurn(text="done")])

    def call_model(messages, active_schemas):
        seen_messages.append([dict(m) for m in messages])
        return next(turns)

    invoked = []

    def invoke_tool(call):
        invoked.append(call)
        return ToolResult(ok=True, content=f"ok:{call.args['v']}")

    deps = make_deps([], invoke=invoke_tool)
    deps.call_model = call_model
    out = run_agent_loop(CFG, [{"role": "user", "content": "go"}], [CALC],
                         deps)
    assert out.status == RUN_DONE and out.final_text == "done"
    assert [c.call_id for c in invoked] == ["idA", "idB"]  # one turn, both dispatched
    second_turn_history = seen_messages[1]
    assistant = second_turn_history[1]
    assert assistant["tool_calls"][0]["id"] == "idA"      # provider echo verbatim
    tool_msgs = [m for m in second_turn_history if m.get("role") == "tool"]
    assert [(m["tool_call_id"], m["content"]) for m in tool_msgs] == [
        ("idA", "ok:1"), ("idB", "ok:2")]
    assert not any(m.get("role") == "user" and
                  str(m.get("content", "")).startswith("Tool result for")
                  for m in second_turn_history[1:])


def test_fence_history_convention_unchanged():
    """A fence-parsed call (call_id='') keeps the plain-text convention:
    assistant text verbatim, user-role 'Tool result for ...' line, and NO
    new keys leak into fence-mode history messages."""
    seen = []
    fence_text = fence("calculator", {"expression": "6*7"})

    def call_model(messages, active_schemas):
        seen.append([dict(m) for m in messages])
        return (ModelTurn(text=fence_text) if len(seen) == 1
                else ModelTurn(text="It is 42."))

    deps = make_deps([])
    deps.call_model = call_model
    out = run_agent_loop(CFG, [{"role": "user", "content": "hi"}], [CALC],
                         deps)
    assert out.status == RUN_DONE and out.final_text == "It is 42."
    history = seen[1]
    assert history[1] == {"role": "assistant", "content": fence_text}
    assert set(history[1].keys()) == {"role", "content"}
    assert history[2]["role"] == "user"
    assert history[2]["content"].startswith("Tool result for")


def test_load_tools_same_batch_duplicate_names_admit_one_into_active():
    """PR #655 review: the loop's own last line of defense must also dedupe
    by name WITHIN one load batch (a caller can hand the same schema back
    twice via name/id aliases), so ``active`` never gains a duplicate."""
    deps = make_deps([ModelTurn(text=fence(
        "load_tools", {"ids": ["calculator", "builtin:calculator"]})),
        ModelTurn(text=fence("calculator", {"expression": "1"})),
        ModelTurn(text="done")])
    deps.load_schemas = lambda ids: [CALC, CALC]  # duplicate in ONE batch
    out = run_agent_loop(CFG, [{"role": "user", "content": "hi"}], [], deps)
    assert out.status == RUN_DONE
    load_result = [s for s in out.steps
                   if s.kind == STEP_TOOL_RESULT and s.tool_name == "load_tools"][0]
    assert load_result.result == "loaded: calculator"  # once, not twice
