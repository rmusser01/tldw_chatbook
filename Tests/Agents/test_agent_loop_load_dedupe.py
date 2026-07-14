# Tests/Agents/test_agent_loop_load_dedupe.py
"""Regression test for Task 13 fix A: loop-side load_tools dedupe.

F1-b (plan-a-final-review addendum): the loop's own `active` list could
accept a duplicate re-load of an already-active schema, desyncing
list-vs-set semantics at the cap boundary. The `load_tools` branch must
filter out any schema whose name is already in `active` BEFORE slicing
against the room budget.
"""
import json

from tldw_chatbook.Agents.agent_models import AgentConfig, RunBudget, ToolSchema
from tldw_chatbook.Agents.agent_runtime import LoopDeps, run_agent_loop, FENCE_OPEN


def _fence(name, args):
    return f'{FENCE_OPEN}\n{json.dumps({"name": name, "arguments": args})}\n```'


def test_load_tools_never_duplicates_an_active_schema():
    active = [ToolSchema(id="p:foo", name="foo", description="d", parameters={})]
    turns = iter([
        type("M", (), {"text": _fence("load_tools", {"ids": ["p:foo"]}), "tool_calls": ()})(),
        type("M", (), {"text": "done", "tool_calls": ()})(),
    ])
    seen_active_sizes = []

    def call_model(messages, active_schemas):
        seen_active_sizes.append(len(active_schemas))
        return next(turns)

    deps = LoopDeps(
        call_model=call_model,
        invoke_tool=lambda call: None,
        spawn=lambda task, **k: None,
        find_tools=lambda q: [],
        load_schemas=lambda ids: [ToolSchema(id="p:foo", name="foo", description="d", parameters={})],
        should_cancel=lambda: False,
        clock=lambda: 0.0)
    outcome = run_agent_loop(
        AgentConfig(model="m", system_prompt="s", allowed_tools=("foo", "load_tools"),
                    budget=RunBudget(max_active_tools=8)),
        [{"role": "user", "content": "hi"}], active, deps)
    # The already-active "foo" schema is not appended a second time.
    assert seen_active_sizes[-1] == 1


def test_load_dedupe_at_cap_boundary_still_admits_a_new_tool():
    """load([t0]) twice + load([t1]) at cap 2: t1 is admitted, `active`
    never gains a duplicate name, and the redundant t0 re-load does not
    starve t1 of the room it needs (F1-b cap-boundary lockstep)."""
    t0 = ToolSchema(id="p:t0", name="t0", description="d", parameters={})
    t1 = ToolSchema(id="p:t1", name="t1", description="d", parameters={})
    catalog = {"p:t0": t0, "p:t1": t1}
    turns = iter([
        type("M", (), {"text": _fence("load_tools", {"ids": ["p:t0"]}), "tool_calls": ()})(),
        type("M", (), {"text": _fence("load_tools", {"ids": ["p:t0"]}), "tool_calls": ()})(),
        type("M", (), {"text": _fence("load_tools", {"ids": ["p:t1"]}), "tool_calls": ()})(),
        type("M", (), {"text": "done", "tool_calls": ()})(),
    ])
    seen_active_sizes = []

    def call_model(messages, active_schemas):
        seen_active_sizes.append(len(active_schemas))
        return next(turns)

    deps = LoopDeps(
        call_model=call_model,
        invoke_tool=lambda call: None,
        spawn=lambda task, **k: None,
        find_tools=lambda q: [],
        load_schemas=lambda ids: [catalog[i] for i in ids if i in catalog],
        should_cancel=lambda: False,
        clock=lambda: 0.0)
    outcome = run_agent_loop(
        AgentConfig(model="m", system_prompt="s",
                    allowed_tools=("t0", "t1", "load_tools"),
                    budget=RunBudget(max_active_tools=2, max_steps=20)),
        [{"role": "user", "content": "hi"}], [], deps)
    # 0 active at the first call, 1 after t0 loads, still 1 after the
    # redundant t0 re-load (no-op, no room consumed), then 2 once t1 is
    # admitted.
    assert seen_active_sizes == [0, 1, 1, 2]
    assert outcome.status == "done"
