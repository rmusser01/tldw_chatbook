"""Working-set replacement regressions for ``load_tools``."""

import json

from tldw_chatbook.Agents.agent_models import (
    STEP_TOOL_RESULT,
    AgentConfig,
    ModelTurn,
    RunBudget,
    ToolCall,
    ToolLoadSelection,
    ToolResult,
    ToolSchema,
)
from tldw_chatbook.Agents.agent_runtime import FENCE_OPEN, LoopDeps, run_agent_loop


def _fence(name, args):
    return f"{FENCE_OPEN}\n{json.dumps({'name': name, 'arguments': args})}\n```"


def _schema(name: str) -> ToolSchema:
    return ToolSchema(id=f"p:{name}", name=name, description="d", parameters={})


def test_load_tools_replaces_the_catalog_working_set_and_permission_names():
    foo, bar = _schema("foo"), _schema("bar")
    catalog = {foo.id: foo, bar.id: bar}
    turns = iter(
        [
            ModelTurn(text=_fence("load_tools", {"ids": [foo.id]})),
            ModelTurn(text=_fence("load_tools", {"ids": [bar.id]})),
            ModelTurn(text="done"),
        ]
    )
    seen_active: list[tuple[str, ...]] = []
    committed: list[frozenset[str]] = []
    deps = LoopDeps(
        call_model=lambda _messages, active: (
            seen_active.append(tuple(schema.name for schema in active)) or next(turns)
        ),
        invoke_tool=lambda call: ToolResult(ok=True, content=call.name),
        spawn=lambda task, **kwargs: ToolResult(ok=True),
        find_tools=lambda query: [],
        load_schemas=lambda ids, _messages, _call: ToolLoadSelection(
            accepted=tuple(catalog[item] for item in ids if item in catalog)
        ),
        replace_disclosed_names=committed.append,
        should_cancel=lambda: False,
        clock=lambda: 0.0,
    )

    outcome = run_agent_loop(
        AgentConfig(
            model="m",
            system_prompt="s",
            allowed_tools=("foo", "bar"),
            budget=RunBudget(max_steps=20),
        ),
        [{"role": "user", "content": "hi"}],
        [],
        deps,
    )

    assert outcome.status == "done"
    assert seen_active == [(), ("foo",), ("bar",)]
    assert committed == [frozenset({"foo"}), frozenset({"bar"})]
    assert all("no room" not in (step.result or "") for step in outcome.steps)


def test_budget_omission_and_invalid_selection_preserve_old_working_set():
    foo = _schema("foo")
    selections = iter(
        [
            ToolLoadSelection(omitted_for_budget=("p:large",)),
            ToolLoadSelection(invalid_inputs=("junk",)),
        ]
    )
    turns = iter(
        [
            ModelTurn(text=_fence("load_tools", {"ids": ["p:large"]})),
            ModelTurn(text=_fence("load_tools", {"ids": ["junk"]})),
            ModelTurn(text="done"),
        ]
    )
    seen_active: list[tuple[str, ...]] = []
    committed: list[frozenset[str]] = []
    steps = []
    deps = LoopDeps(
        call_model=lambda _messages, active: (
            seen_active.append(tuple(schema.name for schema in active)) or next(turns)
        ),
        invoke_tool=lambda call: ToolResult(ok=True),
        spawn=lambda task, **kwargs: ToolResult(ok=True),
        find_tools=lambda query: [],
        load_schemas=lambda _ids, _messages, _call: next(selections),
        replace_disclosed_names=committed.append,
        should_cancel=lambda: False,
        clock=lambda: 0.0,
        on_step=steps.append,
    )

    run_agent_loop(
        AgentConfig(model="m", system_prompt="s", allowed_tools=("foo",)),
        [{"role": "user", "content": "hi"}],
        [foo],
        deps,
    )

    assert seen_active == [("foo",), ("foo",), ("foo",)]
    assert committed == []
    results = [step.result for step in steps if step.kind == STEP_TOOL_RESULT]
    assert results == [
        "not loaded (request budget): p:large",
        "ERROR: invalid tool ids: junk",
    ]


def test_mixed_batch_refuses_load_but_executes_ordinary_call_under_old_set():
    foo, bar = _schema("foo"), _schema("bar")
    calls = [
        ToolCall("load_tools", {"ids": [bar.id]}, call_id="load"),
        ToolCall("foo", {}, call_id="ordinary"),
    ]
    turns = iter([ModelTurn(tool_calls=calls), ModelTurn(text="done")])
    invoked: list[str] = []
    committed: list[frozenset[str]] = []
    steps = []
    deps = LoopDeps(
        call_model=lambda _messages, _active: next(turns),
        invoke_tool=lambda call: invoked.append(call.name)
        or ToolResult(ok=True, content="ok"),
        spawn=lambda task, **kwargs: ToolResult(ok=True),
        find_tools=lambda query: [],
        load_schemas=lambda _ids, _messages, _call: ToolLoadSelection(
            accepted=(bar,)
        ),
        replace_disclosed_names=committed.append,
        should_cancel=lambda: False,
        clock=lambda: 0.0,
        on_step=steps.append,
    )

    run_agent_loop(
        AgentConfig(model="m", system_prompt="s", allowed_tools=("foo", "bar")),
        [{"role": "user", "content": "hi"}],
        [foo],
        deps,
    )

    assert invoked == ["foo"]
    assert committed == []
    results = [step.result for step in steps if step.kind == STEP_TOOL_RESULT]
    assert any("call load_tools alone" in (result or "") for result in results)


def test_repeated_load_batch_preserves_old_set():
    foo, bar = _schema("foo"), _schema("bar")
    turns = iter(
        [
            ModelTurn(
                tool_calls=[
                    ToolCall("load_tools", {"ids": [bar.id]}, call_id="one"),
                    ToolCall("load_tools", {"ids": [bar.id]}, call_id="two"),
                ]
            ),
            ModelTurn(text="done"),
        ]
    )
    committed: list[frozenset[str]] = []
    deps = LoopDeps(
        call_model=lambda _messages, _active: next(turns),
        invoke_tool=lambda call: ToolResult(ok=True),
        spawn=lambda task, **kwargs: ToolResult(ok=True),
        find_tools=lambda query: [],
        load_schemas=lambda _ids, _messages, _call: ToolLoadSelection(
            accepted=(bar,)
        ),
        replace_disclosed_names=committed.append,
        should_cancel=lambda: False,
        clock=lambda: 0.0,
    )

    run_agent_loop(
        AgentConfig(model="m", system_prompt="s", allowed_tools=("foo", "bar")),
        [{"role": "user", "content": "hi"}],
        [foo],
        deps,
    )

    assert committed == []
