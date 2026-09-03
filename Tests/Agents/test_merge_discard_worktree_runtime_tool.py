# Tests/Agents/test_merge_discard_worktree_runtime_tool.py
"""merge_agent_worktree / discard_agent_worktree: dispatch-layer pins.

TASK-28238 phase 2 Task 5. Mirrors
``Tests/Agents/test_search_run_log_runtime_tool.py``'s shape: the loop-
level dispatch wiring (LoopDeps field -> STEP_TOOL_CALL -> deps callable)
is cheap to pin directly with ``run_agent_loop`` and a scripted
``ModelTurn``, independent of the full AgentService/FleetCoordinator
integration in ``Tests/Agents/test_fleet_runtime.py`` (which covers the
real merge/discard service closures, confirm gating, and real git repos).
"""

from tldw_chatbook.Agents.agent_models import (
    DISCARD_AGENT_WORKTREE_TOOL_NAME,
    MERGE_AGENT_WORKTREE_TOOL_NAME,
    RUN_DONE,
    RUNTIME_TOOL_NAMES,
    AgentConfig,
    ModelTurn,
    RunBudget,
    ToolCall,
    ToolResult,
)
from tldw_chatbook.Agents.agent_runtime import run_agent_loop
from tldw_chatbook.Agents.tool_catalog import (
    DISCARD_AGENT_WORKTREE_SCHEMA,
    MERGE_AGENT_WORKTREE_SCHEMA,
)

from Tests.Agents.test_agent_runtime import make_deps


def test_names_are_registered_as_runtime_tools():
    assert MERGE_AGENT_WORKTREE_TOOL_NAME in RUNTIME_TOOL_NAMES
    assert DISCARD_AGENT_WORKTREE_TOOL_NAME in RUNTIME_TOOL_NAMES
    assert MERGE_AGENT_WORKTREE_SCHEMA.name == MERGE_AGENT_WORKTREE_TOOL_NAME
    assert DISCARD_AGENT_WORKTREE_SCHEMA.name == DISCARD_AGENT_WORKTREE_TOOL_NAME
    props = MERGE_AGENT_WORKTREE_SCHEMA.parameters["properties"]
    assert "handle_id" in props and "mode" in props
    assert props["mode"]["enum"] == ["apply", "merge"]
    assert MERGE_AGENT_WORKTREE_SCHEMA.parameters["required"] == ["handle_id"]
    discard_props = DISCARD_AGENT_WORKTREE_SCHEMA.parameters["properties"]
    assert "handle_id" in discard_props


_CFG = AgentConfig(
    model="m", system_prompt="s", budget=RunBudget(max_steps=8, max_model_turns=8)
)


def test_loop_dispatches_merge_to_the_injected_callable():
    seen = {}

    def handler(handle_id, mode):
        seen["handle_id"] = handle_id
        seen["mode"] = mode
        return ToolResult(ok=True, content="merged")

    turns = [
        ModelTurn(
            text="",
            tool_calls=(
                ToolCall(
                    name=MERGE_AGENT_WORKTREE_TOOL_NAME,
                    args={"handle_id": "h1", "mode": "merge"},
                    call_id="c1",
                ),
            ),
            assistant_message={"role": "assistant", "content": ""},
        ),
        ModelTurn(text="done"),
    ]
    deps = make_deps(turns)
    deps.merge_agent_worktree = handler
    outcome = run_agent_loop(_CFG, [{"role": "user", "content": "go"}], [], deps)
    assert seen == {"handle_id": "h1", "mode": "merge"}
    assert outcome.status == RUN_DONE
    assert outcome.final_text == "done"


def test_merge_defaults_mode_to_apply_when_omitted():
    seen = {}

    def handler(handle_id, mode):
        seen["handle_id"] = handle_id
        seen["mode"] = mode
        return ToolResult(ok=True, content="merged")

    turns = [
        ModelTurn(
            text="",
            tool_calls=(
                ToolCall(
                    name=MERGE_AGENT_WORKTREE_TOOL_NAME,
                    args={"handle_id": "h1"},
                    call_id="c1",
                ),
            ),
            assistant_message={"role": "assistant", "content": ""},
        ),
        ModelTurn(text="done"),
    ]
    deps = make_deps(turns)
    deps.merge_agent_worktree = handler
    run_agent_loop(_CFG, [{"role": "user", "content": "go"}], [], deps)
    assert seen == {"handle_id": "h1", "mode": "apply"}


def test_loop_dispatches_discard_to_the_injected_callable():
    seen = {}

    def handler(handle_id):
        seen["handle_id"] = handle_id
        return ToolResult(ok=True, content="discarded")

    turns = [
        ModelTurn(
            text="",
            tool_calls=(
                ToolCall(
                    name=DISCARD_AGENT_WORKTREE_TOOL_NAME,
                    args={"handle_id": "h1"},
                    call_id="c1",
                ),
            ),
            assistant_message={"role": "assistant", "content": ""},
        ),
        ModelTurn(text="done"),
    ]
    deps = make_deps(turns)
    deps.discard_agent_worktree = handler
    outcome = run_agent_loop(_CFG, [{"role": "user", "content": "go"}], [], deps)
    assert seen == {"handle_id": "h1"}
    assert outcome.status == RUN_DONE


def test_unwired_merge_falls_through_to_the_permission_gate():
    # deps.merge_agent_worktree is None -> the else branch -> deps.invoke_tool.
    invoked = []

    def invoke(call):
        invoked.append(call.name)
        return ToolResult(ok=False, error=f"Tool not permitted: {call.name}")

    turns = [
        ModelTurn(
            text="",
            tool_calls=(
                ToolCall(
                    name=MERGE_AGENT_WORKTREE_TOOL_NAME,
                    args={"handle_id": "h1"},
                    call_id="c1",
                ),
            ),
            assistant_message={"role": "assistant", "content": ""},
        ),
        ModelTurn(text="done"),
    ]
    deps = make_deps(turns, invoke=invoke)
    run_agent_loop(_CFG, [{"role": "user", "content": "go"}], [], deps)
    assert invoked == [MERGE_AGENT_WORKTREE_TOOL_NAME]


def test_unwired_discard_falls_through_to_the_permission_gate():
    invoked = []

    def invoke(call):
        invoked.append(call.name)
        return ToolResult(ok=False, error=f"Tool not permitted: {call.name}")

    turns = [
        ModelTurn(
            text="",
            tool_calls=(
                ToolCall(
                    name=DISCARD_AGENT_WORKTREE_TOOL_NAME,
                    args={"handle_id": "h1"},
                    call_id="c1",
                ),
            ),
            assistant_message={"role": "assistant", "content": ""},
        ),
        ModelTurn(text="done"),
    ]
    deps = make_deps(turns, invoke=invoke)
    run_agent_loop(_CFG, [{"role": "user", "content": "go"}], [], deps)
    assert invoked == [DISCARD_AGENT_WORKTREE_TOOL_NAME]
