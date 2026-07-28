# Tests/Agents/test_search_run_log_runtime_tool.py
"""search_run_log: primary-only, no catalog slot, dispatched by the loop."""

import pytest

from tldw_chatbook.Agents.agent_models import (
    AGENT_KIND_PRIMARY,
    AGENT_KIND_SUBAGENT,
    RUNTIME_TOOL_NAMES,
    SEARCH_RUN_LOG_TOOL_NAME,
    AgentConfig,
    ModelTurn,
    RunBudget,
    ToolCall,
    ToolResult,
)
from tldw_chatbook.Agents.agent_runtime import run_agent_loop
from tldw_chatbook.Agents.tool_catalog import SEARCH_RUN_LOG_TOOL_SCHEMA

from Tests.Agents.test_agent_runtime import make_deps


def test_name_is_registered_as_a_runtime_tool():
    assert SEARCH_RUN_LOG_TOOL_NAME in RUNTIME_TOOL_NAMES
    assert SEARCH_RUN_LOG_TOOL_SCHEMA.name == SEARCH_RUN_LOG_TOOL_NAME
    props = SEARCH_RUN_LOG_TOOL_SCHEMA.parameters["properties"]
    assert "contains" in props and "pattern" in props and "from_record" in props


def test_loop_dispatches_to_the_injected_callable():
    seen = {}

    def handler(args):
        seen.update(args)
        return ToolResult(ok=True, content="record 000412 [model]")

    turns = [
        ModelTurn(
            text="",
            tool_calls=(
                ToolCall(
                    name=SEARCH_RUN_LOG_TOOL_NAME,
                    args={"contains": "refused"},
                    call_id="c1",
                ),
            ),
            assistant_message={"role": "assistant", "content": ""},
        ),
        ModelTurn(text="answered"),
    ]
    deps = make_deps(turns)
    deps.search_run_log = handler
    config = AgentConfig(
        model="m", system_prompt="s", budget=RunBudget(max_steps=8, max_model_turns=8)
    )
    outcome = run_agent_loop(config, [{"role": "user", "content": "go"}], [], deps)
    assert seen == {"contains": "refused"}
    assert outcome.final_text == "answered"


def test_unwired_name_falls_through_to_the_permission_gate():
    # deps.search_run_log is None -> the else branch -> deps.invoke_tool.
    invoked = []

    def invoke(call):
        invoked.append(call.name)
        return ToolResult(ok=False, error=f"Tool not permitted: {call.name}")

    turns = [
        ModelTurn(
            text="",
            tool_calls=(
                ToolCall(name=SEARCH_RUN_LOG_TOOL_NAME, args={}, call_id="c1"),
            ),
            assistant_message={"role": "assistant", "content": ""},
        ),
        ModelTurn(text="done"),
    ]
    deps = make_deps(turns, invoke=invoke)
    config = AgentConfig(
        model="m", system_prompt="s", budget=RunBudget(max_steps=8, max_model_turns=8)
    )
    run_agent_loop(config, [{"role": "user", "content": "go"}], [], deps)
    assert invoked == [SEARCH_RUN_LOG_TOOL_NAME]
