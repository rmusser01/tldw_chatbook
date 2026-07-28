# Tests/Agents/test_search_run_log_runtime_tool.py
"""search_run_log: primary-only, no catalog slot, dispatched by the loop."""

import json

import pytest

from tldw_chatbook.Agents.agent_models import (
    AGENT_KIND_PRIMARY,
    AGENT_KIND_SUBAGENT,
    RUN_DONE,
    RUNTIME_TOOL_NAMES,
    SEARCH_RUN_LOG_TOOL_NAME,
    SPAWN_TOOL_NAME,
    AgentConfig,
    ModelTurn,
    RunBudget,
    ToolCall,
    ToolResult,
)
from tldw_chatbook.Agents.agent_runtime import run_agent_loop
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.tool_catalog import (
    SEARCH_RUN_LOG_TOOL_SCHEMA,
    BuiltinToolProvider,
    ToolCatalogRegistry,
)
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

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


# -- Sub-agent isolation, gated to the top-level agent -----------------------
#
# Mirrors Tests/Agents/test_install_skill_runtime_tool.py::
# test_subagent_cannot_call_install_skill -- this task's own header says
# search_run_log mirrors install_skill exactly, so its isolation test does
# too. The AGENT_KIND_PRIMARY gate exists in TWO independent places in
# agent_service.py's _run_one: the schema pin (the runtime_schemas.append
# under the `agent_kind == AGENT_KIND_PRIMARY and ...` condition) and the
# LoopDeps wiring (`search_run_log=(search_run_log if agent_kind ==
# AGENT_KIND_PRIMARY else None)`). Either can regress independently, so
# this test pins BOTH halves rather than just the end-to-end outcome:
#   (a) the schema must never be disclosed to a child at all -- a
#       fence-protocol child's own rendered system prompt (which embeds
#       every schema it was given, by name) must not mention
#       "search_run_log";
#   (b) a child that calls the name anyway (scripted here regardless of
#       (a), the same way test_subagent_cannot_call_install_skill forces
#       the call) must be refused through the ordinary permission path
#       (deps.invoke_tool's "Tool not permitted" message) rather than
#       executing -- proven from the child run's own persisted
#       tool_result steps, not merely the parent's final answer.
#
# A child sharing the parent's log_dir through the two-phase bind is what
# makes this matter: without BOTH gates, a sub-agent handed this tool could
# read its PARENT's entire run history, directly contradicting what
# spawn_subagent promises its children ("It sees only the task text you
# pass").


def _fence(name, args):
    return f"```tool_call\n{json.dumps({'name': name, 'arguments': args})}\n```"


def _svc_fence(name, args):
    return {"choices": [{"message": {"content": _fence(name, args)}}]}


def test_subagent_cannot_call_search_run_log(tmp_path, monkeypatch):
    from tldw_chatbook.Agents import run_log as run_log_module

    # Deterministic, hermetic writer: the run log resolves under tmp_path
    # instead of the real (developer-machine) sandbox root, so `is_active`
    # is controlled by this test, not by whatever happens to be writable
    # on whatever machine runs the suite.
    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: tmp_path)

    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = ToolCatalogRegistry()
    reg.register_provider(BuiltinToolProvider())

    calls = []
    script = [
        _svc_fence(SPAWN_TOOL_NAME, {"task": "native task"}),  # parent spawns
        _svc_fence(SEARCH_RUN_LOG_TOOL_NAME, {"contains": "x"}),  # child tries
        {"choices": [{"message": {"content": "child gave up"}}]},
        {"choices": [{"message": {"content": "final"}}]},
    ]

    def chat(**kwargs):
        calls.append(kwargs)
        return script.pop(0)

    service = AgentService(db, reg, chat_call=chat)
    _rid, outcome = service.run_turn(
        conversation_id="c1",
        messages=[{"role": "user", "content": "go"}],
        config=AgentConfig(
            model="m",
            system_prompt="s",
            allowed_tools=("calculator", SPAWN_TOOL_NAME),
            budget=RunBudget(),
        ),
        api_endpoint="llama_cpp",  # non-native: fence protocol, schemas render into the system prompt
    )
    assert outcome.status == RUN_DONE

    # (a) schema gate: the child's OWN call (calls[1] -- the parent's spawn
    # dispatch runs the child's whole loop inline before dispatch returns,
    # so this is the second chat_call invocation overall) must never have
    # been offered search_run_log at all.
    child_system_prompt = calls[1]["messages_payload"][0]["content"]
    assert SEARCH_RUN_LOG_TOOL_NAME not in child_system_prompt

    # (b) dispatch gate: the child's call, made regardless of (a), must be
    # refused through the ordinary permission path, never executed.
    child_runs = [r for r in db.list_runs("c1") if r["agent_kind"] == "subagent"]
    assert len(child_runs) == 1
    tool_results = [
        s["result"] for s in child_runs[0]["steps"] if s["kind"] == "tool_result"
    ]
    assert any(
        f"Tool not permitted: {SEARCH_RUN_LOG_TOOL_NAME}" in r for r in tool_results
    )
