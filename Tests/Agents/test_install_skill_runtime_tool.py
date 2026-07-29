"""install_skill: the fifth runtime tool — name, schema, dispatch, gating.

Model: Tests/Agents/test_skill_file_runtime_tool.py (the 4th runtime tool).
install_skill is NOT a ToolProvider — its schema is pinned into
runtime_schemas (never disclosure-gated) only for the top-level agent
(agent_kind == primary), and its closure lives on LoopDeps.install_skill.
"""

import json

from tldw_chatbook.Agents.agent_models import (
    INSTALL_SKILL_TOOL_NAME,
    RUNTIME_TOOL_NAMES,
    RUN_LOG_SLICE_TOOL_NAME,
    RUN_LOG_STATS_TOOL_NAME,
    RUN_SKILL_SCRIPT_TOOL_NAME,
    SEARCH_RUN_LOG_TOOL_NAME,
    SPAWN_TOOL_NAME,
    FIND_TOOLS_NAME,
    LOAD_TOOLS_NAME,
    SKILL_FILE_TOOL_NAME,
    AgentConfig,
    ModelTurn,
    RUN_DONE,
    RunBudget,
    ToolCatalogEntry,
    ToolResult,
    ToolSchema,
)
from tldw_chatbook.Agents.tool_catalog import INSTALL_SKILL_TOOL_SCHEMA
from tldw_chatbook.Agents.agent_runtime import LoopDeps, run_agent_loop


def test_install_skill_name_in_runtime_tool_names():
    assert INSTALL_SKILL_TOOL_NAME == "install_skill"
    assert RUNTIME_TOOL_NAMES == {
        SPAWN_TOOL_NAME,
        FIND_TOOLS_NAME,
        LOAD_TOOLS_NAME,
        SKILL_FILE_TOOL_NAME,
        INSTALL_SKILL_TOOL_NAME,
        RUN_SKILL_SCRIPT_TOOL_NAME,
        SEARCH_RUN_LOG_TOOL_NAME,
        RUN_LOG_STATS_TOOL_NAME,
        RUN_LOG_SLICE_TOOL_NAME,
    }


def test_install_skill_schema_shape():
    s = INSTALL_SKILL_TOOL_SCHEMA
    assert s.id == "runtime:install_skill"
    assert s.name == INSTALL_SKILL_TOOL_NAME
    assert s.parameters["required"] == ["url"]
    assert s.parameters["properties"]["url"]["type"] == "string"
    # Description must tell the model the key facts.
    assert "pending" in s.description.lower()
    assert "confirm" in s.description.lower()


_CALC = ToolSchema(
    id="builtin:calculator", name="calculator", description="math",
    parameters={"type": "object"},
)


def _fence(name, args):
    return f"```tool_call\n{json.dumps({'name': name, 'arguments': args})}\n```"


def _deps(turns, *, install_skill=None, invoke=None):
    script = list(turns)

    def call_model(messages, active_schemas):
        return script.pop(0)

    return LoopDeps(
        call_model=call_model,
        invoke_tool=invoke or (lambda c: ToolResult(ok=False, error=f"Tool not permitted: {c.name}")),
        spawn=lambda task: ToolResult(ok=True, content="sub"),
        find_tools=lambda q: [],
        load_schemas=lambda ids: [],
        should_cancel=lambda: False,
        clock=lambda: 0.0,
        install_skill=install_skill,
    )


_CFG = AgentConfig(model="m", system_prompt="s", allowed_tools=("calculator",))


def test_install_skill_dispatches_to_deps_when_wired():
    seen = []

    def installer(url):
        seen.append(url)
        return ToolResult(ok=True, content=f"installed {url}")

    out = run_agent_loop(
        _CFG,
        [{"role": "user", "content": "hi"}],
        [_CALC],
        _deps(
            [
                ModelTurn(text=_fence("install_skill", {"url": "https://github.com/o/r"})),
                ModelTurn(text="done"),
            ],
            install_skill=installer,
        ),
    )
    assert out.status == RUN_DONE
    assert seen == ["https://github.com/o/r"]
    assert any(s.kind == "tool_result" and "installed" in (s.result or "") for s in out.steps)


def test_install_skill_falls_through_when_not_wired():
    out = run_agent_loop(
        _CFG,
        [{"role": "user", "content": "hi"}],
        [_CALC],
        _deps(
            [
                ModelTurn(text=_fence("install_skill", {"url": "https://github.com/o/r"})),
                ModelTurn(text="done"),
            ],
            install_skill=None,  # not wired -> generic invoke_tool path
        ),
    )
    assert out.status == RUN_DONE
    results = [s.result for s in out.steps if s.kind == "tool_result"]
    assert any("Tool not permitted: install_skill" in (r or "") for r in results)


# -- AgentService wiring, gated to the top-level agent ----------------------

from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider, ToolCatalogRegistry
from tldw_chatbook.Agents.agent_models import SPAWN_TOOL_NAME
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


def _svc_fence(name, args):
    return {"choices": [{"message": {"content": _fence(name, args)}}]}


def test_top_level_agent_dispatches_install_skill(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = ToolCatalogRegistry()
    reg.register_provider(BuiltinToolProvider())
    seen = []

    def installer(url):
        seen.append(url)
        return ToolResult(ok=True, content=f"installed {url}")

    script = [
        _svc_fence("install_skill", {"url": "https://github.com/o/r"}),
        {"choices": [{"message": {"content": "Done."}}]},
    ]
    service = AgentService(
        db, reg, chat_call=lambda **k: script.pop(0), install_skill_tool=installer
    )
    _rid, outcome = service.run_turn(
        conversation_id="c1",
        messages=[{"role": "user", "content": "install it"}],
        config=AgentConfig(
            model="m", system_prompt="s",
            allowed_tools=("calculator",), budget=RunBudget(),
        ),
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    assert seen == ["https://github.com/o/r"]


def test_subagent_cannot_call_install_skill(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = ToolCatalogRegistry()
    reg.register_provider(BuiltinToolProvider())

    def installer(url):
        raise AssertionError("subagent must never reach the installer")

    script = [
        _svc_fence(SPAWN_TOOL_NAME, {"task": "native task"}),   # parent spawns
        _svc_fence("install_skill", {"url": "https://github.com/o/r"}),  # child tries
        {"choices": [{"message": {"content": "child gave up"}}]},
        {"choices": [{"message": {"content": "final"}}]},
    ]
    service = AgentService(
        db, reg, chat_call=lambda **k: script.pop(0), install_skill_tool=installer
    )
    _rid, outcome = service.run_turn(
        conversation_id="c1",
        messages=[{"role": "user", "content": "go"}],
        config=AgentConfig(
            model="m", system_prompt="s",
            allowed_tools=("calculator", SPAWN_TOOL_NAME), budget=RunBudget(),
        ),
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    child_runs = [r for r in db.list_runs("c1") if r["agent_kind"] == "subagent"]
    assert len(child_runs) == 1
    tool_results = [
        s["result"] for s in child_runs[0]["steps"] if s["kind"] == "tool_result"
    ]
    assert any("Tool not permitted: install_skill" in r for r in tool_results)
