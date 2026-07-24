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
