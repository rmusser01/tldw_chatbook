import json
import pytest
from tldw_chatbook.Agents.agent_models import AgentConfig, RunBudget, SPAWN_TOOL_NAME, RUN_DONE, ToolResult
from tldw_chatbook.Agents.agent_runtime import FENCE_OPEN
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider, ToolCatalogRegistry
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


def _fence(name, args):
    return f'{FENCE_OPEN}\n{json.dumps({"name": name, "arguments": args})}\n```'


class _FakeSkillRunner:
    def __init__(self):
        self.spawned_with = None

    def is_skill_tool(self, name):
        return name == "code-review"

    def run(self, name, args, spawn):
        self.spawned_with = args
        return spawn(f"RENDERED[{args}]", allowed_tools=("calculator",))


def test_skill_tool_routes_through_spawn(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = ToolCatalogRegistry(); reg.register_provider(BuiltinToolProvider())
    script = [
        {"choices": [{"message": {"content": _fence("code-review", {"args": "the diff"})}}]},
        {"choices": [{"message": {"content": "child answer"}}]},   # sub-agent turn
        {"choices": [{"message": {"content": "Done reviewing."}}]},  # primary final
    ]
    runner = _FakeSkillRunner()
    service = AgentService(db, reg, chat_call=lambda **k: script.pop(0), skill_runner=runner)
    _run_id, outcome = service.run_turn(
        conversation_id="c1", messages=[{"role": "user", "content": "review"}],
        config=AgentConfig(model="m", system_prompt="s",
                           allowed_tools=("calculator", "code-review", SPAWN_TOOL_NAME),
                           budget=RunBudget()),
        api_endpoint="llama_cpp")
    assert outcome.status == RUN_DONE
    assert runner.spawned_with == "the diff"
    assert db.count_subagent_runs("c1") == 1          # skill ran as a budget-counted sub-agent


def test_skill_tool_respects_subagent_budget(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = ToolCatalogRegistry(); reg.register_provider(BuiltinToolProvider())
    # Two skill calls with max_subagents=1: the second must be refused.
    script = [
        {"choices": [{"message": {"content": _fence("code-review", {"args": "a"})}}]},
        {"choices": [{"message": {"content": "child a"}}]},
        {"choices": [{"message": {"content": _fence("code-review", {"args": "b"})}}]},
        {"choices": [{"message": {"content": "child b never"}}]},
        {"choices": [{"message": {"content": "final"}}]},
    ]
    service = AgentService(db, reg, chat_call=lambda **k: script.pop(0),
                           skill_runner=_FakeSkillRunner())
    _r, outcome = service.run_turn(
        conversation_id="c1", messages=[{"role": "user", "content": "go"}],
        config=AgentConfig(model="m", system_prompt="s",
                           allowed_tools=("calculator", "code-review", SPAWN_TOOL_NAME),
                           budget=RunBudget(max_subagents=1, max_steps=12)),
        api_endpoint="llama_cpp")
    assert outcome.status == RUN_DONE
    assert db.count_subagent_runs("c1") == 1          # second skill spawn refused by budget
