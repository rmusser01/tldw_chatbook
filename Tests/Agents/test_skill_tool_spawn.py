import json
import pytest
from tldw_chatbook.Agents.agent_models import (
    AgentConfig, DIRECT_DISCLOSE_THRESHOLD, FIND_TOOLS_NAME, LOAD_TOOLS_NAME,
    RUN_DONE, RunBudget, SPAWN_TOOL_NAME, ToolCatalogEntry, ToolResult,
    ToolSchema,
)
from tldw_chatbook.Agents.agent_runtime import FENCE_OPEN
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.tool_catalog import (
    BuiltinToolProvider, SkillToolProvider, ToolCatalogRegistry,
)
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


def _registry_with_code_review_skill():
    # "code-review" must be a real catalog entry (not just a name floating
    # in config.allowed_tools) so it can actually be DISCLOSED -- the same
    # way a builtin is -- rather than merely permitted. Mirrors how
    # console_agent_bridge._compose_run_registry_and_allowed wires a real
    # SkillToolProvider in production (Task-12 review Finding 1).
    reg = ToolCatalogRegistry()
    reg.register_provider(BuiltinToolProvider())
    reg.register_provider(SkillToolProvider(
        [{"name": "code-review", "description": "Reviews a diff.",
          "argument_hint": "the diff"}]))
    return reg


def test_skill_tool_routes_through_spawn(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = _registry_with_code_review_skill()
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
    reg = _registry_with_code_review_skill()
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


# --- Task-12 review Finding 1: skill dispatch must honor disclosed_names,
# not just config.allowed_tools -- exactly like an ordinary catalog tool. ---

class _NCatalogProvider:
    """Catalog of N generic tools, to force the find/load disclosure path.

    DIRECT_DISCLOSE_THRESHOLD is 8; a catalog bigger than that defers all
    disclosure to find_tools/load_tools instead of direct-disclosing
    everything up front (see tool_catalog.initial_disclosure).
    """

    def __init__(self, names):
        self._names = list(names)

    def list_catalog(self):
        return [ToolCatalogEntry(id=f"fake:{n}", name=n,
                                 one_line_description=f"tool {n}",
                                 source="fake")
                for n in self._names]

    def load_schema(self, tool_id):
        name = tool_id.split(":", 1)[1]
        return ToolSchema(id=tool_id, name=name, description="fake",
                          parameters={"type": "object"})

    def invoke(self, tool_id, args):
        return ToolResult(ok=True, content=f"invoked {tool_id}")


class _NamedSkillRunner:
    def __init__(self, skill_name):
        self._skill_name = skill_name
        self.ran_with = None

    def is_skill_tool(self, name):
        return name == self._skill_name

    def run(self, name, args, spawn):
        self.ran_with = args
        return spawn(f"RENDERED[{args}]")


def _nine_entry_names():
    # 9 > DIRECT_DISCLOSE_THRESHOLD(8) -- forces the find/load path.
    assert DIRECT_DISCLOSE_THRESHOLD == 8
    return ["code-review"] + [f"filler{i}" for i in range(8)]


def test_undisclosed_skill_tool_is_refused_without_find_load(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    names = _nine_entry_names()
    registry = ToolCatalogRegistry()
    registry.register_provider(_NCatalogProvider(names))
    config = AgentConfig(model="m", system_prompt="s",
                         allowed_tools=tuple(names),
                         budget=RunBudget(max_steps=6))
    script = [
        # Calls the skill cold -- never disclosed via find_tools/load_tools,
        # and the catalog is too big for direct-disclosure.
        {"choices": [{"message": {"content": _fence("code-review", {"args": "the diff"})}}]},
        {"choices": [{"message": {"content": "gave up"}}]},
    ]
    runner = _NamedSkillRunner("code-review")
    service = AgentService(db, registry, chat_call=lambda **k: script.pop(0),
                           skill_runner=runner)
    run_id, outcome = service.run_turn(
        conversation_id="c1", messages=[{"role": "user", "content": "q"}],
        config=config, api_endpoint="llama_cpp")
    assert outcome.status == RUN_DONE
    run = db.get_run(run_id)
    results = [s for s in run["steps"] if s["kind"] == "tool_result"]
    assert "Tool not permitted: code-review" in results[0]["result"]
    assert runner.ran_with is None                     # never actually spawned
    assert db.count_subagent_runs("c1") == 0


def test_skill_tool_executes_after_find_load_discloses_it(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    names = _nine_entry_names()
    registry = ToolCatalogRegistry()
    registry.register_provider(_NCatalogProvider(names))
    config = AgentConfig(model="m", system_prompt="s",
                         allowed_tools=tuple(names) + (SPAWN_TOOL_NAME,),
                         budget=RunBudget(max_steps=10))
    script = [
        {"choices": [{"message": {"content": _fence(
            LOAD_TOOLS_NAME, {"ids": ["fake:code-review"]})}}]},
        {"choices": [{"message": {"content": _fence(
            "code-review", {"args": "the diff"})}}]},
        {"choices": [{"message": {"content": "child answer"}}]},   # sub-agent turn
        {"choices": [{"message": {"content": "Done reviewing."}}]},  # primary final
    ]
    runner = _NamedSkillRunner("code-review")
    service = AgentService(db, registry, chat_call=lambda **k: script.pop(0),
                           skill_runner=runner)
    run_id, outcome = service.run_turn(
        conversation_id="c1", messages=[{"role": "user", "content": "review"}],
        config=config, api_endpoint="llama_cpp")
    assert outcome.status == RUN_DONE
    assert runner.ran_with == "the diff"
    assert db.count_subagent_runs("c1") == 1


# --- Task-12 review Finding 2: max_subagents must bound the COMBINED count
# of native spawn_subagent runs and skill-tool runs, not each independently.
# Order-agnostic: test both orders. ---

def test_combined_budget_native_spawn_then_skill_call(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = _registry_with_code_review_skill()
    script = [
        {"choices": [{"message": {"content": _fence(
            SPAWN_TOOL_NAME, {"task": "native task"})}}]},
        {"choices": [{"message": {"content": "native child answer"}}]},
        {"choices": [{"message": {"content": _fence(
            "code-review", {"args": "the diff"})}}]},
        {"choices": [{"message": {"content": "final"}}]},
    ]
    runner = _FakeSkillRunner()
    service = AgentService(db, reg, chat_call=lambda **k: script.pop(0),
                           skill_runner=runner)
    _r, outcome = service.run_turn(
        conversation_id="c1", messages=[{"role": "user", "content": "go"}],
        config=AgentConfig(model="m", system_prompt="s",
                           allowed_tools=("calculator", "code-review", SPAWN_TOOL_NAME),
                           budget=RunBudget(max_subagents=1, max_steps=12)),
        api_endpoint="llama_cpp")
    assert outcome.status == RUN_DONE
    assert db.count_subagent_runs("c1") == 1            # only the native spawn ran
    assert runner.spawned_with is None                  # the skill call never actually ran
    run = db.get_run(_r)
    results = [s for s in run["steps"] if s["kind"] == "tool_result"]
    assert any("sub-agent budget exhausted" in r["result"] for r in results)


def test_combined_budget_skill_call_then_native_spawn(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = _registry_with_code_review_skill()
    script = [
        {"choices": [{"message": {"content": _fence(
            "code-review", {"args": "the diff"})}}]},
        {"choices": [{"message": {"content": "skill child answer"}}]},
        {"choices": [{"message": {"content": _fence(
            SPAWN_TOOL_NAME, {"task": "native task"})}}]},
        {"choices": [{"message": {"content": "final"}}]},
    ]
    runner = _FakeSkillRunner()
    service = AgentService(db, reg, chat_call=lambda **k: script.pop(0),
                           skill_runner=runner)
    _r, outcome = service.run_turn(
        conversation_id="c1", messages=[{"role": "user", "content": "go"}],
        config=AgentConfig(model="m", system_prompt="s",
                           allowed_tools=("calculator", "code-review", SPAWN_TOOL_NAME),
                           budget=RunBudget(max_subagents=1, max_steps=12)),
        api_endpoint="llama_cpp")
    assert outcome.status == RUN_DONE
    assert db.count_subagent_runs("c1") == 1            # only the skill's spawn ran
    assert runner.spawned_with == "the diff"
    run = db.get_run(_r)
    results = [s for s in run["steps"] if s["kind"] == "tool_result"]
    assert any("sub-agent budget exhausted" in r["result"] for r in results)


# --- Pre-merge review MINOR 3: an ordinary (native spawn_subagent, not
# skill-driven) child's allow-list must exclude skill-tool names too, not
# just the spawn tool itself -- mirroring the skill-driven child's own
# explicit builtins-only allow-list (SkillRunner.run's `intersect_skill_tools`
# call never re-admits skill names). Previously the child inherited every
# skill name the parent had, so a child's attempt to call one only happened
# to be refused by the incidental max_subagents=0 depth-1 clamp (a numeric
# budget check, not a permission boundary) rather than the "Tool not
# permitted" gate every other disallowed tool hits. ---

def test_native_spawn_child_cannot_call_a_skill_tool(tmp_path):
    """A child spawned via the native spawn_subagent tool (not a skill's
    own spawn) must have skill names excluded from its allow-list, exactly
    like the spawn tool name itself already was. Proves the refusal
    happens at the permission GATE ("Tool not permitted"), never falling
    through to the budget-exhausted branch, and that skill_runner.run is
    never reached (the counting fake's `spawned_with` stays ``None``)."""
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = _registry_with_code_review_skill()
    script = [
        {"choices": [{"message": {"content": _fence(
            SPAWN_TOOL_NAME, {"task": "native task"})}}]},
        # Inside the child: the model attempts the skill tool directly.
        {"choices": [{"message": {"content": _fence(
            "code-review", {"args": "the diff"})}}]},
        {"choices": [{"message": {"content": "child gave up"}}]},
        {"choices": [{"message": {"content": "final"}}]},
    ]
    runner = _FakeSkillRunner()
    service = AgentService(db, reg, chat_call=lambda **k: script.pop(0),
                           skill_runner=runner)
    _r, outcome = service.run_turn(
        conversation_id="c1", messages=[{"role": "user", "content": "go"}],
        config=AgentConfig(model="m", system_prompt="s",
                           allowed_tools=("calculator", "code-review", SPAWN_TOOL_NAME),
                           budget=RunBudget()),
        api_endpoint="llama_cpp")
    assert outcome.status == RUN_DONE
    assert runner.spawned_with is None    # never actually rendered/run

    runs = db.list_runs("c1")
    child_runs = [r for r in runs if r["agent_kind"] == "subagent"]
    assert len(child_runs) == 1
    tool_results = [s["result"] for s in child_runs[0]["steps"] if s["kind"] == "tool_result"]
    assert any("Tool not permitted: code-review" in r for r in tool_results)
    assert not any("sub-agent budget exhausted" in r for r in tool_results)
