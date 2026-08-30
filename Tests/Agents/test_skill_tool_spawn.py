import json
from tldw_chatbook.Agents.agent_models import (
    AgentConfig,
    LOAD_TOOLS_NAME,
    RUN_DONE,
    RunBudget,
    SPAWN_TOOL_NAME,
    ToolCatalogEntry,
    ToolResult,
    ToolSchema,
)
from tldw_chatbook.Agents.agent_runtime import FENCE_OPEN
from tldw_chatbook.Agents.agent_service import AgentService, FirstRequestSchemaPlan
from tldw_chatbook.Agents.tool_catalog import (
    BuiltinToolProvider,
    FIND_TOOLS_SCHEMA,
    LOAD_TOOLS_SCHEMA,
    SkillToolProvider,
    ToolCatalogRegistry,
)
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

from Tests.Agents.conftest import join_fleet_children, pin_max_live_subagents
from Tests.Agents.test_agent_service import FleetChat, verbatim


def _fence(name, args):
    return f"{FENCE_OPEN}\n{json.dumps({'name': name, 'arguments': args})}\n```"


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
    reg.register_provider(
        SkillToolProvider(
            [
                {
                    "name": "code-review",
                    "description": "Reviews a diff.",
                    "argument_hint": "the diff",
                }
            ]
        )
    )
    return reg


def test_skill_tool_routes_through_spawn(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = _registry_with_code_review_skill()
    script = [
        {
            "choices": [
                {"message": {"content": _fence("code-review", {"args": "the diff"})}}
            ]
        },
        {"choices": [{"message": {"content": "child answer"}}]},  # sub-agent turn
        {"choices": [{"message": {"content": "Done reviewing."}}]},  # primary final
    ]
    runner = _FakeSkillRunner()
    service = AgentService(
        db, reg, chat_call=lambda **k: script.pop(0), skill_runner=runner
    )
    _run_id, outcome = service.run_turn(
        conversation_id="c1",
        messages=[{"role": "user", "content": "review"}],
        config=AgentConfig(
            model="m",
            system_prompt="s",
            allowed_tools=("calculator", "code-review", SPAWN_TOOL_NAME),
            budget=RunBudget(),
        ),
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    assert runner.spawned_with == "the diff"
    assert db.count_subagent_runs("c1") == 1  # skill ran as a budget-counted sub-agent


def test_skill_spawn_capture_failure_uses_the_parent_diagnostic(tmp_path, monkeypatch):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    original_insert = db.insert_steps_at_indices

    def fail_spawn_capture(run_id, indexed_steps):
        if any(
            step["kind"] == "tool_call" and step.get("tool_name") == "code-review"
            for _index, step in indexed_steps
        ):
            raise RuntimeError("persistent skill spawn capture failure")
        return original_insert(run_id, indexed_steps)

    monkeypatch.setattr(db, "insert_steps_at_indices", fail_spawn_capture)
    reg = _registry_with_code_review_skill()
    script = [
        {"choices": [{"message": {"content": _fence("code-review", {"args": "x"})}}]},
        {"choices": [{"message": {"content": "child answer"}}]},
        {"choices": [{"message": {"content": "done"}}]},
    ]
    service = AgentService(
        db, reg, chat_call=lambda **_kwargs: script.pop(0), skill_runner=_FakeSkillRunner()
    )
    parent_id, outcome = service.run_turn(
        conversation_id="skill-spawn-capture",
        messages=[{"role": "user", "content": "review"}],
        config=AgentConfig(
            model="m",
            system_prompt="s",
            allowed_tools=("calculator", "code-review", SPAWN_TOOL_NAME),
            budget=RunBudget(),
        ),
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    rows = db.list_runs("skill-spawn-capture", include_superseded=True)
    parent = next(row for row in rows if row["id"] == parent_id)
    child = next(row for row in rows if row["agent_kind"] == "subagent")
    diagnostic = next(step for step in parent["steps"] if step["kind"] == "capture_failed")
    diagnostic_id = f"agent-step:{parent_id}:{diagnostic['index']}"
    assert child["spawn_event_id"] == diagnostic_id


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
    service = AgentService(
        db, reg, chat_call=lambda **k: script.pop(0), skill_runner=_FakeSkillRunner()
    )
    _r, outcome = service.run_turn(
        conversation_id="c1",
        messages=[{"role": "user", "content": "go"}],
        config=AgentConfig(
            model="m",
            system_prompt="s",
            allowed_tools=("calculator", "code-review", SPAWN_TOOL_NAME),
            budget=RunBudget(max_subagents=1, max_steps=12),
        ),
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    assert db.count_subagent_runs("c1") == 1  # second skill spawn refused by budget


# --- Task-12 review Finding 1: skill dispatch must honor disclosed_names,
# not just config.allowed_tools -- exactly like an ordinary catalog tool. ---


class _NCatalogProvider:
    """Catalog of N generic tools for explicit discovery-path tests."""

    def __init__(self, names):
        self._names = list(names)

    def list_catalog(self):
        return [
            ToolCatalogEntry(
                id=f"fake:{n}", name=n, one_line_description=f"tool {n}", source="fake"
            )
            for n in self._names
        ]

    def load_schema(self, tool_id):
        name = tool_id.split(":", 1)[1]
        return ToolSchema(
            id=tool_id, name=name, description="fake", parameters={"type": "object"}
        )

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


def _discovery_names():
    return ["code-review"] + [f"filler{i}" for i in range(24)]


def _discovery_plan() -> FirstRequestSchemaPlan:
    return FirstRequestSchemaPlan(
        active_schemas=(),
        runtime_schemas=(FIND_TOOLS_SCHEMA, LOAD_TOOLS_SCHEMA),
        offer_find_load=True,
        log_active=False,
        system_prompt="s",
    )


def test_undisclosed_skill_tool_is_refused_without_find_load(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    names = _discovery_names()
    registry = ToolCatalogRegistry()
    registry.register_provider(_NCatalogProvider(names))
    config = AgentConfig(
        model="m",
        system_prompt="s",
        allowed_tools=tuple(names),
        budget=RunBudget(max_steps=6),
    )
    script = [
        # Calls the skill cold -- never disclosed via find_tools/load_tools,
        # and the catalog is too big for direct-disclosure.
        {
            "choices": [
                {"message": {"content": _fence("code-review", {"args": "the diff"})}}
            ]
        },
        {"choices": [{"message": {"content": "gave up"}}]},
    ]
    runner = _NamedSkillRunner("code-review")
    service = AgentService(
        db, registry, chat_call=lambda **k: script.pop(0), skill_runner=runner
    )
    run_id, outcome = service.run_turn(
        conversation_id="c1",
        messages=[{"role": "user", "content": "q"}],
        config=config,
        api_endpoint="llama_cpp",
        first_request_schema_plan=_discovery_plan(),
    )
    assert outcome.status == RUN_DONE
    run = db.get_run(run_id)
    results = [s for s in run["steps"] if s["kind"] == "tool_result"]
    assert "Tool not permitted: code-review" in results[0]["result"]
    assert runner.ran_with is None  # never actually spawned
    assert db.count_subagent_runs("c1") == 0


def test_skill_tool_executes_after_find_load_discloses_it(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    names = _discovery_names()
    registry = ToolCatalogRegistry()
    registry.register_provider(_NCatalogProvider(names))
    config = AgentConfig(
        model="m",
        system_prompt="s",
        allowed_tools=tuple(names) + (SPAWN_TOOL_NAME,),
        budget=RunBudget(max_steps=10),
    )
    script = [
        {
            "choices": [
                {
                    "message": {
                        "content": _fence(
                            LOAD_TOOLS_NAME, {"ids": ["fake:code-review"]}
                        )
                    }
                }
            ]
        },
        {
            "choices": [
                {"message": {"content": _fence("code-review", {"args": "the diff"})}}
            ]
        },
        {"choices": [{"message": {"content": "child answer"}}]},  # sub-agent turn
        {"choices": [{"message": {"content": "Done reviewing."}}]},  # primary final
    ]
    runner = _NamedSkillRunner("code-review")
    service = AgentService(
        db, registry, chat_call=lambda **k: script.pop(0), skill_runner=runner
    )
    run_id, outcome = service.run_turn(
        conversation_id="c1",
        messages=[{"role": "user", "content": "review"}],
        config=config,
        api_endpoint="llama_cpp",
        first_request_schema_plan=_discovery_plan(),
    )
    assert outcome.status == RUN_DONE
    assert runner.ran_with == "the diff"
    assert db.count_subagent_runs("c1") == 1


# --- Task-12 review Finding 2: max_subagents must bound the COMBINED count
# of native spawn_subagent runs and skill-tool runs, not each independently.
# Order-agnostic: test both orders. ---


def test_combined_budget_native_spawn_then_skill_call(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = _registry_with_code_review_skill()
    # PR2a Task 6.5: the fleet is ON by default, so the native child runs
    # on its own thread -- one ordered queue is no longer deterministic.
    # Addressed per agent instead; the replies themselves are unchanged,
    # and the COMBINED ceiling this test is about is unaffected by where
    # the child runs (the counter is incremented in the one shared `spawn`
    # closure, before either path branches).
    chat = FleetChat(
        [
            {
                "choices": [
                    {
                        "message": {
                            "content": _fence(SPAWN_TOOL_NAME, {"task": "native task"})
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "message": {
                            "content": _fence("code-review", {"args": "the diff"})
                        }
                    }
                ]
            },
            {"choices": [{"message": {"content": "final"}}]},
        ],
        {"native task": [{"choices": [{"message": {"content": "native child answer"}}]}]},
        reply=verbatim,
    )
    runner = _FakeSkillRunner()
    service = AgentService(db, reg, chat_call=chat, skill_runner=runner)
    _r, outcome = service.run_turn(
        conversation_id="c1",
        messages=[{"role": "user", "content": "go"}],
        config=AgentConfig(
            model="m",
            system_prompt="s",
            allowed_tools=("calculator", "code-review", SPAWN_TOOL_NAME),
            budget=RunBudget(max_subagents=1, max_steps=12),
        ),
        api_endpoint="llama_cpp",
    )
    join_fleet_children(service)  # PR3a-1 Task 2: the child outlives the turn
    assert outcome.status == RUN_DONE
    assert db.count_subagent_runs("c1") == 1  # only the native spawn ran
    assert runner.spawned_with is None  # the skill call never actually ran
    run = db.get_run(_r)
    results = [s for s in run["steps"] if s["kind"] == "tool_result"]
    assert any("sub-agent budget exhausted" in r["result"] for r in results)


def test_combined_budget_skill_call_then_native_spawn(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = _registry_with_code_review_skill()
    # PR2a Task 6.5: addressed per agent (see the sibling test above). The
    # SKILL's own spawn is threaded too -- both paths go through the one
    # `spawn` closure -- so the skill child is addressed by the task text
    # `_FakeSkillRunner.run` renders for it.
    chat = FleetChat(
        [
            {
                "choices": [
                    {
                        "message": {
                            "content": _fence("code-review", {"args": "the diff"})
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "message": {
                            "content": _fence(SPAWN_TOOL_NAME, {"task": "native task"})
                        }
                    }
                ]
            },
            {"choices": [{"message": {"content": "final"}}]},
        ],
        {
            "RENDERED[the diff]": [
                {"choices": [{"message": {"content": "skill child answer"}}]}
            ]
        },
        reply=verbatim,
    )
    runner = _FakeSkillRunner()
    service = AgentService(db, reg, chat_call=chat, skill_runner=runner)
    _r, outcome = service.run_turn(
        conversation_id="c1",
        messages=[{"role": "user", "content": "go"}],
        config=AgentConfig(
            model="m",
            system_prompt="s",
            allowed_tools=("calculator", "code-review", SPAWN_TOOL_NAME),
            budget=RunBudget(max_subagents=1, max_steps=12),
        ),
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    assert db.count_subagent_runs("c1") == 1  # only the skill's spawn ran
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
    # PR2a Task 6.5: addressed per agent (see the sibling tests above).
    chat = FleetChat(
        [
            {
                "choices": [
                    {
                        "message": {
                            "content": _fence(SPAWN_TOOL_NAME, {"task": "native task"})
                        }
                    }
                ]
            },
            {"choices": [{"message": {"content": "final"}}]},
        ],
        {
            "native task": [
                # Inside the child: the model attempts the skill tool directly.
                {
                    "choices": [
                        {
                            "message": {
                                "content": _fence("code-review", {"args": "the diff"})
                            }
                        }
                    ]
                },
                {"choices": [{"message": {"content": "child gave up"}}]},
            ]
        },
        reply=verbatim,
    )
    runner = _FakeSkillRunner()
    service = AgentService(db, reg, chat_call=chat, skill_runner=runner)
    _r, outcome = service.run_turn(
        conversation_id="c1",
        messages=[{"role": "user", "content": "go"}],
        config=AgentConfig(
            model="m",
            system_prompt="s",
            allowed_tools=("calculator", "code-review", SPAWN_TOOL_NAME),
            budget=RunBudget(),
        ),
        api_endpoint="llama_cpp",
    )
    join_fleet_children(service)  # PR3a-1 Task 2: the child outlives the turn
    assert outcome.status == RUN_DONE
    assert runner.spawned_with is None  # never actually rendered/run

    runs = db.list_runs("c1")
    child_runs = [r for r in runs if r["agent_kind"] == "subagent"]
    assert len(child_runs) == 1
    tool_results = [
        step for step in child_runs[0]["steps"] if step["kind"] == "tool_result"
    ]
    permission_refusal = next(
        step
        for step in tool_results
        if "Tool not permitted: code-review" in step["result"]
    )
    assert permission_refusal["tool_outcome"] == "blocked"
    assert not any(
        "sub-agent budget exhausted" in step["result"] for step in tool_results
    )


# --- PR2a Task 6.5: a SKILL call keeps its contract under a live fleet ---


def test_skill_call_runs_inline_and_returns_the_output_not_a_handle(
    tmp_path, monkeypatch
):
    """With the fleet ON, a skill call still returns the skill's OUTPUT.

    `spawn_subagent` and a skill tool share one `spawn` closure, so turning
    the fleet on would have made a skill call return `started <id>: ...`
    and require `wait_agents` to collect. That silently breaks the skill
    contract: nothing tells the model a skill is asynchronous, so it
    answers from the literal handle string while `_settle_fleet` discards
    the real work -- a wrong answer, not an error. The service therefore
    hands `skill_runner.run` a spawn pre-bound to the inline path.

    Pinned end-to-end through a REAL fleet (max_live_subagents = 3, no
    injected coordinator), because "the fleet is on" is the precondition
    that makes this regressable at all.
    """
    pin_max_live_subagents(monkeypatch, 3)
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = _registry_with_code_review_skill()
    chat = FleetChat(
        [
            {
                "choices": [
                    {
                        "message": {
                            "content": _fence("code-review", {"args": "the diff"})
                        }
                    }
                ]
            },
            {"choices": [{"message": {"content": "Done reviewing."}}]},
        ],
        {"RENDERED[the diff]": [{"choices": [{"message": {"content": "child answer"}}]}]},
        reply=verbatim,
    )
    runner = _FakeSkillRunner()
    service = AgentService(db, reg, chat_call=chat, skill_runner=runner)
    run_id, outcome = service.run_turn(
        conversation_id="c1",
        messages=[{"role": "user", "content": "review"}],
        config=AgentConfig(
            model="m",
            system_prompt="s",
            allowed_tools=("calculator", "code-review", SPAWN_TOOL_NAME),
            budget=RunBudget(),
        ),
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    # A fleet really was built for this turn ...
    assert service._fleet is not None
    # ... and the skill call still came back with the child's own answer,
    # in the SKILL's own tool_result -- no handle, no wait_agents needed.
    results = [
        s["result"]
        for s in db.get_run(run_id)["steps"]
        if s["kind"] == "tool_result" and s["tool_name"] == "code-review"
    ]
    assert results == ["child answer"]
    assert not any("started " in r for r in results)
    assert db.count_subagent_runs("c1") == 1
