"""run_skill_script: the sixth runtime tool -- name, schema, dispatch, reach.

Model: Tests/Agents/test_skill_file_runtime_tool.py (system-prompt content
is how a prior test proved a runtime schema is actually PINNED, not just
dispatched) and Tests/Agents/test_install_skill_runtime_tool.py (loop-level
LoopDeps tests plus AgentService-level reach tests through a scripted-model
harness). The brief's own draft test called a
``service._runtime_schema_names(agent_kind=...)`` helper that does not
exist anywhere in this codebase; rather than invent one that would have to
duplicate ``AgentService._run_one``'s real (config/budget-dependent, not
just agent_kind-dependent) pinning predicate, this file asserts against the
REAL pinned schemas and REAL dispatch by driving ``AgentService.run_turn``
with the suite's existing fake chat_call/registry scaffolding -- the same
choice ``test_skill_file_runtime_tool.py`` and
``test_install_skill_runtime_tool.py`` already made.

The crux design decision under test: unlike ``install_skill`` (gated
``agent_kind == AGENT_KIND_PRIMARY`` so a spawned sub-agent never receives
it -- see ``test_install_skill_runtime_tool.test_subagent_cannot_call_
install_skill``), ``run_skill_script`` is wired with NO agent_kind gate at
all. ``test_subagent_can_also_dispatch_and_be_pinned_run_skill_script``
below proves BOTH halves of that (the schema pin into the system prompt,
and the LoopDeps dispatch wiring) reach a spawned sub-agent exactly like
they reach the primary agent.
"""

import json

from tldw_chatbook.Agents.agent_models import (
    RUNTIME_TOOL_NAMES,
    RUN_DONE,
    RUN_SKILL_SCRIPT_TOOL_NAME,
    SPAWN_TOOL_NAME,
    AgentConfig,
    ModelTurn,
    RunBudget,
    ToolResult,
    ToolSchema,
)
from tldw_chatbook.Agents.agent_runtime import LoopDeps, run_agent_loop
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.tool_catalog import (
    RUN_SKILL_SCRIPT_TOOL_SCHEMA,
    BuiltinToolProvider,
    ToolCatalogRegistry,
)
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

# --- Step 1 unit tests (name + schema shape) --------------------------------


def test_run_skill_script_name_in_runtime_tool_names():
    assert RUN_SKILL_SCRIPT_TOOL_NAME == "run_skill_script"
    assert RUN_SKILL_SCRIPT_TOOL_NAME in RUNTIME_TOOL_NAMES


def test_run_skill_script_schema_shape():
    schema = RUN_SKILL_SCRIPT_TOOL_SCHEMA
    assert schema.id == "runtime:run_skill_script"
    assert schema.name == RUN_SKILL_SCRIPT_TOOL_NAME
    assert schema.description.strip()
    # The description is what the model reads to decide whether/how to call
    # this -- it must convey the confirm gate and the sandboxing, not just
    # the parameter names.
    assert "confirm" in schema.description.lower()
    props = schema.parameters["properties"]
    assert set(props) == {"skill_name", "script_path", "args"}
    assert props["args"]["type"] == "array"
    assert schema.parameters["required"] == ["skill_name", "script_path"]


def test_shadow_guard_lists_the_new_name():
    from tldw_chatbook.Library.library_skills_state import _SHADOWED_BUILTIN_NAMES

    assert "run_skill_script" in _SHADOWED_BUILTIN_NAMES


# --- Loop-level tests: LoopDeps.run_skill_script + dispatch -----------------

_CALC = ToolSchema(
    id="builtin:calculator",
    name="calculator",
    description="math",
    parameters={"type": "object"},
)


def _fence(name, args):
    return f"```tool_call\n{json.dumps({'name': name, 'arguments': args})}\n```"


def _deps(turns, *, run_skill_script=None):
    script = list(turns)

    def call_model(messages, active_schemas):
        return script.pop(0)

    return LoopDeps(
        call_model=call_model,
        invoke_tool=lambda c: ToolResult(
            ok=False, error=f"Tool not permitted: {c.name}"
        ),
        spawn=lambda task: ToolResult(ok=True, content="sub"),
        find_tools=lambda q: [],
        load_schemas=lambda ids: [],
        should_cancel=lambda: False,
        clock=lambda: 0.0,
        run_skill_script=run_skill_script,
    )


_CFG = AgentConfig(model="m", system_prompt="s", allowed_tools=("calculator",))


def test_dispatch_routes_to_the_wired_callable():
    seen = []

    def fake_run(skill_name, script_path, args):
        seen.append((skill_name, script_path, args))
        return ToolResult(ok=True, content="ran")

    out = run_agent_loop(
        _CFG,
        [{"role": "user", "content": "hi"}],
        [_CALC],
        _deps(
            [
                ModelTurn(
                    text=_fence(
                        "run_skill_script",
                        {
                            "skill_name": "demo",
                            "script_path": "scripts/hello.py",
                            "args": ["x"],
                        },
                    )
                ),
                ModelTurn(text="done"),
            ],
            run_skill_script=fake_run,
        ),
    )
    assert out.status == RUN_DONE
    assert seen == [("demo", "scripts/hello.py", ["x"])]
    assert any(
        s.kind == "tool_result" and "ran" in (s.result or "") for s in out.steps
    )


def test_dispatch_falls_through_when_not_wired():
    out = run_agent_loop(
        _CFG,
        [{"role": "user", "content": "hi"}],
        [_CALC],
        _deps(
            [
                ModelTurn(
                    text=_fence(
                        "run_skill_script",
                        {"skill_name": "demo", "script_path": "scripts/hello.py"},
                    )
                ),
                ModelTurn(text="done"),
            ],
            run_skill_script=None,  # not wired -> generic invoke_tool path
        ),
    )
    assert out.status == RUN_DONE
    results = [s.result for s in out.steps if s.kind == "tool_result"]
    assert any(
        "Tool not permitted: run_skill_script" in (r or "") for r in results
    )


def test_missing_args_defaults_to_empty_list():
    seen = []

    def fake_run(skill_name, script_path, args):
        seen.append(args)
        return ToolResult(ok=True, content="ran")

    out = run_agent_loop(
        _CFG,
        [{"role": "user", "content": "hi"}],
        [_CALC],
        _deps(
            [
                ModelTurn(
                    text=_fence(
                        "run_skill_script",
                        {"skill_name": "demo", "script_path": "scripts/hello.py"},
                    )
                ),
                ModelTurn(text="done"),
            ],
            run_skill_script=fake_run,
        ),
    )
    assert out.status == RUN_DONE
    assert seen == [[]]


def test_scalar_args_is_coerced_to_a_one_item_list():
    seen = []

    def fake_run(skill_name, script_path, args):
        seen.append(args)
        return ToolResult(ok=True, content="ran")

    out = run_agent_loop(
        _CFG,
        [{"role": "user", "content": "hi"}],
        [_CALC],
        _deps(
            [
                ModelTurn(
                    text=_fence(
                        "run_skill_script",
                        {
                            "skill_name": "demo",
                            "script_path": "scripts/hello.py",
                            "args": "solo",
                        },
                    )
                ),
                ModelTurn(text="done"),
            ],
            run_skill_script=fake_run,
        ),
    )
    assert out.status == RUN_DONE
    assert seen == [["solo"]]


# -- AgentService wiring: all-agents scope (NOT gated to the top level) -----


def _registry_with_builtins():
    reg = ToolCatalogRegistry()
    reg.register_provider(BuiltinToolProvider())
    return reg


def _svc_fence(name, args):
    return {"choices": [{"message": {"content": _fence(name, args)}}]}


def test_top_level_agent_dispatches_run_skill_script(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = _registry_with_builtins()
    seen = []

    def runner(skill_name, script_path, args):
        seen.append((skill_name, script_path, args))
        return ToolResult(ok=True, content="stdout: hi")

    script = [
        _svc_fence(
            "run_skill_script",
            {
                "skill_name": "demo",
                "script_path": "scripts/hello.py",
                "args": ["x"],
            },
        ),
        {"choices": [{"message": {"content": "Done."}}]},
    ]
    calls = []

    def chat_call(**kwargs):
        calls.append(kwargs)
        return script.pop(0)

    service = AgentService(db, reg, chat_call=chat_call, run_skill_script_tool=runner)
    _rid, outcome = service.run_turn(
        conversation_id="c1",
        messages=[{"role": "user", "content": "run it"}],
        config=AgentConfig(
            model="m",
            system_prompt="s",
            allowed_tools=("calculator",),
            budget=RunBudget(),
        ),
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    assert seen == [("demo", "scripts/hello.py", ["x"])]

    # Active from the FIRST provider call, never disclosure-gated -- proves
    # the SCHEMA (not just the callable) reached the run, mirroring how
    # test_skill_file_runtime_tool.py proves skill_file's pin.
    first_system_content = calls[0]["messages_payload"][0]["content"]
    assert RUN_SKILL_SCRIPT_TOOL_NAME in first_system_content


def test_run_skill_script_absent_and_falls_through_when_not_wired(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = _registry_with_builtins()

    script = [
        _svc_fence(
            "run_skill_script",
            {"skill_name": "demo", "script_path": "scripts/hello.py"},
        ),
        {"choices": [{"message": {"content": "Done."}}]},
    ]
    calls = []

    def chat_call(**kwargs):
        calls.append(kwargs)
        return script.pop(0)

    # run_skill_script_tool not passed at all -- this service was never
    # wired for it.
    service = AgentService(db, reg, chat_call=chat_call)
    rid, outcome = service.run_turn(
        conversation_id="c1",
        messages=[{"role": "user", "content": "run it"}],
        config=AgentConfig(
            model="m",
            system_prompt="s",
            allowed_tools=("calculator",),
            budget=RunBudget(),
        ),
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE

    first_system_content = calls[0]["messages_payload"][0]["content"]
    assert RUN_SKILL_SCRIPT_TOOL_NAME not in first_system_content

    run = db.get_run(rid)
    results = [s["result"] for s in run["steps"] if s["kind"] == "tool_result"]
    assert any("Tool not permitted: run_skill_script" in r for r in results)


def test_subagent_can_also_dispatch_and_be_pinned_run_skill_script(tmp_path):
    """All-agents caller scope (the crux decision): a spawned sub-agent
    reaches run_skill_script exactly like the primary agent does -- both
    the schema pin (system-prompt disclosure) and the dispatch wiring
    (LoopDeps.run_skill_script). Contrast with
    test_install_skill_runtime_tool.test_subagent_cannot_call_install_skill,
    where the equivalent child call is refused."""
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = _registry_with_builtins()
    seen = []

    def runner(skill_name, script_path, args):
        seen.append((skill_name, script_path, args))
        return ToolResult(ok=True, content="stdout: hi")

    script = [
        _svc_fence(SPAWN_TOOL_NAME, {"task": "child task"}),  # parent spawns
        _svc_fence(
            "run_skill_script",
            {
                "skill_name": "demo",
                "script_path": "scripts/hello.py",
                "args": ["x"],
            },
        ),  # child dispatches
        {"choices": [{"message": {"content": "child done"}}]},
        {"choices": [{"message": {"content": "parent done"}}]},
    ]
    calls = []

    def chat_call(**kwargs):
        calls.append(kwargs)
        return script.pop(0)

    service = AgentService(db, reg, chat_call=chat_call, run_skill_script_tool=runner)
    _rid, outcome = service.run_turn(
        conversation_id="c1",
        messages=[{"role": "user", "content": "go"}],
        config=AgentConfig(
            model="m",
            system_prompt="s",
            allowed_tools=("calculator", SPAWN_TOOL_NAME),
            budget=RunBudget(),
        ),
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE

    # Dispatch reached the CHILD.
    assert seen == [("demo", "scripts/hello.py", ["x"])]

    # The schema was pinned into the CHILD's own first turn too. calls[0] is
    # the parent's first call (before it spawns); calls[1] is the child's
    # first call.
    child_first_system_content = calls[1]["messages_payload"][0]["content"]
    assert RUN_SKILL_SCRIPT_TOOL_NAME in child_first_system_content

    child_runs = [r for r in db.list_runs("c1") if r["agent_kind"] == "subagent"]
    assert len(child_runs) == 1
    tool_results = [
        s["result"] for s in child_runs[0]["steps"] if s["kind"] == "tool_result"
    ]
    assert any("stdout: hi" in r for r in tool_results)
