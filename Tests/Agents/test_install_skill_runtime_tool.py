"""install_skill: the fifth runtime tool — name, schema, dispatch, gating.

Model: Tests/Agents/test_skill_file_runtime_tool.py (the 4th runtime tool).
install_skill is NOT a ToolProvider — its schema is pinned into
runtime_schemas (never disclosure-gated) only for the top-level agent
(agent_kind == primary), and its closure lives on LoopDeps.install_skill.
"""

import json

from tldw_chatbook.Agents.agent_models import (
    CHECK_AGENTS_TOOL_NAME,
    INSTALL_SKILL_TOOL_NAME,
    RUNTIME_TOOL_NAMES,
    WAIT_AGENTS_TOOL_NAME,
    RUN_LOG_SLICE_TOOL_NAME,
    RUN_LOG_STATS_TOOL_NAME,
    RUN_SKILL_SCRIPT_TOOL_NAME,
    SEARCH_RUN_LOG_TOOL_NAME,
    SEND_TO_AGENT_TOOL_NAME,
    SPAWN_TOOL_NAME,
    FIND_TOOLS_NAME,
    LOAD_TOOLS_NAME,
    SKILL_FILE_TOOL_NAME,
    FORK_CHAT_TOOL_NAME,
    NEW_CHAT_TOOL_NAME,
    AgentConfig,
    ModelTurn,
    RUN_DONE,
    RunBudget,
    ToolCatalogEntry,
    ToolLoadSelection,
    ToolResult,
    ToolSchema,
    READ_AGENT_MESSAGES_TOOL_NAME,
    LIST_PEER_AGENTS_TOOL_NAME,
    SEND_TO_PEER_TOOL_NAME,
    REPORT_TO_SUPERVISOR_TOOL_NAME,
    PREPARE_MANAGED_SKILL_PROMOTION_TOOL_NAME,
    DISCARD_AGENT_WORKTREE_TOOL_NAME,
    MERGE_AGENT_WORKTREE_TOOL_NAME,
)
from tldw_chatbook.Agents.tool_catalog import INSTALL_SKILL_TOOL_SCHEMA
from tldw_chatbook.Agents.agent_runtime import LoopDeps, run_agent_loop

from Tests.Agents.test_agent_service import FleetChat, verbatim


def test_install_skill_name_in_runtime_tool_names():
    assert INSTALL_SKILL_TOOL_NAME == "install_skill"
    assert (
        RUNTIME_TOOL_NAMES
        == {  # ADR-348 chat fork/spawn + dev-baseline completion (fleet message/
            # worktree constants were in RUNTIME_TOOL_NAMES but unlisted here).
            MERGE_AGENT_WORKTREE_TOOL_NAME,
            DISCARD_AGENT_WORKTREE_TOOL_NAME,
            PREPARE_MANAGED_SKILL_PROMOTION_TOOL_NAME,
            REPORT_TO_SUPERVISOR_TOOL_NAME,
            READ_AGENT_MESSAGES_TOOL_NAME,
            LIST_PEER_AGENTS_TOOL_NAME,
            SEND_TO_PEER_TOOL_NAME,
            SPAWN_TOOL_NAME,
            FIND_TOOLS_NAME,
            LOAD_TOOLS_NAME,
            SKILL_FILE_TOOL_NAME,
            INSTALL_SKILL_TOOL_NAME,
            RUN_SKILL_SCRIPT_TOOL_NAME,
            SEARCH_RUN_LOG_TOOL_NAME,
            RUN_LOG_STATS_TOOL_NAME,
            RUN_LOG_SLICE_TOOL_NAME,
            WAIT_AGENTS_TOOL_NAME,
            CHECK_AGENTS_TOOL_NAME,
            SEND_TO_AGENT_TOOL_NAME,
            FORK_CHAT_TOOL_NAME,
            NEW_CHAT_TOOL_NAME,
        }
    )


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
    id="builtin:calculator",
    name="calculator",
    description="math",
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
        invoke_tool=invoke
        or (lambda c: ToolResult(ok=False, error=f"Tool not permitted: {c.name}")),
        spawn=lambda task: ToolResult(ok=True, content="sub"),
        find_tools=lambda q: [],
        load_schemas=lambda _ids, _messages, _call: ToolLoadSelection(),
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
                ModelTurn(
                    text=_fence("install_skill", {"url": "https://github.com/o/r"})
                ),
                ModelTurn(text="done"),
            ],
            install_skill=installer,
        ),
    )
    assert out.status == RUN_DONE
    assert seen == ["https://github.com/o/r"]
    assert any(
        s.kind == "tool_result" and "installed" in (s.result or "") for s in out.steps
    )


def test_install_skill_falls_through_when_not_wired():
    out = run_agent_loop(
        _CFG,
        [{"role": "user", "content": "hi"}],
        [_CALC],
        _deps(
            [
                ModelTurn(
                    text=_fence("install_skill", {"url": "https://github.com/o/r"})
                ),
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

from Tests.Agents.conftest import join_fleet_children


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
            model="m",
            system_prompt="s",
            allowed_tools=("calculator",),
            budget=RunBudget(),
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

    # PR2a Task 6.5: the fleet is ON by default, so the child runs on its
    # own thread -- one ordered queue is no longer deterministic. Addressed
    # per agent instead; the replies themselves are unchanged.
    chat = FleetChat(
        [
            _svc_fence(SPAWN_TOOL_NAME, {"task": "native task"}),  # parent spawns
            {"choices": [{"message": {"content": "final"}}]},
        ],
        {
            "native task": [
                _svc_fence(
                    "install_skill", {"url": "https://github.com/o/r"}
                ),  # child tries
                {"choices": [{"message": {"content": "child gave up"}}]},
            ]
        },
        reply=verbatim,
    )
    service = AgentService(db, reg, chat_call=chat, install_skill_tool=installer)
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
    join_fleet_children(service)  # PR3a-1 Task 2: the child outlives the turn
    assert outcome.status == RUN_DONE
    child_runs = [r for r in db.list_runs("c1") if r["agent_kind"] == "subagent"]
    assert len(child_runs) == 1
    tool_results = [
        s["result"] for s in child_runs[0]["steps"] if s["kind"] == "tool_result"
    ]
    assert any("Tool not permitted: install_skill" in r for r in tool_results)


# -- TASK-36950.1: each special-tool handler runs without the loop -----------

from dataclasses import replace as _replace
from types import SimpleNamespace

import pytest

from tldw_chatbook.Agents.agent_models import STEP_SPAWN, STEP_TOOL_CALL, ToolCall
from tldw_chatbook.Agents.agent_runtime import (
    ORDINARY_TOOL,
    SPECIAL_TOOL_HANDLERS,
    LoopState,
)


def _state(config=_CFG, **deps):
    """A LoopState whose `add` records (kind, kwargs) instead of persisting."""
    steps = []

    def add(kind, **kw):
        steps.append((kind, kw))
        return SimpleNamespace(index=len(steps) + 40)

    state = LoopState(
        config=config,
        deps=_replace(_deps([]), **deps),
        add=add,
        messages=[{"role": "user", "content": "hi"}],
        active=[_CALC],
        current_call_correlation="corr-1",
        display_call_arguments={"shown": "display"},
    )
    return state, steps


def _dispatch(name, args, state):
    return SPECIAL_TOOL_HANDLERS[name](ToolCall(name=name, args=args), state)


def test_every_runtime_tool_is_dispatched_through_the_table():
    assert set(SPECIAL_TOOL_HANDLERS) == RUNTIME_TOOL_NAMES | {
        "search_archived_conversations",
        "request_conversation_restore",
    }


_GATED = {
    WAIT_AGENTS_TOOL_NAME: "wait_agents",
    CHECK_AGENTS_TOOL_NAME: "check_agents",
    MERGE_AGENT_WORKTREE_TOOL_NAME: "merge_agent_worktree",
    DISCARD_AGENT_WORKTREE_TOOL_NAME: "discard_agent_worktree",
    SKILL_FILE_TOOL_NAME: "read_skill_file",
    INSTALL_SKILL_TOOL_NAME: "install_skill",
    RUN_SKILL_SCRIPT_TOOL_NAME: "run_skill_script",
    "search_archived_conversations": "search_archived_conversations",
    "request_conversation_restore": "request_conversation_restore",
    FORK_CHAT_TOOL_NAME: "fork_chat",
    NEW_CHAT_TOOL_NAME: "new_chat",
    SEARCH_RUN_LOG_TOOL_NAME: "search_run_log",
    RUN_LOG_STATS_TOOL_NAME: "run_log_stats",
    RUN_LOG_SLICE_TOOL_NAME: "run_log_slice",
}


@pytest.mark.parametrize("name", sorted(_GATED))
def test_unwired_gated_handler_defers_to_the_ordinary_path(name):
    state, steps = _state(**{_GATED[name]: None})
    assert _dispatch(name, {}, state) is ORDINARY_TOOL
    assert steps == []


def test_display_args_handler_records_display_args_and_passes_raw_args():
    seen = []
    state, steps = _state(
        run_log_stats=lambda args: seen.append(args) or ToolResult(ok=True, content="s")
    )
    result = _dispatch(RUN_LOG_STATS_TOOL_NAME, {"q": 1}, state)
    assert result.content == "s" and seen == [{"q": 1}]
    assert steps == [
        (STEP_TOOL_CALL, {"tool_name": RUN_LOG_STATS_TOOL_NAME, "args": {"shown": "display"}})
    ]


def test_raw_args_handler_records_and_passes_the_call_args():
    state, steps = _state(fork_chat=lambda args: ToolResult(ok=True, content=str(args)))
    assert _dispatch(FORK_CHAT_TOOL_NAME, {"t": "x"}, state).content == "{'t': 'x'}"
    assert steps == [(STEP_TOOL_CALL, {"tool_name": FORK_CHAT_TOOL_NAME, "args": {"t": "x"}})]


def test_correlation_handlers_bind_the_current_call_id():
    from tldw_chatbook.Agents.run_context import current_tool_call_id

    state, _ = _state(
        request_conversation_restore=lambda args, call_id: ToolResult(
            ok=True, content=call_id
        ),
        prepare_managed_skill_promotion=lambda args: ToolResult(
            ok=True, content=current_tool_call_id()
        ),
    )
    assert _dispatch("request_conversation_restore", {}, state).content == "corr-1"
    assert (
        _dispatch(PREPARE_MANAGED_SKILL_PROMOTION_TOOL_NAME, {}, state).content
        == "corr-1"
    )


def test_unwired_promotion_is_blocked_not_deferred():
    state, steps = _state(prepare_managed_skill_promotion=None)
    result = _dispatch(PREPARE_MANAGED_SKILL_PROMOTION_TOOL_NAME, {}, state)
    assert result is not ORDINARY_TOOL and not result.ok
    assert len(steps) == 1


def test_coercing_handlers_normalize_model_arguments():
    waited, merged = [], []
    state, _ = _state(
        wait_agents=lambda ids: waited.append(ids) or ToolResult(ok=True),
        merge_agent_worktree=lambda handle, mode: merged.append((handle, mode))
        or ToolResult(ok=True),
    )
    _dispatch(WAIT_AGENTS_TOOL_NAME, {"ids": "a1"}, state)
    _dispatch(WAIT_AGENTS_TOOL_NAME, {"ids": []}, state)
    _dispatch(MERGE_AGENT_WORKTREE_TOOL_NAME, {"handle_id": 7}, state)
    assert waited == [["a1"], None]
    assert merged == [("7", "apply")]


def test_send_to_agent_steers_at_its_step_or_refuses_when_unwired():
    state, _ = _state(
        send_to_agent=lambda target, message: ToolResult(ok=True),
        send_to_agent_at_step=lambda target, message, index: ToolResult(
            ok=True, content=f"{target}|{message}|{index}"
        ),
    )
    result = _dispatch(SEND_TO_AGENT_TOOL_NAME, {"id": 3, "message": None}, state)
    assert result.content == "3||41"
    state, steps = _state(send_to_agent=None)
    result = _dispatch(SEND_TO_AGENT_TOOL_NAME, {"id": "a"}, state)
    assert result.error == "Tool not permitted: send_to_agent"
    assert result.dispatch_state == "not_started"
    assert steps == [(STEP_TOOL_CALL, {"tool_name": SEND_TO_AGENT_TOOL_NAME, "args": {}})]


def test_message_tool_handler_picks_its_callback_or_refuses():
    state, _ = _state(
        read_agent_messages=lambda args: ToolResult(ok=True, content="inbox")
    )
    assert _dispatch(READ_AGENT_MESSAGES_TOOL_NAME, {}, state).content == "inbox"
    result = _dispatch(SEND_TO_PEER_TOOL_NAME, {}, state)
    assert not result.ok and result.dispatch_state == "not_started"


def test_find_tools_formats_the_catalog():
    entry = ToolCatalogEntry(
        id="builtin:calculator", name="calculator", one_line_description="math", source="b"
    )
    state, _ = _state(find_tools=lambda query: [entry] if query == "calc" else [])
    result = _dispatch(FIND_TOOLS_NAME, {"query": "calc"}, state)
    assert result.content == "builtin:calculator — calculator: math"


def test_load_tools_replaces_the_working_set_only_when_alone_in_its_batch():
    new = ToolSchema(id="builtin:clock", name="clock", description="t", parameters={})
    committed = []
    state, _ = _state(
        load_schemas=lambda ids, messages, call: ToolLoadSelection(accepted=(new,)),
        replace_disclosed_names=committed.append,
    )
    refused = _dispatch(LOAD_TOOLS_NAME, {"ids": "builtin:clock"}, state)
    assert refused.error == "call load_tools alone in its own tool batch"
    assert state.active == [_CALC]
    state.load_batch_exclusive = True
    assert _dispatch(LOAD_TOOLS_NAME, {"ids": "builtin:clock"}, state).ok
    assert state.active == [new] and committed == [frozenset({"clock"})]


def test_run_skill_script_binds_the_output_observer():
    from tldw_chatbook.Agents.tool_output import current_tool_output_sink

    state, _ = _state(
        run_skill_script=lambda skill, path, args: ToolResult(
            ok=True, content=f"{current_tool_output_sink() is not None} {args}"
        )
    )
    state.output_observer = lambda text: None
    result = _dispatch(RUN_SKILL_SCRIPT_TOOL_NAME, {"args": 5}, state)
    assert result.content == "True ['5']"


def test_spawn_counts_admitted_children_on_the_state():
    config = AgentConfig(
        model="m",
        system_prompt="s",
        allowed_tools=(SPAWN_TOOL_NAME,),
        budget=RunBudget(max_subagents=1),
    )
    state, steps = _state(config)
    assert _dispatch(SPAWN_TOOL_NAME, {"task": "t"}, state).ok
    assert state.spawned == 1
    assert [kind for kind, _ in steps] == [STEP_SPAWN]
    exhausted = _dispatch(SPAWN_TOOL_NAME, {"task": "t2"}, state)
    assert exhausted.error == "sub-agent budget exhausted"
    assert state.spawned == 1 and len(steps) == 1
