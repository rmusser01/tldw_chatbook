# Tests/Agents/test_run_log_stats_slice_runtime_tools.py
"""run_log_stats/run_log_slice: Phase 2's aggregation/slicing runtime tools.

Mirrors Tests/Agents/test_search_run_log_runtime_tool.py's structure -- both
new tools are registered exactly like search_run_log (task-1271's own
mandate): name constant + RUNTIME_TOOL_NAMES membership + tool_catalog
schema + LoopDeps field + dispatch branch + the SAME primary-agent-only
service gate. See agent_service.py's `log_active` block for that gate.
"""

import json
import re

import pytest

from tldw_chatbook.Agents.agent_models import (
    AGENT_KIND_PRIMARY,
    AGENT_KIND_SUBAGENT,
    RUN_DONE,
    RUN_LOG_SLICE_TOOL_NAME,
    RUN_LOG_STATS_TOOL_NAME,
    RUNTIME_TOOL_NAMES,
    SPAWN_TOOL_NAME,
    AgentConfig,
    ModelTurn,
    RunBudget,
    ToolCall,
    ToolResult,
)
from tldw_chatbook.Agents.agent_runtime import run_agent_loop
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.run_log_search import MAX_SLICE_RECORDS
from tldw_chatbook.Agents.tool_catalog import (
    RUN_LOG_SLICE_TOOL_SCHEMA,
    RUN_LOG_STATS_TOOL_SCHEMA,
    BuiltinToolProvider,
    ToolCatalogRegistry,
)
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from Tests.Agents.test_agent_runtime import make_deps


def _fence(name, args):
    return f"```tool_call\n{json.dumps({'name': name, 'arguments': args})}\n```"


def _svc_fence(name, args):
    return {"choices": [{"message": {"content": _fence(name, args)}}]}


# -- Registration --------------------------------------------------------


def test_names_are_registered_as_runtime_tools():
    assert RUN_LOG_STATS_TOOL_NAME in RUNTIME_TOOL_NAMES
    assert RUN_LOG_SLICE_TOOL_NAME in RUNTIME_TOOL_NAMES
    assert RUN_LOG_STATS_TOOL_SCHEMA.name == RUN_LOG_STATS_TOOL_NAME
    assert RUN_LOG_SLICE_TOOL_SCHEMA.name == RUN_LOG_SLICE_TOOL_NAME
    stats_props = RUN_LOG_STATS_TOOL_SCHEMA.parameters["properties"]
    assert "group_by" in stats_props
    assert "tool" in stats_props and "type" in stats_props and "status" in stats_props
    slice_props = RUN_LOG_SLICE_TOOL_SCHEMA.parameters["properties"]
    assert "from_record" in slice_props and "to_record" in slice_props
    assert RUN_LOG_SLICE_TOOL_SCHEMA.parameters["required"] == ["from_record"]


# -- Pure loop dispatch (fake deps) --------------------------------------


def test_loop_dispatches_run_log_stats_to_the_injected_callable():
    seen = {}

    def handler(args):
        seen.update(args)
        return ToolResult(ok=True, content="1 record(s), grouped by tool:\n  calc: count=1")

    turns = [
        ModelTurn(
            text="",
            tool_calls=(
                ToolCall(name=RUN_LOG_STATS_TOOL_NAME, args={"group_by": "tool"}, call_id="c1"),
            ),
            assistant_message={"role": "assistant", "content": ""},
        ),
        ModelTurn(text="answered"),
    ]
    deps = make_deps(turns)
    deps.run_log_stats = handler
    config = AgentConfig(
        model="m", system_prompt="s", budget=RunBudget(max_steps=8, max_model_turns=8)
    )
    outcome = run_agent_loop(config, [{"role": "user", "content": "go"}], [], deps)
    assert seen == {"group_by": "tool"}
    assert outcome.final_text == "answered"


def test_loop_dispatches_run_log_slice_to_the_injected_callable():
    seen = {}

    def handler(args):
        seen.update(args)
        return ToolResult(ok=True, content="records 000001-000003 of this run's log:\n\n...")

    turns = [
        ModelTurn(
            text="",
            tool_calls=(
                ToolCall(
                    name=RUN_LOG_SLICE_TOOL_NAME,
                    args={"from_record": 1, "to_record": 3},
                    call_id="c1",
                ),
            ),
            assistant_message={"role": "assistant", "content": ""},
        ),
        ModelTurn(text="answered"),
    ]
    deps = make_deps(turns)
    deps.run_log_slice = handler
    config = AgentConfig(
        model="m", system_prompt="s", budget=RunBudget(max_steps=8, max_model_turns=8)
    )
    outcome = run_agent_loop(config, [{"role": "user", "content": "go"}], [], deps)
    assert seen == {"from_record": 1, "to_record": 3}
    assert outcome.final_text == "answered"


@pytest.mark.parametrize("name", [RUN_LOG_STATS_TOOL_NAME, RUN_LOG_SLICE_TOOL_NAME])
def test_unwired_name_falls_through_to_the_permission_gate(name):
    # deps.run_log_stats/run_log_slice is None -> the else branch ->
    # deps.invoke_tool, exactly like an unwired search_run_log.
    invoked = []

    def invoke(call):
        invoked.append(call.name)
        return ToolResult(ok=False, error=f"Tool not permitted: {call.name}")

    turns = [
        ModelTurn(
            text="",
            tool_calls=(ToolCall(name=name, args={}, call_id="c1"),),
            assistant_message={"role": "assistant", "content": ""},
        ),
        ModelTurn(text="done"),
    ]
    deps = make_deps(turns, invoke=invoke)
    config = AgentConfig(
        model="m", system_prompt="s", budget=RunBudget(max_steps=8, max_model_turns=8)
    )
    run_agent_loop(config, [{"role": "user", "content": "go"}], [], deps)
    assert invoked == [name]


# -- Sub-agent isolation, gated to the top-level agent -----------------------
#
# Mirrors test_search_run_log_runtime_tool.py::test_subagent_cannot_call_
# search_run_log: the AGENT_KIND_PRIMARY gate exists in TWO places in
# agent_service.py's _run_one (the schema pin under `log_active`, and the
# LoopDeps wiring `run_log_stats=(... if agent_kind == AGENT_KIND_PRIMARY
# else None)`), so this pins BOTH halves rather than just the outcome:
#   (a) neither schema is ever disclosed to a child;
#   (b) a child that calls either name anyway is refused through the
#       ordinary permission path, never executed.
#
# A dedicated mutation check (removing the LoopDeps-wiring gate, confirming
# THIS test fails, then restoring it) was run manually during development
# per task-1271's own instructions; not committed as a permanent test since
# it would require monkeypatching agent_service internals in a way that
# duplicates rather than strengthens this end-to-end assertion.


def test_subagent_cannot_call_run_log_stats_or_run_log_slice(tmp_path, monkeypatch):
    from tldw_chatbook.Agents import run_log as run_log_module

    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: tmp_path)

    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = ToolCatalogRegistry()
    reg.register_provider(BuiltinToolProvider())

    script = [
        _svc_fence(SPAWN_TOOL_NAME, {"task": "native task"}),  # parent spawns
        _svc_fence(RUN_LOG_STATS_TOOL_NAME, {"group_by": "tool"}),  # child tries stats
        _svc_fence(RUN_LOG_SLICE_TOOL_NAME, {"from_record": 1}),  # child tries slice
        {"choices": [{"message": {"content": "child gave up"}}]},
        {"choices": [{"message": {"content": "final"}}]},
    ]
    calls = []

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
        api_endpoint="llama_cpp",  # fence protocol: schemas render into the system prompt
    )
    assert outcome.status == RUN_DONE

    # (a) schema gate: the child's own system prompt (calls[1] -- the
    # parent's spawn dispatch runs the child's whole loop inline before
    # dispatch returns, so this is the second chat_call invocation overall)
    # must never mention either tool.
    child_system_prompt = calls[1]["messages_payload"][0]["content"]
    assert RUN_LOG_STATS_TOOL_NAME not in child_system_prompt
    assert RUN_LOG_SLICE_TOOL_NAME not in child_system_prompt

    # (b) dispatch gate: both of the child's calls, made regardless of (a),
    # must be refused through the ordinary permission path.
    child_runs = [r for r in db.list_runs("c1") if r["agent_kind"] == "subagent"]
    assert len(child_runs) == 1
    tool_results = [
        s["result"] for s in child_runs[0]["steps"] if s["kind"] == "tool_result"
    ]
    assert any(f"Tool not permitted: {RUN_LOG_STATS_TOOL_NAME}" in r for r in tool_results)
    assert any(f"Tool not permitted: {RUN_LOG_SLICE_TOOL_NAME}" in r for r in tool_results)


# -- Real closures, empty / single-segment / multi-segment logs --------------


@pytest.fixture
def wired(tmp_path, monkeypatch):
    from tldw_chatbook.Agents import run_log as run_log_module

    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: tmp_path)
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = ToolCatalogRegistry()
    reg.register_provider(BuiltinToolProvider())
    return db, reg, tmp_path


@pytest.fixture
def real_closures(wired, monkeypatch):
    """The ACTUAL run_log_stats/run_log_slice closures a real AgentService
    wires into LoopDeps, captured via a run_agent_loop spy -- mirrors
    test_search_run_log_runtime_tool.py::real_search_run_log.
    """
    from tldw_chatbook.Agents import agent_service as agent_service_module

    db, registry, _root = wired
    captured: dict = {}
    real_run_agent_loop = agent_service_module.run_agent_loop

    def spy_run_agent_loop(config, messages, active, deps):
        captured["deps"] = deps
        return real_run_agent_loop(config, messages, active, deps)

    monkeypatch.setattr(agent_service_module, "run_agent_loop", spy_run_agent_loop)

    service = AgentService(
        db, registry, chat_call=lambda **kw: {"choices": [{"message": {"content": "ok"}}]}
    )
    service.run_turn(
        conversation_id="c1",
        messages=[{"role": "user", "content": "hi"}],
        config=AgentConfig(model="m", system_prompt="s", budget=RunBudget()),
        api_endpoint="openai",
    )
    deps = captured["deps"]
    assert deps.run_log_stats is not None
    assert deps.run_log_slice is not None
    return deps.run_log_stats, deps.run_log_slice


def test_real_run_log_stats_sees_the_single_model_turn_already_logged(real_closures):
    # "Single-segment" case: by the time the closure is reachable at all, at
    # least one model record has already been captured (the turn that
    # decided to call this tool) -- so the log is never literally empty in
    # this closure's own lifetime, but it IS small/single-segment. Confirm
    # the real end-to-end path (writer -> disk -> load_records ->
    # compute_stats -> format_stats) produces a sane, bounded result.
    run_log_stats, _run_log_slice = real_closures
    result = run_log_stats({"group_by": "type"})
    assert result.ok is True
    assert "grouped by type" in result.content
    assert "model" in result.content  # at least the model turn itself


def test_real_run_log_slice_renders_the_first_record(real_closures):
    _run_log_stats, run_log_slice = real_closures
    result = run_log_slice({"from_record": 1, "to_record": 1})
    assert result.ok is True
    assert "record 000001" in result.content


def test_real_closures_on_a_missing_log_return_a_clean_error(monkeypatch, tmp_path):
    """When the writer never activates (e.g. run_log_enabled=False), the
    SCHEMA is never disclosed and the system-prompt section is suppressed
    (log_active is False) -- but the LoopDeps WIRING for a primary agent is
    gated only on `agent_kind == AGENT_KIND_PRIMARY`, mirroring
    search_run_log exactly (agent_service.py: `search_run_log=(... if
    agent_kind == AGENT_KIND_PRIMARY else None)`, no `log_active` check).
    So a primary run's `deps.run_log_stats`/`deps.run_log_slice` are real,
    callable closures even when the log itself never activated; each must
    degrade to a normal error ToolResult when called, mirroring
    search_run_log's own "No run log is available." path, never raise.
    """
    from tldw_chatbook.Agents.run_log import RunLogWriter

    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = ToolCatalogRegistry()
    reg.register_provider(BuiltinToolProvider())
    inactive_writer = RunLogWriter(dir_name="agent-runs")
    monkeypatch.setattr(
        "tldw_chatbook.Agents.run_log._setting",
        lambda key, default: False if key == "run_log_enabled" else default,
    )
    service = AgentService(
        db,
        reg,
        chat_call=lambda **kw: {"choices": [{"message": {"content": "done"}}]},
        run_log_writer=inactive_writer,
    )
    from tldw_chatbook.Agents import agent_service as agent_service_module

    captured: dict = {}
    real_run_agent_loop = agent_service_module.run_agent_loop

    def spy(config, messages, active, deps):
        captured["deps"] = deps
        return real_run_agent_loop(config, messages, active, deps)

    monkeypatch.setattr(agent_service_module, "run_agent_loop", spy)
    service.run_turn(
        conversation_id="c1",
        messages=[{"role": "user", "content": "hi"}],
        config=AgentConfig(model="m", system_prompt="s", budget=RunBudget()),
        api_endpoint="openai",
    )
    assert inactive_writer.is_active is False
    deps = captured["deps"]
    # Wired (not None): the agent_kind gate alone wires it, matching
    # search_run_log's own established behaviour.
    assert deps.run_log_stats is not None
    assert deps.run_log_slice is not None
    stats_result = deps.run_log_stats({"group_by": "tool"})
    assert stats_result.ok is False
    assert "No run log is available" in stats_result.error
    slice_result = deps.run_log_slice({"from_record": 1})
    assert slice_result.ok is False
    assert "No run log is available" in slice_result.error


def test_multi_segment_log_stats_and_slice_see_every_record(tmp_path, monkeypatch):
    """Multi-segment case: force the writer to roll into several
    logs.NNNN.txt files, then confirm both `compute_stats` and
    `slice_records` -- via `load_records`, the same loader the real
    closures use -- see every record across every segment, not just the
    last (or first) one.
    """
    from tldw_chatbook.Agents import run_log as run_log_module
    from tldw_chatbook.Agents.run_log_search import compute_stats, load_records, slice_records

    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: tmp_path)
    writer = run_log_module.RunLogWriter(segment_bytes=400)
    writer.bind("run-multiseg")
    assert writer.is_active

    total = 30
    for i in range(total):
        writer.append(
            run_id="run-multiseg",
            kind="primary",
            type="tool_result",
            content=f"record body number {i} " + ("x" * 20),
            tool="alpha" if i % 2 == 0 else "beta",
            status="error" if i % 5 == 0 else "ok",
        )
    writer.close()

    segments = sorted(writer.log_dir.glob("logs.*.txt"))
    assert len(segments) > 1, "test setup did not actually force a segment roll"

    records = load_records(writer.log_dir)
    assert len(records) == total
    assert [r.number for r in records] == list(range(1, total + 1))

    # compute_stats sees records from every segment, not just one.
    groups, total_matched, omitted = compute_stats(records, group_by="tool")
    assert omitted == 0  # only 2 distinct tool names, nowhere near MAX_STATS_GROUPS
    assert total_matched == total
    by_key = {g.key: g for g in groups}
    assert by_key["alpha"].count + by_key["beta"].count == total
    assert sum(g.error_count for g in groups) == len([i for i in range(total) if i % 5 == 0])

    # slice_records spans a segment boundary correctly (segment_bytes=400
    # rolls well before 30 records of ~45 bytes each are written).
    selected, matched, lo, hi = slice_records(records, from_record=1, to_record=total)
    assert matched == total
    assert [r.number for r in selected] == list(range(1, min(total, MAX_SLICE_RECORDS) + 1))


# -- Junk arguments through the REAL service closures ------------------------
#
# Mirrors test_search_run_log_runtime_tool.py's F2 section: every argument
# is coerced defensively (str(...) for string filters; int(... or 0) inside
# try/except for numeric ones), so a bad value must return a clean
# ToolResult error, never raise into the run. Extends that file's coverage
# with a HUGE value and a NEGATIVE value (task-1271's own ask) -- both are
# valid ints that clamp/default safely rather than erroring, distinct from
# the type-mismatched cases below.


# C (Qodo review, PR #1078): `int(...)` on from_record/to_record is wrapped
# in `except (TypeError, ValueError, OverflowError)`. `float('inf')`/
# `float('-inf')` raise OverflowError specifically (NOT TypeError/
# ValueError) -- `float('nan')` above already raises ValueError, already
# covered pre-fix; inf/-inf were the actual gap. Both directions included
# since nothing about the coercion path treats them differently.


@pytest.mark.parametrize(
    "bad_args",
    [
        {"from_record": "not-a-number"},
        {"to_record": "not-a-number"},
        {"from_record": {"nested": "object"}},
        {"to_record": {"nested": "object"}},
        {"from_record": [1, 2, 3]},
        {"to_record": [1, 2, 3]},
        {"from_record": float("nan")},
        {"from_record": float("inf")},
        {"to_record": float("inf")},
        {"from_record": float("-inf")},
        {"to_record": float("-inf")},
    ],
)
def test_run_log_stats_unparseable_numeric_args_return_a_clean_error(real_closures, bad_args):
    run_log_stats, _slice = real_closures
    result = run_log_stats(bad_args)
    assert result.ok is False
    assert "Invalid stats arguments" in result.error


@pytest.mark.parametrize(
    "bad_args",
    [
        {"from_record": "not-a-number"},
        {"to_record": "not-a-number"},
        {"from_record": {"nested": "object"}},
        {"to_record": {"nested": "object"}},
        {"from_record": [1, 2, 3]},
        {"to_record": [1, 2, 3]},
        {"from_record": float("nan")},
        {"from_record": float("inf")},
        {"to_record": float("inf")},
        {"from_record": float("-inf")},
        {"to_record": float("-inf")},
    ],
)
def test_run_log_slice_unparseable_numeric_args_return_a_clean_error(real_closures, bad_args):
    _stats, run_log_slice = real_closures
    result = run_log_slice(bad_args)
    assert result.ok is False
    assert "Invalid slice arguments" in result.error


@pytest.mark.parametrize(
    "ok_args",
    [
        {"from_record": None},
        {"to_record": None},
        {"from_record": 10 ** 18},  # huge int
        {"to_record": 10 ** 18},  # huge int
        {"from_record": 10 ** 30},  # enormous int, well past int64 range
        {"to_record": 10 ** 30},
        {"from_record": 1e300},  # enormous float, still finite -- int() succeeds
        {"to_record": 1e300},
        {"from_record": -5},  # negative
        {"to_record": -5},  # negative
        {},
        {"unexpected_key": "whatever"},
    ],
)
def test_run_log_stats_null_huge_and_negative_args_never_raise(real_closures, ok_args):
    run_log_stats, _slice = real_closures
    result = run_log_stats(ok_args)
    assert result.ok is True


@pytest.mark.parametrize(
    "ok_args",
    [
        {"from_record": None},
        {"to_record": None},
        {"from_record": 10 ** 18},  # huge int -- resolves to an empty-but-clean slice
        {"to_record": 10 ** 18},
        {"from_record": 10 ** 30},  # enormous int, well past int64 range
        {"to_record": 10 ** 30},
        {"from_record": 1e300},  # enormous float, still finite -- int() succeeds
        {"to_record": 1e300},
        {"from_record": -5},  # negative -- clamped to record 1
        {"to_record": -5},
        {},
        {"unexpected_key": "whatever"},
    ],
)
def test_run_log_slice_null_huge_and_negative_args_never_raise(real_closures, ok_args):
    _stats, run_log_slice = real_closures
    result = run_log_slice(ok_args)
    assert result.ok is True


@pytest.mark.parametrize(
    "bad_args",
    [
        {"group_by": 123},
        {"group_by": None},
        {"group_by": {"nested": "object"}},
        {"group_by": [1, 2, 3]},
        {"tool": {"nested": "object"}},
        {"type": [1, 2, 3]},
        {"status": None},
        {"kind": None},
    ],
)
def test_run_log_stats_string_typed_args_are_defensively_coerced(real_closures, bad_args):
    run_log_stats, _slice = real_closures
    result = run_log_stats(bad_args)
    assert result.ok is True


# -- Boundedness as the log grows --------------------------------------------
#
# task-1271's core requirement: "No call may return output that scales with
# log size." Both real closures are driven against a log large enough that
# an unbounded implementation (one line/record per record) would produce a
# vastly larger result than a bounded one.


def test_real_run_log_stats_and_slice_stay_bounded_on_a_large_log(tmp_path, monkeypatch):
    from tldw_chatbook.Agents import run_log as run_log_module
    from tldw_chatbook.Agents import agent_service as agent_service_module

    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: tmp_path)
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = ToolCatalogRegistry()
    reg.register_provider(BuiltinToolProvider())

    captured: dict = {}
    real_run_agent_loop = agent_service_module.run_agent_loop

    def spy(config, messages, active, deps):
        captured["deps"] = deps
        return real_run_agent_loop(config, messages, active, deps)

    monkeypatch.setattr(agent_service_module, "run_agent_loop", spy)

    service = AgentService(
        db, reg, chat_call=lambda **kw: {"choices": [{"message": {"content": "ok"}}]}
    )
    service.run_turn(
        conversation_id="c1",
        messages=[{"role": "user", "content": "hi"}],
        config=AgentConfig(model="m", system_prompt="s", budget=RunBudget()),
        api_endpoint="openai",
    )
    writer = service.run_log_writer
    assert writer.is_active

    # Append a log "large enough that an unbounded implementation would
    # blow up": thousands of records collapsing into a handful of tools.
    tool_names = ["read_file", "write_file", "grep_files", "calculator", "web_search"]
    for i in range(4000):
        writer.append(
            run_id=writer.log_dir.name,
            kind="primary",
            type="tool_result",
            content=f"synthetic record {i} " + ("z" * 30),
            tool=tool_names[i % len(tool_names)],
            status="error" if i % 11 == 0 else "ok",
        )

    run_log_stats = captured["deps"].run_log_stats
    run_log_slice = captured["deps"].run_log_slice

    # type="tool_result" excludes the single "model" record run_turn's own
    # "hi" -> "ok" round already logged (record 1, tool="") -- isolating
    # this assertion to exactly the 4000 synthetic tool_result records and
    # their 5 distinct tool names.
    stats_result = run_log_stats({"group_by": "tool", "type": "tool_result"})
    assert stats_result.ok is True
    # One line per distinct tool (bounded), never one per record: a header
    # plus exactly len(tool_names) group lines, nowhere near 4000+ lines.
    assert stats_result.content.count("\n") == len(tool_names)
    assert len(stats_result.content) < 5000  # comfortably bounded

    slice_result = run_log_slice({"from_record": 1, "to_record": 100000})
    assert slice_result.ok is True
    # MAX_SLICE_RECORDS caps the record COUNT; the rendered text is still
    # bounded far below "4000 records' worth of content". Counts rendered
    # record HEADERS specifically (format_results' own
    # "record NNNNNN [...]" pattern) rather than a bare substring: a
    # synthetic record's own content text ("synthetic record 0 ...")
    # would otherwise collide with a naive "record 0" search.
    header_count = len(re.findall(r"record \d{6} \[", slice_result.content))
    assert header_count == MAX_SLICE_RECORDS


def test_real_run_log_stats_caps_groups_when_tool_names_far_exceed_the_cap(
    tmp_path, monkeypatch
):
    """A (Qodo review, PR #1078), end-to-end: `compute_stats`' own group cap
    (pinned directly against the pure function in test_run_log_search.py)
    must also hold through the REAL `run_log_stats` closure the loop
    actually calls -- tool names are attacker/model/MCP-server controlled,
    so nothing bounds their distinct count except this cap.
    """
    from tldw_chatbook.Agents import run_log as run_log_module
    from tldw_chatbook.Agents import agent_service as agent_service_module
    from tldw_chatbook.Agents.run_log_search import MAX_STATS_GROUPS

    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: tmp_path)
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = ToolCatalogRegistry()
    reg.register_provider(BuiltinToolProvider())

    captured: dict = {}
    real_run_agent_loop = agent_service_module.run_agent_loop

    def spy(config, messages, active, deps):
        captured["deps"] = deps
        return real_run_agent_loop(config, messages, active, deps)

    monkeypatch.setattr(agent_service_module, "run_agent_loop", spy)

    service = AgentService(
        db, reg, chat_call=lambda **kw: {"choices": [{"message": {"content": "ok"}}]}
    )
    service.run_turn(
        conversation_id="c1",
        messages=[{"role": "user", "content": "hi"}],
        config=AgentConfig(model="m", system_prompt="s", budget=RunBudget()),
        api_endpoint="openai",
    )
    writer = service.run_log_writer
    assert writer.is_active

    # Far more distinct tool names than MAX_STATS_GROUPS -- exactly the
    # shape a model calling many differently-named MCP tools would produce.
    distinct_tool_count = MAX_STATS_GROUPS + 60
    for i in range(distinct_tool_count):
        writer.append(
            run_id=writer.log_dir.name,
            kind="primary",
            type="tool_result",
            content=f"call to tool {i}",
            tool=f"mcp_tool_{i}",
            status="ok",
        )

    run_log_stats = captured["deps"].run_log_stats
    result = run_log_stats({"group_by": "tool", "type": "tool_result"})
    assert result.ok is True
    # Bounded: 1 header line + MAX_STATS_GROUPS group lines + 1 omission
    # trailer line = MAX_STATS_GROUPS + 1 newlines -- never one line per
    # distinct tool name (which would be distinct_tool_count + 1 lines,
    # far more than this).
    assert result.content.count("\n") == MAX_STATS_GROUPS + 1
    # The omission is NAMED, not silently dropped: the exact remainder
    # count appears, and the word "omitted" appears -- a model reading this
    # must be able to tell the result is partial, not "the whole picture is
    # exactly the top group and nothing else exists".
    omitted_count = distinct_tool_count - MAX_STATS_GROUPS
    assert str(omitted_count) in result.content
    assert "omitted" in result.content


# -- B (Qodo review, PR #1078): an unsupported group_by must not be --------
# -- rendered under its own (wrong) label -----------------------------------
#
# `compute_stats` itself already falls back an unrecognised `group_by` to
# "tool" -- that part was never broken. The bug was in THIS closure: it
# passed the caller's ORIGINAL string through to `format_stats`' `group_by=`
# label regardless of what `compute_stats` actually grouped by, so an
# unsupported value produced tool counts confidently mislabelled as
# whatever the model asked for. Fixed by normalising `group_by` here,
# before calling `compute_stats`, so the same (already-fallen-back) value
# feeds both the aggregation and the label.


def test_run_log_stats_unsupported_group_by_is_labelled_as_tool_not_echoed(real_closures):
    run_log_stats, _slice = real_closures
    result = run_log_stats({"group_by": "not_a_real_dimension"})
    assert result.ok is True
    assert "grouped by tool" in result.content
    assert "grouped by not_a_real_dimension" not in result.content


@pytest.mark.parametrize("group_by", ["tool", "type", "status", "kind"])
def test_run_log_stats_every_supported_group_by_is_labelled_correctly(real_closures, group_by):
    run_log_stats, _slice = real_closures
    result = run_log_stats({"group_by": group_by})
    assert result.ok is True
    assert f"grouped by {group_by}" in result.content


# -- E/F (Qodo review, PR #1078, DECLINED with a ruling): raw-dict-plus- ----
# -- defensive-cast is kept, but every argument slot of BOTH tools must be --
# -- proven safe against the full JSON-decodable value space ---------------
#
# The review wanted model-supplied tool arguments routed through a Pydantic
# model instead of the defensive str()/int() coercion this closure (and
# every sibling runtime-tool closure -- install_skill, run_skill_script,
# skill_file, search_run_log) already uses. Declined for consistency (see
# run_log_stats'/run_log_slice's own docstrings in agent_service.py) --
# but declining a validation LAYER does not excuse skipping proof that the
# coercion actually holds. This drives every argument BOTH tools accept,
# one at a time, through EVERY value type a JSON-decoded `ToolCall.args`
# could put there (str, int, float, bool, None, list, dict) -- covering
# "a string where an int belongs", None, a nested object, a list, a
# boolean, a negative, and an out-of-range value, per-argument, through the
# REAL service closures (not a fake) -- and confirms every single
# combination returns a clean ToolResult, never an exception escaping the
# call.

_HOSTILE_JSON_VALUES = [
    pytest.param("not-a-number", id="string-where-int-belongs"),
    pytest.param(None, id="none"),
    pytest.param({"nested": "object"}, id="nested-object"),
    pytest.param([1, 2, 3], id="list"),
    pytest.param(True, id="boolean"),
    pytest.param(-5, id="negative"),
    pytest.param(10 ** 30, id="out-of-range"),
]

RUN_LOG_STATS_ARG_NAMES = (
    "group_by",
    "tool",
    "type",
    "status",
    "kind",
    "from_record",
    "to_record",
)
RUN_LOG_SLICE_ARG_NAMES = ("from_record", "to_record")


@pytest.mark.parametrize("value", _HOSTILE_JSON_VALUES)
@pytest.mark.parametrize("arg_name", RUN_LOG_STATS_ARG_NAMES)
def test_run_log_stats_every_argument_survives_every_hostile_json_value(
    real_closures, arg_name, value
):
    run_log_stats, _slice = real_closures
    # The call itself is the assertion: if any argument slot's coercion is
    # unsafe for this value, this raises and the test fails with that
    # exception's traceback -- exactly the "never raise into the run"
    # contract this proves.
    result = run_log_stats({arg_name: value})
    assert isinstance(result, ToolResult)


@pytest.mark.parametrize("value", _HOSTILE_JSON_VALUES)
@pytest.mark.parametrize("arg_name", RUN_LOG_SLICE_ARG_NAMES)
def test_run_log_slice_every_argument_survives_every_hostile_json_value(
    real_closures, arg_name, value
):
    _stats, run_log_slice = real_closures
    result = run_log_slice({arg_name: value})
    assert isinstance(result, ToolResult)
