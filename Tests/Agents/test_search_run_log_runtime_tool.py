# Tests/Agents/test_search_run_log_runtime_tool.py
"""search_run_log: primary-only, no catalog slot, dispatched by the loop."""

import json
import re

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
    # TASK-1250: offset is the schema-level knob that lets a model page
    # deterministically through a record larger than the render ceiling.
    assert "offset" in props


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


# -- Final-review CRITICAL 1 / IMPORTANT 6: exercise the REAL closure --------
#
# Every test above either injects a fake deps.search_run_log or asserts on
# the schema; none drives agent_service.py's REAL closure. That gap is why
# the closure's 400-char rendering ceiling (format_results' own default,
# never overridden by the closure) survived seven reviews: following a
# truncation trailer returned LESS content than the truncation it was
# supposed to repair -- defeating spec §6.1, the single change that makes an
# additive Phase 1 pay off at all.


class _AllowGate:
    """Bypasses BuiltinToolProvider's approval machinery for these tests.

    `read_file`/`grep_files` carry the "reads" risk tag, which floors them
    to `ask` under the REAL gate (`BuiltinToolProvider()`'s lazily-built
    default) -- see test_builtin_gate_live_tools.py. These tests care about
    the run-log closure and the file tools' own containment, not the
    approval round trip, so they hand the provider a gate that always
    allows.
    """

    def check(self, tool):
        return None


def test_real_closure_recovers_full_content_beyond_both_caps(tmp_path, monkeypatch):
    """CRITICAL 1's regression test: drive a real tool result large enough
    to be truncated in history, follow the trailer's own record pointer
    through the REAL search_run_log closure, and confirm the recovered
    content reaches a marker placed well beyond format_results' OLD
    400-char rendering default.

    Note on "beyond 16,000 chars" (the run's tool-result ceiling): this is
    NOT independently achievable and is not attempted here. The recovered
    render is deliberately capped at THIS SAME ceiling
    (`config.budget.max_tool_result_chars`) that the ORIGINAL append-time
    truncation used, and the search_run_log call's own result is re-capped
    at the identical ceiling when IT is appended to history (the loop's
    ordinary `_truncate_tool_result`, applied uniformly to every tool
    result). So nothing the model ever reads in a message can exceed that
    ceiling, by design ("this cannot blow the context" -- see the finding).
    The marker is instead placed comfortably ABOVE 400 and comfortably
    BELOW the ceiling: exactly the band the old bug made unreachable and
    the fix restores.
    """
    from tldw_chatbook.Agents import run_log as run_log_module
    import tldw_chatbook.Tools.file_operation_tools as file_tools
    import tldw_chatbook.config as config_module

    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: tmp_path)
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    monkeypatch.setattr(file_tools, "_resolve_sandbox_config", lambda: str(sandbox))

    def fake_get_cli_setting(section, key=None, default=None):
        if section == "tools" and key == "read_file_enabled":
            return True
        return default

    monkeypatch.setattr(config_module, "get_cli_setting", fake_get_cli_setting)

    # Marker sits at ~char 10,000: well past the old 400-char rendering
    # bug, safely inside the run's 16,000-char ceiling so it is genuinely
    # recoverable, while the trailing filler pushes the WHOLE result past
    # 16,000 so a real append-time truncation (and a real format_results
    # truncation on the recovery side) both actually fire.
    marker = "END_MARKER_7f3a9c"
    big_content = "A" * 10_000 + marker + "C" * 40_000
    (sandbox / "big.txt").write_text(big_content, encoding="utf-8")

    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = ToolCatalogRegistry()
    reg.register_provider(BuiltinToolProvider(gate=_AllowGate()))

    calls = []

    def chat(**kwargs):
        calls.append(kwargs)
        round_idx = len(calls)
        if round_idx == 1:
            return _svc_fence("read_file", {"file_path": "big.txt"})
        if round_idx == 2:
            # Follow the trailer's OWN pointer, extracted from the ACTUAL
            # truncated tool-result message the loop appended -- not a
            # hardcoded record number -- so this test tracks the real
            # capture order rather than assuming it.
            last_msg = kwargs["messages_payload"][-1]["content"]
            match = re.search(r"from_record=(\d+)", last_msg)
            assert match, f"trailer did not name a record: {last_msg!r}"
            record_number = int(match.group(1))
            return _svc_fence(
                SEARCH_RUN_LOG_TOOL_NAME,
                {"from_record": record_number, "to_record": record_number},
            )
        return {"choices": [{"message": {"content": "done"}}]}

    service = AgentService(db, reg, chat_call=chat)
    _rid, outcome = service.run_turn(
        conversation_id="c1",
        messages=[{"role": "user", "content": "read the file"}],
        config=AgentConfig(
            model="m",
            system_prompt="s",
            allowed_tools=("read_file",),
            budget=RunBudget(),
        ),
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE

    # Round 2's payload carries the history-truncated result: confirm a
    # genuine truncation happened (there is more content than the ceiling
    # allows), and that it names a recovery record.
    truncated_msg = calls[1]["messages_payload"][-1]["content"]
    assert "truncated" in truncated_msg
    assert "search_run_log" in truncated_msg

    # Round 3's payload carries search_run_log's rendered result: the
    # marker -- at ~char 10,000, 25x past format_results' OLD 400-char
    # default -- must be PRESENT, proving the closure now renders at the
    # run's real ceiling rather than the old hardcoded default.
    recovered_msg = calls[2]["messages_payload"][-1]["content"]
    assert marker in recovered_msg
    assert len(recovered_msg) > 400, (
        "recovered message must be far larger than the old 400-char bug"
    )


def test_parent_can_filter_its_log_to_subagent_records_via_kind(tmp_path, monkeypatch):
    """IMPORTANT 5's regression test: `kind` is implemented in
    `search_records` and justified by spec §4.1 ("a parent can search its
    child's entire trace"), but the schema omitted it and the closure never
    passed it through -- so it was unreachable end to end. Drives a real
    spawn, then has the PARENT (never the child -- search_run_log stays
    primary-only) filter its own log to `kind=subagent` and confirms it
    finds the child's own record.
    """
    from tldw_chatbook.Agents import run_log as run_log_module

    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: tmp_path)

    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = ToolCatalogRegistry()
    reg.register_provider(BuiltinToolProvider())

    child_marker = "CHILD_ONLY_MARKER_4b8e"
    script = [
        _svc_fence(SPAWN_TOOL_NAME, {"task": "investigate"}),  # parent spawns
        {"choices": [{"message": {"content": child_marker}}]},  # child's final answer
        _svc_fence(SEARCH_RUN_LOG_TOOL_NAME, {"kind": "subagent"}),  # parent searches
        {"choices": [{"message": {"content": "done"}}]},  # parent's final answer
    ]

    def chat(**kwargs):
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
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE

    # Verify against the persisted run tree, not just the scripted flow:
    # the PARENT's own tool_result for its search_run_log call must
    # contain the CHILD's marker text.
    parent_runs = [r for r in db.list_runs("c1") if r["agent_kind"] == "primary"]
    assert len(parent_runs) == 1
    search_results = [
        s["result"]
        for s in parent_runs[0]["steps"]
        if s["kind"] == "tool_result" and s.get("tool_name") == SEARCH_RUN_LOG_TOOL_NAME
    ]
    assert search_results, "expected a search_run_log tool_result step"
    assert any(child_marker in r for r in search_results)


def test_real_closure_never_raises_on_a_junk_offset(tmp_path, monkeypatch):
    """TASK-1250: `offset` must be coerced defensively, the same way the
    closure already coerces from_record/to_record/context -- a model
    sending a non-numeric value must come back as an ordinary error
    ToolResult, never an exception that aborts the run.
    """
    from tldw_chatbook.Agents import run_log as run_log_module

    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: tmp_path)

    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = ToolCatalogRegistry()
    reg.register_provider(BuiltinToolProvider())

    script = [
        _svc_fence(
            SEARCH_RUN_LOG_TOOL_NAME, {"contains": "x", "offset": "not-a-number"}
        ),
        {"choices": [{"message": {"content": "done"}}]},
    ]

    def chat(**kwargs):
        return script.pop(0)

    service = AgentService(db, reg, chat_call=chat)
    _rid, outcome = service.run_turn(
        conversation_id="c1",
        messages=[{"role": "user", "content": "go"}],
        config=AgentConfig(
            model="m",
            system_prompt="s",
            allowed_tools=("calculator",),
            budget=RunBudget(),
        ),
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE

    parent_runs = [r for r in db.list_runs("c1") if r["agent_kind"] == "primary"]
    assert len(parent_runs) == 1
    tool_results = [
        s["result"]
        for s in parent_runs[0]["steps"]
        if s["kind"] == "tool_result" and s.get("tool_name") == SEARCH_RUN_LOG_TOOL_NAME
    ]
    assert tool_results, "expected a search_run_log tool_result step"
    assert any("Invalid search arguments" in r for r in tool_results)


# -- F2 (Qodo #2, PR #1066 review -- DECLINED, confirmed by test) ------------
#
# Qodo wanted these raw dict args routed through a Pydantic model before use.
# Declined: every OTHER runtime tool this service wires (install_skill,
# run_skill_script, skill_file) takes the same raw-dict-plus-defensive-cast
# shape, and every argument here is ALREADY coerced defensively -- a bad
# value returns a clean ToolResult error rather than raising. This section
# drives the REAL closure (captured off a REAL AgentService the same way
# test_on_record_returns_the_assigned_record_number does) with a string
# where an int is expected, null, a nested object, and a list, for BOTH the
# string-typed metadata filters and the int-typed range/paging arguments --
# confirming none of them raise. If any gap had been found, it would have
# been fixed here rather than merely asserted away.


@pytest.fixture
def wired(tmp_path, monkeypatch):
    from tldw_chatbook.Agents import run_log as run_log_module

    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: tmp_path)
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = ToolCatalogRegistry()
    reg.register_provider(BuiltinToolProvider())
    return db, reg, tmp_path


@pytest.fixture
def real_search_run_log(wired, monkeypatch):
    """The ACTUAL search_run_log closure a real AgentService wires into
    LoopDeps, captured via a run_agent_loop spy -- mirrors
    test_run_log_service_wiring.py::test_on_record_returns_the_assigned_record_number.
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
    assert deps.search_run_log is not None
    return deps.search_run_log


@pytest.mark.parametrize(
    "bad_args",
    [
        {"from_record": "not-a-number"},
        {"to_record": "not-a-number"},
        {"context": "not-a-number"},
        {"offset": "not-a-number"},
        {"from_record": {"nested": "object"}},
        {"to_record": {"nested": "object"}},
        {"context": {"nested": "object"}},
        {"offset": {"nested": "object"}},
        {"from_record": [1, 2, 3]},
        {"to_record": [1, 2, 3]},
        {"context": [1, 2, 3]},
        {"offset": [1, 2, 3]},
        {"from_record": float("nan")},
        # task-1272 Phase 3 review (carried-over finding, sibling PR): a
        # model can send `float('inf')` for any of these -- `int(float(
        # 'inf'))` raises OverflowError, NOT TypeError/ValueError, which
        # this closure's `except (TypeError, ValueError)` previously did
        # not catch, so it escaped uncaught into the run instead of
        # degrading to a clean tool error like every other malformed
        # value here. `run_log_stats`/`run_log_slice` (Phase 2) already
        # caught this; `search_run_log` (Phase 1, merged earlier) did not.
        {"from_record": float("inf")},
        {"to_record": float("inf")},
        {"context": float("inf")},
        {"offset": float("inf")},
    ],
)
def test_unparseable_numeric_args_return_a_clean_error_never_raise(
    real_search_run_log, bad_args
):
    result = real_search_run_log(bad_args)
    assert result.ok is False
    assert "Invalid search arguments" in result.error


@pytest.mark.parametrize(
    "null_args",
    [
        {"from_record": None},
        {"to_record": None},
        {"context": None},
        {"offset": None},
    ],
)
def test_null_numeric_args_are_coerced_to_zero_not_an_error(
    real_search_run_log, null_args
):
    # `args.get(key) or 0` treats an explicit `null`/None the same as a
    # missing key -- a deliberate, already-safe coercion (not a gap): the
    # model sent nothing usable, so "no filter/no offset" is the correct
    # reading, distinct from a genuinely malformed value like a string or a
    # nested object above, which DOES surface as an error.
    result = real_search_run_log(null_args)
    assert result.ok is True


@pytest.mark.parametrize(
    "bad_args",
    [
        {"contains": 123},
        {"contains": None},
        {"contains": {"nested": "object"}},
        {"contains": [1, 2, 3]},
        {"pattern": 123},
        {"pattern": None},
        {"pattern": {"nested": "object"}},
        {"tool": 123},
        {"tool": None},
        {"tool": {"nested": "object"}},
        {"tool": [1, 2, 3]},
        {"type": None},
        {"status": None},
        {"kind": None},
    ],
)
def test_string_typed_args_are_defensively_coerced_never_raise(
    real_search_run_log, bad_args
):
    result = real_search_run_log(bad_args)
    # str(...) on any of these never raises, so these must reach a normal
    # (possibly empty) result -- never an exception.
    assert result.ok is True


def test_missing_args_dict_entirely_does_not_raise(real_search_run_log):
    # An empty dict -- every key falls back to its documented default.
    result = real_search_run_log({})
    assert result.ok is True


def test_args_is_not_even_a_dict_of_the_expected_shape(real_search_run_log):
    # A model could plausibly hand back a flat, oddly-shaped dict (extra
    # unrecognised keys) -- ignored, never raises.
    result = real_search_run_log({"unexpected_key": "whatever", "contains": "x"})
    assert result.ok is True
