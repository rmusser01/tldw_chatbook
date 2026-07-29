# Tests/Agents/test_run_log_cross_run_search.py
"""task-1273: `search_run_log`'s `scope="conversation"` cross-run search.

Builds on task-870 (`run_log.resolve_existing_log_dir`) and the Phase 2
query layer (`run_log_search.py`) -- see the "Revised analysis" section of
task-1273 for why this composes without a schema change: `AgentRunsDB.
list_runs(conversation_id)` already enumerates a conversation's runs, and
`resolve_existing_log_dir(run_id)` already locates an arbitrary run's log
directory by id, read-only.

Mirrors Tests/Agents/test_search_run_log_runtime_tool.py's structure: the
REAL closure a real AgentService wires, captured via a run_agent_loop spy
for the argument-shape tests, and full `service.run_turn` scripts (like
test_subagent_cannot_call_search_run_log and
test_parent_can_filter_its_log_to_subagent_records_via_kind there) for the
end-to-end scenarios that need an actual prior run planted in the DB before
the current run starts.
"""

import json

import pytest

from tldw_chatbook.Agents.agent_models import (
    AGENT_KIND_PRIMARY,
    RUN_DONE,
    AgentConfig,
    RunBudget,
    ToolResult,
)
from tldw_chatbook.Agents.agent_models import SEARCH_RUN_LOG_TOOL_NAME, SPAWN_TOOL_NAME
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.tool_catalog import (
    SEARCH_RUN_LOG_TOOL_SCHEMA,
    BuiltinToolProvider,
    ToolCatalogRegistry,
)
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


def _fence(name, args):
    return f"```tool_call\n{json.dumps({'name': name, 'arguments': args})}\n```"


def _svc_fence(name, args):
    return {"choices": [{"message": {"content": _fence(name, args)}}]}


def test_scope_is_a_declared_parameter():
    props = SEARCH_RUN_LOG_TOOL_SCHEMA.parameters["properties"]
    assert "scope" in props


# -- Argument-shape tests, via the REAL closure captured off a REAL
# AgentService (mirrors test_search_run_log_runtime_tool.py's `wired`/
# `real_search_run_log` fixtures -- duplicated locally rather than imported
# cross-file, matching this suite's existing per-file convention). ---------


@pytest.fixture
def real_search_run_log(tmp_path, monkeypatch):
    """The ACTUAL search_run_log closure a real AgentService wires into
    LoopDeps, captured via a run_agent_loop spy, for a conversation ("c1")
    that starts with no other runs.
    """
    from tldw_chatbook.Agents import agent_service as agent_service_module
    from tldw_chatbook.Agents import run_log as run_log_module

    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: tmp_path)
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = ToolCatalogRegistry()
    reg.register_provider(BuiltinToolProvider())

    captured: dict = {}
    real_run_agent_loop = agent_service_module.run_agent_loop

    def spy_run_agent_loop(config, messages, active, deps):
        captured["deps"] = deps
        return real_run_agent_loop(config, messages, active, deps)

    monkeypatch.setattr(agent_service_module, "run_agent_loop", spy_run_agent_loop)

    service = AgentService(
        db, reg, chat_call=lambda **kw: {"choices": [{"message": {"content": "ok"}}]}
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


def test_default_scope_is_byte_identical_to_explicit_run_scope(real_search_run_log):
    """Default behaviour must not change: omitting `scope` entirely must
    produce EXACTLY the same ToolResult as `scope="run"` -- both take the
    literal unchanged code path (the "run" branch below task-1273's new
    `if scope == "conversation":` fork was never touched)."""
    omitted = real_search_run_log({"contains": "x"})
    explicit = real_search_run_log({"contains": "x", "scope": "run"})
    assert omitted == explicit
    assert isinstance(omitted, ToolResult)


def test_conversation_scope_with_zero_prior_runs_searches_only_current(
    real_search_run_log,
):
    """A conversation with no earlier runs (only the current one, which the
    fixture's own `run_turn` already created): `scope="conversation"` must
    degrade gracefully to "search the one run there is", not error."""
    result = real_search_run_log({"scope": "conversation"})
    assert result.ok is True
    assert "Searched 1 of 1 run(s)" in result.content
    assert "could not be located" not in result.content


@pytest.mark.parametrize(
    "bad_scope",
    ["", None, "everything", "RUN", 123, ["conversation"], {"a": 1}, 12.5],
)
def test_junk_scope_values_never_raise(real_search_run_log, bad_scope):
    result = real_search_run_log({"contains": "x", "scope": bad_scope})
    assert result.ok is True


# -- End-to-end scenarios: a prior primary run planted in the DB (and, for
# the "found" case, a real log directory) BEFORE the current run starts. --


def _plant_older_run(db, run_log_module, conversation_id: str, *, with_log: bool, content: str = "") -> str:
    """Create an older PRIMARY run's DB row, optionally with a real log.

    Args:
        db: The shared AgentRunsDB.
        run_log_module: `tldw_chatbook.Agents.run_log`, already pointed at
            the test's tmp_path root via `resolve_log_root`.
        conversation_id: The conversation this run belongs to.
        with_log: When True, binds a real RunLogWriter and appends
            `content` -- simulating a run whose log is still reachable
            under the current root. When False, the DB row exists but no
            log directory is ever created -- simulating task-1273's one
            honest limitation (the root changed, or predates run-log).
        content: Record content to append when `with_log` is True.

    Returns:
        The older run's id.
    """
    older_run_id = db.create_run(
        conversation_id=conversation_id, agent_kind=AGENT_KIND_PRIMARY
    )
    db.set_status(older_run_id, "done", result="done")
    if with_log:
        writer = run_log_module.RunLogWriter()
        writer.bind(older_run_id)
        writer.append(run_id=older_run_id, kind="primary", type="model", content=content)
    return older_run_id


def test_hit_in_an_older_run_is_found_and_attributed(tmp_path, monkeypatch):
    from tldw_chatbook.Agents import run_log as run_log_module

    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: tmp_path)
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = ToolCatalogRegistry()
    reg.register_provider(BuiltinToolProvider())

    marker = "OLDER_RUN_MARKER_9f2a"
    older_run_id = _plant_older_run(
        db, run_log_module, "c1", with_log=True, content=f"tried {marker} last time"
    )

    script = [
        _svc_fence(
            SEARCH_RUN_LOG_TOOL_NAME, {"contains": marker, "scope": "conversation"}
        ),
        {"choices": [{"message": {"content": "done"}}]},
    ]

    def chat(**kwargs):
        return script.pop(0)

    service = AgentService(db, reg, chat_call=chat)
    _rid, outcome = service.run_turn(
        conversation_id="c1",
        messages=[{"role": "user", "content": "what did I try last time?"}],
        config=AgentConfig(
            model="m",
            system_prompt="s",
            allowed_tools=("calculator",),
            budget=RunBudget(),
        ),
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE

    primary_runs = [r for r in db.list_runs("c1") if r["agent_kind"] == "primary"]
    current = [r for r in primary_runs if r["id"] != older_run_id]
    assert len(current) == 1
    tool_results = [
        s["result"]
        for s in current[0]["steps"]
        if s["kind"] == "tool_result" and s.get("tool_name") == SEARCH_RUN_LOG_TOOL_NAME
    ]
    assert tool_results, "expected a search_run_log tool_result step"
    text = tool_results[0]
    assert marker in text
    assert "Searched 2 of 2 run(s)" in text
    assert "could not be located" not in text
    # Attribution: the hit must be labelled as coming from the OLDER run,
    # by its id, and never rendered as if it were "this run".
    assert f"an earlier run ({older_run_id})" in text


def test_unresolvable_older_run_is_reported_not_silently_skipped(tmp_path, monkeypatch):
    """The one honest limitation (task-1273): a run whose log cannot be
    located under the current root must be counted and reported, never
    silently dropped from the response."""
    from tldw_chatbook.Agents import run_log as run_log_module

    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: tmp_path)
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = ToolCatalogRegistry()
    reg.register_provider(BuiltinToolProvider())

    missing_run_id = _plant_older_run(db, run_log_module, "c1", with_log=False)

    # `tool=` (an exact metadata filter), not `contains=`: a `contains=`
    # value would be embedded verbatim in the model's own tool_call record
    # (its arguments ARE the query), self-matching the CURRENT run
    # regardless of what string is chosen -- an artifact of this
    # script-driven test technique, not of the closure under test. A tool
    # name that never appears in any record's metadata has no such problem.
    script = [
        _svc_fence(
            SEARCH_RUN_LOG_TOOL_NAME,
            {"tool": "nonexistent_tool_xyz", "scope": "conversation"},
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

    primary_runs = [r for r in db.list_runs("c1") if r["agent_kind"] == "primary"]
    current = [r for r in primary_runs if r["id"] != missing_run_id]
    assert len(current) == 1
    tool_results = [
        s["result"]
        for s in current[0]["steps"]
        if s["kind"] == "tool_result" and s.get("tool_name") == SEARCH_RUN_LOG_TOOL_NAME
    ]
    assert tool_results, "expected a search_run_log tool_result step"
    text = tool_results[0]
    # Only the current run was actually searched; the older one is reported
    # as unlocatable, never conflated with "no matches".
    assert "Searched 1 of 2 run(s)" in text
    assert "1 could not be located" in text
    assert "No matching records." in text


def test_multiple_older_runs_mixed_resolvable_and_not(tmp_path, monkeypatch):
    """Two older runs: one whose log is found (no marker hit, but counted
    as searched) and one whose log cannot be located -- both must be
    accounted for distinctly in the coverage line, alongside the current
    run's own search.
    """
    from tldw_chatbook.Agents import run_log as run_log_module

    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: tmp_path)
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = ToolCatalogRegistry()
    reg.register_provider(BuiltinToolProvider())

    found_run_id = _plant_older_run(
        db, run_log_module, "c1", with_log=True, content="nothing interesting here"
    )
    missing_run_id = _plant_older_run(db, run_log_module, "c1", with_log=False)

    script = [
        _svc_fence(
            SEARCH_RUN_LOG_TOOL_NAME, {"contains": "nope", "scope": "conversation"}
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

    primary_runs = [r for r in db.list_runs("c1") if r["agent_kind"] == "primary"]
    current = [
        r for r in primary_runs if r["id"] not in (found_run_id, missing_run_id)
    ]
    assert len(current) == 1
    tool_results = [
        s["result"]
        for s in current[0]["steps"]
        if s["kind"] == "tool_result" and s.get("tool_name") == SEARCH_RUN_LOG_TOOL_NAME
    ]
    assert tool_results
    text = tool_results[0]
    # 3 runs total: current + found_run_id + missing_run_id.
    assert "Searched 2 of 3 run(s)" in text
    assert "1 could not be located" in text


# -- Sub-agent gating: scope="conversation" must not widen a sub-agent's
# reach any more than plain search_run_log already refuses it. Mirrors
# test_search_run_log_runtime_tool.py::test_subagent_cannot_call_search_run_log
# exactly, but the child's scripted call explicitly requests conversation
# scope -- proving the gate holds independent of which scope is requested.


def test_subagent_cannot_call_search_run_log_with_conversation_scope(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Agents import run_log as run_log_module

    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: tmp_path)
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = ToolCatalogRegistry()
    reg.register_provider(BuiltinToolProvider())

    calls = []
    script = [
        _svc_fence(SPAWN_TOOL_NAME, {"task": "native task"}),  # parent spawns
        _svc_fence(
            SEARCH_RUN_LOG_TOOL_NAME, {"contains": "x", "scope": "conversation"}
        ),  # child tries cross-run reach
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
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE

    # (a) schema gate: the child never sees search_run_log at all (scope is
    # just an argument on an undisclosed tool).
    child_system_prompt = calls[1]["messages_payload"][0]["content"]
    assert SEARCH_RUN_LOG_TOOL_NAME not in child_system_prompt

    # (b) dispatch gate: the child's call, made regardless of (a), is
    # refused through the ordinary permission path -- it never reaches the
    # search_run_log closure, so cross-run search never even executes for
    # a sub-agent.
    child_runs = [r for r in db.list_runs("c1") if r["agent_kind"] == "subagent"]
    assert len(child_runs) == 1
    tool_results = [
        s["result"] for s in child_runs[0]["steps"] if s["kind"] == "tool_result"
    ]
    assert any(
        f"Tool not permitted: {SEARCH_RUN_LOG_TOOL_NAME}" in r for r in tool_results
    )


# -- Pure-function coverage for `search_across_runs`/`format_cross_run_results`
# directly -- the shared-budget behaviours (limit and deadline SHARED across
# every run, not reset per run) are real correctness properties but awkward
# to provoke reliably through a full scripted `service.run_turn`, so they
# are exercised here against the functions themselves, mirroring how
# test_run_log_search.py tests `search_records`/`compute_stats`/
# `slice_records` directly rather than only through the runtime-tool
# closures.


def _write_run(run_log_module, run_id: str, contents: list[str]):
    writer = run_log_module.RunLogWriter()
    writer.bind(run_id)
    for content in contents:
        writer.append(run_id=run_id, kind="primary", type="model", content=content)
    return writer.log_dir


def test_search_across_runs_shares_the_hit_limit_across_runs(tmp_path, monkeypatch):
    from tldw_chatbook.Agents import run_log as run_log_module
    from tldw_chatbook.Agents.run_log_search import search_across_runs

    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: tmp_path)
    dir_a = _write_run(run_log_module, "run-a", [f"marker {i}" for i in range(30)])
    dir_b = _write_run(run_log_module, "run-b", [f"marker {i}" for i in range(30)])

    result = search_across_runs(
        [("run-b", dir_b), ("run-a", dir_a)],
        current_run_id="run-b",
        contains="marker",
        limit=10,
    )
    # 60 total matching records exist across the two runs; the shared
    # `limit` must still cap the combined output at 10 -- the per-call
    # output ceiling must not scale with the number of runs searched.
    assert len(result.hits) == 10
    # Both runs were nonetheless SCANNED (their logs located and searched),
    # even though run-a's own hits were all cut by the already-spent limit.
    assert result.searched_run_ids == ["run-b", "run-a"]
    assert result.unresolved_run_ids == []
    assert result.not_searched_run_ids == []


def test_search_across_runs_shares_the_deadline_across_runs(tmp_path, monkeypatch):
    """An exhausted shared wall-clock budget must stop further scanning --
    resetting the deadline per run would let a `scope="conversation"` call
    cost `len(runs) * MAX_SEARCH_SECONDS` in the worst case, defeating the
    single-run "cheap, in-process" guarantee. A run whose log genuinely
    exists but was never reached this way is `not_searched`, never
    conflated with `unresolved` (whose log could not be located at all).
    """
    from tldw_chatbook.Agents import run_log as run_log_module
    from tldw_chatbook.Agents.run_log_search import search_across_runs

    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: tmp_path)
    dir_a = _write_run(run_log_module, "run-a", ["hello"])
    dir_b = _write_run(run_log_module, "run-b", ["hello"])

    result = search_across_runs(
        [("run-a", dir_a), ("run-b", dir_b), ("run-c", None)],
        current_run_id="run-a",
        contains="hello",
        deadline_seconds=0.0,
    )
    assert result.hits == []
    assert result.searched_run_ids == []
    assert set(result.not_searched_run_ids) == {"run-a", "run-b"}
    # run-c's log was never locatable in the first place -- that fact does
    # not depend on the deadline at all, and must stay in its own bucket.
    assert result.unresolved_run_ids == ["run-c"]


def test_format_cross_run_results_states_coverage_and_attributes_hits():
    from tldw_chatbook.Agents.run_log_format import RunLogRecord
    from tldw_chatbook.Agents.run_log_search import (
        CrossRunHit,
        CrossRunSearchResult,
        format_cross_run_results,
    )

    record = RunLogRecord(
        number=1, run_id="run-old", kind="primary", type="model", ts="-",
        content="hello world",
    )
    result = CrossRunSearchResult(
        hits=[CrossRunHit(record=record, source_run_id="run-old", is_current_run=False)],
        searched_run_ids=["run-cur", "run-old"],
        unresolved_run_ids=["run-missing"],
        not_searched_run_ids=["run-skipped"],
    )
    text = format_cross_run_results(result)
    assert "Searched 2 of 4 run(s) in this conversation" in text
    assert "1 could not be located" in text
    assert "1 not attempted this call" in text
    assert "an earlier run (run-old)" in text
    assert "hello world" in text


def test_format_cross_run_results_no_hits_still_states_coverage():
    """An empty `hits` list must never read as "searched everything, found
    nothing" when some runs were not actually searched -- task-1273's core
    requirement."""
    from tldw_chatbook.Agents.run_log_search import (
        CrossRunSearchResult,
        format_cross_run_results,
    )

    result = CrossRunSearchResult(
        hits=[], searched_run_ids=["a"], unresolved_run_ids=["b"],
        not_searched_run_ids=[],
    )
    text = format_cross_run_results(result)
    assert "Searched 1 of 2 run(s)" in text
    assert "1 could not be located" in text
    assert "No matching records." in text


def test_format_cross_run_results_folds_omitted_run_count_into_not_attempted():
    """`omitted_run_count` (runs beyond MAX_CROSS_RUN_RUNS, known only as a
    COUNT -- see finding A) and `not_searched_run_ids` (runs cut by the
    shared deadline, known by exact id) must both land in the SAME "not
    attempted" coverage note and the same total, even though the caller
    only ever has ids for one of the two."""
    from tldw_chatbook.Agents.run_log_search import (
        CrossRunSearchResult,
        format_cross_run_results,
    )

    result = CrossRunSearchResult(
        hits=[], searched_run_ids=["a"], unresolved_run_ids=[],
        not_searched_run_ids=["b"],
    )
    text = format_cross_run_results(result, omitted_run_count=3)
    # 1 searched + 0 unresolved + (1 not_searched + 3 omitted) = 5 total.
    assert "Searched 1 of 5 run(s)" in text
    assert "4 not attempted this call" in text


# -- Review findings on PR #1088 ---------------------------------------------
#
# A (Performance, agent_service.py): the conversation-scope path called
# AgentRunsDB.list_runs(conversation_id) with no limit, materialising every
# run a conversation has ever had before capping to MAX_CROSS_RUN_RUNS
# client-side. Fixed by pushing both the `agent_kind` filter and the
# `limit` into the query (AgentRunsDB.list_runs/count_runs), so the DB
# returns at most what the cap can use, plus one cheap COUNT(*) query for
# the exact omitted total (never every omitted row).
#
# B (Reliability, run_log_search.py): the shared deadline was checked
# before `load_records()`, but loading itself is unbounded I/O not counted
# against the budget. Fixed by making `load_records` itself deadline-aware
# (stops between segments, raises RunLogSearchTimeout) AND recomputing the
# remaining deadline again immediately after each load, before searching --
# either exhaustion routes that run to `not_searched`, never to a partial
# scan silently presented as complete.


def test_more_runs_than_the_cap_reports_the_excess_correctly(tmp_path, monkeypatch):
    """Finding A: a conversation with more PRIMARY runs than
    MAX_CROSS_RUN_RUNS must still report an EXACT omitted count via the
    bounded `count_runs` query -- not silently claim full coverage, and not
    require fetching every omitted run just to count them."""
    from tldw_chatbook.Agents import run_log as run_log_module
    from tldw_chatbook.Agents.run_log_search import MAX_CROSS_RUN_RUNS

    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: tmp_path)
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = ToolCatalogRegistry()
    reg.register_provider(BuiltinToolProvider())

    extra = 3
    older_count = MAX_CROSS_RUN_RUNS + extra
    for i in range(older_count):
        _plant_older_run(
            db, run_log_module, "c1", with_log=True, content=f"older {i}"
        )

    script = [
        _svc_fence(SEARCH_RUN_LOG_TOOL_NAME, {"scope": "conversation"}),
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

    all_primary = [r for r in db.list_runs("c1") if r["agent_kind"] == "primary"]
    total_primary = len(all_primary)  # older_count + 1 (current)
    assert total_primary == older_count + 1
    current = max(all_primary, key=lambda r: r["created_at"])
    tool_results = [
        s["result"]
        for s in current["steps"]
        if s["kind"] == "tool_result" and s.get("tool_name") == SEARCH_RUN_LOG_TOOL_NAME
    ]
    assert tool_results, "expected a search_run_log tool_result step"
    text = tool_results[0]
    # The window holds MAX_CROSS_RUN_RUNS runs (current + the 9 most recent
    # older ones), all with real logs -- so all MAX_CROSS_RUN_RUNS are
    # SEARCHED. The remaining (extra + 1) older runs fall outside the
    # window and are reported via the exact `count_runs` total, not fetched.
    assert f"Searched {MAX_CROSS_RUN_RUNS} of {total_primary} run(s)" in text
    assert "could not be located" not in text
    assert f"{extra + 1} not attempted this call" in text


def test_load_records_is_deadline_aware(tmp_path, monkeypatch):
    """The stricter half of finding B: `load_records` itself enforces
    `deadline_seconds`, raising RunLogSearchTimeout rather than silently
    returning a partial (and indistinguishable-from-complete) record list.
    """
    from tldw_chatbook.Agents import run_log as run_log_module
    from tldw_chatbook.Agents.run_log_search import RunLogSearchTimeout, load_records

    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: tmp_path)
    log_dir = _write_run(run_log_module, "run-x", ["hello"])

    with pytest.raises(RunLogSearchTimeout):
        load_records(log_dir, deadline_seconds=0.0)

    # Unbounded (the default, and every call before task-1273) is
    # completely unaffected -- byte-identical to prior behaviour.
    assert len(load_records(log_dir)) == 1


def test_slow_load_is_recorded_as_not_searched_not_scanned_or_dropped(
    tmp_path, monkeypatch
):
    """Finding B, at the `search_across_runs` level: a run whose LOAD
    consumes the shared budget must land in `not_searched_run_ids` -- never
    silently scanned against a partial (truncated) record list (which would
    look like a complete search that found nothing), and never simply
    missing from every bucket (which would break the searched + unresolved
    + not_searched == total accounting the coverage line relies on).
    """
    from tldw_chatbook.Agents import run_log as run_log_module
    from tldw_chatbook.Agents import run_log_search as run_log_search_module
    from tldw_chatbook.Agents.run_log_search import (
        RunLogSearchTimeout,
        search_across_runs,
    )

    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: tmp_path)
    fast_dir = _write_run(run_log_module, "run-fast", ["marker hello"])
    slow_dir = _write_run(run_log_module, "run-slow", ["marker hello too"])

    real_load_records = run_log_search_module.load_records

    def fake_load_records(log_dir, *, deadline_seconds=None):
        if log_dir == slow_dir:
            # Simulate a load that alone exhausts whatever budget remained
            # -- the exact contract `load_records` itself now honours for
            # real (see test_load_records_is_deadline_aware above); faked
            # here so the test is deterministic rather than relying on
            # real slow disk I/O.
            raise RunLogSearchTimeout("simulated: load alone spent the budget")
        return real_load_records(log_dir, deadline_seconds=deadline_seconds)

    monkeypatch.setattr(run_log_search_module, "load_records", fake_load_records)

    result = search_across_runs(
        [("run-fast", fast_dir), ("run-slow", slow_dir)],
        current_run_id="run-fast",
        contains="marker",
    )
    # The fast run completed normally: found, searched, contributed a hit.
    assert result.searched_run_ids == ["run-fast"]
    assert len(result.hits) == 1
    assert result.hits[0].source_run_id == "run-fast"
    # The slow run: not searched (its load never finished), not unresolved
    # (its log WAS locatable -- resolved_runs handed in a real Path), and
    # contributed no hits despite genuinely containing a "marker" record.
    assert result.not_searched_run_ids == ["run-slow"]
    assert result.unresolved_run_ids == []
    assert all(h.source_run_id != "run-slow" for h in result.hits)
