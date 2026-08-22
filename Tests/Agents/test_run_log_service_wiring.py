# Tests/Agents/test_run_log_service_wiring.py
"""The service owns the writer: one counter per run tree, every caller logged."""

import json
from pathlib import Path

import pytest

from tldw_chatbook.Agents import agent_service as agent_service_module
from tldw_chatbook.Agents import run_log as run_log_module
from tldw_chatbook.Agents.agent_models import (
    RUN_DONE,
    SPAWN_TOOL_NAME,
    AgentConfig,
    RunBudget,
)
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.run_log import RunLogWriter
from tldw_chatbook.Agents.run_log_format import iter_records
from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider, ToolCatalogRegistry
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


@pytest.fixture
def wired(tmp_path, monkeypatch):
    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: tmp_path)
    db = AgentRunsDB(tmp_path / "runs.db")
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    return db, registry, tmp_path


def chat_call_returning(text):
    def call(**kwargs):
        return {"choices": [{"message": {"content": text}}]}

    return call


def fence(name, args):
    return f"```tool_call\n{json.dumps({'name': name, 'arguments': args})}\n```"


def scripted_chat(replies):
    """Returns each reply from ``replies`` in order across successive calls."""
    remaining = list(replies)

    def call(**kwargs):
        return {"choices": [{"message": {"content": remaining.pop(0)}}]}

    return call


def all_records(run_dir: Path):
    records = []
    for segment in sorted(run_dir.glob("logs.*.txt")):
        records.extend(iter_records(segment.read_bytes()))
    return records


def read_all(root: Path):
    # TASK-1270: the writer now dots the directory name unconditionally
    # (both the sandbox-fallback and bound-workspace case), so this
    # fixture-path helper follows suit -- this is a location detail for
    # the test harness, not a security assertion (the security invariant
    # itself lives in test_run_log_sandbox_isolation.py /
    # test_run_log_workspace_isolation.py).
    run_dirs = list((root / ".agent-runs").iterdir())
    run_dirs = [d for d in run_dirs if d.is_dir()]
    assert len(run_dirs) == 1
    records = []
    for segment in sorted(run_dirs[0].glob("logs.*.txt")):
        records.extend(iter_records(segment.read_bytes()))
    return records


def test_a_plain_run_writes_records_without_the_caller_wiring_anything(wired):
    db, registry, root = wired
    service = AgentService(db, registry, chat_call=chat_call_returning("hello"))
    service.run_turn(
        conversation_id="conv1",
        messages=[{"role": "user", "content": "hi"}],
        config=AgentConfig(model="m", system_prompt="s", budget=RunBudget()),
        api_endpoint="openai",
    )
    records = read_all(root)
    assert [r.type for r in records] == ["model"]
    assert records[0].content == "hello"
    assert records[0].kind == "primary"


def test_record_numbers_are_unique_across_the_whole_run_tree(wired):
    db, registry, root = wired
    writer = RunLogWriter()
    service = AgentService(
        db, registry, chat_call=chat_call_returning("x"), run_log_writer=writer
    )
    service.run_turn(
        conversation_id="conv1",
        messages=[{"role": "user", "content": "hi"}],
        config=AgentConfig(model="m", system_prompt="s", budget=RunBudget()),
        api_endpoint="openai",
    )
    # Simulate a child appending through the same shared writer.
    writer.append(run_id="child", kind="subagent", type="model", content="child work")
    numbers = [r.number for r in read_all(root)]
    assert numbers == sorted(set(numbers))


def test_disabled_writer_leaves_the_run_untouched(wired, monkeypatch):
    db, registry, root = wired
    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: None)
    service = AgentService(db, registry, chat_call=chat_call_returning("hello"))
    _run_id, outcome = service.run_turn(
        conversation_id="conv1",
        messages=[{"role": "user", "content": "hi"}],
        config=AgentConfig(model="m", system_prompt="s", budget=RunBudget()),
        api_endpoint="openai",
    )
    assert outcome.final_text == "hello"
    assert not (root / "agent-runs").exists()


def test_a_real_spawn_shares_the_parent_log_directory_and_counter(wired, inline_spawns):
    """Round-1 review fix: the prior version of this suite only ever proved
    sharing by manually calling ``writer.append`` on the object the test
    already held -- a real ``spawn()`` never ran. A mutation giving
    non-primary runs their own fresh writer instance (see this file's git
    history / the fix report) still passed all 51 pre-fix tests. This test
    drives an ACTUAL sub-agent spawn through the fence tool-call protocol
    and checks the parent's and child's records land in one directory with
    one strictly-increasing, gap-free, duplicate-free number sequence.
    """
    db, registry, root = wired
    replies = [
        fence(SPAWN_TOOL_NAME, {"task": "child task"}),  # parent turn 1: spawns
        "child says hi",  # CHILD turn 1
        "parent final",  # parent turn 2
    ]
    service = AgentService(db, registry, chat_call=scripted_chat(replies))
    config = AgentConfig(
        model="m",
        system_prompt="s",
        allowed_tools=(SPAWN_TOOL_NAME,),
        budget=RunBudget(),
    )
    _run_id, outcome = service.run_turn(
        conversation_id="conv1",
        messages=[{"role": "user", "content": "delegate this"}],
        config=config,
        api_endpoint="llama_cpp",  # non-native: dispatches spawn via the fence
    )
    assert outcome.status == RUN_DONE and outcome.subagents_spawned == 1

    run_dirs = [d for d in (root / ".agent-runs").iterdir() if d.is_dir()]
    assert len(run_dirs) == 1, "parent and child must share ONE log directory"
    records = all_records(run_dirs[0])

    numbers = [r.number for r in records]
    assert numbers == list(range(1, len(numbers) + 1)), (
        "record numbers must be gap-free, duplicate-free, and strictly "
        f"increasing across the whole tree; got {numbers}"
    )
    kinds = {r.kind for r in records}
    assert kinds == {"primary", "subagent"}, (
        "both the parent's and the child's records must actually be present "
        f"-- got kinds {kinds}"
    )


def test_parent_spawn_tool_call_record_precedes_the_childs_own_records(
    wired, inline_spawns
):
    """F5 (Qodo #5, PR #1066 review): end-to-end counterpart of the pure-loop
    test in test_run_log_on_record.py, driven through a REAL nested spawn
    and the REAL RunLogWriter. `deps.spawn()` runs the child's ENTIRE loop
    inline before returning to the parent's dispatch of spawn_subagent, so
    before the F5 fix (both _emit_record calls at the content-assembly
    point, AFTER dispatch) the child's own model/tool_call/tool_result
    records -- written during that still-in-progress parent dispatch --
    landed in the log BEFORE the parent's own tool_call record that caused
    the spawn at all: backwards chronological order. After the fix, the
    parent's tool_call record for spawn_subagent (its record number is
    assigned BEFORE `deps.spawn()` runs) must precede every one of the
    child's records.
    """
    db, registry, root = wired
    replies = [
        fence(SPAWN_TOOL_NAME, {"task": "child task"}),  # parent turn 1: spawns
        "child says hi",  # CHILD turn 1
        "parent final",  # parent turn 2
    ]
    service = AgentService(db, registry, chat_call=scripted_chat(replies))
    config = AgentConfig(
        model="m",
        system_prompt="s",
        allowed_tools=(SPAWN_TOOL_NAME,),
        budget=RunBudget(),
    )
    _run_id, outcome = service.run_turn(
        conversation_id="conv1",
        messages=[{"role": "user", "content": "delegate this"}],
        config=config,
        api_endpoint="llama_cpp",  # non-native: dispatches spawn via the fence
    )
    assert outcome.status == RUN_DONE and outcome.subagents_spawned == 1

    run_dirs = [d for d in (root / ".agent-runs").iterdir() if d.is_dir()]
    assert len(run_dirs) == 1
    records = all_records(run_dirs[0])

    parent_spawn_call = next(
        r
        for r in records
        if r.kind == "primary" and r.type == "tool_call" and r.tool == SPAWN_TOOL_NAME
    )
    child_first = min(
        (r for r in records if r.kind == "subagent"), key=lambda r: r.number
    )
    assert parent_spawn_call.number < child_first.number, (
        "the parent's spawn_subagent tool_call record must be the FIRST "
        f"record of this spawn, not appear after the child's own records; "
        f"got parent={parent_spawn_call.number}, child first={child_first.number}"
    )


def test_on_record_returns_the_assigned_record_number(wired, monkeypatch):
    """Round-1 review fix: a mutation that calls ``append(...)`` and then
    ``return None`` instead of returning its result still passed all 51
    pre-fix tests -- nothing pinned the return value. A later task threads
    this number into a truncation trailer, so dropping it silently breaks
    that. This test captures the actual ``LoopDeps`` the service builds and
    calls its ``on_record`` directly, asserting the return is the writer's
    assigned (non-``None``) record number.
    """
    db, registry, root = wired
    captured: dict = {}
    real_run_agent_loop = agent_service_module.run_agent_loop

    def spy_run_agent_loop(config, messages, active, deps):
        captured["deps"] = deps
        return real_run_agent_loop(config, messages, active, deps)

    monkeypatch.setattr(agent_service_module, "run_agent_loop", spy_run_agent_loop)

    service = AgentService(db, registry, chat_call=chat_call_returning("hello"))
    service.run_turn(
        conversation_id="conv1",
        messages=[{"role": "user", "content": "hi"}],
        config=AgentConfig(model="m", system_prompt="s", budget=RunBudget()),
        api_endpoint="openai",
    )
    deps = captured["deps"]
    assert deps.on_record is not None
    result = deps.on_record("model", {"content": "probe"})
    assert isinstance(result, int) and result > 0, (
        f"on_record must return the writer's assigned record number, got {result!r}"
    )


def test_real_run_log_omits_sensitive_tool_args_and_results(wired, monkeypatch):
    db, registry, root = wired
    captured: dict = {}
    real_run_agent_loop = agent_service_module.run_agent_loop

    def spy_run_agent_loop(config, messages, active, deps):
        captured["deps"] = deps
        return real_run_agent_loop(config, messages, active, deps)

    monkeypatch.setattr(agent_service_module, "run_agent_loop", spy_run_agent_loop)
    service = AgentService(db, registry, chat_call=chat_call_returning("hello"))
    service.run_turn(
        conversation_id="conv1",
        messages=[{"role": "user", "content": "hi"}],
        config=AgentConfig(model="m", system_prompt="s", budget=RunBudget()),
        api_endpoint="openai",
    )
    on_record = captured["deps"].on_record
    sensitive = (
        "chain of thought: private internal plan",
        "chain-of-thought: private internal plan",
        "CHAIN_OF_THOUGHT: private internal plan",
        "<think>private internal plan</think>",
        "<THINK>private internal plan</THINK>",
        json.dumps({"reasoning": "private internal plan"}),
        json.dumps({"meta": {"reasoning_content": "private internal plan"}}),
        "{'chain_of_thought': 'private internal plan'}",
        "reasoning: private internal plan",
        "  reasoning_content = private internal plan",
        "meta:\n  chain_of_thought: private internal plan",
        "{reasoning: private internal plan}",
        "chain-of-thought = private internal plan",
        "ghp_" + "a" * 36,
        "AKIA" + "A" * 16,
        "eyJabcdefghij.abcdefghij.abcdefghij",
        "-----BEGIN PRIVATE KEY-----\nprivate-key-body",
        "/private/var/db/secrets.txt",
        "file:///private/tmp/secret.txt",
        r"C:\Users\alice\secret.txt",
        r"\\server\share\secret.txt",
        'File "package/module.py", line 42, in run',
        json.dumps({"meta": {"file_path": "/api/private"}}),
        "cat /docs/private/local",
        "open /help/private/local",
        "[" * 1_000 + '{"file_path":"/api/private"}' + "]" * 1_000,
    )
    for value in sensitive:
        assert on_record("tool_call", {"content": json.dumps({"value": value})}) is None
        assert on_record("tool_result", {"content": value}) is None
    safe_values = (
        "safe output: reasoning about three visible matches",
        "rendered HTML: <div>safe</div>",
    )
    for value in safe_values:
        assert isinstance(on_record("tool_result", {"content": value}), int)

    records = read_all(root)
    persisted = "\n".join(record.content for record in records)
    for value in sensitive:
        assert value not in persisted
    assert "private internal plan" not in persisted
    assert "private-key-body" not in persisted
    assert all(any(record.content == value for record in records) for value in safe_values)


def test_run_turn_called_twice_on_one_service_gets_two_separate_logs(wired):
    """Round-1 review fix (item 3): ``bind()`` latches permanently, so a
    writer built once in ``__init__`` and reused across two ``run_turn``
    calls on the same ``AgentService`` would silently append the second
    tree's records into the first tree's (already-bound) directory and
    overwrite its manifest. The writer must be constructed fresh per
    ``run_turn`` call (per spec §3.1) unless one was explicitly injected via
    the constructor.
    """
    db, registry, root = wired
    service = AgentService(db, registry, chat_call=chat_call_returning("hello"))
    config = AgentConfig(model="m", system_prompt="s", budget=RunBudget())
    service.run_turn(
        conversation_id="conv1",
        messages=[{"role": "user", "content": "first"}],
        config=config,
        api_endpoint="openai",
    )
    service.run_turn(
        conversation_id="conv1",
        messages=[{"role": "user", "content": "second"}],
        config=config,
        api_endpoint="openai",
    )
    run_dirs = [d for d in (root / ".agent-runs").iterdir() if d.is_dir()]
    assert len(run_dirs) == 2, "each run_turn call must get its own log directory"
    for run_dir in run_dirs:
        manifest = json.loads((run_dir / "MANIFEST").read_text())
        assert manifest["record_count"] == 1, (
            "the second tree's manifest must not include the first tree's "
            f"records; got {manifest}"
        )
        numbers = [r.number for r in all_records(run_dir)]
        assert numbers == [1], (
            "numbering must restart per tree, not continue across "
            f"run_turn calls; got {numbers}"
        )


def test_tool_is_offered_to_the_primary_agent_only(wired, monkeypatch):
    """Primary + at least one other disclosable schema (here: the builtin
    ``calculator``, allow-listed so it lands in ``active``) -> offered.

    Controller ruling (post-review of the original spec): search_run_log is
    additionally gated on the run having at least one OTHER disclosable
    schema (``runtime_schemas or active`` non-empty), mirroring every other
    runtime tool's own gate (spawn_subagent on max_subagents>0, find_tools/
    load_tools on offer_find_load, skill_file on a non-empty authorized
    set). This test now allow-lists ``calculator`` so ``active`` is
    non-empty and the assertion is exercising "primary vs subagent",
    exactly as it did before that ruling -- not the separate "nothing else
    disclosed" case, which ``test_tool_is_not_offered_when_nothing_else_is_disclosed``
    below covers.
    """
    from tldw_chatbook.Agents.agent_models import SEARCH_RUN_LOG_TOOL_NAME

    db, registry, root = wired
    offered = []

    def capture(**kwargs):
        names = [t["function"]["name"] for t in kwargs.get("tools", [])]
        offered.append(names)
        return {"choices": [{"message": {"content": "ok"}}]}

    service = AgentService(db, registry, chat_call=capture)
    service.run_turn(
        conversation_id="conv1",
        messages=[{"role": "user", "content": "hi"}],
        config=AgentConfig(
            model="m",
            system_prompt="s",
            allowed_tools=("calculator",),
            budget=RunBudget(max_subagents=0),
        ),
        api_endpoint="openai",
    )
    assert any(SEARCH_RUN_LOG_TOOL_NAME in names for names in offered)


def test_tool_is_not_offered_when_nothing_else_is_disclosed(wired, monkeypatch):
    """Controller ruling: an otherwise-tool-less primary run (empty
    allow-list, max_subagents=0, no skills/install/run-script wiring) must
    NOT be offered search_run_log even though the writer is active.

    Such a run can only ever produce model-turn log records -- it has no
    tool results, so nothing was ever truncated and there is nothing to
    recover -- so the tool would buy it nothing while changing the
    provider payload of a deliberately tool-less run (task-243 minor m3:
    a native-capable endpoint with no disclosable schemas must send no
    ``tools=`` kwarg at all).
    """
    db, registry, root = wired
    offered = []

    def capture(**kwargs):
        offered.append(kwargs)
        return {"choices": [{"message": {"content": "ok"}}]}

    service = AgentService(db, registry, chat_call=capture)
    service.run_turn(
        conversation_id="conv1",
        messages=[{"role": "user", "content": "hi"}],
        config=AgentConfig(
            model="m",
            system_prompt="s",
            budget=RunBudget(max_subagents=0),
        ),
        api_endpoint="openai",
    )
    assert len(offered) == 1
    assert "tools" not in offered[0], (
        "a deliberately tool-less run must never gain search_run_log as its "
        f"sole disclosed tool; got kwargs {offered[0]!r}"
    )


# -- TASK-16788: the allow-list governs the CATALOG, not the runtime layer ---
#
# Recorded decision (Docs/superpowers/specs/2026-08-16-expansion-residue-
# design.md): the run-log tools are the same family as spawn_subagent /
# find_tools / load_tools / skill_file, ALL of which are appended to
# `runtime_schemas` after `_run_one`'s allow-list filter and dispatched by
# `run_agent_loop` in dedicated branches before `invoke_tool` can apply that
# filter. Filtering only the run-log tools would make one runtime tool
# behave unlike its family; filtering the whole family would break skills
# and sub-agents. So the behaviour is DOCUMENTED (on
# `AgentConfig.allowed_tools`) and pinned here, red if someone later
# "fixes" it silently. The confound this cost a real experiment is recorded
# in Docs/superpowers/qa/2026-08-15-rag-agentic-expansion/report.md.


def test_run_log_tools_are_offered_under_an_empty_allow_list(wired):
    """An EMPTY `allowed_tools` still gets the run-log tools.

    The two halves are asserted in the same run, so the test cannot pass by
    the allow-list silently having no effect at all: every offered name must
    be a RUNTIME tool (nothing from the catalog survived the filter -- see
    `test_tool_is_offered_to_the_primary_agent_only` for the same harness
    with `calculator` allow-listed and offered), AND all three run-log
    schemas must be present.

    `log_active` is satisfied the ordinary way: primary agent, an active
    writer (the `wired` fixture points `resolve_log_root` at tmp_path), and
    a non-empty `runtime_schemas` -- here the spawn schema, which the
    default `max_subagents` admits regardless of the allow-list too.
    """
    from tldw_chatbook.Agents.agent_models import (
        RUN_LOG_SLICE_TOOL_NAME,
        RUN_LOG_STATS_TOOL_NAME,
        RUNTIME_TOOL_NAMES,
        SEARCH_RUN_LOG_TOOL_NAME,
    )

    db, registry, _root = wired
    offered = []

    def capture(**kwargs):
        offered.append([t["function"]["name"] for t in kwargs.get("tools", [])])
        return {"choices": [{"message": {"content": "ok"}}]}

    service = AgentService(db, registry, chat_call=capture)
    service.run_turn(
        conversation_id="conv1",
        messages=[{"role": "user", "content": "hi"}],
        config=AgentConfig(
            model="m",
            system_prompt="s",
            allowed_tools=(),  # nothing from the catalog is permitted
            budget=RunBudget(),
        ),
        api_endpoint="openai",  # native: the offered set IS the tools= kwarg
    )
    assert len(offered) == 1
    names = offered[0]
    assert set(names) <= RUNTIME_TOOL_NAMES, (
        "an empty allow-list must leave the catalog half empty; offered "
        f"{sorted(set(names) - RUNTIME_TOOL_NAMES)} anyway"
    )
    assert SEARCH_RUN_LOG_TOOL_NAME in names
    assert RUN_LOG_STATS_TOOL_NAME in names
    assert RUN_LOG_SLICE_TOOL_NAME in names


def test_a_run_log_call_dispatches_although_the_allow_list_is_empty():
    """The other half of the contract: the CALL is not caught later either.

    `invoke_tool` refuses any name outside `config.allowed_tools`, but a
    run-log call never reaches it -- `run_agent_loop` has a dedicated branch
    for each run-log name ahead of the generic fallback. Pinned with an
    explicit empty allow-list and an `invoke_tool` that records every call
    it is handed: the injected handler must run and `invoke_tool` must stay
    untouched.
    """
    from tldw_chatbook.Agents.agent_models import (
        SEARCH_RUN_LOG_TOOL_NAME,
        ModelTurn,
        ToolCall,
        ToolResult,
    )
    from tldw_chatbook.Agents.agent_runtime import run_agent_loop

    from Tests.Agents.test_agent_runtime import make_deps

    handled = []
    fell_through = []

    def handler(args):
        handled.append(dict(args))
        return ToolResult(ok=True, content="record 000412 [model]")

    def invoke(call):
        fell_through.append(call.name)
        return ToolResult(ok=False, error=f"Tool not permitted: {call.name}")

    deps = make_deps(
        [
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
        ],
        invoke=invoke,
    )
    deps.search_run_log = handler
    outcome = run_agent_loop(
        AgentConfig(
            model="m",
            system_prompt="s",
            allowed_tools=(),  # explicitly empty, not merely defaulted
            budget=RunBudget(max_steps=8, max_model_turns=8),
        ),
        [{"role": "user", "content": "go"}],
        [],
        deps,
    )
    assert handled == [{"contains": "refused"}]
    assert fell_through == [], (
        "a run-log call must dispatch in its own branch, never through "
        f"invoke_tool's allow-list check; got {fell_through}"
    )
    assert outcome.final_text == "answered"
