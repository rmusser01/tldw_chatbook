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
    run_dirs = list((root / "agent-runs").iterdir())
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


def test_a_real_spawn_shares_the_parent_log_directory_and_counter(wired):
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

    run_dirs = [d for d in (root / "agent-runs").iterdir() if d.is_dir()]
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
    run_dirs = [d for d in (root / "agent-runs").iterdir() if d.is_dir()]
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
    from tldw_chatbook.Agents.agent_models import SEARCH_RUN_LOG_TOOL_NAME

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
