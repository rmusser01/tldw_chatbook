# Tests/Agents/test_fleet_runtime.py
"""Fleet runtime: threaded sub-agents + ``wait_agents``/``check_agents``.

PR2a Task 6. These exercise the CONCURRENT spawn path, which is opt-in:
a run is threaded only when the service was handed a ``FleetCoordinator``
(or ``[agents] max_live_subagents`` is raised above 1). Every test here
therefore builds its service with an explicit coordinator; the inline
path -- unchanged, byte-identical, and guarded by the pre-existing
``Tests/Agents/test_agent_service.py`` spawn suite -- is re-asserted once
here too (``test_without_a_coordinator_spawn_stays_inline``) so a
regression in the gate itself is caught in this file rather than only in
the older one.

Scripting note: ``test_agent_service.ScriptedChat`` pops one shared
ordered list, which stops being deterministic the moment children run on
their own threads (whichever thread wins the race takes the next reply).
``FleetChat`` below keeps the same shape but ADDRESSES replies -- an
ordered script for the parent, a per-task script for each child -- so
every assertion here is deterministic under real threads.
"""

import threading
import time

import pytest

from tldw_chatbook.Agents.agent_models import (
    RUN_CANCELLED,
    RUN_DONE,
    RUN_ERROR,
    SPAWN_TOOL_NAME,
    TERMINAL_RUN_STATUSES,
    AgentConfig,
    AgentDefinition,
    RunBudget,
    ToolCatalogEntry,
    ToolResult,
    ToolSchema,
)
from tldw_chatbook.Agents.agent_models import (
    CHECK_AGENTS_TOOL_NAME,
    FENCE_TOOL_RESULT_PREFIX,
    RUNTIME_TOOL_NAMES,
    WAIT_AGENTS_TOOL_NAME,
)
from tldw_chatbook.Agents import agent_service
from tldw_chatbook.Agents.agent_service import (
    SUBAGENT_SYSTEM_PROMPT,
    AgentService,
)
from tldw_chatbook.Agents.fleet_coordinator import FleetCoordinator
from tldw_chatbook.Agents.run_context import current_run_id
from tldw_chatbook.Agents.tool_catalog import (
    CHECK_AGENTS_SCHEMA,
    WAIT_AGENTS_SCHEMA,
    BuiltinToolProvider,
    SkillToolProvider,
    ToolCatalogRegistry,
)
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

from Tests.Agents.test_agent_service import ScriptedChat, fence, provider_reply

# The child's system prompt is the sub-agent prompt (plus, for a named
# agent, its instructions) followed by the rendered fence protocol -- so a
# prefix match on its first sentence is what identifies a child's provider
# call, the same identity contract `console_agent_bridge._is_subagent`
# relies on.
_SUBAGENT_PREFIX = SUBAGENT_SYSTEM_PROMPT.split(".")[0]

_JOIN_TIMEOUT = 5.0


def _child_task(payload: list[dict]) -> str | None:
    """The task text of the child whose provider call this payload is.

    Args:
        payload: The ``messages_payload`` handed to ``chat_api_call``.

    Returns:
        The child's task text (its first user message), or ``None`` when
        this payload belongs to the primary agent.
    """
    if not payload:
        return None
    system = payload[0]
    if system.get("role") != "system":
        return None
    if not str(system.get("content", "")).startswith(_SUBAGENT_PREFIX):
        return None
    for message in payload[1:]:
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return None


class FleetChat:
    """Addressed scripted provider: one script per agent, not one queue.

    Replies may be plain strings/dicts (as ``ScriptedChat``) or zero-arg
    callables, which are invoked at call time -- that is how a test gates a
    child on an ``Event`` or a ``Barrier`` while the parent keeps running.
    """

    def __init__(self, parent_replies, child_replies=None):
        self.parent_replies = list(parent_replies)
        self.child_replies = {
            task: list(script) for task, script in (child_replies or {}).items()
        }
        self.calls: list[dict] = []
        self.child_calls: dict[str, list[dict]] = {}
        self._lock = threading.Lock()

    def __call__(self, **kwargs):
        payload = kwargs["messages_payload"]
        task = _child_task(payload)
        with self._lock:
            self.calls.append(kwargs)
            if task is None:
                assert self.parent_replies, "parent script exhausted"
                item = self.parent_replies.pop(0)
            else:
                self.child_calls.setdefault(task, []).append(kwargs)
                script = self.child_replies.get(task)
                assert script, f"no scripted reply left for child task {task!r}"
                item = script.pop(0)
        # Called OUTSIDE the lock: a gated reply blocks here, and holding
        # the lock would serialize the very concurrency under test.
        if callable(item):
            item = item()
        return provider_reply(item)


class RunIdProbeProvider:
    """A one-tool provider that records the run id bound at invoke time.

    The PR2a Task 5 carry-forward guard: a tool dispatched from a child's
    own thread must see the CHILD's run id, or every approval stamp the
    gates keyed by run becomes unreachable and an approved tool fails
    closed.
    """

    def __init__(self):
        self.seen: list[str] = []
        self._lock = threading.Lock()

    def list_catalog(self):
        return [
            ToolCatalogEntry(
                id="probe:whoami",
                name="whoami",
                one_line_description="Report the dispatching run id.",
                source="probe",
            )
        ]

    def load_schema(self, tool_id):
        return ToolSchema(
            id="probe:whoami",
            name="whoami",
            description="Report the dispatching run id.",
            parameters={"type": "object", "properties": {}},
        )

    def invoke(self, tool_id, args):
        run_id = current_run_id()
        with self._lock:
            self.seen.append(run_id)
        return ToolResult(ok=True, content=run_id or "<none>")


@pytest.fixture()
def db(tmp_path):
    return AgentRunsDB(tmp_path / "runs.db", client_id="test")


def make_fleet_service(
    db,
    parent_replies,
    child_replies=None,
    max_live=3,
    providers=(),
    revoke_approvals=None,
    review_tool_calls=None,
):
    """An AgentService wired for the fleet (explicit coordinator = opt in)."""
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    for provider in providers:
        registry.register_provider(provider)
    chat = FleetChat(parent_replies, child_replies)
    coordinator = FleetCoordinator(max_live=max_live, clock=time.monotonic)
    service = AgentService(
        db=db,
        registry=registry,
        chat_call=chat,
        fleet_coordinator=coordinator,
        revoke_approvals=revoke_approvals,
        review_tool_calls=review_tool_calls,
    )
    return service, chat, coordinator


def make_inline_service(db, replies):
    """An AgentService with NO coordinator -- the pre-PR inline path.

    Uses the ORDINARY single-queue ``ScriptedChat``: with no fleet there
    are no threads, so strict reply ordering is exactly what should still
    hold -- and asserting against it is the point.
    """
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    chat = ScriptedChat(replies)
    return AgentService(db=db, registry=registry, chat_call=chat), chat


# max_steps/max_model_turns raised off the 8-step default: a fleet turn
# spends 3 steps per round (model + tool_call + tool_result) and these
# scripts run several rounds. max_subagents is raised above the fleet cap
# under test so the CAP, not the per-turn spawn budget, is what refuses.
FLEET_CFG = AgentConfig(
    model="test-model",
    system_prompt="You are helpful.",
    allowed_tools=("calculator", "get_current_datetime", SPAWN_TOOL_NAME),
    budget=RunBudget(max_steps=40, max_model_turns=40, max_subagents=4),
)


def _tool_results(run: dict, tool_name: str) -> list[str]:
    """That tool's results as recorded in the run's STEP log.

    Note the step log caps each result at 2000 chars, so this is only
    usable for short-substring assertions -- anything about result SIZE
    must read history via ``_history_tool_results`` instead.
    """
    return [
        step["result"]
        for step in run["steps"]
        if step["kind"] == "tool_result" and step["tool_name"] == tool_name
    ]


def _history_tool_results(chat: FleetChat, tool_name: str) -> list[str]:
    """That tool's results exactly as they entered the model's history.

    This is the seam the result budget is enforced at
    (``agent_runtime._truncate_tool_result``), so it is the only honest
    place to assert what the supervisor actually received.
    """
    prefix = f"{FENCE_TOOL_RESULT_PREFIX}{tool_name}: "
    payload = chat.calls[-1]["messages_payload"]
    return [
        str(message["content"])[len(prefix) :]
        for message in payload
        if str(message.get("content", "")).startswith(prefix)
    ]


# -- registration (the runtime-tool mandate: name + set + schema) ---------


def test_both_fleet_tools_are_registered_runtime_tools():
    assert WAIT_AGENTS_TOOL_NAME == "wait_agents"
    assert CHECK_AGENTS_TOOL_NAME == "check_agents"
    assert WAIT_AGENTS_TOOL_NAME in RUNTIME_TOOL_NAMES
    assert CHECK_AGENTS_TOOL_NAME in RUNTIME_TOOL_NAMES
    assert WAIT_AGENTS_SCHEMA.name == WAIT_AGENTS_TOOL_NAME
    assert CHECK_AGENTS_SCHEMA.name == CHECK_AGENTS_TOOL_NAME
    # `ids` is OPTIONAL: an omitted `ids` means "every child".
    assert WAIT_AGENTS_SCHEMA.parameters.get("required", []) == []
    assert WAIT_AGENTS_SCHEMA.parameters["properties"]["ids"]["type"] == "array"
    assert CHECK_AGENTS_SCHEMA.parameters["properties"] == {}


# -- the core concurrency behaviours --------------------------------------


def test_two_children_run_concurrently_and_wait_collects_both(db):
    # A 2-party barrier is the concurrency PROOF: if the children were
    # still serialized, the first would block on it until the timeout and
    # break the barrier, failing the run instead of answering.
    barrier = threading.Barrier(2, timeout=_JOIN_TIMEOUT)

    def gated(text):
        def reply():
            barrier.wait()
            return text

        return reply

    service, chat, coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "task one"}),
            fence(SPAWN_TOOL_NAME, {"task": "task two"}),
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "combined answer",
        ],
        {
            "task one": [gated("answer one")],
            "task two": [gated("answer two")],
        },
    )
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=FLEET_CFG,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    assert outcome.final_text == "combined answer"
    assert db.count_subagent_runs("c") == 2
    # Both children's results reached the parent's wait result.
    wait_results = _tool_results(db.get_run(run_id), WAIT_AGENTS_TOOL_NAME)
    assert len(wait_results) == 1
    assert "answer one" in wait_results[0]
    assert "answer two" in wait_results[0]
    # ... and therefore into the payload of the parent's final call.
    final_payload = str(chat.calls[-1]["messages_payload"])
    assert "answer one" in final_payload and "answer two" in final_payload
    assert coordinator.all_finished()
    assert [h.status for h in coordinator.snapshot()] == [RUN_DONE, RUN_DONE]


def test_spawn_returns_handle_without_blocking(db):
    """The spawn tool result names a handle, not the child's answer."""
    release = threading.Event()

    def blocked_child():
        assert release.wait(_JOIN_TIMEOUT)
        return "answer one"

    def final_answer():
        release.set()
        return "done"

    service, _chat, coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "task one"}),
            # Still running here -- the parent got control straight back.
            fence(CHECK_AGENTS_TOOL_NAME, {}),
            final_answer,
        ],
        {"task one": [blocked_child]},
    )
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=FLEET_CFG,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    spawn_results = _tool_results(db.get_run(run_id), SPAWN_TOOL_NAME)
    assert len(spawn_results) == 1
    handle_id = coordinator.snapshot()[0].handle_id
    assert spawn_results[0].startswith("started ")
    assert handle_id in spawn_results[0]
    assert "task one" in spawn_results[0]
    # The child's answer is NOT what spawn returned.
    assert "answer one" not in spawn_results[0]
    # The check_agents call ran while the child was still blocked, proving
    # the parent kept control rather than waiting on the child.
    check_results = _tool_results(db.get_run(run_id), CHECK_AGENTS_TOOL_NAME)
    assert check_results and "running" in check_results[0]


def test_check_agents_reports_status_without_blocking(db):
    release = threading.Event()

    def blocked_child():
        assert release.wait(_JOIN_TIMEOUT)
        return "child answer"

    def release_then_wait():
        release.set()
        return fence(WAIT_AGENTS_TOOL_NAME, {})

    service, _chat, coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "slow task"}),
            fence(CHECK_AGENTS_TOOL_NAME, {}),  # while the child is blocked
            release_then_wait,
            fence(CHECK_AGENTS_TOOL_NAME, {}),  # after it finished
            "done",
        ],
        {"slow task": [blocked_child]},
    )
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=FLEET_CFG,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    checks = _tool_results(db.get_run(run_id), CHECK_AGENTS_TOOL_NAME)
    assert len(checks) == 2
    handle_id = coordinator.snapshot()[0].handle_id
    assert handle_id in checks[0] and "running" in checks[0]
    assert "slow task" in checks[0]
    # check_agents never blocks: the first snapshot was taken while the
    # child was still gated, and it did NOT contain the child's answer.
    assert "child answer" not in checks[0]
    assert handle_id in checks[1] and RUN_DONE in checks[1]


def test_live_cap_refuses_beyond_max_live_subagents(db):
    release = threading.Event()

    def blocked_child():
        assert release.wait(_JOIN_TIMEOUT)
        return "answer one"

    def final_answer():
        release.set()
        return "done"

    service, _chat, coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "task one"}),
            fence(SPAWN_TOOL_NAME, {"task": "task two"}),  # refused: at cap
            final_answer,
        ],
        {"task one": [blocked_child]},
        max_live=1,
    )
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=FLEET_CFG,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    spawn_results = _tool_results(db.get_run(run_id), SPAWN_TOOL_NAME)
    assert len(spawn_results) == 2
    assert spawn_results[0].startswith("started ")
    assert "ERROR" in spawn_results[1]
    assert "wait_agents" in spawn_results[1]  # tells the model how to recover
    # The refused spawn created no run and reserved no handle.
    assert db.count_subagent_runs("c") == 1
    assert len(coordinator.snapshot()) == 1


def test_live_cap_refusal_does_not_burn_the_per_turn_spawn_budget(db):
    """A cap refusal is retryable, so it must not consume a spawn slot.

    The per-turn ceiling here is 2 spawns; the fleet cap is 1. Three spawn
    calls are made and the middle one is refused at the cap -- if that
    refusal consumed a slot, the third (made after the first child was
    collected) would hit "sub-agent budget exhausted" instead of starting.

    Named spawns, deliberately: ``run_agent_loop`` keeps its own redundant
    secondary counter, which for an UNNAMED spawn increments even on a
    refusal (pre-existing, byte-identical behaviour this PR does not
    touch). Only the named path lets the service's authoritative counter
    be observed on its own.
    """
    db.create_agent_definition(
        AgentDefinition(
            name="researcher",
            description="Searches.",
            instructions="Cite sources.",
        )
    )
    release_one = threading.Event()

    def first_child():
        assert release_one.wait(_JOIN_TIMEOUT)
        return "answer one"

    def release_then_wait():
        release_one.set()
        return fence(WAIT_AGENTS_TOOL_NAME, {})

    tight = AgentConfig(
        model="test-model",
        system_prompt="You are helpful.",
        allowed_tools=(SPAWN_TOOL_NAME,),
        budget=RunBudget(max_steps=40, max_model_turns=40, max_subagents=2),
    )
    service, _chat, _coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "task one", "agent": "researcher"}),
            # Refused at the cap -- no run, no handle, no spawn slot.
            fence(SPAWN_TOOL_NAME, {"task": "task two", "agent": "researcher"}),
            release_then_wait,
            fence(SPAWN_TOOL_NAME, {"task": "task three", "agent": "researcher"}),
            "done",
        ],
        {"task one": [first_child], "task three": ["answer three"]},
        max_live=1,
    )
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=tight,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    spawn_results = _tool_results(db.get_run(run_id), SPAWN_TOOL_NAME)
    assert len(spawn_results) == 3
    assert spawn_results[0].startswith("started ")
    assert "wait_agents" in spawn_results[1]  # the cap refusal
    assert spawn_results[2].startswith("started ")  # NOT budget-exhausted
    assert "budget exhausted" not in spawn_results[2]
    assert db.count_subagent_runs("c") == 2


def test_end_of_turn_waits_for_stragglers(db):
    """The turn must not return with a child still running."""
    started = threading.Event()

    def slow_child():
        started.set()
        time.sleep(0.2)
        return "late answer"

    service, _chat, coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "slow task"}),
            "parent answered early",  # never calls wait_agents
        ],
        {"slow task": [slow_child]},
    )
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=FLEET_CFG,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    assert outcome.final_text == "parent answered early"
    assert started.is_set()
    # No handle and no run row may be left live once run_turn returns.
    assert coordinator.all_finished()
    assert [h.status for h in coordinator.snapshot()] == [RUN_DONE]
    rows = db.list_runs("c", include_superseded=True)
    assert rows and all(row["status"] in TERMINAL_RUN_STATUSES for row in rows)
    child = next(row for row in rows if row["agent_kind"] == "subagent")
    assert child["status"] == RUN_DONE
    assert child["result"] == "late answer"
    assert child["parent_run_id"] == run_id


def test_end_of_turn_cancels_and_abandons_a_wedged_child(db):
    """A child that never returns is cancelled, abandoned, and recorded.

    The parent's wall-clock budget is 1s, so the end-of-turn wait expires
    almost immediately; the child ignores its cooperative cancel entirely
    (it is blocked inside the provider call), so it must be abandoned after
    the join timeout with its handle AND its run row marked cancelled --
    never left ``running``.
    """
    never = threading.Event()
    entered = threading.Event()

    def wedged_child():
        entered.set()
        never.wait(30.0)  # released in the finally below, not by the run
        return "unreachable"

    short = AgentConfig(
        model="test-model",
        system_prompt="You are helpful.",
        allowed_tools=(SPAWN_TOOL_NAME,),
        budget=RunBudget(
            max_steps=40,
            max_model_turns=40,
            max_subagents=2,
            max_wall_seconds=1.0,
        ),
    )
    service, _chat, coordinator = make_fleet_service(
        db,
        [fence(SPAWN_TOOL_NAME, {"task": "wedged"}), "parent done"],
        {"wedged": [wedged_child]},
    )
    try:
        _run_id, outcome = service.run_turn(
            conversation_id="c",
            messages=[{"role": "user", "content": "go"}],
            config=short,
            api_endpoint="llama_cpp",
        )
        assert entered.is_set()
        assert outcome.status == RUN_DONE
        assert coordinator.all_finished()
        assert [h.status for h in coordinator.snapshot()] == [RUN_CANCELLED]
        rows = db.list_runs("c", include_superseded=True)
        assert all(row["status"] in TERMINAL_RUN_STATUSES for row in rows)
        child = next(row for row in rows if row["agent_kind"] == "subagent")
        assert child["status"] == RUN_CANCELLED
    finally:
        never.set()


def test_single_child_under_a_live_fleet_routes_through_wait_agents(db):
    """One child, fleet ON: same run rows as inline, result via wait_agents.

    Deliberately NOT named "path is unchanged": under a live fleet the
    path is NOT unchanged. The run ROWS match the inline path exactly
    (lineage, task, status, result, clean context), but the supervisor
    reaches the result through an extra wait_agents call instead of
    getting it back from spawn. The byte-identical acceptance criterion
    is guarded by ``test_without_a_coordinator_spawn_stays_inline`` below
    and by the untouched pre-existing spawn suites -- not by this test.
    """
    service, chat, _coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "compute 6*7"}),
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "The sub-agent says 42.",
        ],
        {"compute 6*7": ["sub answer: 42"]},
    )
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "delegate this"}],
        config=FLEET_CFG,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    assert outcome.final_text == "The sub-agent says 42."
    assert outcome.subagents_spawned == 1
    # Same run rows as the inline path (cf.
    # test_agent_service.test_spawn_creates_linked_child_with_clean_context).
    runs = db.list_runs("c")
    child = next(r for r in runs if r["agent_kind"] == "subagent")
    assert child["parent_run_id"] == run_id
    assert child["task"] == "compute 6*7"
    assert child["status"] == RUN_DONE and child["result"] == "sub answer: 42"
    assert db.count_subagent_runs("c") == 1
    # Same clean context: the child saw only its task + its own prompt.
    child_call = chat.child_calls["compute 6*7"][0]["messages_payload"]
    assert child_call[0]["role"] == "system"
    assert child_call[0]["content"].startswith(_SUBAGENT_PREFIX)
    assert child_call[1] == {"role": "user", "content": "compute 6*7"}
    assert not any("delegate this" in m["content"] for m in child_call)
    # Same result text reaches the parent -- via wait_agents rather than
    # via the spawn call's own return value.
    wait_results = _tool_results(db.get_run(run_id), WAIT_AGENTS_TOOL_NAME)
    assert wait_results and "sub answer: 42" in wait_results[0]


def test_without_a_coordinator_spawn_stays_inline(db):
    """No coordinator => the pre-PR inline path, result returned by spawn.

    This is the gate that keeps the existing spawn suites byte-identical:
    the fleet is opt-in, and an un-opted-in service must not thread.
    """
    service, _chat = make_inline_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "compute 6*7"}),
            "sub answer: 42",  # consumed by the child, INLINE and in order
            "The sub-agent says 42.",
        ],
    )
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "delegate this"}],
        config=FLEET_CFG,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    spawn_results = _tool_results(db.get_run(run_id), SPAWN_TOOL_NAME)
    # The spawn call itself returned the child's answer -- no handle.
    assert spawn_results == ["sub answer: 42"]
    child = next(r for r in db.list_runs("c") if r["agent_kind"] == "subagent")
    assert child["result"] == "sub answer: 42"


def test_fleet_tools_are_not_offered_without_a_coordinator(db):
    service, chat = make_inline_service(db, ["just answering"])
    service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=FLEET_CFG,
        api_endpoint="llama_cpp",
    )
    system_prompt = chat.calls[0]["messages_payload"][0]["content"]
    assert WAIT_AGENTS_TOOL_NAME not in system_prompt
    assert CHECK_AGENTS_TOOL_NAME not in system_prompt


def test_fleet_tools_are_primary_only(db):
    """A child never receives wait_agents/check_agents (isolation)."""
    service, chat, _coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "child task"}),
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "done",
        ],
        {
            "child task": [
                fence(WAIT_AGENTS_TOOL_NAME, {}),  # the child tries anyway
                "child recovered",
            ]
        },
    )
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=FLEET_CFG,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    parent_prompt = chat.calls[0]["messages_payload"][0]["content"]
    assert WAIT_AGENTS_TOOL_NAME in parent_prompt
    assert CHECK_AGENTS_TOOL_NAME in parent_prompt
    child_prompt = chat.child_calls["child task"][0]["messages_payload"][0][
        "content"
    ]
    assert WAIT_AGENTS_TOOL_NAME not in child_prompt
    assert CHECK_AGENTS_TOOL_NAME not in child_prompt
    # The child's hallucinated call falls through to the ordinary
    # permission path, exactly like any other undisclosed tool name.
    child = next(r for r in db.list_runs("c") if r["agent_kind"] == "subagent")
    refusals = _tool_results(child, WAIT_AGENTS_TOOL_NAME)
    assert refusals and "not permitted" in refusals[0]
    assert db.get_run(run_id)["status"] == RUN_DONE


# -- the config switch: the ONLY production-reachable path ----------------
#
# Every other test in this file injects a coordinator. In production
# nothing does (yet), so `[agents] max_live_subagents` -> `run_turn`'s
# `max_live > 1` branch is the only way a real user turns the fleet on --
# and it is the exact line the eventual default-flip will change.


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("3", 3),
        (3, 3),
        # A TOML float must not silently disable the fleet.
        (3.0, 3),
        ("2.9", 2),
        ("1", 1),
        (0, 1),
        (-5, 1),
        # Unparseable -> the documented default, never a raise.
        ("plenty", agent_service.DEFAULT_MAX_LIVE_SUBAGENTS),
        (None, agent_service.DEFAULT_MAX_LIVE_SUBAGENTS),
        ("", agent_service.DEFAULT_MAX_LIVE_SUBAGENTS),
        (float("inf"), agent_service.DEFAULT_MAX_LIVE_SUBAGENTS),
        (float("nan"), agent_service.DEFAULT_MAX_LIVE_SUBAGENTS),
    ],
)
def test_coerce_max_live_subagents(configured, expected):
    assert agent_service._coerce_max_live_subagents(configured) == expected


def _patch_max_live(monkeypatch, value):
    """Make `[agents] max_live_subagents` read as `value`.

    Every OTHER `_setting` key (the run-log eviction knobs read by
    `_make_call_model`) must keep returning its own default, or this
    fixture would silently reconfigure unrelated behaviour.
    """
    real_key = agent_service.MAX_LIVE_SUBAGENTS_KEY

    def fake_setting(key, default):
        return value if key == real_key else default

    monkeypatch.setattr(agent_service, "_setting", fake_setting)


def test_config_above_one_builds_a_fleet_and_threads_spawns(db, monkeypatch):
    """`max_live_subagents = 3` with NO injected coordinator: fleet ON."""
    _patch_max_live(monkeypatch, "3")
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    chat = FleetChat(
        [
            fence(SPAWN_TOOL_NAME, {"task": "task one"}),
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "done",
        ],
        {"task one": ["answer one"]},
    )
    service = AgentService(db=db, registry=registry, chat_call=chat)
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=FLEET_CFG,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    assert service._fleet is not None
    # Threaded, not inline: spawn handed back a handle ...
    spawn_results = _tool_results(db.get_run(run_id), SPAWN_TOOL_NAME)
    assert spawn_results and spawn_results[0].startswith("started ")
    # ... the tools were offered ...
    system_prompt = chat.calls[0]["messages_payload"][0]["content"]
    assert WAIT_AGENTS_TOOL_NAME in system_prompt
    assert CHECK_AGENTS_TOOL_NAME in system_prompt
    # ... and the result still came back.
    waits = _tool_results(db.get_run(run_id), WAIT_AGENTS_TOOL_NAME)
    assert waits and "answer one" in waits[0]


@pytest.mark.parametrize("configured", ["1", 1, 0, "nonsense", None])
def test_config_of_one_or_junk_keeps_the_inline_path(db, monkeypatch, configured):
    """Anything that resolves to <= 1 -- including junk -- means no fleet.

    This is the shipped default, and the guarantee the pre-existing spawn
    suites rely on: no coordinator, no fleet tools, spawn returns the
    child's own answer.
    """
    _patch_max_live(monkeypatch, configured)
    service, chat = make_inline_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "compute 6*7"}),
            "sub answer: 42",
            "The sub-agent says 42.",
        ],
    )
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=FLEET_CFG,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    assert service._fleet is None
    spawn_results = _tool_results(db.get_run(run_id), SPAWN_TOOL_NAME)
    assert spawn_results == ["sub answer: 42"]
    system_prompt = chat.calls[0]["messages_payload"][0]["content"]
    assert WAIT_AGENTS_TOOL_NAME not in system_prompt
    assert CHECK_AGENTS_TOOL_NAME not in system_prompt


def test_injected_coordinator_wins_over_the_config(db, monkeypatch):
    """An injected coordinator is the opt-in; config sizing is the fallback."""
    _patch_max_live(monkeypatch, "1")
    service, _chat, coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "task one"}),
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "done",
        ],
        {"task one": ["answer one"]},
    )
    _run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=FLEET_CFG,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    assert service._fleet is coordinator
    assert len(coordinator.snapshot()) == 1


# -- result budgeting (spec §5) -------------------------------------------


def test_wait_agents_splits_the_history_budget_across_children(db):
    """5 children x 4000 chars must not blow (or be cut by) the 16k cap."""
    bodies = {f"task {i}": ["Z" * 4000] for i in range(1, 6)}
    parent = [fence(SPAWN_TOOL_NAME, {"task": f"task {i}"}) for i in range(1, 6)]
    parent += [fence(WAIT_AGENTS_TOOL_NAME, {}), "done"]
    cfg = AgentConfig(
        model="test-model",
        system_prompt="You are helpful.",
        allowed_tools=(SPAWN_TOOL_NAME,),
        budget=RunBudget(
            max_steps=60,
            max_model_turns=60,
            max_subagents=5,
            max_tool_result_chars=6000,
        ),
    )
    service, chat, coordinator = make_fleet_service(db, parent, bodies, max_live=5)
    _run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=cfg,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    wait_results = _history_tool_results(chat, WAIT_AGENTS_TOOL_NAME)
    assert len(wait_results) == 1
    combined = wait_results[0]
    # Budgeted BEFORE the history-append seam, so that seam never fired:
    # its trailer is absent and every child is still identified. Without
    # the even split the first children would have filled the 6000 chars
    # and the last ones would have been cut away entirely.
    assert len(combined) <= cfg.budget.max_tool_result_chars
    assert "[truncated: wait_agents returned" not in combined
    for handle in coordinator.snapshot():
        assert handle.handle_id in combined
    # Each child got an EQUAL share of the body budget (and each was
    # shortened, since 5 x 4000 cannot fit in 6000) -- not "the first two
    # children in full, the rest cut off".
    entries = combined.split("\n\n")
    bodies = [entry.split("\n", 1)[1] for entry in entries[:5]]
    assert len({len(body) for body in bodies}) == 1
    assert all(body.endswith("[truncated]") for body in bodies)
    assert all(len(body) > 500 for body in bodies)
    assert combined.count("[truncated]") == 5
    # ... and the model is told how to get one child's full result.
    assert "wait_agents" in combined


def test_wait_agents_refetches_one_child_at_the_full_per_child_cap(db):
    """`wait_agents([id])` returns that child at max_subagent_result_chars."""
    bodies = {"task one": ["A" * 3000], "task two": ["B" * 3000]}
    cfg = AgentConfig(
        model="test-model",
        system_prompt="You are helpful.",
        allowed_tools=(SPAWN_TOOL_NAME,),
        budget=RunBudget(
            max_steps=40,
            max_model_turns=40,
            max_subagents=2,
            max_subagent_result_chars=4000,
            max_tool_result_chars=4000,
        ),
    )
    captured: dict[str, str] = {}

    def refetch_first():
        # The handle ids only exist once both children have been started,
        # so the id to re-fetch is read at call time from the coordinator.
        handle = next(
            h for h in coordinator.snapshot() if h.task == "task one"
        )
        captured["handle_id"] = handle.handle_id
        return fence(WAIT_AGENTS_TOOL_NAME, {"ids": [handle.handle_id]})

    service, chat, coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "task one"}),
            fence(SPAWN_TOOL_NAME, {"task": "task two"}),
            fence(WAIT_AGENTS_TOOL_NAME, {}),  # both, budget-split
            refetch_first,  # one, at the full per-child cap
            "done",
        ],
        bodies,
        max_live=2,
    )
    _run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=cfg,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    waits = _history_tool_results(chat, WAIT_AGENTS_TOOL_NAME)
    assert len(waits) == 2
    combined, refetched = waits
    # Split: each of the two got roughly half, so neither is complete.
    assert combined.count("A") < 3000 and combined.count("B") < 3000
    # Re-fetch: the whole 3000-char body, and only that child.
    assert refetched.count("A") == 3000
    assert "B" not in refetched
    assert captured["handle_id"] in refetched


def test_wait_agents_reports_unknown_ids(db):
    service, _chat, _coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "task one"}),
            fence(WAIT_AGENTS_TOOL_NAME, {"ids": ["nope"]}),
            "done",
        ],
        {"task one": ["answer one"]},
    )
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=FLEET_CFG,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    waits = _tool_results(db.get_run(run_id), WAIT_AGENTS_TOOL_NAME)
    assert waits and "nope" in waits[0] and "ERROR" in waits[0]


def test_wait_agents_with_no_children_says_so(db):
    service, _chat, _coordinator = make_fleet_service(
        db, [fence(WAIT_AGENTS_TOOL_NAME, {}), "done"], {}
    )
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=FLEET_CFG,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    waits = _tool_results(db.get_run(run_id), WAIT_AGENTS_TOOL_NAME)
    assert waits and "spawn_subagent" in waits[0]


# -- cancellation and the wall-clock bound --------------------------------


def test_wait_agents_cancellation_stops_children_and_ends_the_run(db):
    """User cancellation while waiting propagates to the children.

    The child keeps calling a tool (with varying arguments, so the cycle
    detector never fires), which gives it the step boundaries at which a
    cooperative cancel is actually noticed.
    """
    cancelled = threading.Event()

    def cancel_then_wait():
        cancelled.set()
        return fence(WAIT_AGENTS_TOOL_NAME, {})

    cfg = AgentConfig(
        model="test-model",
        system_prompt="You are helpful.",
        allowed_tools=("calculator", SPAWN_TOOL_NAME),
        budget=RunBudget(max_steps=200, max_model_turns=200, max_subagents=2),
    )
    service, _chat, coordinator = make_fleet_service(
        db,
        [fence(SPAWN_TOOL_NAME, {"task": "busy"}), cancel_then_wait],
        {
            "busy": [
                fence("calculator", {"expression": f"1+{n}"}) for n in range(60)
            ]
        },
    )
    _run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=cfg,
        api_endpoint="llama_cpp",
        should_cancel=cancelled.is_set,
    )
    assert outcome.status == RUN_CANCELLED
    assert coordinator.all_finished()
    assert coordinator.snapshot()[0].status == RUN_CANCELLED
    rows = db.list_runs("c", include_superseded=True)
    assert all(row["status"] in TERMINAL_RUN_STATUSES for row in rows)
    child = next(row for row in rows if row["agent_kind"] == "subagent")
    assert child["status"] == RUN_CANCELLED


def test_cancelling_and_abandoning_a_child_revokes_its_approval_cards(db):
    """PR2a Task 7: a stopped child's pending approval card is revoked.

    The approval wait lives on the child's own per-call daemon thread, so
    a card left on screen after the child is cancelled is a card the user
    can still press Approve on -- and the tool would run for real for a
    run that already reads ``cancelled``. The service does not know what a
    card is; it only knows which run it just stopped, and hands that id to
    the injected ``revoke_approvals`` seam (wired by the Console bridge to
    ``ConsoleChatController.revoke_approval_rounds_for_run``).

    The child here is wedged inside its provider call, so it is both
    cooperatively cancelled AND abandoned -- the two moments that must
    revoke.
    """
    never = threading.Event()
    entered = threading.Event()

    def wedged_child():
        entered.set()
        never.wait(30.0)  # released in the finally below, not by the run
        return "unreachable"

    revoked: list[str] = []
    short = AgentConfig(
        model="test-model",
        system_prompt="You are helpful.",
        allowed_tools=(SPAWN_TOOL_NAME,),
        budget=RunBudget(
            max_steps=40,
            max_model_turns=40,
            max_subagents=2,
            max_wall_seconds=1.0,
        ),
    )
    service, _chat, coordinator = make_fleet_service(
        db,
        [fence(SPAWN_TOOL_NAME, {"task": "wedged"}), "parent done"],
        {"wedged": [wedged_child]},
        revoke_approvals=revoked.append,
    )
    try:
        _run_id, outcome = service.run_turn(
            conversation_id="c",
            messages=[{"role": "user", "content": "go"}],
            config=short,
            api_endpoint="llama_cpp",
        )
        assert entered.is_set()
        assert outcome.status == RUN_DONE
        child_run_ids = [handle.run_id for handle in coordinator.snapshot()]
        assert child_run_ids and all(child_run_ids), (
            "precondition: the child's run id reached its handle"
        )
        # Revoked for exactly the children this turn stopped -- never for
        # the parent, whose own card (if any) belongs to a live run.
        assert set(revoked) == set(child_run_ids)
        assert _run_id not in revoked
    finally:
        never.set()


def test_wait_agents_is_bounded_by_the_runs_remaining_wall_clock(
    db, monkeypatch
):
    """A wedged child must not hold wait_agents past the run's budget."""
    # Keep the post-cancel drain short: the behaviour under test is the
    # wall-clock BOUND, not how long an abandoned thread is humoured.
    monkeypatch.setattr(agent_service, "FLEET_JOIN_TIMEOUT_SECONDS", 0.2)
    never = threading.Event()

    def wedged_child():
        never.wait(30.0)
        return "unreachable"

    cfg = AgentConfig(
        model="test-model",
        system_prompt="You are helpful.",
        allowed_tools=(SPAWN_TOOL_NAME,),
        budget=RunBudget(
            max_steps=40,
            max_model_turns=40,
            max_subagents=2,
            max_wall_seconds=1.0,
        ),
    )
    service, _chat, coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "wedged"}),
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "gave up",
        ],
        {"wedged": [wedged_child]},
    )
    try:
        started_at = time.monotonic()
        run_id, outcome = service.run_turn(
            conversation_id="c",
            messages=[{"role": "user", "content": "go"}],
            config=cfg,
            api_endpoint="llama_cpp",
        )
        elapsed = time.monotonic() - started_at
        # Bounded by the 1s budget (plus the shortened drain/join), not by
        # the child's own 30s block.
        assert elapsed < 10.0
        # The wait burned the run's remaining wall-clock, so the loop's own
        # budget check ends the run -- terminal either way, never hung.
        assert outcome.status in TERMINAL_RUN_STATUSES
        # Read the step log, not history: the run may end before another
        # provider call carries this result into a payload.
        waits = _tool_results(db.get_run(run_id), WAIT_AGENTS_TOOL_NAME)
        assert waits and "time budget ran out" in waits[0]
        assert coordinator.all_finished()
        rows = db.list_runs("c", include_superseded=True)
        assert all(row["status"] in TERMINAL_RUN_STATUSES for row in rows)
    finally:
        never.set()


# -- failure isolation -----------------------------------------------------


def test_child_thread_exception_finishes_the_handle_as_error(db):
    """An exception ESCAPING _run_one must never strand the parent's join."""
    service, _chat, coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "boom"}),
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "handled",
        ],
        {"boom": ["never reached"]},
    )
    real_run_one = service._run_one

    def exploding_run_one(**kwargs):
        if kwargs.get("agent_kind") == "subagent":
            raise RuntimeError("child thread blew up")
        return real_run_one(**kwargs)

    service._run_one = exploding_run_one

    _run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=FLEET_CFG,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE  # the PARENT survives
    assert coordinator.all_finished()
    handle = coordinator.snapshot()[0]
    assert handle.status == RUN_ERROR
    assert "child thread blew up" in handle.error


class _SpyRunLogWriter:
    """Minimal stand-in recording the run tree's two cleanup calls."""

    def __init__(self):
        self.is_active = False
        self.log_dir = None
        self.bound: str | None = None
        self.manifests: list[dict] = []
        self.closed = 0

    def bind(self, run_id):
        self.bound = run_id

    def append(self, **kwargs):
        return None

    def write_manifest(self, data):
        self.manifests.append(data)

    def close(self):
        self.closed += 1


def test_thread_start_failure_is_contained_and_the_turn_still_finalizes(
    db, monkeypatch
):
    """Thread exhaustion must not strand a handle or skip run finalization.

    Registering the thread before ``start()`` succeeded meant
    ``_settle_fleet`` would later join an unstarted thread -- a
    RuntimeError out of ``run_turn`` that skips ``write_manifest()`` and
    ``run_log_writer.close()``, leaking a file descriptor. The reserved
    handle was never finished either, so the settle loop first burned the
    whole remaining wall-clock waiting for a child that does not exist.
    """
    db.create_agent_definition(
        AgentDefinition(
            name="researcher",
            description="Searches.",
            instructions="Cite sources.",
        )
    )
    # Fail only the FIRST fleet thread, and only fleet threads: everything
    # else in the process (including _call_with_timeout's "tool-*" workers,
    # sqlite, and loguru) must keep working normally.
    real_start = threading.Thread.start
    failures: list[str] = []

    def flaky_start(self):
        if self.name.startswith("fleet-") and not failures:
            failures.append(self.name)
            raise RuntimeError("can't start new thread")
        return real_start(self)

    monkeypatch.setattr(threading.Thread, "start", flaky_start)

    spy = _SpyRunLogWriter()
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    chat = FleetChat(
        [
            fence(SPAWN_TOOL_NAME, {"task": "doomed", "agent": "researcher"}),
            fence(SPAWN_TOOL_NAME, {"task": "task two", "agent": "researcher"}),
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "done",
        ],
        {"task two": ["answer two"]},
    )
    coordinator = FleetCoordinator(max_live=3, clock=time.monotonic)
    service = AgentService(
        db=db,
        registry=registry,
        chat_call=chat,
        run_log_writer=spy,
        fleet_coordinator=coordinator,
    )
    # ONE spawn slot: the failed spawn must give its slot back, or the
    # second (real) spawn would be refused as budget-exhausted.
    cfg = AgentConfig(
        model="test-model",
        system_prompt="You are helpful.",
        allowed_tools=(SPAWN_TOOL_NAME,),
        budget=RunBudget(
            max_steps=40,
            max_model_turns=40,
            max_subagents=1,
            max_wall_seconds=60.0,
        ),
    )
    started_at = time.monotonic()
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=cfg,
        api_endpoint="llama_cpp",
    )
    elapsed = time.monotonic() - started_at

    assert failures, "the fleet thread's start() was never exercised"
    assert outcome.status == RUN_DONE
    # The turn was NOT held open by a handle nobody would ever finish.
    assert elapsed < 10.0
    # Finalization still happened -- manifest written, writer closed once.
    assert len(spy.manifests) == 1
    assert spy.manifests[0]["run_id"] == run_id
    assert spy.closed == 1
    # The stranded handle was finished, not left live.
    assert coordinator.all_finished()
    doomed = next(h for h in coordinator.snapshot() if h.task == "doomed")
    assert doomed.status == RUN_ERROR
    assert "could not start" in doomed.error
    # The model was told, and no child run row was ever created for it.
    spawn_results = _tool_results(db.get_run(run_id), SPAWN_TOOL_NAME)
    assert len(spawn_results) == 2
    assert "could not start sub-agent" in spawn_results[0]
    # The spawn slot was given back: the second spawn started for real.
    assert spawn_results[1].startswith("started ")
    assert db.count_subagent_runs("c") == 1
    waits = _tool_results(db.get_run(run_id), WAIT_AGENTS_TOOL_NAME)
    assert waits and "answer two" in waits[0]


def test_failed_child_is_reported_in_wait_result(db):
    service, _chat, _coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "good"}),
            fence(SPAWN_TOOL_NAME, {"task": "bad"}),
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "done",
        ],
        {
            "good": ["good answer"],
            # An empty final text with no tool call ends the run DONE, so
            # force a non-done child by exhausting its model turns instead.
            "bad": [
                fence("calculator", {"expression": "1+1"}),
                fence("calculator", {"expression": "1+1"}),
                fence("calculator", {"expression": "1+1"}),
            ],
        },
    )
    cfg = AgentConfig(
        model="test-model",
        system_prompt="You are helpful.",
        allowed_tools=("calculator", SPAWN_TOOL_NAME),
        budget=RunBudget(
            max_steps=40, max_model_turns=40, max_subagents=3
        ),
    )
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=cfg,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    waits = _tool_results(db.get_run(run_id), WAIT_AGENTS_TOOL_NAME)
    assert waits and "good answer" in waits[0]
    # The failed child is named with its terminal status, not silently
    # dropped -- the supervisor decides what to do about it.
    assert "stuck" in waits[0]


# -- the PR2a Task 5 carry-forward guard ----------------------------------


def test_child_tool_call_binds_the_childs_own_run_id(db):
    """A tool run from a child's thread must see the CHILD's run id.

    Both permission gates key this turn's verdicts by ``(run_id, tool)``.
    A child thread that failed to bind its own run id would read ``""``,
    find no stamp, and fail an APPROVED tool closed -- silently.
    """
    probe = RunIdProbeProvider()
    cfg = AgentConfig(
        model="test-model",
        system_prompt="You are helpful.",
        allowed_tools=("whoami", SPAWN_TOOL_NAME),
        budget=RunBudget(max_steps=40, max_model_turns=40, max_subagents=2),
    )
    service, _chat, _coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "probe task"}),
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "done",
        ],
        {"probe task": [fence("whoami", {}), "child done"]},
        providers=(probe,),
    )
    _run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=cfg,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    child = next(r for r in db.list_runs("c") if r["agent_kind"] == "subagent")
    assert probe.seen == [child["id"]]
    assert probe.seen[0] != ""


def test_the_review_hook_runs_with_its_own_runs_id_bound(db):
    """The approval bridge the review hook calls must see the run id.

    PR2a Task 7: an approval card is armed from inside
    ``review_tool_calls`` (and from ``MCPToolProvider.invoke``'s
    single-call fallback), and each armed round records WHICH RUN armed
    it so a cancelled child's card can be revoked without touching a live
    sibling's. The hook's own ``run_id`` parameter cannot reach that
    bridge -- it is a pre-bound callable whose signature the hook builders
    do not own -- so the service binds the same ``run_context``
    ContextVar around the hook that it already binds around tool
    execution. Unbound, ownership would silently read ``""`` and every
    revoke would be a no-op: fail-OPEN, which is the direction that
    leaves a cancelled child's card pressable.
    """
    seen: list[tuple[str, str]] = []

    def review(calls, run_id):
        seen.append((run_id, current_run_id()))
        return {}

    cfg = AgentConfig(
        model="test-model",
        system_prompt="You are helpful.",
        allowed_tools=("calculator", SPAWN_TOOL_NAME),
        budget=RunBudget(max_steps=40, max_model_turns=40, max_subagents=2),
    )
    service, _chat, _coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "probe task"}),
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "done",
        ],
        {"probe task": [fence("calculator", {"expression": "1+1"}), "child done"]},
        review_tool_calls=review,
    )
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=cfg,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    assert seen, "the review hook never ran"
    # Every batch: the id bound for the hook IS the run the hook was told
    # it was reviewing -- never "" and never a sibling's.
    assert all(bound == hook_run_id for hook_run_id, bound in seen), seen
    child = next(r for r in db.list_runs("c") if r["agent_kind"] == "subagent")
    bound_ids = {bound for _hook_run_id, bound in seen}
    assert child["id"] in bound_ids, "the child's own batch was reviewed unbound"
    assert run_id in bound_ids


# -- deferred self-review items -------------------------------------------


class _MutuallyExclusiveProbeRunner:
    """A SkillRunner that calls spawn with both mutually exclusive kwargs."""

    def __init__(self):
        self.raised: Exception | None = None

    def is_skill_tool(self, name):
        return name == "code-review"

    def run(self, name, args, spawn):
        try:
            spawn("t", allowed_tools=("calculator",), agent="researcher")
        except Exception as exc:  # noqa: BLE001 — recorded, then asserted
            self.raised = exc
            return ToolResult(ok=True, content="refused as expected")
        return ToolResult(ok=False, error="spawn accepted both kwargs")


def test_spawn_rejects_agent_and_allowed_tools_together(db):
    """The structural invariant raises explicitly -- never a bare assert.

    A bare ``assert`` is stripped under ``python -O``, which would turn a
    future caller's mistake into a silently mis-configured child instead
    of a loud refusal.
    """
    db.create_agent_definition(
        AgentDefinition(
            name="researcher",
            description="Searches.",
            instructions="Cite sources.",
        )
    )
    runner = _MutuallyExclusiveProbeRunner()
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    # "code-review" must be a real catalog entry so it is DISCLOSED, not
    # merely permitted -- otherwise invoke_tool's skill branch refuses it
    # before the runner (and therefore the spawn closure) is ever reached.
    registry.register_provider(
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
    chat = FleetChat([fence("code-review", {"args": "the diff"}), "done"], {})
    service = AgentService(
        db=db,
        registry=registry,
        chat_call=chat,
        skill_runner=runner,
        fleet_coordinator=FleetCoordinator(max_live=3, clock=time.monotonic),
    )
    cfg = AgentConfig(
        model="test-model",
        system_prompt="You are helpful.",
        allowed_tools=("calculator", "code-review", SPAWN_TOOL_NAME),
        budget=RunBudget(max_steps=40, max_model_turns=40, max_subagents=2),
    )
    _run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=cfg,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    assert isinstance(runner.raised, ValueError)
    assert "mutually exclusive" in str(runner.raised)
    # No child was created for the rejected call.
    assert db.count_subagent_runs("c") == 0


def test_agent_definitions_are_loaded_once_per_turn(db):
    """Fleet spec §4: the roster loads ONCE per turn, not once per spawn.

    A call-count guard, not a behavioural proxy: the previous coverage
    only proved the roster the model SEES is stable, which a per-spawn
    re-read would also satisfy while quietly issuing one DB query per
    child.
    """
    db.create_agent_definition(
        AgentDefinition(
            name="researcher",
            description="Searches.",
            instructions="Cite sources.",
        )
    )
    calls: list[tuple] = []
    real = db.list_agent_definitions

    def counting(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    db.list_agent_definitions = counting

    service, _chat, _coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "task one", "agent": "researcher"}),
            fence(SPAWN_TOOL_NAME, {"task": "task two", "agent": "researcher"}),
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "done",
        ],
        {"task one": ["answer one"], "task two": ["answer two"]},
    )
    _run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=FLEET_CFG,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    assert db.count_subagent_runs("c") == 2
    assert len(calls) == 1, f"roster re-read {len(calls)} times in one turn"
    assert calls[0][1] == {"enabled_only": True}
