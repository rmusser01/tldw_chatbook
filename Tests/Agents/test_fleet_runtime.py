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
``test_agent_service.FleetChat`` keeps the same shape but ADDRESSES
replies -- an ordered script for the parent, a per-task script for each
child -- so every assertion here is deterministic under real threads. It
was written here and moved next to ``ScriptedChat`` in Task 6.5, when
flipping the default made nine other suites need it too.
"""

import json
import threading
import time
from collections import Counter

import pytest

from Tests.Agents.test_agent_service import (
    SUBAGENT_PROMPT_PREFIX,
    FleetChat,
    ScriptedChat,
    fence,
)
from tldw_chatbook.Agents import agent_service
from tldw_chatbook.Agents.agent_models import (
    CHECK_AGENTS_TOOL_NAME,
    FENCE_TOOL_RESULT_PREFIX,
    RUN_CANCELLED,
    RUN_DONE,
    RUN_ERROR,
    RUN_RUNNING,
    RUN_SKILL_SCRIPT_TOOL_NAME,
    RUNTIME_TOOL_NAMES,
    SPAWN_TOOL_NAME,
    TERMINAL_RUN_STATUSES,
    WAIT_AGENTS_TOOL_NAME,
    AgentConfig,
    AgentDefinition,
    RunBudget,
    ToolCatalogEntry,
    ToolResult,
    ToolSchema,
)
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.fleet_coordinator import FleetCoordinator
from tldw_chatbook.Agents.local_tool_provider import (
    LocalToolProvider,
    _default_specs,
)
from tldw_chatbook.Agents.run_context import current_run_id
from tldw_chatbook.Agents.session_todo_store import SessionTodoStore
from tldw_chatbook.Agents.tool_catalog import (
    CHECK_AGENTS_SCHEMA,
    WAIT_AGENTS_SCHEMA,
    BuiltinToolProvider,
    SkillToolProvider,
    ToolCatalogRegistry,
)
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Chat.trajectory import derive_trajectory
from tldw_chatbook.MCP.permission_store import EffectiveToolState

from Tests.Agents.conftest import (
    join_fleet_children,
    pin_agent_settings,
    pin_max_live_subagents,
    pin_turn_scoped_children,
)
_JOIN_TIMEOUT = 5.0


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


class _FleetTodoBarrierStore(SessionTodoStore):
    """Pause two real provider calls at the shared store method boundary."""

    def __init__(self) -> None:
        super().__init__()
        self._barrier: threading.Barrier | None = None
        self._barrier_operation: str | None = None
        self._arrival_lock = threading.Lock()
        self._arrivals: list[tuple[str, str]] = []

    def arm(self, operation: str) -> None:
        self._barrier_operation = operation
        self._barrier = threading.Barrier(2, timeout=_JOIN_TIMEOUT)

    def arrivals(self, operation: str) -> set[str]:
        with self._arrival_lock:
            return {run_id for name, run_id in self._arrivals if name == operation}

    def _meet(self, operation: str) -> None:
        if self._barrier_operation != operation or self._barrier is None:
            return
        run_id = current_run_id()
        with self._arrival_lock:
            self._arrivals.append((operation, run_id))
        self._barrier.wait()

    def create(self, **kwargs):
        self._meet("create")
        return super().create(**kwargs)

    def update(self, **kwargs):
        self._meet("update")
        return super().update(**kwargs)


def _todo_provider(workspace, store: SessionTodoStore) -> LocalToolProvider:
    """Build one real provider whose four task handlers close over ``store``."""
    todo_specs = [
        spec
        for spec in _default_specs(workspace, todo_store=store)
        if spec.name in {"todo_create", "todo_update", "todo_get", "todo_list"}
    ]
    return LocalToolProvider(
        workspace_root=workspace,
        specs=todo_specs,
        resolve_state=lambda _hub: EffectiveToolState(
            state="allow", origin="tool_override"
        ),
    )


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
    run_skill_script_tool=None,
    allow_unconsumed=False,
    run_log_writer=None,
):
    """An AgentService wired for the fleet (explicit coordinator = opt in).

    `allow_unconsumed` forwards to `FleetChat`: set it in a test that
    deliberately strands scripted turns (a cancelled, wedged, or exploding
    child), so the teardown consumption sweep does not flag them. Mis-keyed
    scripts stay fatal either way.

    `run_log_writer` injects a writer double (a `_SpyRunLogWriter`) for the
    tests that assert on the run tree's finalization order.
    """
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    for provider in providers:
        registry.register_provider(provider)
    chat = FleetChat(parent_replies, child_replies, allow_unconsumed=allow_unconsumed)
    coordinator = FleetCoordinator(max_live=max_live, clock=time.monotonic)
    service = AgentService(
        db=db,
        registry=registry,
        chat_call=chat,
        fleet_coordinator=coordinator,
        revoke_approvals=revoke_approvals,
        review_tool_calls=review_tool_calls,
        run_skill_script_tool=run_skill_script_tool,
        run_log_writer=run_log_writer,
    )
    return service, chat, coordinator


def _patch_max_live(monkeypatch, value):
    """Make `[agents] max_live_subagents` read as `value`.

    Delegates to the shared conftest helper rather than re-patching
    `_setting` itself: since PR3a-1 Task 2 a second knob
    (`subagents_outlive_turn`) is pinned by some of these tests too, and
    two independent whole-function patches would silently drop each other.
    """
    pin_max_live_subagents(monkeypatch, value)


def make_inline_service(db, replies, monkeypatch, *, max_live=1):
    """An AgentService on the pre-PR inline path: no coordinator, cap 1.

    `monkeypatch` is REQUIRED, not optional: since Task 6.5 the shipped
    default is 3, so merely omitting the injected coordinator no longer
    buys the inline path -- `run_turn` would size a fleet from config.
    Every inline-path test therefore has to say `max_live_subagents = 1`
    out loud, which is also what a user opting out actually writes.

    Uses the ORDINARY single-queue ``ScriptedChat``: with no fleet there
    are no threads, so strict reply ordering is exactly what should still
    hold -- and asserting against it is the point.

    Args:
        max_live: the raw `[agents] max_live_subagents` value to pin.
            Defaults to the opt-out, `1`; the config-coercion test passes
            other values that must ALSO resolve to the inline path.
    """
    _patch_max_live(monkeypatch, max_live)
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

#: How long a parked approval card waits for the test to answer it.
_CARD_TIMEOUT = 10.0

#: `FLEET_CFG` with a wall clock short enough that a REGRESSION (a settle
#: that waits for a child parked on a card nobody will answer until the
#: turn is over) fails in 20s instead of the 240s default. PR3a-1 Task 5
#: correction: a THREADED survivor's own budget is no longer clamped
#: from this config's `max_wall_seconds=20.0` at all -- it gets
#: `contain_child_budget`'s independent ceiling
#: (`agent_service.DEFAULT_CHILD_MAX_WALL_SECONDS`, 1800s by default),
#: unrelated to this 20s value. This config's short wall clock only
#: bounds a TURN-SCOPED child (`clamp_child_budget` still derives from
#: it) and the parent's own settle-loop timing.
CARD_CFG = AgentConfig(
    model="test-model",
    system_prompt="You are helpful.",
    allowed_tools=("calculator", SPAWN_TOOL_NAME),
    budget=RunBudget(
        max_steps=40,
        max_model_turns=40,
        max_subagents=2,
        max_wall_seconds=20.0,
    ),
)

#: `FLEET_CFG` with a 1s wall clock, so a TURN-SCOPED settle stops waiting
#: for a straggler almost immediately instead of humouring it for the
#: default budget. Used where both settle modes must run the same script.
SHORT_WALL_CFG = AgentConfig(
    model="test-model",
    system_prompt="You are helpful.",
    allowed_tools=("calculator", "get_current_datetime", SPAWN_TOOL_NAME),
    budget=RunBudget(
        max_steps=40,
        max_model_turns=40,
        max_subagents=4,
        max_wall_seconds=1.0,
    ),
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


def _wait_until(predicate, message, timeout=_JOIN_TIMEOUT):
    """Poll until ``predicate()`` holds, or fail with ``message``.

    Since PR3a-1 Task 2 a child can still be working when `run_turn`
    returns, so any assertion about a child's finished state has to say
    when it expects that state to exist.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(message)


def _child_row(db, conversation_id="c"):
    """The single sub-agent run row for this conversation."""
    rows = db.list_runs(conversation_id, include_superseded=True)
    return next(row for row in rows if row["agent_kind"] == "subagent")


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
    parent = db.get_run(run_id)
    spawn_events = {
        f"agent-step:{run_id}:{step['index']}"
        for step in parent["steps"]
        if step["kind"] == "spawn"
    }
    children = [row for row in db.list_runs("c") if row["agent_kind"] == "subagent"]
    assert {row["spawn_event_id"] for row in children} == spawn_events


def test_parent_and_fleet_child_share_todo_store_for_concurrent_creates(db, tmp_path):
    """Parent and child enter one store concurrently and retain both creates."""
    store = _FleetTodoBarrierStore()
    store.arm("create")
    provider = _todo_provider(tmp_path, store)
    config = AgentConfig(
        model="test-model",
        system_prompt="You are helpful.",
        allowed_tools=("todo_create", SPAWN_TOOL_NAME),
        budget=RunBudget(max_steps=40, max_model_turns=40, max_subagents=2),
    )
    service, _chat, coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "create child task"}),
            fence("todo_create", {"content": "Parent task"}),
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "both tasks created",
        ],
        {
            "create child task": [
                fence("todo_create", {"content": "Child task"}),
                "child created its task",
            ]
        },
        providers=(provider,),
    )

    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "create both tasks"}],
        config=config,
        api_endpoint="llama_cpp",
    )

    assert outcome.status == RUN_DONE
    assert outcome.final_text == "both tasks created"
    child = next(row for row in db.list_runs("c") if row["agent_kind"] == "subagent")
    assert child["status"] == RUN_DONE
    assert child["id"] != run_id
    assert store.arrivals("create") == {run_id, child["id"]}

    parent_results = _tool_results(db.get_run(run_id), "todo_create")
    child_results = _tool_results(child, "todo_create")
    assert len(parent_results) == len(child_results) == 1
    parent_record = json.loads(parent_results[0])
    child_record = json.loads(child_results[0])
    assert parent_record == {
        "id": parent_record["id"],
        "version": 1,
        "content": "Parent task",
        "status": "pending",
    }
    assert child_record == {
        "id": child_record["id"],
        "version": 1,
        "content": "Child task",
        "status": "pending",
    }
    assert store.get(parent_record["id"]) == parent_record
    assert store.get(child_record["id"]) == child_record
    created = [parent_record, child_record]
    assert {record["id"] for record in created} == {"1", "2"}
    assert {record["content"] for record in created} == {
        "Parent task",
        "Child task",
    }
    assert {record["status"] for record in created} == {"pending"}
    assert {record["version"] for record in created} == {1}
    assert {(record["id"], record["content"]) for record in store.list_after(None)} == {
        (record["id"], record["content"]) for record in created
    }
    assert coordinator.all_finished()


def test_parent_and_fleet_child_preserve_updates_to_distinct_tasks(db, tmp_path):
    """Concurrent version-1 updates to separate IDs both survive at version 2."""
    store = _FleetTodoBarrierStore()
    first = store.create(content="Parent-owned task")
    second = store.create(content="Child-owned task")
    store.arm("update")
    provider = _todo_provider(tmp_path, store)
    config = AgentConfig(
        model="test-model",
        system_prompt="You are helpful.",
        allowed_tools=("todo_update", SPAWN_TOOL_NAME),
        budget=RunBudget(max_steps=40, max_model_turns=40, max_subagents=2),
    )
    service, _chat, coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "update child task"}),
            fence(
                "todo_update",
                {
                    "id": first["id"],
                    "expected_version": 1,
                    "status": "completed",
                },
            ),
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "both tasks updated",
        ],
        {
            "update child task": [
                fence(
                    "todo_update",
                    {
                        "id": second["id"],
                        "expected_version": 1,
                        "status": "in_progress",
                    },
                ),
                "child updated its task",
            ]
        },
        providers=(provider,),
    )

    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "update both tasks"}],
        config=config,
        api_endpoint="llama_cpp",
    )

    assert outcome.status == RUN_DONE
    child = next(row for row in db.list_runs("c") if row["agent_kind"] == "subagent")
    assert child["status"] == RUN_DONE
    assert child["id"] != run_id
    assert store.arrivals("update") == {run_id, child["id"]}

    parent_results = _tool_results(db.get_run(run_id), "todo_update")
    child_results = _tool_results(child, "todo_update")
    assert len(parent_results) == len(child_results) == 1
    parent_record = json.loads(parent_results[0])
    child_record = json.loads(child_results[0])
    assert parent_record == {
        "id": first["id"],
        "version": 2,
        "content": "Parent-owned task",
        "status": "completed",
    }
    assert child_record == {
        "id": second["id"],
        "version": 2,
        "content": "Child-owned task",
        "status": "in_progress",
    }
    assert store.get(first["id"]) == parent_record
    assert store.get(second["id"]) == child_record
    assert coordinator.all_finished()


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
    handle = coordinator.snapshot()[0]
    handle_id = handle.handle_id
    assert handle.run_id
    assert spawn_results[0].startswith("started ")
    assert handle_id not in spawn_results[0]
    assert f"run:{handle.run_id}" in spawn_results[0]
    assert any(handle_id in (step.result or "") for step in outcome.steps)
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
    handle = coordinator.snapshot()[0]
    assert handle.run_id
    assert handle.handle_id not in checks[0]
    assert f"run:{handle.run_id}" in checks[0] and "running" in checks[0]
    assert "slow task" in checks[0]
    # check_agents never blocks: the first snapshot was taken while the
    # child was still gated, and it did NOT contain the child's answer.
    assert "child answer" not in checks[0]
    assert handle.handle_id not in checks[1]
    assert f"run:{handle.run_id}" in checks[1] and RUN_DONE in checks[1]


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
    # PR3a-1 Task 2: the accepted child outlives the turn by default, so
    # its run ROW may not exist yet when run_turn returns -- the
    # end-of-turn wait used to guarantee it did. This test's subject is
    # the cap refusal, not the settle, so it waits for the accepted child
    # rather than pinning a settle mode it does not care about.
    _wait_until(coordinator.all_finished, "the accepted child never finished")
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
    service, _chat, coordinator = make_fleet_service(
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
    # The third child outlives the turn by default (PR3a-1 Task 2); wait
    # for it rather than counting rows that are still being written.
    _wait_until(coordinator.all_finished, "the third child never finished")
    assert db.count_subagent_runs("c") == 2


def test_end_of_turn_waits_for_stragglers(db, monkeypatch):
    """Turn-scoped: the turn must not return with a child still running.

    PR3a-1 Task 2 made survival the default, so this -- the phase-2 rule
    in full -- is now what `[agents] subagents_outlive_turn = false`
    buys, and this test is its guard. Everything below the pin is
    unchanged from phase 2 on purpose: the settle path must stay
    byte-identical under the kill switch.
    """
    pin_turn_scoped_children(monkeypatch)
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


def test_end_of_turn_cancels_and_abandons_a_wedged_child(db, monkeypatch):
    """Turn-scoped: a child that never returns is cancelled and abandoned.

    The parent's wall-clock budget is 1s, so the end-of-turn wait expires
    almost immediately; the child ignores its cooperative cancel entirely
    (it is blocked inside the provider call), so it must be abandoned after
    the join timeout with its handle AND its run row marked cancelled --
    never left ``running``.

    Pinned turn-scoped (PR3a-1 Task 2): under the shipped default this
    child would simply keep running, which is what
    ``test_a_straggler_outlives_its_turn_by_default`` asserts.
    """
    pin_turn_scoped_children(monkeypatch)
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


# -- crossing the turn boundary (PR3a-1 Task 2) ---------------------------
#
# The shipped default is now `[agents] subagents_outlive_turn = true`: a
# child still running when the turn returns KEEPS RUNNING. The tests above
# pin the kill switch and describe what it buys; these describe the
# default. Each of them gates its child inside its first provider call, so
# "still running when run_turn returned" is a fact the test controls rather
# than a race it hopes to win.


def _gated_child(entered, release, reply="late answer", timeout=10.0):
    """A child provider reply that blocks until the test releases it.

    Args:
        entered: set once the child has actually reached its model call --
            the precondition every survival assertion below rests on.
        release: the test's go signal.
        reply: what the child answers once released.
        timeout: fail loudly rather than hang the suite forever if the
            test forgets to release it.

    Returns:
        A zero-arg callable for a ``FleetChat`` child script.
    """

    def child():
        entered.set()
        if not release.wait(timeout):
            raise AssertionError("child was never released by the test")
        return reply

    return child


def _after(entered, reply, timeout=_JOIN_TIMEOUT):
    """A PARENT reply held until the child is provably live.

    Without this the parent can answer -- and the turn can end -- before
    the child thread has reached ``create_run``, which would leave these
    tests asserting about a child that has no run row yet and a scripted
    child turn nobody ever asked for. Gating removes the race instead of
    sleeping through it.

    Args:
        entered: the child's "I am at my model call" signal.
        reply: what the parent answers once the child is live.
        timeout: fail loudly rather than hang if the child never starts.

    Returns:
        A zero-arg callable for a ``FleetChat`` parent script.
    """

    def parent():
        if not entered.wait(timeout):
            raise AssertionError("the child never reached its model call")
        return reply

    return parent


def test_a_straggler_outlives_its_turn_by_default(db):
    """The turn returns; the child is neither cancelled nor forced.

    This is the whole feature: `run_turn` comes back with the supervisor's
    answer while the child is still working, and NOTHING about the child
    is touched on the way out -- its handle still reads ``running`` and so
    does its run row. Phase 2 would have waited for it and then marked
    both ``cancelled``.
    """
    entered = threading.Event()
    release = threading.Event()
    service, _chat, coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "slow task"}),
            _after(entered, "parent answered early"),  # no wait_agents
        ],
        {"slow task": [_gated_child(entered, release)]},
    )
    try:
        started_at = time.monotonic()
        run_id, outcome = service.run_turn(
            conversation_id="c",
            messages=[{"role": "user", "content": "go"}],
            config=FLEET_CFG,
            api_endpoint="llama_cpp",
        )
        elapsed = time.monotonic() - started_at
        assert entered.is_set(), "precondition: the child reached its model call"
        assert outcome.status == RUN_DONE
        assert outcome.final_text == "parent answered early"
        # The turn did not wait out the child's 10s block, nor the 5s join.
        assert elapsed < 3.0, f"the turn was held open for {elapsed:.2f}s"
        # Still live -- the point of the change.
        assert not coordinator.all_finished()
        assert [h.status for h in coordinator.snapshot()] == [RUN_RUNNING]
        child = _child_row(db)
        assert child["status"] == RUN_RUNNING
        assert child["parent_run_id"] == run_id
    finally:
        release.set()
        _wait_until(
            coordinator.all_finished, "the released child never finished"
        )


def test_a_survivor_persists_a_real_terminal_status_after_its_turn(db):
    """Finishing after the turn is a REAL completion, not a leak.

    The child answers for real once released -- long after `run_turn`
    returned -- and both its handle and its run row land on ``done`` with
    its actual result, which is what `wait_agents`/`check_agents` and the
    fleet panel will read from.
    """
    entered = threading.Event()
    release = threading.Event()
    service, _chat, coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "slow task"}),
            _after(entered, "parent answered early"),
        ],
        {"slow task": [_gated_child(entered, release, reply="late answer")]},
    )
    try:
        _run_id, outcome = service.run_turn(
            conversation_id="c",
            messages=[{"role": "user", "content": "go"}],
            config=FLEET_CFG,
            api_endpoint="llama_cpp",
        )
        assert outcome.status == RUN_DONE
        assert _child_row(db)["status"] == RUN_RUNNING
    finally:
        release.set()
    _wait_until(coordinator.all_finished, "the released child never finished")
    handle = coordinator.snapshot()[0]
    assert handle.status == RUN_DONE
    assert handle.result == "late answer"
    child = _child_row(db)
    assert child["status"] == RUN_DONE
    assert child["result"] == "late answer"


def test_a_survivors_pending_approval_is_answerable_after_its_turn(db):
    """A live child's approval card outlives the turn AND still works.

    This is the property `_cancel_fleet_handles`' survivor exclusion
    exists for, tested where the Console actually parks a run: INSIDE
    `review_tool_calls`, which is where an approval card is armed and
    where the child's thread waits for the human. The card here is
    answered only AFTER `run_turn` has returned -- the case the exclusion
    protects -- and the gate must then release for real: the tool
    dispatches, its result reaches the child's next provider call, and
    the child lands `done`.

    An earlier version of this test only asserted that `revoke_approvals`
    was never called while a child blocked in its PROVIDER call. That
    child never armed a card at all, so the assertion held even with the
    exclusion deleted and `_revoke_handle_approvals` no-opped -- vacuous.
    """
    parked = threading.Event()
    answered = threading.Event()
    revoked: list[str] = []

    def review(calls, run_id):
        # Only the CHILD calls calculator; the parent's own spawn review
        # must not park, or the turn would never reach its answer.
        if any(call.name == "calculator" for call in calls):
            parked.set()
            if not answered.wait(_CARD_TIMEOUT):
                raise AssertionError("the card was never answered")
        return {}

    service, chat, coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "probe task"}),
            # The parent answers while the child sits on its card.
            _after(parked, "parent answered early"),
        ],
        {
            "probe task": [
                fence("calculator", {"expression": "1+1"}),
                "child done",
            ]
        },
        revoke_approvals=revoked.append,
        review_tool_calls=review,
    )
    try:
        _run_id, outcome = service.run_turn(
            conversation_id="c",
            messages=[{"role": "user", "content": "go"}],
            config=CARD_CFG,
            api_endpoint="llama_cpp",
        )
        assert outcome.status == RUN_DONE
        assert parked.is_set(), "precondition: the child armed a card"
        # The turn ended with the card still pending: it must NOT have
        # been failed closed on the way out, and the child must still be
        # live to receive the answer.
        assert revoked == [], f"a live child's card was revoked: {revoked}"
        assert not coordinator.all_finished()
        assert _child_row(db)["status"] == RUN_RUNNING
    finally:
        # The human approves -- one turn late, which is the whole point.
        answered.set()
    _wait_until(coordinator.all_finished, "the approved child never finished")

    assert revoked == []
    # The gate released for real: the tool ran and produced its answer...
    child_row = _child_row(db)
    calc_results = _tool_results(db.get_run(child_row["id"]), "calculator")
    assert calc_results and "2" in calc_results[0], calc_results
    # ... that answer reached the child's NEXT provider call ...
    child_turns = chat.child_calls["probe task"]
    assert len(child_turns) == 2, "the child never got a turn after the card"
    assert any(
        "2" in str(message.get("content", ""))
        for message in child_turns[1]["messages_payload"]
    )
    # ... and the child finished properly, a turn after the one that
    # spawned it.
    assert coordinator.snapshot()[0].status == RUN_DONE
    assert child_row["status"] == RUN_DONE
    assert child_row["result"] == "child done"


def test_stopping_the_turn_still_stops_its_children(db, monkeypatch):
    """User cancellation reaches the fleet UNDER THE KILL SWITCH.

    PR3b Task 5 (spec Sec 8) changed the SHIPPED-default fate this test
    used to pin: with `subagents_outlive_turn` on, a user Stop now stops
    the supervisor only and the children keep working (see
    `Tests/Agents/test_fleet_stop_semantics.py`, whose probes were
    measured red/green at the merge-base). What this test guards since
    then is the kill switch's half of that contract: pinned turn-scoped,
    a cancelled turn settles exactly as phase 2 did, byte-identically.
    """
    pin_turn_scoped_children(monkeypatch)
    # The child is blocked inside its provider call, so it can only be
    # abandoned, never joined; 0.2s of grace makes the point in 0.2s.
    monkeypatch.setattr(agent_service, "FLEET_JOIN_TIMEOUT_SECONDS", 0.2)
    entered = threading.Event()
    release = threading.Event()
    cancelled = threading.Event()

    def spawn_then_cancel():
        # Cancel once the child is provably live, so the turn ends with a
        # running child AND a cancellation -- the combination under test.
        if not entered.wait(_JOIN_TIMEOUT):
            raise AssertionError("the child never reached its model call")
        cancelled.set()
        return "parent stopped"

    service, _chat, coordinator = make_fleet_service(
        db,
        [fence(SPAWN_TOOL_NAME, {"task": "slow task"}), spawn_then_cancel],
        {"slow task": [_gated_child(entered, release)]},
    )
    try:
        _run_id, outcome = service.run_turn(
            conversation_id="c",
            messages=[{"role": "user", "content": "go"}],
            config=FLEET_CFG,
            api_endpoint="llama_cpp",
            should_cancel=cancelled.is_set,
        )
        assert entered.is_set()
        assert outcome.status == RUN_CANCELLED
        # Wedged inside its provider call, so it is abandoned rather than
        # unwound -- but never left live, and never left `running`.
        assert coordinator.all_finished()
        assert [h.status for h in coordinator.snapshot()] == [RUN_CANCELLED]
        assert _child_row(db)["status"] == RUN_CANCELLED
    finally:
        release.set()


@pytest.mark.parametrize(
    "outlive, expect_finished",
    [(True, False), (False, True)],
)
def test_the_kill_switch_decides_the_same_childs_fate(
    db, monkeypatch, outlive, expect_finished
):
    """One script, one child, two configs -- opposite outcomes.

    The regression guard in its strongest form: everything except
    `[agents] subagents_outlive_turn` is held identical -- one script, one
    child, one config object -- so the knob is provably the only thing
    deciding whether the turn settles this child or leaves it running.
    """
    pin_agent_settings(
        monkeypatch,
        **{agent_service.SUBAGENTS_OUTLIVE_TURN_KEY: outlive},
    )
    monkeypatch.setattr(agent_service, "FLEET_JOIN_TIMEOUT_SECONDS", 0.2)
    entered = threading.Event()
    release = threading.Event()
    service, _chat, coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "slow task"}),
            _after(entered, "parent answered early"),
        ],
        {"slow task": [_gated_child(entered, release)]},
    )
    try:
        _run_id, outcome = service.run_turn(
            conversation_id="c",
            messages=[{"role": "user", "content": "go"}],
            config=SHORT_WALL_CFG,
            api_endpoint="llama_cpp",
        )
        assert entered.is_set()
        assert outcome.status == RUN_DONE
        assert coordinator.all_finished() is expect_finished
        expected_row = RUN_CANCELLED if expect_finished else RUN_RUNNING
        assert _child_row(db)["status"] == expected_row
    finally:
        release.set()
        if outlive:
            _wait_until(
                coordinator.all_finished, "the released child never finished"
            )


def test_a_survivor_is_out_of_reach_of_the_next_turns_settle(db):
    """`mine` scoping: a later turn must not settle an earlier turn's child.

    The comment on `mine = list(self._fleet_cancels)` called this defensive
    when children could not outlive their turn. It is load-bearing now: the
    injected coordinator here is long-lived (as the Console's will be), so
    turn two can SEE turn one's survivor -- and must leave it alone, even
    though turn two is itself turn-scoped.
    """
    entered = threading.Event()
    release = threading.Event()
    service, _chat, coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "slow task"}),
            _after(entered, "parent answered early"),
        ],
        {"slow task": [_gated_child(entered, release)]},
    )
    try:
        service.run_turn(
            conversation_id="c",
            messages=[{"role": "user", "content": "go"}],
            config=FLEET_CFG,
            api_endpoint="llama_cpp",
        )
        assert entered.is_set()
        # A second turn on the same service, spawning nothing at all.
        _chat.parent_replies.append("second answer")
        _run_id, outcome = service.run_turn(
            conversation_id="c",
            messages=[{"role": "user", "content": "again"}],
            config=FLEET_CFG,
            api_endpoint="llama_cpp",
        )
        assert outcome.final_text == "second answer"
        # Turn one's child is untouched by turn two's settle.
        assert [h.status for h in coordinator.snapshot()] == [RUN_RUNNING]
        assert _child_row(db)["status"] == RUN_RUNNING
    finally:
        release.set()
        _wait_until(
            coordinator.all_finished, "the released child never finished"
        )
    assert _child_row(db)["status"] == RUN_DONE


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
    assert child_call[0]["content"].startswith(SUBAGENT_PROMPT_PREFIX)
    assert child_call[1] == {"role": "user", "content": "compute 6*7"}
    assert not any("delegate this" in m["content"] for m in child_call)
    # Same result text reaches the parent -- via wait_agents rather than
    # via the spawn call's own return value.
    wait_results = _tool_results(db.get_run(run_id), WAIT_AGENTS_TOOL_NAME)
    assert wait_results and "sub answer: 42" in wait_results[0]


def test_max_live_of_one_keeps_spawn_inline(db, monkeypatch):
    """`max_live_subagents = 1` => the pre-PR inline path, result from spawn.

    This is the KILL SWITCH gate. Before Task 6.5 the same guarantee was
    reached by simply not injecting a coordinator (the default was 1);
    since the default is 3, opting out is an explicit config value, and
    this test now pins that value rather than the absence of an injection.
    What it guarantees is unchanged: at a cap of 1 no coordinator is
    built, the child runs inline and synchronously, and `spawn` itself
    returns the child's answer instead of a handle.
    """
    service, _chat = make_inline_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "compute 6*7"}),
            "sub answer: 42",  # consumed by the child, INLINE and in order
            "The sub-agent says 42.",
        ],
        monkeypatch,
    )
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "delegate this"}],
        config=FLEET_CFG,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    assert service._fleet is None
    spawn_results = _tool_results(db.get_run(run_id), SPAWN_TOOL_NAME)
    # The spawn call itself returned the child's answer -- no handle.
    assert spawn_results == ["sub answer: 42"]
    child = next(r for r in db.list_runs("c") if r["agent_kind"] == "subagent")
    assert child["result"] == "sub answer: 42"
    parent_spawn = next(
        step for step in db.get_run(run_id)["steps"] if step["kind"] == "spawn"
    )
    assert child["spawn_event_id"] == f"agent-step:{run_id}:{parent_spawn['index']}"


def test_fleet_tools_are_not_offered_at_max_live_of_one(db, monkeypatch):
    """Same kill-switch pin: at a cap of 1 neither fleet tool is disclosed."""
    service, chat = make_inline_service(db, ["just answering"], monkeypatch)
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
    child_prompt = chat.child_calls["child task"][0]["messages_payload"][0]["content"]
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
# Every other test in this file injects a coordinator. Nothing in
# production does, so `[agents] max_live_subagents` -> `run_turn`'s
# `max_live > 1` branch is the whole of a real user's control over the
# fleet. Task 6.5 moved DEFAULT_MAX_LIVE_SUBAGENTS 1 -> 3, so that branch
# is now taken by DEFAULT and a cap of 1 is the opt-OUT.


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


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (True, True),
        (False, False),
        # TOML strings and env-var strings both land here as text.
        ("true", True),
        ("False", False),
        ("YES", True),
        ("off", False),
        ("1", True),
        ("0", False),
        # Unparseable -> the documented default, never a raise: a typo in
        # the config file must not decide a run's containment silently.
        ("maybe", agent_service.DEFAULT_SUBAGENTS_OUTLIVE_TURN),
        (None, agent_service.DEFAULT_SUBAGENTS_OUTLIVE_TURN),
        ("", agent_service.DEFAULT_SUBAGENTS_OUTLIVE_TURN),
    ],
)
def test_coerce_subagents_outlive_turn(configured, expected):
    assert (
        agent_service._coerce_subagents_outlive_turn(configured) is expected
    )


def test_children_outlive_their_turn_by_default():
    """The shipped default is survival -- asserted, not assumed."""
    assert agent_service.DEFAULT_SUBAGENTS_OUTLIVE_TURN is True
    assert agent_service.SUBAGENTS_OUTLIVE_TURN_KEY == "subagents_outlive_turn"


# -- containment for a THREADED survivor: replacing clamp_child_budget's --
# -- parent-remainder clamp on THAT path only (spec Sec 5)             ----
#
# PR3a-1 Task 5 (spec Sec 5 "Containment"), scope corrected after review
# (Defect 1): a THREADED, non-inline background child deliberately
# outlives its parent (Task 2's default), so its own wall-clock ceiling
# can no longer be `min(child, parent's remaining budget)` for THAT
# child -- that made a surviving child's effective bound an accident of
# WHEN in the turn it was spawned. The replacement:
# `agent_models.contain_child_budget` gives a threaded child its OWN
# independent ceiling, resolved from `[agents] child_max_wall_seconds`
# (default `DEFAULT_CHILD_MAX_WALL_SECONDS`) -- same `_setting`-driven
# config chain as `max_live_subagents` above. A TURN-SCOPED or
# `inline=True` child is UNAFFECTED by any of this: `AgentService.spawn`
# branches on `fleet is None or inline`, and that child still gets
# `clamp_child_budget`'s old parent-remainder clamp, byte-identical to
# every release before this task (see
# `test_clamp_child_budget_for_the_turn_scoped_path_*` in
# `test_agent_models.py`, and `test_an_inline_childs_budget_still_clamps_
# to_the_parents_remainder` below).
#
# SPEND (`max_total_tokens`, passed through unchanged) is untouched by
# this task and already bounds each run independently of the parent's
# lifetime. COUNT was NOT bounded across turns when Task 5 shipped --
# `[agents] max_live_subagents` capped live children WITHIN one
# `run_turn` call only (Task 5 review, Defect 2, disproved by execution:
# two consecutive `run_turn` calls each spawning 2 blocking children
# yielded 4 simultaneously running against a cap of 2). PR3a-1 Task 6a
# fixed it by moving the coordinator's ownership up to
# `ConsoleAgentBridge`, one per conversation, injected into every
# service; `test_live_children_are_capped_across_turns` below is that
# test, INVERTED to assert the cap now holds.


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("900", 900.0),
        (900, 900.0),
        (900.5, 900.5),
        ("0.5", 0.5),
        # The 1s FLOOR is deliberately NOT this function's job -- it lives
        # in `contain_child_budget` (single source of truth, same place
        # `clamp_child_budget`'s own floor already lived), so a
        # non-positive config value passes through unfloored here.
        (0, 0.0),
        (-5, -5.0),
        # Unparseable -> the documented default, never a raise.
        ("plenty", agent_service.DEFAULT_CHILD_MAX_WALL_SECONDS),
        (None, agent_service.DEFAULT_CHILD_MAX_WALL_SECONDS),
        ("", agent_service.DEFAULT_CHILD_MAX_WALL_SECONDS),
        (float("inf"), agent_service.DEFAULT_CHILD_MAX_WALL_SECONDS),
        (float("nan"), agent_service.DEFAULT_CHILD_MAX_WALL_SECONDS),
    ],
)
def test_coerce_child_max_wall_seconds(configured, expected):
    assert agent_service._coerce_child_max_wall_seconds(configured) == expected


def test_a_threaded_childs_own_wall_clock_matches_the_config_default(db):
    """A THREADED child's persisted budget carries the INDEPENDENT
    default ceiling, not the parent's own (`FLEET_CFG.budget.
    max_wall_seconds == 240.0`, deliberately different from the default
    below so this distinguishes old vs new behaviour). `make_fleet_service`
    builds an explicit coordinator, so `fleet is not None` and this spawn
    call is non-inline -- the `contain_child_budget` branch. Also pins
    the depth-1 guarantee end to end.
    """
    service, chat, coordinator = make_fleet_service(
        db,
        [fence(SPAWN_TOOL_NAME, {"task": "child task"}), "handled"],
        {"child task": ["child done"]},
    )
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=FLEET_CFG,
        api_endpoint="llama_cpp",
    )
    join_fleet_children(service)
    child = next(r for r in db.list_runs("c") if r["agent_kind"] == "subagent")
    assert (
        child["budget"]["max_wall_seconds"]
        == agent_service.DEFAULT_CHILD_MAX_WALL_SECONDS
    )
    assert child["budget"]["max_wall_seconds"] != FLEET_CFG.budget.max_wall_seconds
    assert child["budget"]["max_subagents"] == 0  # depth-1 preserved


def test_a_threaded_childs_wall_clock_ceiling_respects_a_config_override(
    db, monkeypatch
):
    """The config key actually reaches the THREADED spawn call, not just
    the default (`make_fleet_service` -> non-inline, so this exercises
    `contain_child_budget`, not `clamp_child_budget`)."""
    pin_agent_settings(monkeypatch, child_max_wall_seconds="77.0")
    service, chat, coordinator = make_fleet_service(
        db,
        [fence(SPAWN_TOOL_NAME, {"task": "child task"}), "handled"],
        {"child task": ["child done"]},
    )
    service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=FLEET_CFG,
        api_endpoint="llama_cpp",
    )
    join_fleet_children(service)
    child = next(r for r in db.list_runs("c") if r["agent_kind"] == "subagent")
    assert child["budget"]["max_wall_seconds"] == 77.0


def test_a_threaded_childs_other_budget_fields_still_inherit_the_parents(db):
    """`contain_child_budget`'s own "everything but wall clock and
    subagent count is unchanged" half, end to end through the real
    spawn() closure on the THREADED path (`make_fleet_service` below
    builds an explicit coordinator, so `fleet is not None` and this
    spawn call is non-inline): the child still inherits the parent's
    round budget, same fields `clamp_child_budget` passes through
    unchanged on the turn-scoped/inline path."""
    cfg = AgentConfig(
        model="test-model",
        system_prompt="You are helpful.",
        allowed_tools=(SPAWN_TOOL_NAME,),
        budget=RunBudget(
            max_steps=40,
            max_model_turns=17,
            max_subagents=2,
            max_total_tokens=5000,
            max_tool_call_seconds=45.0,
        ),
    )
    service, chat, coordinator = make_fleet_service(
        db,
        [fence(SPAWN_TOOL_NAME, {"task": "child task"}), "handled"],
        {"child task": ["child done"]},
    )
    service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=cfg,
        api_endpoint="llama_cpp",
    )
    join_fleet_children(service)
    child = next(r for r in db.list_runs("c") if r["agent_kind"] == "subagent")
    assert child["budget"]["max_model_turns"] == 17
    assert child["budget"]["max_steps"] == 40
    assert child["budget"]["max_total_tokens"] == 5000
    assert child["budget"]["max_tool_call_seconds"] == 45.0


def test_an_inline_childs_budget_still_clamps_to_the_parents_remainder(
    db, monkeypatch
):
    """PR3a-1 Task 5 review, Defect 1 (Major, blocking) -- the regression
    this pins.

    The first version of this task swapped the child-budget call at
    `AgentService.spawn`'s ONE spawn site with no branch, so it hit the
    INLINE path too: every skill call (`functools.partial(spawn,
    inline=True)`), and every spawn when the fleet is off entirely
    (`fleet is None`, this test's own setup via `max_live_subagents = 1`).
    An inline child is turn-scoped by construction -- it blocks the
    parent inside `deps.spawn`, and there is no `_settle_fleet` to bound
    it externally -- so handing it `contain_child_budget`'s INDEPENDENT
    ceiling (unrelated to how much of the parent's own wall clock was
    left) violated the plan's Global Constraint verbatim: "Turn-scoped
    behaviour must stay byte-identical when no child outlives its turn."
    Proved by execution at review time: a child with a 30s ceiling ran
    1.5s past a parent whose own ceiling was 1.0s and returned RUN_DONE;
    reverted to the parent-remainder clamp, it correctly went `stuck`
    instead of returning done.

    This test pins the FIXED behaviour: an inline child's persisted
    budget still comes from `clamp_child_budget` (parent-remainder),
    never from `contain_child_budget`'s independent
    `DEFAULT_CHILD_MAX_WALL_SECONDS` default.
    """
    cfg = AgentConfig(
        model="test-model",
        system_prompt="You are helpful.",
        allowed_tools=("calculator", SPAWN_TOOL_NAME),
        budget=RunBudget(
            max_steps=10, max_model_turns=10, max_subagents=1, max_wall_seconds=100.0
        ),
    )
    service, _chat = make_inline_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "child task"}),
            "sub answer",  # consumed by the child, INLINE and in order
            "handled",
        ],
        monkeypatch,
    )
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=cfg,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    assert service._fleet is None  # confirms this really is the inline path
    child = next(r for r in db.list_runs("c") if r["agent_kind"] == "subagent")
    # NOT the independent default -- the defect this pins would have made
    # this equal DEFAULT_CHILD_MAX_WALL_SECONDS (1800.0) regardless of the
    # parent's own 100.0s ceiling.
    assert (
        child["budget"]["max_wall_seconds"]
        != agent_service.DEFAULT_CHILD_MAX_WALL_SECONDS
    )
    # Clamped to (approximately) the parent's own remaining wall clock at
    # spawn time -- never exceeding the parent's own ceiling, the
    # pre-Task-5 invariant, restored byte-identical.
    assert 0 < child["budget"]["max_wall_seconds"] <= cfg.budget.max_wall_seconds
    assert child["budget"]["max_subagents"] == 0  # depth-1 still holds


def test_live_children_are_capped_across_turns(db, monkeypatch):
    """PR3a-1 Task 6a -- the INVERSION of Task 5's
    `test_live_children_are_not_capped_across_turns`, kept as an
    inversion rather than a rewrite so the gap it pinned cannot quietly
    reopen.

    What that test proved by execution (and Task 5's review disproved a
    claim with): `[agents] max_live_subagents` bounded live children
    WITHIN one `run_turn` call only. `AgentService._run_one` builds a
    brand-new `FleetCoordinator` every `run_turn` that did not have one
    injected, and Console built a new `AgentService` per `run_reply` with
    no `fleet_coordinator=` at all -- so two turns each spawning 2
    children ran 4 at once against a cap of 2, and aggregate live
    children scaled with MESSAGES SENT, bounded by nothing.

    What changed, and why the construction below is now the
    production-faithful one: Task 6a moved the coordinator's ownership UP
    to `ConsoleAgentBridge`, which keeps ONE per conversation and injects
    it into the fresh `AgentService` it still builds for every
    `run_reply`. So this test builds TWO separate services -- one per
    turn, exactly as Console does -- sharing ONE coordinator. It
    deliberately still does not use `make_fleet_service`: that helper
    builds a single service and would prove only that one service reuses
    its own coordinator, which was never the failing case. The failing
    case was the SERVICE being replaced between turns, and that is what
    is reproduced here.

    Sequential by construction, not concurrent threads: PR3a-1 Task 2
    made a still-running child outlive the `run_turn` call that spawned
    it BY DEFAULT, so turn 1's `run_turn()` returns once its own primary
    answers -- without waiting for its two spawned children, which stay
    running in the background -- before this test ever calls turn 2's
    `run_turn()`. Both children stay blocked on the same `Event` until
    released at teardown. `FleetChat`'s parent-reply queue is per
    PRIMARY-run-at-a-time (task-vs-no-task addressing, not run-id), so
    two truly concurrent primaries sharing one `FleetChat` would race on
    it -- sequential calls sidestep that entirely and still prove the
    bound, since this is about COUNT accounting, not concurrency.
    """
    released = threading.Event()

    def blocked_child():
        released.wait(10.0)
        return "released"

    _patch_max_live(monkeypatch, 2)
    cfg = AgentConfig(
        model="test-model",
        system_prompt="You are helpful.",
        allowed_tools=(SPAWN_TOOL_NAME,),
        budget=RunBudget(max_steps=10, max_model_turns=10, max_subagents=2),
    )
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    chat = FleetChat(
        [
            fence(SPAWN_TOOL_NAME, {"task": "a"}),
            fence(SPAWN_TOOL_NAME, {"task": "b"}),
            "turn 1 done",
        ]
        + [
            fence(SPAWN_TOOL_NAME, {"task": "c"}),
            fence(SPAWN_TOOL_NAME, {"task": "d"}),
            "turn 2 done",
        ],
        {
            "a": [blocked_child],
            "b": [blocked_child],
            "c": [blocked_child],
            "d": [blocked_child],
        },
        allow_unconsumed=True,
    )
    # The bridge's own construction, reduced to its load-bearing part: one
    # coordinator per CONVERSATION, one service per TURN.
    fleet = FleetCoordinator(max_live=2, clock=time.monotonic)
    service_1 = AgentService(
        db=db, registry=registry, chat_call=chat, fleet_coordinator=fleet
    )
    service_2 = AgentService(
        db=db, registry=registry, chat_call=chat, fleet_coordinator=fleet
    )
    try:
        _run_id_1, outcome_1 = service_1.run_turn(
            conversation_id="c",
            messages=[{"role": "user", "content": "go 1"}],
            config=cfg,
            api_endpoint="llama_cpp",
        )
        assert outcome_1.status == RUN_DONE
        _wait_until(
            lambda: len(
                [h for h in fleet.snapshot() if h.status == RUN_RUNNING]
            )
            == 2,
            "turn 1's 2 children never both started running",
        )

        # Turn 2: a SECOND, independent service -- a new `run_reply` in
        # Console terms -- on the SAME conversation. Its two spawns must
        # now be REFUSED: turn 1's survivors still hold both slots.
        run_id_2, outcome_2 = service_2.run_turn(
            conversation_id="c",
            messages=[{"role": "user", "content": "go 2"}],
            config=cfg,
            api_endpoint="llama_cpp",
        )
        assert outcome_2.status == RUN_DONE
        assert service_2._fleet is fleet, (
            "the injected coordinator must be honored as-is, not rebuilt"
        )

        # The headline, inverted: 2 live children across two turns
        # against a cap of 2, where before Task 6a it was 4.
        running = [h for h in fleet.snapshot() if h.status == RUN_RUNNING]
        assert len(running) == 2, (
            f"the cap of 2 must hold ACROSS turns; saw {len(running)} "
            f"live: {[h.task for h in running]}"
        )
        assert sorted(h.task for h in running) == ["a", "b"], (
            "the live children must still be turn 1's -- a cap that "
            "held by killing the survivors would be worse than no cap"
        )

        # And the refusal is REAL, not merely uncounted: no child run row
        # was ever created for turn 2's tasks. (`reserve()` returns None
        # before any thread starts, so a capped spawn costs nothing.)
        child_tasks = sorted(
            row["task"]
            for row in db.list_runs("c", include_superseded=True)
            if row["agent_kind"] == "subagent"
        )
        assert child_tasks == ["a", "b"], child_tasks
        # The supervisor is TOLD why, in a retryable form -- an invisible
        # refusal would just look like a broken spawn tool.
        spawn_results = _tool_results(db.get_run(run_id_2), SPAWN_TOOL_NAME)
        assert spawn_results and all(
            "live sub-agent limit reached" in result for result in spawn_results
        ), spawn_results
    finally:
        # Unblocks turn 1's children (turn 2 never started any).
        released.set()
        drain_deadline = time.monotonic() + 2.0
        while time.monotonic() < drain_deadline:
            if not [h for h in fleet.snapshot() if h.status == RUN_RUNNING]:
                break
            time.sleep(0.02)


def test_a_later_turns_settle_does_not_reach_an_earlier_turns_survivor(
    db, monkeypatch
):
    """PR3a-1 Task 6a regression guard for `_settle_fleet`'s
    `mine = list(self._fleet_cancels)` scoping (Task 2), which a
    long-lived coordinator makes load-bearing rather than merely
    defensive.

    Before Task 6a the scoping could not be observed at all: each turn
    built its own coordinator, so "every handle in the fleet" and "this
    turn's handles" were the same set by construction. Now they are not
    -- turn 2 shares a coordinator that still holds turn 1's live
    survivor -- and settling by coordinator membership instead of by
    `_fleet_cancels` would cancel, join and abandon that survivor at the
    end of turn 2, marking its run row `cancelled` while its thread is
    still working. "The next message you send kills your background
    agents" is the feature deleting itself.

    Why turn 2 runs under the KILL SWITCH (`subagents_outlive_turn =
    false`) while turn 1 did not -- this is the whole trick, and without
    it the test is vacuous: on the default path `_surviving_handles`
    returns EVERY pending handle, so a wrongly-widened `mine` still
    spares the survivor and the mutation goes undetected (verified: it
    does). The scoping only bites when a turn genuinely settles -- the
    kill switch, or a user Stop. So turn 1 spawns a survivor under the
    default, then the switch is flipped and turn 2 settles for real,
    which is also a realistic sequence (the user flips the knob, or a
    later turn is cancelled) rather than a contrivance.

    Mutation-checked: rewriting `_settle_fleet`'s `mine` to
    `[h.handle_id for h in fleet.snapshot()]` fails this test on its own
    assertions -- the survivor's cancel Event comes back set, its handle
    terminal and its run row `cancelled`.
    """
    released = threading.Event()

    def blocked_child():
        released.wait(10.0)
        return "released"

    outlive = {"value": True}

    def fake_setting(key, default):
        if key == agent_service.MAX_LIVE_SUBAGENTS_KEY:
            return 3
        if key == agent_service.SUBAGENTS_OUTLIVE_TURN_KEY:
            return outlive["value"]
        return default

    monkeypatch.setattr(agent_service, "_setting", fake_setting)
    cfg = AgentConfig(
        model="test-model",
        system_prompt="You are helpful.",
        allowed_tools=(SPAWN_TOOL_NAME,),
        budget=RunBudget(max_steps=10, max_model_turns=10, max_subagents=2),
    )
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    chat = FleetChat(
        [fence(SPAWN_TOOL_NAME, {"task": "survivor"}), "turn 1 done"]
        + [fence(SPAWN_TOOL_NAME, {"task": "turn 2 child"}), "turn 2 done"],
        {"survivor": [blocked_child], "turn 2 child": ["quick answer"]},
        allow_unconsumed=True,
    )
    fleet = FleetCoordinator(max_live=3, clock=time.monotonic)
    service_1 = AgentService(
        db=db, registry=registry, chat_call=chat, fleet_coordinator=fleet
    )
    service_2 = AgentService(
        db=db, registry=registry, chat_call=chat, fleet_coordinator=fleet
    )
    try:
        _run_id_1, outcome_1 = service_1.run_turn(
            conversation_id="c",
            messages=[{"role": "user", "content": "go 1"}],
            config=cfg,
            api_endpoint="llama_cpp",
        )
        assert outcome_1.status == RUN_DONE
        _wait_until(
            lambda: len(
                [h for h in fleet.snapshot() if h.status == RUN_RUNNING]
            )
            == 1,
            "turn 1's child never started running",
        )
        survivor = next(
            h for h in fleet.snapshot() if h.status == RUN_RUNNING
        )
        # Turn 1's own service still holds the survivor's cancel Event --
        # this is exactly what the bridge keeps a finished run's service
        # for, and what turn 2 must not touch.
        cancel_event = service_1._fleet_cancels[survivor.handle_id]

        outlive["value"] = False  # turn 2 settles its own children for real
        _run_id_2, outcome_2 = service_2.run_turn(
            conversation_id="c",
            messages=[{"role": "user", "content": "go 2"}],
            config=cfg,
            api_endpoint="llama_cpp",
        )
        assert outcome_2.status == RUN_DONE
        # Turn 2 really did settle: its own child is terminal, waited for
        # inside its own turn the phase-2 way.
        turn_2_child = next(
            h for h in fleet.snapshot() if h.task == "turn 2 child"
        )
        assert turn_2_child.status == RUN_DONE, turn_2_child

        # ... and turn 1's survivor is untouched by ALL of it.
        assert not cancel_event.is_set(), (
            "turn 2's settle cancelled turn 1's survivor -- `mine` "
            "scoping lost"
        )
        after = fleet.get(survivor.handle_id)
        assert after is not None and after.status == RUN_RUNNING, after
        row = next(
            r
            for r in db.list_runs("c", include_superseded=True)
            if r["agent_kind"] == "subagent" and r["task"] == "survivor"
        )
        assert row["status"] == RUN_RUNNING, row["status"]

        # ... and it still finishes on its own terms afterwards.
        released.set()
        _wait_until(
            lambda: fleet.get(survivor.handle_id).status == RUN_DONE,
            "the survivor never completed after being released",
        )
        assert fleet.get(survivor.handle_id).result == "released"
    finally:
        released.set()


def test_only_the_service_that_spawned_a_child_can_cancel_it(db, monkeypatch):
    """PR3a-1 Task 6a: `cancel_subagent` reports what it can DELIVER.

    With a per-conversation coordinator, a later turn's service can SEE
    every live handle -- including a survivor it did not start -- but the
    cancel Event lives in the service that spawned it and nowhere else.
    Answering `True` from the wrong service would set no Event, stop
    nothing, and take the user's Cancel button down to a no-op that looks
    like a success: the exact silent-failure class this PR's audit was
    written to hunt. The bridge relies on the honest `False` to fall
    through to the real owner (`ConsoleAgentBridge.cancel_subagent`).
    """
    released = threading.Event()

    def blocked_child():
        released.wait(10.0)
        return "released"

    _patch_max_live(monkeypatch, 3)
    cfg = AgentConfig(
        model="test-model",
        system_prompt="You are helpful.",
        allowed_tools=(SPAWN_TOOL_NAME,),
        budget=RunBudget(max_steps=10, max_model_turns=10, max_subagents=2),
    )
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    chat = FleetChat(
        [fence(SPAWN_TOOL_NAME, {"task": "survivor"}), "turn 1 done"]
        + ["turn 2 done"],
        {"survivor": [blocked_child]},
        allow_unconsumed=True,
    )
    fleet = FleetCoordinator(max_live=3, clock=time.monotonic)
    service_1 = AgentService(
        db=db, registry=registry, chat_call=chat, fleet_coordinator=fleet
    )
    service_2 = AgentService(
        db=db, registry=registry, chat_call=chat, fleet_coordinator=fleet
    )
    try:
        service_1.run_turn(
            conversation_id="c",
            messages=[{"role": "user", "content": "go 1"}],
            config=cfg,
            api_endpoint="llama_cpp",
        )
        _wait_until(
            lambda: len(
                [h for h in fleet.snapshot() if h.status == RUN_RUNNING]
            )
            == 1,
            "turn 1's child never started running",
        )
        survivor = next(
            h for h in fleet.snapshot() if h.status == RUN_RUNNING
        )
        service_2.run_turn(
            conversation_id="c",
            messages=[{"role": "user", "content": "go 2"}],
            config=cfg,
            api_endpoint="llama_cpp",
        )

        # Turn 2's service SEES it ...
        assert any(
            h.handle_id == survivor.handle_id
            for h in service_2.fleet_snapshot()
        )
        # ... does not own it ...
        assert service_2.live_subagent_handles() == []
        assert service_2.cancel_subagent(survivor.handle_id) is False
        assert not service_1._fleet_cancels[survivor.handle_id].is_set(), (
            "a refusal must not have set the Event anyway"
        )
        # ... and turn 1's service, which does, still can.
        assert [h.handle_id for h in service_1.live_subagent_handles()] == [
            survivor.handle_id
        ]
        assert service_1.cancel_subagent(survivor.handle_id) is True
        assert service_1._fleet_cancels[survivor.handle_id].is_set()
    finally:
        released.set()
        _wait_until(
            lambda: fleet.get(survivor.handle_id).status != RUN_RUNNING,
            "the cancelled survivor never unwound",
        )


def test_check_agents_shows_an_earlier_turns_survivor_in_its_own_section(
    db, monkeypatch
):
    """PR3a-1 Task 6a: a survivor must be VISIBLE to the supervisor that
    comes after it, not merely alive.

    `check_agents` scoped its whole answer to `my_handle_ids`, so with a
    per-conversation coordinator a child still working from an earlier
    message would be absent from the one surface that answers "what is
    still running?" -- present in the process, invisible to the agent.
    It is reported in a separate labelled section because `wait_agents`
    deliberately still refuses it (collecting a foreign child's result is
    PR 3a-2's delivery work).
    """
    released = threading.Event()

    def blocked_child():
        released.wait(10.0)
        return "released"

    _patch_max_live(monkeypatch, 3)
    cfg = AgentConfig(
        model="test-model",
        system_prompt="You are helpful.",
        allowed_tools=(SPAWN_TOOL_NAME, CHECK_AGENTS_TOOL_NAME),
        budget=RunBudget(max_steps=10, max_model_turns=10, max_subagents=2),
    )
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    chat = FleetChat(
        [fence(SPAWN_TOOL_NAME, {"task": "long job"}), "turn 1 done"]
        + [fence(CHECK_AGENTS_TOOL_NAME, {}), "turn 2 done"],
        {"long job": [blocked_child]},
        allow_unconsumed=True,
    )
    fleet = FleetCoordinator(max_live=3, clock=time.monotonic)
    service_1 = AgentService(
        db=db, registry=registry, chat_call=chat, fleet_coordinator=fleet
    )
    service_2 = AgentService(
        db=db, registry=registry, chat_call=chat, fleet_coordinator=fleet
    )
    try:
        service_1.run_turn(
            conversation_id="c",
            messages=[{"role": "user", "content": "go 1"}],
            config=cfg,
            api_endpoint="llama_cpp",
        )
        _wait_until(
            lambda: len(
                [h for h in fleet.snapshot() if h.status == RUN_RUNNING]
            )
            == 1,
            "turn 1's child never started running",
        )
        run_id_2, _outcome_2 = service_2.run_turn(
            conversation_id="c",
            messages=[{"role": "user", "content": "go 2"}],
            config=cfg,
            api_endpoint="llama_cpp",
        )
        checks = _tool_results(db.get_run(run_id_2), CHECK_AGENTS_TOOL_NAME)
        assert checks, "turn 2 never called check_agents"
        rendered = "\n".join(checks)
        assert "No sub-agents have been started yet." not in rendered, rendered
        assert "Still running from an earlier turn" in rendered, rendered
        assert "long job" in rendered, rendered
    finally:
        released.set()


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


@pytest.mark.parametrize("configured", ["1", 1, 0, -5])
def test_config_of_one_or_less_keeps_the_inline_path(db, monkeypatch, configured):
    """Anything that RESOLVES to <= 1 means no fleet: the opt-out.

    Before Task 6.5 this parametrization also carried `"nonsense"` and
    `None`, because unparseable config fell back to a default of 1. The
    default is now 3, so junk resolves to a FLEET -- that half moved to
    `test_config_of_junk_now_lands_on_the_default_fleet` below rather than
    being dropped. `-5` replaces them here: a negative is still floored to
    1 by `_coerce_max_live_subagents`, so it still means inline.

    The guarantee itself is unchanged and is what a user opting out gets:
    no coordinator, no fleet tools, spawn returns the child's own answer.
    """
    service, chat = make_inline_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "compute 6*7"}),
            "sub answer: 42",
            "The sub-agent says 42.",
        ],
        monkeypatch,
        max_live=configured,
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


@pytest.mark.parametrize("configured", ["nonsense", None, ""])
def test_config_of_junk_now_lands_on_the_default_fleet(db, monkeypatch, configured):
    """Junk config still never raises -- it lands on the DEFAULT, now 3.

    The other half of the old `test_config_of_one_or_junk_keeps_the_
    inline_path`: its junk cases asserted "no fleet" only because the
    default they fall back to WAS 1. The coverage worth keeping is that a
    malformed `[agents] max_live_subagents` never stops a run and always
    resolves to a defined size -- so this asserts the same inputs against
    what the default now is: a live fleet, threading its spawns.
    """
    _patch_max_live(monkeypatch, configured)
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
    assert service._fleet._max_live == agent_service.DEFAULT_MAX_LIVE_SUBAGENTS
    spawn_results = _tool_results(db.get_run(run_id), SPAWN_TOOL_NAME)
    assert spawn_results and spawn_results[0].startswith("started ")


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
        handle = next(h for h in coordinator.snapshot() if h.task == "task one")
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


def test_wait_agents_cancellation_stops_children_and_ends_the_run(
    db, monkeypatch
):
    """User cancellation around a wait propagates -- UNDER THE KILL SWITCH.

    The child keeps calling a tool (with varying arguments, so the cycle
    detector never fires), which gives it the step boundaries at which a
    cooperative cancel is actually noticed.

    PR3b Task 5 pinned this turn-scoped: on the shipped default a Stop
    now spares the children (`test_fleet_stop_semantics`), so the kill
    propagation asserted here is what `subagents_outlive_turn = false`
    buys. Honest mechanics note, verified while writing that suite's
    probes: this script flips `cancelled` BEFORE returning the wait
    fence, so the parent dies at the loop's pre-dispatch cancellation
    gate and `wait_agents` never actually runs -- the children die
    through the child-side parent poll and the end-of-turn settle, both
    of which this kill-switch path keeps. `wait_agents`' own cancel
    branch is exercised (both key directions) by
    `test_fleet_stop_semantics`' in-wait triggers.
    """
    pin_turn_scoped_children(monkeypatch)
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
        {"busy": [fence("calculator", {"expression": f"1+{n}"}) for n in range(60)]},
        # The child is CANCELLED mid-flight: its remaining scripted turns
        # are meant to go unused -- that is the point of the test.
        allow_unconsumed=True,
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


def test_cancelling_and_abandoning_a_child_revokes_its_approval_cards(
    db, monkeypatch
):
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

    Pinned turn-scoped (PR3a-1 Task 2): revocation is what STOPPING a
    child does, and a surviving child is not stopped -- its card belongs
    to a run that is still live, so revoking it would fail a legitimate
    tool call closed (asserted by
    ``test_a_survivors_approval_cards_are_not_revoked``).
    """
    pin_turn_scoped_children(monkeypatch)
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
        # Counted, not set-collapsed (review M2): the two revoke sites are
        # SEPARATE moments and each must be pinned on its own. A set
        # comparison stayed green when either one was deleted. This child
        # is wedged, so it is revoked exactly twice -- once by
        # `_cancel_fleet_handles`' cooperative cancel, once by
        # `_settle_fleet`'s abandon of what the join could not reclaim.
        assert Counter(revoked) == Counter({run: 2 for run in child_run_ids}), (
            f"expected each stopped child revoked at both moments: {revoked}"
        )
        # Never the parent, whose own card (if any) belongs to a live run.
        assert _run_id not in revoked
    finally:
        never.set()


# -- PR2b Task 5: per-row cancel + cost rollup ----------------------------


def test_cancel_subagent_revokes_approval_cards_mid_run(db):
    """PR2b Task 5: a UI-initiated per-row cancel (``AgentService.
    cancel_subagent``) revokes the child's pending approval cards
    SYNCHRONOUSLY, mid-run -- the same PR2a guarantee
    ``_cancel_fleet_handles`` already provides at end-of-turn (see
    ``test_cancelling_and_abandoning_a_child_revokes_its_approval_cards``
    above), now reachable on demand for ONE specific handle without
    waiting for the whole run to finish. This is the Console rail's
    per-row Cancel action's actual production path
    (``ConsoleAgentBridge.cancel_subagent`` -> here).
    """
    never = threading.Event()
    entered = threading.Event()

    def wedged_child():
        entered.set()
        never.wait(30.0)  # released in the finally below, not by the run
        return "unreachable"

    revoked: list[str] = []
    cfg = AgentConfig(
        model="test-model",
        system_prompt="You are helpful.",
        allowed_tools=(SPAWN_TOOL_NAME,),
        budget=RunBudget(
            max_steps=40,
            max_model_turns=40,
            max_subagents=2,
            max_wall_seconds=30.0,
        ),
    )
    service, _chat, coordinator = make_fleet_service(
        db,
        [fence(SPAWN_TOOL_NAME, {"task": "wedged"}), "parent done"],
        {"wedged": [wedged_child]},
        revoke_approvals=revoked.append,
    )

    result: dict = {}

    def do_run():
        result["outcome"] = service.run_turn(
            conversation_id="c",
            messages=[{"role": "user", "content": "go"}],
            config=cfg,
            api_endpoint="llama_cpp",
        )

    runner = threading.Thread(target=do_run, name="test-cancel-subagent-run")
    runner.start()
    try:
        assert entered.wait(5), "child never reached its wedged call"
        # `attach_run` (fired from inside `_run_one`) races this thread's
        # own read of the handle -- poll briefly rather than assume it has
        # already landed by the time `entered` is set.
        handle = None
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            snapshot = coordinator.snapshot()
            if snapshot and snapshot[0].run_id:
                handle = snapshot[0]
                break
            time.sleep(0.01)
        assert handle is not None and handle.run_id, (
            "the child's handle never attached a run id"
        )
        assert handle.status == "running"

        # An unknown handle id is a clean no-op, discriminated from the
        # real, live one below.
        assert service.cancel_subagent("not-a-real-handle") is False
        assert revoked == []

        assert service.cancel_subagent(handle.handle_id) is True
        # The revoke happens SYNCHRONOUSLY inside `cancel_subagent` --
        # observable immediately, well before the run itself finishes (the
        # child is still wedged; nothing else has had a chance to revoke
        # anything yet).
        assert revoked == [handle.run_id]
    finally:
        never.set()
    runner.join(10)
    assert not runner.is_alive(), "run_turn never returned"
    _wait_until(
        lambda: db.get_run(handle.run_id)["status"] == RUN_CANCELLED,
        "cancelled child never persisted its terminal status",
    )
    child = db.get_run(handle.run_id)
    assert child["status"] == RUN_CANCELLED
    assert any(
        step["kind"] == "agent_run_cancelled" for step in child["steps"]
    )
    path = db.db_path
    db.close()
    reopened = AgentRunsDB(path, client_id="cancel-reload")
    runs = reopened.list_runs("c", include_superseded=True)
    records = [
        record
        for turn in derive_trajectory(
            messages=[],
            usage_by_id={},
            traj_rows=[],
            variant_sets=[],
            compaction_records=[],
            agent_runs=runs,
            agent_steps=[
                {**step, "run_id": row["id"], "conversation_id": "c"}
                for row in runs
                for step in row["steps"]
            ],
        ).turns
        for record in turn.records
    ]
    assert any(
        record.run_id == handle.run_id and record.kind == "agent_run_cancelled"
        for record in records
    )
    reopened.close()


def test_cancel_subagent_returns_false_with_no_fleet_yet(db):
    """A service that has never run a turn has no live fleet to cancel
    against -- a clean `False`, not an `AttributeError` on `self._fleet`."""
    service, _chat, _coordinator = make_fleet_service(
        db, ["never used"], allow_unconsumed=True
    )
    assert service.cancel_subagent("whatever") is False


def test_cancel_subagent_returns_false_for_an_already_terminal_handle(db):
    """A child that already finished normally cannot be cancelled again --
    `_pending_handles` (the same liveness test `_settle_fleet` itself
    uses) reports it done, so `cancel_subagent` no-ops rather than issuing
    a pointless (and, per `_revoke_handle_approvals`, redundant-revoke)
    cancel against a handle nothing is waiting on anymore."""
    service, _chat, coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "task one"}),
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "combined answer",
        ],
        {"task one": ["answer one"]},
    )
    service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=FLEET_CFG,
        api_endpoint="llama_cpp",
    )
    handle = coordinator.snapshot()[0]
    assert handle.status == RUN_DONE
    assert service.cancel_subagent(handle.handle_id) is False


def test_finished_children_record_their_measured_token_spend_on_the_handle(db):
    """PR2b Task 5 (cost rollup): ``FleetHandle.total_tokens`` is
    populated from each child's own ``RunOutcome.total_tokens`` once it
    finishes -- the live source ``Console_Modules/agent.py`` sums for the
    fleet rail's per-row token segment and the Console cost ticker's fleet
    aggregate (``ConsoleAgentController._console_agent_fleet_token_
    total``).
    """
    service, _chat, coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "task one"}),
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "combined answer",
        ],
        {"task one": ["a real answer with enough text to have a real token count"]},
    )
    service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=FLEET_CFG,
        api_endpoint="llama_cpp",
    )
    assert coordinator.all_finished()
    handle = coordinator.snapshot()[0]
    assert handle.status == RUN_DONE
    # No provider `usage` block in this scripted reply, so the runtime
    # estimated it (`_usage_total_tokens` -> the local estimator fallback,
    # `agent_service.py`) -- still a REAL, non-placeholder measured figure.
    assert handle.total_tokens > 0


def test_wait_agents_is_bounded_by_the_runs_remaining_wall_clock(
    db, monkeypatch
):
    """A wedged child must not hold wait_agents past the run's budget.

    Pinned turn-scoped (PR3a-1 Task 2) so the tail assertions -- every
    handle and every run row terminal once the turn returns -- still
    describe this test's own subject rather than the new default.
    """
    pin_turn_scoped_children(monkeypatch)
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
        # The run is cut short by its 1s wall clock, so the parent's last
        # scripted turn ("gave up") is deliberately never reached.
        allow_unconsumed=True,
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
        # "never reached" is literal: _run_one is monkeypatched to raise
        # before the child ever asks for a reply.
        {"boom": ["never reached"]},
        allow_unconsumed=True,
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


def test_setup_phase_exception_still_persists_a_terminal_db_status(db, monkeypatch):
    """A setup-phase exception must not leave the child's DB row `running`.

    `_run_one`'s own `except Exception` (which calls `_persist`) wraps
    ONLY the `run_agent_loop(...)` call. Anything between `create_run()`
    and that try block -- notably `initial_disclosure`, which walks the
    tool catalog's cache/lock path -- is unprotected: raising there
    unwinds `_run_one` entirely, past `_persist`, straight into
    `run_child`'s `except BaseException`. Unlike
    `test_child_thread_exception_finishes_the_handle_as_error` above
    (which replaces the whole of `_run_one`, so `create_run()` never
    runs and no DB row ever exists), this raises AFTER `create_run()`
    and `attach_run()` have already fired -- a DB row exists, in
    `running` status, and nothing but `run_child`'s `finally` can ever
    mark it terminal. Before the fix that `finally` only called
    `fleet.finish()` (in-memory), so the row stayed `running` for the
    life of the process -- only `reconcile_orphaned_runs` on the NEXT
    app restart would clear it, violating spec Sec 3 invariant 3
    ("DB is truth").
    """
    service, _chat, coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "boom"}),
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "handled",
        ],
        # "never reached" is literal: initial_disclosure blows up before
        # the child ever asks the model for a reply.
        {"boom": ["never reached"]},
        allow_unconsumed=True,
    )
    real_initial_disclosure = agent_service.initial_disclosure

    def exploding_initial_disclosure(registry, budget):
        # Only a depth-1 CHILD's budget has max_subagents == 0 -- zeroed
        # by whichever containment function built it (this test's child
        # is threaded, via `contain_child_budget`; an inline child would
        # get the same zero from `clamp_child_budget` instead, PR3a-1
        # Task 5) -- the primary's is > 0. This fires only
        # for the child -- standing in for a misbehaving provider's
        # `list_catalog()` recursing into the tool catalog's RLock
        # (Task 4's own documented trigger for this exact exception).
        if budget.max_subagents == 0:
            raise RecursionError("setup-phase blew up")
        return real_initial_disclosure(registry, budget)

    monkeypatch.setattr(
        agent_service, "initial_disclosure", exploding_initial_disclosure
    )

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
    assert handle.run_id, "attach_run must have fired before the exception"

    # The headline assertion: the child's DB row reached a terminal
    # status. Before the fix this read "running" forever.
    child_row = db.get_run(handle.run_id)
    assert child_row is not None
    assert child_row["status"] in TERMINAL_RUN_STATUSES
    assert child_row["status"] != "running"


class _SpyRunLogWriter:
    """Minimal stand-in recording the run tree's two cleanup calls.

    Args:
        observe: optional zero-arg callable invoked from INSIDE
            ``write_manifest``; whatever it returns is appended to
            ``observed``. That is the only honest way to assert what was
            true *at manifest time* rather than after the turn -- the
            ordering constraint the settle sits under.
    """

    def __init__(self, observe=None):
        self.is_active = False
        self.log_dir = None
        self.bound: str | None = None
        self.manifests: list[dict] = []
        self.closed = 0
        self._observe = observe
        self.observed: list = []

    def bind(self, run_id):
        self.bound = run_id

    def append(self, **kwargs):
        return None

    def write_manifest(self, data):
        if self._observe is not None:
            self.observed.append(self._observe())
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
    # The model was told, and the pre-dispatch durable owner records the
    # failed launch even though the child thread never ran.
    spawn_results = _tool_results(db.get_run(run_id), SPAWN_TOOL_NAME)
    assert len(spawn_results) == 2
    assert "could not start sub-agent" in spawn_results[0]
    # The spawn slot was given back: the second spawn started for real.
    assert spawn_results[1].startswith("started ")
    assert db.count_subagent_runs("c") == 2
    doomed_row = next(row for row in db.list_runs("c") if row["task"] == "doomed")
    assert doomed_row["status"] == RUN_ERROR
    assert [
        step["kind"]
        for step in doomed_row["steps"]
        if step["kind"].startswith("agent_run_")
    ] == ["agent_run_reserved", "agent_run_created", "agent_run_failed"]
    waits = _tool_results(db.get_run(run_id), WAIT_AGENTS_TOOL_NAME)
    assert waits and "answer two" in waits[0]


def test_thread_start_and_transient_terminal_status_failure_are_both_contained(
    db, monkeypatch
):
    real_start = threading.Thread.start
    start_failed = False

    def fail_fleet_start_once(self):
        nonlocal start_failed
        if self.name.startswith("fleet-") and not start_failed:
            start_failed = True
            raise RuntimeError("can't start new thread")
        return real_start(self)

    real_set_terminal = db.set_terminal_with_step
    status_attempts = 0

    def fail_child_terminal_once(run_id, status, result, terminal_step):
        nonlocal status_attempts
        row = db.get_run(run_id)
        if row and row["agent_kind"] == "subagent" and status == RUN_ERROR:
            status_attempts += 1
            if status_attempts == 1:
                raise RuntimeError("transient terminal write failure")
        return real_set_terminal(run_id, status, result, terminal_step)

    monkeypatch.setattr(threading.Thread, "start", fail_fleet_start_once)
    monkeypatch.setattr(db, "set_terminal_with_step", fail_child_terminal_once)
    service, _chat, coordinator = make_fleet_service(
        db,
        [fence(SPAWN_TOOL_NAME, {"task": "doomed"}), "parent completed"],
        {},
    )

    _parent_id, outcome = service.run_turn(
        conversation_id="thread-and-status-failure",
        messages=[{"role": "user", "content": "delegate"}],
        config=FLEET_CFG,
        api_endpoint="llama_cpp",
    )

    assert outcome.status == RUN_DONE
    assert status_attempts == 2
    assert coordinator.all_finished()
    child = next(
        row
        for row in db.list_runs("thread-and-status-failure")
        if row["agent_kind"] == "subagent"
    )
    assert child["status"] == RUN_ERROR
    assert any(step["kind"] == "agent_run_failed" for step in child["steps"])


def test_thread_start_and_persistent_terminal_failure_reconciles_on_reopen(
    db, monkeypatch
):
    real_start = threading.Thread.start

    def fail_fleet_start(self):
        if self.name.startswith("fleet-"):
            raise RuntimeError("can't start new thread")
        return real_start(self)

    def fail_child_terminal(run_id, status, result, terminal_step):
        row = db.get_run(run_id)
        if row and row["agent_kind"] == "subagent" and status == RUN_ERROR:
            raise RuntimeError("persistent terminal write failure")
        return False

    monkeypatch.setattr(threading.Thread, "start", fail_fleet_start)
    monkeypatch.setattr(db, "set_terminal_with_step", fail_child_terminal)
    service, _chat, coordinator = make_fleet_service(
        db,
        [fence(SPAWN_TOOL_NAME, {"task": "doomed"}), "parent completed"],
        {},
    )

    _parent_id, outcome = service.run_turn(
        conversation_id="persistent-thread-status-failure",
        messages=[{"role": "user", "content": "delegate"}],
        config=FLEET_CFG,
        api_endpoint="llama_cpp",
    )

    assert outcome.status == RUN_DONE
    assert coordinator.all_finished()
    child = next(
        row
        for row in db.list_runs("persistent-thread-status-failure")
        if row["agent_kind"] == "subagent"
    )
    assert child["status"] == "running"
    assert any(step["kind"] == "capture_failed" for step in child["steps"])
    path = db.db_path
    db.close()
    AgentRunsDB._swept_paths.discard(str(path))

    reopened = AgentRunsDB(path, client_id="persistent-launch-reload")
    repaired = reopened.get_run(child["id"])
    assert repaired["status"] == RUN_ERROR
    assert repaired["result"] == "Interrupted by app restart"
    assert len(
        [step for step in repaired["steps"] if step["kind"] == "capture_failed"]
    ) == 2
    reopened.close()


def test_a_settled_child_is_settled_before_the_manifest_is_written(
    db, monkeypatch
):
    """The ordering constraint, asserted AT manifest time.

    Whatever the turn still settles must be finished before
    `write_manifest`/`close` run -- otherwise the run tree's manifest
    describes a tree still in motion and the writer is closed under a
    child that is still appending. Observed from inside `write_manifest`,
    because after `run_turn` returns every ordering looks the same.
    """
    pin_turn_scoped_children(monkeypatch)
    entered = threading.Event()
    release = threading.Event()
    live: dict = {}
    spy = _SpyRunLogWriter(
        observe=lambda: [h.status for h in live["coordinator"].snapshot()]
    )
    service, _chat, coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "slow task"}),
            _after(entered, "parent answered early"),
        ],
        {"slow task": [_gated_child(entered, release, reply="on time")]},
        run_log_writer=spy,
    )
    live["coordinator"] = coordinator
    # Released from another thread shortly after the turn ends, so the
    # child finishes NORMALLY inside the settle's wait -- the ordinary
    # case, not the abandonment path.
    releaser = threading.Thread(target=lambda: (time.sleep(0.1), release.set()))
    releaser.start()
    try:
        _run_id, outcome = service.run_turn(
            conversation_id="c",
            messages=[{"role": "user", "content": "go"}],
            config=FLEET_CFG,
            api_endpoint="llama_cpp",
        )
    finally:
        release.set()
        releaser.join(_JOIN_TIMEOUT)
    assert outcome.status == RUN_DONE
    # Read from INSIDE write_manifest: the child was already terminal.
    assert spy.observed == [[RUN_DONE]], (
        f"handle statuses at manifest time: {spy.observed}"
    )
    assert len(spy.manifests) == 1
    assert spy.closed == 1


def test_a_survivor_does_not_cost_the_turn_its_manifest(db):
    """Survival must not skip finalization -- only defer the child.

    The settle is wrapped precisely so nothing about the fleet can cost
    the run tree its manifest or leak the writer's descriptor. A turn that
    leaves a child running still writes exactly one manifest and closes
    exactly once, and it does so with the survivor still live -- which is
    the seam Task 3 (writer lifetime) owns, recorded here rather than
    left to be discovered.
    """
    entered = threading.Event()
    release = threading.Event()
    spy = _SpyRunLogWriter()
    service, _chat, coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "slow task"}),
            _after(entered, "parent answered early"),
        ],
        {"slow task": [_gated_child(entered, release)]},
        run_log_writer=spy,
    )
    try:
        run_id, outcome = service.run_turn(
            conversation_id="c",
            messages=[{"role": "user", "content": "go"}],
            config=FLEET_CFG,
            api_endpoint="llama_cpp",
        )
        assert outcome.status == RUN_DONE
        assert len(spy.manifests) == 1
        assert spy.manifests[0]["run_id"] == run_id
        assert spy.closed == 1
        # Written while the child is still working, deliberately.
        assert not coordinator.all_finished()
    finally:
        release.set()
        _wait_until(
            coordinator.all_finished, "the released child never finished"
        )


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
        budget=RunBudget(max_steps=40, max_model_turns=40, max_subagents=3),
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


def test_an_in_loop_runtime_tool_sees_its_own_runs_id(db):
    """`run_skill_script` is dispatched IN-LOOP, and it arms a confirm card.

    Review M1: unlike a provider tool, the runtime tools never go through
    ``invoke_tool``'s per-call daemon thread -- ``run_agent_loop`` calls
    them directly on the run's own thread. ``run_skill_script`` raises a
    consent card whose round records ``current_run_id()`` so a cancelled
    child's confirm can be revoked; if the loop thread had no binding,
    that ownership would read ``""``, the revoke would silently match
    nothing, and a clicked Allow would still execute the script.

    The tool is all-agents scope, so the CHILD is the case that matters.
    """
    seen: list[str] = []

    def run_skill_script_tool(skill_name, script_path, args):
        seen.append(current_run_id())
        return ToolResult(ok=True, content="exit_code: 0")

    cfg = AgentConfig(
        model="test-model",
        system_prompt="You are helpful.",
        allowed_tools=(SPAWN_TOOL_NAME,),
        budget=RunBudget(max_steps=40, max_model_turns=40, max_subagents=2),
    )
    service, _chat, _coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "script task"}),
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "done",
        ],
        {
            "script task": [
                fence(
                    RUN_SKILL_SCRIPT_TOOL_NAME,
                    {"skill_name": "demo", "script_path": "run.sh", "args": []},
                ),
                "child done",
            ]
        },
        run_skill_script_tool=run_skill_script_tool,
    )
    _run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=cfg,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    child = next(r for r in db.list_runs("c") if r["agent_kind"] == "subagent")
    assert seen == [child["id"]], (
        f"the child's in-loop runtime tool ran unbound or as another run: {seen}"
    )


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


@pytest.mark.parametrize(
    "configured, expected",
    [
        (True, True),
        (False, False),
        ("true", True),
        ("FALSE", False),
        ("on", True),
        ("off", False),
        ("1", True),
        ("0", False),
        # Unparseable -> the documented default, never a raise, never its
        # opposite: same posture as its sibling switches above.
        ("maybe", agent_service.DEFAULT_AUTOWAKE_ENABLED),
        (None, agent_service.DEFAULT_AUTOWAKE_ENABLED),
        ("", agent_service.DEFAULT_AUTOWAKE_ENABLED),
    ],
)
def test_coerce_autowake_enabled(configured, expected):
    assert agent_service._coerce_autowake_enabled(configured) is expected


def test_autowake_ships_on_by_default():
    """Spec Sec 3 invariant 5 (corrected 2026-08-11): the shipped default
    is auto-wake ON -- asserted, not assumed."""
    assert agent_service.DEFAULT_AUTOWAKE_ENABLED is True
    assert agent_service.AUTOWAKE_ENABLED_KEY == "autowake_enabled"
