# Tests/Agents/test_fleet_send_to_agent.py
"""Fleet PR 3b Task 2: ``send_to_agent`` -- the supervisor steering producer.

The first producer for Task 1's per-child mailbox (spec SS6 "two paths one
mechanism"; SS3 invariant 4 "steering never cancels"). Plan:
Docs/superpowers/plans/2026-08-17-fleet-pr3b-steering.md, Task 2.

The plan-mandated reds live here:
  - the schema is offered only to a fleet-active PRIMARY: absent without a
    fleet, absent for a sub-agent (depth-1: children cannot steer each
    other), and a child's hallucinated call is refused like any other
    undisclosed name;
  - end-to-end through the fake provider: the supervisor calls the tool
    and the child's next model-turn payload carries the
    ``[Steering from supervisor]``-labeled user-role message, LAST, after
    the batch's tool results;
  - refusal shapes, each with its own copy: empty message, oversize
    message, unknown id (the producers validate -- Task 1 deliberately
    left ``post_steering`` unvalidating);
  - id resolution speaks BOTH vocabularies (handle id from spawn results /
    check_agents; run id from completion notices), resolves over the WHOLE
    coordinator (a foreign live survivor is steerable -- steering, unlike
    cancel, needs no per-service state), and prefers the handle id when a
    forged run id collides with another child's handle id;
  - steering never cancels: after a post the child's coordinator status,
    its run row, and its cancel Event are all untouched;
  - steering never satisfies an approval: with a real approval round armed
    and held open, a steering post leaves the verdict pending, the gated
    tool unexecuted, and the entry queued -- delivered only after the
    round resolves (mirroring the wake's not-user-input pin,
    ``console_fleet_wake.WAKE_NOTICE_DISCLAIMER``).
"""

from __future__ import annotations

import threading
import time

import pytest

from Tests.Agents.test_agent_service import FleetChat, fence
from Tests.Agents.test_fleet_runtime import (
    _JOIN_TIMEOUT,
    CARD_CFG,
    FLEET_CFG,
    _child_row,
    _tool_results,
    _wait_until,
    make_fleet_service,
    make_inline_service,
)
from tldw_chatbook.Agents import run_log as run_log_module
from tldw_chatbook.Agents.agent_models import (
    FENCE_TOOL_RESULT_PREFIX,
    MAX_STEERING_CHARS,
    RUN_DONE,
    RUN_RUNNING,
    RUNTIME_TOOL_NAMES,
    SEND_TO_AGENT_TOOL_NAME,
    SPAWN_TOOL_NAME,
    STEERING_SOURCE_SUPERVISOR,
    TERMINAL_RUN_STATUSES,
    WAIT_AGENTS_TOOL_NAME,
    format_steering_message,
)
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.fleet_coordinator import FleetCoordinator
from tldw_chatbook.Agents.run_log_search import load_records
from tldw_chatbook.Agents.tool_catalog import (
    SEND_TO_AGENT_SCHEMA,
    BuiltinToolProvider,
    ToolCatalogRegistry,
)
from tldw_chatbook.Chat.trajectory import derive_trajectory
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

#: How long a parked approval card waits for the test to answer it.
_CARD_TIMEOUT = 10.0


@pytest.fixture()
def db(tmp_path):
    return AgentRunsDB(tmp_path / "runs.db", client_id="test")


def _live_running_handle(coordinator):
    """The single RUNNING handle -- these scripts keep at most one live."""
    return next(h for h in coordinator.snapshot() if h.status == RUN_RUNNING)


def _sends(db, run_id):
    """The supervisor's recorded send_to_agent results, in order."""
    return _tool_results(db.get_run(run_id), SEND_TO_AGENT_TOOL_NAME)


# -- registration (the runtime-tool mandate: name + set + schema) ---------


def test_send_to_agent_is_a_registered_runtime_tool():
    assert SEND_TO_AGENT_TOOL_NAME == "send_to_agent"
    assert SEND_TO_AGENT_TOOL_NAME in RUNTIME_TOOL_NAMES
    assert SEND_TO_AGENT_SCHEMA.name == SEND_TO_AGENT_TOOL_NAME
    assert SEND_TO_AGENT_SCHEMA.id == "runtime:send_to_agent"
    # BOTH parameters are required -- an id-less or message-less steer has
    # no meaning, unlike wait_agents' optional ids.
    assert SEND_TO_AGENT_SCHEMA.parameters["required"] == ["id", "message"]
    assert SEND_TO_AGENT_SCHEMA.parameters["properties"]["id"]["type"] == "string"
    assert (
        SEND_TO_AGENT_SCHEMA.parameters["properties"]["message"]["type"] == "string"
    )


def test_the_schema_teaches_ids_latency_and_never_cancels():
    """The description is the supervisor's whole curriculum: both id
    vocabularies, the delivery latency (next model turn; a long tool call
    delays it), and that steering never cancels or restarts the child."""
    text = SEND_TO_AGENT_SCHEMA.description
    assert "handle id" in text
    assert "run id" in text
    assert "next model turn" in text
    assert "long tool call" in text
    assert "never cancels" in text
    param_doc = SEND_TO_AGENT_SCHEMA.parameters["properties"]["id"]["description"]
    assert "handle id" in param_doc and "run id" in param_doc


# -- gating: fleet-active primaries only ----------------------------------


def test_schema_offered_with_a_fleet_and_absent_without(db, monkeypatch):
    """The schema rides the exact `fleet_active` predicate wait_agents
    uses: offered to a primary with a live fleet, absent on the inline
    path (max_live 1 -- there is no mailbox to post into)."""
    service, chat, _coordinator = make_fleet_service(db, ["just answering"], {})
    service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=FLEET_CFG,
        api_endpoint="llama_cpp",
    )
    fleet_prompt = chat.parent_calls[0]["messages_payload"][0]["content"]
    assert SEND_TO_AGENT_TOOL_NAME in fleet_prompt

    inline_service, inline_chat = make_inline_service(
        db, ["just answering"], monkeypatch
    )
    inline_service.run_turn(
        conversation_id="c2",
        messages=[{"role": "user", "content": "go"}],
        config=FLEET_CFG,
        api_endpoint="llama_cpp",
    )
    inline_prompt = inline_chat.calls[0]["messages_payload"][0]["content"]
    assert SEND_TO_AGENT_TOOL_NAME not in inline_prompt


def test_schema_is_primary_only_and_a_childs_call_is_refused(db):
    """Depth-1: children cannot steer each other. A sub-agent never sees
    the schema, and its hallucinated call falls through to the ordinary
    permission path like any other undisclosed tool name."""
    service, chat, _coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "child task"}),
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "done",
        ],
        {
            "child task": [
                # The child tries to steer a sibling anyway.
                fence(SEND_TO_AGENT_TOOL_NAME, {"id": "some-id", "message": "hi"}),
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
    parent_prompt = chat.parent_calls[0]["messages_payload"][0]["content"]
    assert SEND_TO_AGENT_TOOL_NAME in parent_prompt
    child_prompt = chat.child_calls["child task"][0]["messages_payload"][0]["content"]
    assert SEND_TO_AGENT_TOOL_NAME not in child_prompt
    child = next(r for r in db.list_runs("c") if r["agent_kind"] == "subagent")
    refusals = _tool_results(child, SEND_TO_AGENT_TOOL_NAME)
    assert refusals and "not permitted" in refusals[0]


# -- the supervisor path, end to end --------------------------------------


def test_supervisor_steers_a_live_child_end_to_end(db, tmp_path, monkeypatch):
    """The whole seam: the supervisor calls the tool, the ok copy states
    queued-plus-latency honestly, and the child's next model-turn payload
    ends with the ``[Steering from supervisor]``-labeled user message,
    after the batch's tool result -- Task 1's coherent boundary."""
    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: tmp_path)
    entered = threading.Event()
    steered = threading.Event()
    holder: dict = {}
    steering_message = (
        "reasoning_content: wrap up using /Users/alice/secret.txt and "
        "api_key=sk-private-steering"
    )

    def steer():
        assert entered.wait(_JOIN_TIMEOUT), "the child never reached its model call"
        handle = _live_running_handle(holder["coordinator"])
        holder["handle_id"] = handle.handle_id
        return fence(
            SEND_TO_AGENT_TOOL_NAME,
            {"id": handle.handle_id, "message": steering_message},
        )

    def release_then_wait():
        steered.set()
        return fence(WAIT_AGENTS_TOOL_NAME, {})

    def gated_child():
        entered.set()
        assert steered.wait(_JOIN_TIMEOUT), "the supervisor never finished steering"
        return fence("calculator", {"expression": "6*7"})

    service, chat, coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "steer target"}),
            steer,
            release_then_wait,
            "combined answer",
        ],
        {"steer target": [gated_child, "child answer"]},
    )
    holder["coordinator"] = coordinator
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=FLEET_CFG,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE

    labeled = format_steering_message(STEERING_SOURCE_SUPERVISOR, steering_message)
    assert labeled.startswith("[Steering from supervisor] ")
    child_turns = chat.child_calls["steer target"]
    assert len(child_turns) == 2
    second_payload = child_turns[1]["messages_payload"]
    assert second_payload[-1] == {"role": "user", "content": labeled}
    assert str(second_payload[-2]["content"]).startswith(
        f"{FENCE_TOOL_RESULT_PREFIX}calculator:"
    )
    # The ok copy is the spec's latency honesty: queued, delivered before
    # the child's next model turn -- never "delivered now".
    sends = _sends(db, run_id)
    assert sends and "ERROR" not in sends[0]
    assert "queued" in sends[0] and "next model turn" in sends[0]
    # Consumed from the mailbox, and the label never leaks into the
    # SUPERVISOR's own payloads (the mechanism prepends it for the child).
    assert coordinator.get(holder["handle_id"]).queued_steering == 0
    for call in chat.parent_calls:
        assert not any(
            labeled in str(m.get("content", "")) for m in call["messages_payload"]
        )

    parent = db.get_run(run_id)
    send_step = next(
        step
        for step in parent["steps"]
        if step["kind"] == "tool_call"
        and step["tool_name"] == SEND_TO_AGENT_TOOL_NAME
    )
    send_event_id = f"agent-step:{run_id}:{send_step['index']}"
    child = next(row for row in db.list_runs("c") if row["agent_kind"] == "subagent")
    steering = next(step for step in child["steps"] if step["kind"] == "steering")
    assert steering["parent_event_id"] == send_event_id
    assert steering["source_event_id"] == send_event_id

    path = db.db_path
    db.close()
    reopened = AgentRunsDB(path, client_id="handoff-reload")
    runs = reopened.list_runs("c", include_superseded=True)
    steps = [
        {**step, "run_id": row["id"], "conversation_id": "c"}
        for row in runs
        for step in row["steps"]
    ]
    snapshot = derive_trajectory(
        messages=[],
        usage_by_id={},
        traj_rows=[],
        variant_sets=[],
        compaction_records=[],
        agent_runs=runs,
        agent_steps=steps,
    )
    records = [record for turn in snapshot.turns for record in turn.records]
    projected = next(
        record
        for record in records
        if record.run_id == child["id"] and record.kind == "steering"
    )
    assert projected.parent_event_id == send_event_id
    assert projected.source_event_id == send_event_id
    assert [record.event_id for record in records].index(send_event_id) < [
        record.event_id for record in records
    ].index(projected.event_id)
    logged = "\n".join(
        record.content for record in load_records(service.run_log_writer.log_dir)
    )
    durable = str(runs)
    for forbidden in (
        "reasoning_content",
        "/Users/alice/secret.txt",
        "sk-private-steering",
        holder["handle_id"],
    ):
        assert forbidden not in logged
        assert forbidden not in durable
    reopened.close()


# -- refusal shapes: the producer validates (Task 1 pinned that the
# -- mailbox does NOT) ----------------------------------------------------


def test_an_empty_message_is_refused_and_nothing_is_queued(db):
    entered = threading.Event()
    released = threading.Event()
    holder: dict = {}

    def steer_empty():
        assert entered.wait(_JOIN_TIMEOUT)
        handle = _live_running_handle(holder["coordinator"])
        holder["handle_id"] = handle.handle_id
        return fence(
            SEND_TO_AGENT_TOOL_NAME,
            {"id": handle.handle_id, "message": "   \n\t "},
        )

    def capture_then_wait():
        holder["queued_after_refusal"] = (
            holder["coordinator"].get(holder["handle_id"]).queued_steering
        )
        released.set()
        return fence(WAIT_AGENTS_TOOL_NAME, {})

    def gated_child():
        entered.set()
        assert released.wait(_JOIN_TIMEOUT)
        return "child answer"

    service, _chat, coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "the task"}),
            steer_empty,
            capture_then_wait,
            "done",
        ],
        {"the task": [gated_child]},
    )
    holder["coordinator"] = coordinator
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=FLEET_CFG,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    sends = _sends(db, run_id)
    assert sends and "ERROR" in sends[0] and "empty" in sends[0]
    assert holder["queued_after_refusal"] == 0


def test_an_oversize_message_is_refused_and_the_cap_itself_is_accepted(db):
    """Over the cap: refused, naming the cap. AT the cap: accepted --
    the boundary is exact, not off by one."""
    entered = threading.Event()
    released = threading.Event()
    holder: dict = {}

    def steer_oversize():
        assert entered.wait(_JOIN_TIMEOUT)
        handle = _live_running_handle(holder["coordinator"])
        holder["handle_id"] = handle.handle_id
        return fence(
            SEND_TO_AGENT_TOOL_NAME,
            {"id": handle.handle_id, "message": "x" * (MAX_STEERING_CHARS + 1)},
        )

    def steer_at_cap():
        return fence(
            SEND_TO_AGENT_TOOL_NAME,
            {"id": holder["handle_id"], "message": "y" * MAX_STEERING_CHARS},
        )

    def capture_then_wait():
        holder["queued"] = (
            holder["coordinator"].get(holder["handle_id"]).queued_steering
        )
        released.set()
        return fence(WAIT_AGENTS_TOOL_NAME, {})

    def gated_child():
        entered.set()
        assert released.wait(_JOIN_TIMEOUT)
        return "child answer"

    service, _chat, coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "the task"}),
            steer_oversize,
            steer_at_cap,
            capture_then_wait,
            "done",
        ],
        {"the task": [gated_child]},
    )
    holder["coordinator"] = coordinator
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=FLEET_CFG,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    sends = _sends(db, run_id)
    assert len(sends) == 2
    assert "ERROR" in sends[0] and "too long" in sends[0]
    assert str(MAX_STEERING_CHARS) in sends[0]
    assert "ERROR" not in sends[1] and "queued" in sends[1]
    # Only the at-cap entry was queued; the child never took another model
    # turn, so it is still sitting in the mailbox (Task 1 pinned that an
    # undrained entry survives finish).
    assert holder["queued"] == 1


def test_an_unknown_id_is_refused_naming_the_live_ids(db):
    entered = threading.Event()
    released = threading.Event()
    holder: dict = {}

    def steer_unknown():
        assert entered.wait(_JOIN_TIMEOUT)
        handle = _live_running_handle(holder["coordinator"])
        holder["handle_id"] = handle.handle_id
        return fence(
            SEND_TO_AGENT_TOOL_NAME, {"id": "nope", "message": "hello?"}
        )

    def release_then_wait():
        released.set()
        return fence(WAIT_AGENTS_TOOL_NAME, {})

    def gated_child():
        entered.set()
        assert released.wait(_JOIN_TIMEOUT)
        return "child answer"

    service, _chat, coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "the task"}),
            steer_unknown,
            release_then_wait,
            "done",
        ],
        {"the task": [gated_child]},
    )
    holder["coordinator"] = coordinator
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=FLEET_CFG,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    sends = _sends(db, run_id)
    assert sends and "ERROR" in sends[0]
    assert "'nope'" in sends[0]
    # The error NAMES the known live ids, so the supervisor can correct
    # itself without a check_agents round trip.
    handle = coordinator.get(holder["handle_id"])
    assert handle.run_id
    assert holder["handle_id"] not in sends[0]
    assert f"run:{handle.run_id}" in sends[0]
    # Nothing landed anywhere.
    assert coordinator.get(holder["handle_id"]).queued_steering == 0


def test_a_terminal_id_states_the_child_has_finished(db):
    """Steering a FINISHED, UNRETAINED child -- by its handle id or its
    run id -- is refused with copy saying it finished, with no live ids
    left to name. PR3b Task 4 upgraded the terminal branch to
    continuation, so a retainable finished child now RESUMES (the
    continuation suite owns that path); THIS test pins the honest
    refusal for the child whose transcript is NOT retained, by switching
    retention off (caps 0) before the run -- the same copy a cancelled/
    superseded/oversize/evicted child draws."""
    holder: dict = {}

    def steer_terminal_handle():
        def _finished():
            handles = holder["coordinator"].snapshot()
            return bool(handles) and all(
                h.status in TERMINAL_RUN_STATUSES for h in handles
            )

        _wait_until(_finished, "the quick child never finished")
        [handle] = holder["coordinator"].snapshot()
        holder["handle_id"], holder["run_id"] = handle.handle_id, handle.run_id
        assert holder["run_id"], "precondition: the finished child has a run id"
        return fence(
            SEND_TO_AGENT_TOOL_NAME,
            {"id": handle.handle_id, "message": "too late"},
        )

    def steer_terminal_run_id():
        return fence(
            SEND_TO_AGENT_TOOL_NAME,
            {"id": holder["run_id"], "message": "still too late"},
        )

    service, _chat, coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "quick task"}),
            steer_terminal_handle,
            steer_terminal_run_id,
            "done",
        ],
        {"quick task": ["quick answer"]},
    )
    holder["coordinator"] = coordinator
    # Retention OFF: with the caps at 0 the finished child's transcript is
    # never retained, which is what routes both steers into the honest
    # not-retained refusal instead of Task 4's continuation.
    coordinator.set_retention_caps(0, 0)
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=FLEET_CFG,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    sends = _sends(db, run_id)
    assert len(sends) == 2
    for send in sends:
        assert "ERROR" in send and "finished" in send
        assert "no retained transcript" in send
        assert "Spawn a fresh sub-agent" in send
        assert "Live sub-agent ids: none" in send


# -- id vocabularies: run id resolves too; handle id wins a collision -----


def test_a_run_id_reaches_the_same_mailbox_as_the_handle_id(db):
    """The wake notice speaks run ids (`console_fleet_wake` identity
    vocabulary); a supervisor pasting one must reach the same child."""
    entered = threading.Event()
    steered = threading.Event()
    holder: dict = {}

    def steer_by_run_id():
        assert entered.wait(_JOIN_TIMEOUT)
        _wait_until(
            lambda: _live_running_handle(holder["coordinator"]).run_id is not None,
            "the child never got a run id attached",
        )
        handle = _live_running_handle(holder["coordinator"])
        holder["handle_id"] = handle.handle_id
        return fence(
            SEND_TO_AGENT_TOOL_NAME,
            {"id": handle.run_id, "message": "addressed by run id"},
        )

    def release_then_wait():
        steered.set()
        return fence(WAIT_AGENTS_TOOL_NAME, {})

    def gated_child():
        entered.set()
        assert steered.wait(_JOIN_TIMEOUT)
        return fence("calculator", {"expression": "2+2"})

    service, chat, coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "run id target"}),
            steer_by_run_id,
            release_then_wait,
            "combined answer",
        ],
        {"run id target": [gated_child, "child answer"]},
    )
    holder["coordinator"] = coordinator
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=FLEET_CFG,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    labeled = format_steering_message(
        STEERING_SOURCE_SUPERVISOR, "addressed by run id"
    )
    second_payload = chat.child_calls["run id target"][1]["messages_payload"]
    assert second_payload[-1] == {"role": "user", "content": labeled}
    # The ok copy names the HANDLE the run id resolved to -- proof the two
    # vocabularies land on one mailbox, not two.
    sends = _sends(db, run_id)
    assert sends and "ERROR" not in sends[0]
    handle = coordinator.get(holder["handle_id"])
    assert handle.run_id
    assert holder["handle_id"] not in sends[0]
    assert f"run:{handle.run_id}" in sends[0]
    assert coordinator.get(holder["handle_id"]).queued_steering == 0


def test_a_live_handle_id_beats_a_colliding_run_id(db):
    """Resolution order: handle id FIRST, then a live handle's run id.

    The coordinator minted the handle id for exactly this purpose --
    check_agents, spawn results and the panel rows all speak it -- so a
    (forged here, pathological anywhere) collision where child A's run id
    equals child B's handle id must resolve to B. Mutation target: swap
    the resolution order and this dies."""
    entered_a = threading.Event()
    entered_b = threading.Event()
    released = threading.Event()
    holder: dict = {}

    def child_a():
        entered_a.set()
        assert released.wait(_JOIN_TIMEOUT)
        return "done a"

    def child_b():
        entered_b.set()
        assert released.wait(_JOIN_TIMEOUT)
        return "done b"

    def collide_then_steer():
        assert entered_a.wait(_JOIN_TIMEOUT) and entered_b.wait(_JOIN_TIMEOUT)
        coordinator = holder["coordinator"]
        a = next(h for h in coordinator.snapshot() if h.task == "task a")
        b = next(h for h in coordinator.snapshot() if h.task == "task b")
        holder["a"], holder["b"] = a.handle_id, b.handle_id
        # Forge the collision: A's run id becomes B's handle id.
        coordinator.attach_run(a.handle_id, b.handle_id)
        return fence(
            SEND_TO_AGENT_TOOL_NAME,
            {"id": b.handle_id, "message": "for b only"},
        )

    def capture_then_wait():
        coordinator = holder["coordinator"]
        holder["queued"] = (
            coordinator.get(holder["a"]).queued_steering,
            coordinator.get(holder["b"]).queued_steering,
        )
        released.set()
        return fence(WAIT_AGENTS_TOOL_NAME, {})

    service, _chat, coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "task a"}),
            fence(SPAWN_TOOL_NAME, {"task": "task b"}),
            collide_then_steer,
            capture_then_wait,
            "done",
        ],
        {"task a": [child_a], "task b": [child_b]},
    )
    holder["coordinator"] = coordinator
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=FLEET_CFG,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    # The message landed in B's mailbox and only B's.
    assert holder["queued"] == (0, 1)
    sends = _sends(db, run_id)
    target = coordinator.get(holder["b"])
    assert target.run_id
    assert sends and "ERROR" not in sends[0]
    assert holder["b"] not in sends[0]
    assert f"run:{target.run_id}" in sends[0]
    # Neither child took another model turn, so B never DRAINED the entry
    # -- and A's mailbox stayed empty throughout. PR3b Task 4: at finish
    # time retention CLAIMED B's undelivered remnant (Task 1's pinned
    # window), so the mailbox reads 0 and the entry -- still B's, still
    # supervisor-labeled -- now rides B's retained transcript, where a
    # resume would replay it.
    assert coordinator.get(holder["b"]).queued_steering == 0
    retained_b = coordinator.get_retained(holder["b"])
    assert retained_b is not None
    assert list(retained_b.steering) == [
        (STEERING_SOURCE_SUPERVISOR, "for b only")
    ]
    retained_a = coordinator.get_retained(holder["a"])
    assert retained_a is not None and retained_a.steering == ()
    assert coordinator.get(holder["a"]).queued_steering == 0


def test_junk_args_never_crash_the_loop(db):
    """The in-loop dispatch coerces junk exactly like wait_agents' ids: a
    numeric id/message never raises; the coerced id just fails to match
    and the service's own refusal copy comes back."""
    entered = threading.Event()
    released = threading.Event()
    holder: dict = {}

    def steer_junk():
        assert entered.wait(_JOIN_TIMEOUT)
        handle = _live_running_handle(holder["coordinator"])
        holder["handle_id"] = handle.handle_id
        return fence(SEND_TO_AGENT_TOOL_NAME, {"id": 123, "message": 456})

    def release_then_wait():
        released.set()
        return fence(WAIT_AGENTS_TOOL_NAME, {})

    def gated_child():
        entered.set()
        assert released.wait(_JOIN_TIMEOUT)
        return "child answer"

    service, _chat, coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "the task"}),
            steer_junk,
            release_then_wait,
            "done",
        ],
        {"the task": [gated_child]},
    )
    holder["coordinator"] = coordinator
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=FLEET_CFG,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    sends = _sends(db, run_id)
    assert sends and "ERROR" in sends[0] and "'123'" in sends[0]
    assert coordinator.get(holder["handle_id"]).queued_steering == 0


# -- spec SS3 invariant 4: steering never cancels --------------------------


def test_steering_never_cancels_the_child(db):
    """After a successful post: coordinator status untouched, run row
    untouched, cancel Event unset -- and still unset after the child has
    consumed the message and finished on its own terms."""
    entered = threading.Event()
    steered = threading.Event()
    holder: dict = {}

    def steer():
        assert entered.wait(_JOIN_TIMEOUT)
        handle = _live_running_handle(holder["coordinator"])
        holder["handle_id"] = handle.handle_id
        return fence(
            SEND_TO_AGENT_TOOL_NAME,
            {"id": handle.handle_id, "message": "keep going"},
        )

    def capture_then_wait():
        coordinator = holder["coordinator"]
        hid = holder["handle_id"]
        holder["status_after_post"] = coordinator.get(hid).status
        holder["row_after_post"] = _child_row(db)["status"]
        holder["cancel_set_after_post"] = (
            holder["service"]._fleet_cancels[hid].is_set()
        )
        steered.set()
        return fence(WAIT_AGENTS_TOOL_NAME, {})

    def gated_child():
        entered.set()
        assert steered.wait(_JOIN_TIMEOUT)
        return fence("calculator", {"expression": "1+1"})

    service, chat, coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "steady task"}),
            steer,
            capture_then_wait,
            "combined answer",
        ],
        {"steady task": [gated_child, "child done"]},
    )
    holder["coordinator"] = coordinator
    holder["service"] = service
    _run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=FLEET_CFG,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    # Right after the post: nothing about the child moved.
    assert holder["status_after_post"] == RUN_RUNNING
    assert holder["row_after_post"] == RUN_RUNNING
    assert holder["cancel_set_after_post"] is False
    # And at the end: the child consumed the message and finished DONE on
    # its own terms -- coordinator and run row both say so, which is the
    # invariant's end-state. (Deliberately NOT asserted: the raw cancel
    # Event after run_turn. `_settle_fleet` sets every settling child's
    # Event unconditionally at end of turn -- documented there as inert
    # for an already-finished child -- so that Event is end-of-turn
    # bookkeeping, not steering's doing; the mid-turn probe above is the
    # honest measurement of what the POST touched.)
    hid = holder["handle_id"]
    assert coordinator.get(hid).status == RUN_DONE
    assert _child_row(db)["status"] == RUN_DONE
    labeled = format_steering_message(STEERING_SOURCE_SUPERVISOR, "keep going")
    second_payload = chat.child_calls["steady task"][1]["messages_payload"]
    assert second_payload[-1] == {"role": "user", "content": labeled}


# -- steering never satisfies an approval ---------------------------------


def test_steering_never_satisfies_a_pending_approval(db):
    """A real approval round is armed and HELD OPEN (the child parked
    inside `review_tool_calls`, where the Console parks). Steering posted
    at that moment leaves the verdict pending, the gated tool unexecuted,
    and the entry queued; only after the human answers does the tool run
    -- and only at the NEXT boundary does the steering arrive. This is
    the wake's not-user-input guarantee, mirrored for steering."""
    parked = threading.Event()
    answered = threading.Event()
    review_finished = threading.Event()
    holder: dict = {}

    def review(calls, run_id):
        # Only the CHILD's calculator call parks; the parent's own
        # spawn/steer batches must sail through or the turn deadlocks.
        if any(call.name == "calculator" for call in calls):
            parked.set()
            if not answered.wait(_CARD_TIMEOUT):
                raise AssertionError("the card was never answered")
            review_finished.set()
        return {}

    def steer_at_the_card():
        assert parked.wait(_CARD_TIMEOUT), "the child never armed its card"
        handle = _live_running_handle(holder["coordinator"])
        holder["handle_id"] = handle.handle_id
        return fence(
            SEND_TO_AGENT_TOOL_NAME,
            {"id": handle.handle_id, "message": "steer at the card"},
        )

    def capture():
        coordinator = holder["coordinator"]
        hid = holder["handle_id"]
        holder["review_finished"] = review_finished.is_set()
        holder["answered"] = answered.is_set()
        holder["child_turns"] = len(holder["chat"].child_calls["gated task"])
        child_row = _child_row(db)
        holder["calc_steps"] = _tool_results(
            db.get_run(child_row["id"]), "calculator"
        )
        holder["queued"] = coordinator.get(hid).queued_steering
        holder["cancel_set"] = holder["service"]._fleet_cancels[hid].is_set()
        return "parent answered early"

    service, chat, coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "gated task"}),
            steer_at_the_card,
            capture,
        ],
        {
            "gated task": [
                fence("calculator", {"expression": "1+1"}),
                "child done",
            ]
        },
        review_tool_calls=review,
    )
    holder["coordinator"] = coordinator
    holder["service"] = service
    holder["chat"] = chat
    try:
        _run_id, outcome = service.run_turn(
            conversation_id="c",
            messages=[{"role": "user", "content": "go"}],
            config=CARD_CFG,
            api_endpoint="llama_cpp",
        )
        assert outcome.status == RUN_DONE
        # At the moment after the steering post, with the card still held:
        assert holder["review_finished"] is False, "steering resolved the verdict"
        assert holder["answered"] is False
        assert holder["child_turns"] == 1, "the child took a turn past its card"
        assert holder["calc_steps"] == [], "the gated tool executed"
        assert holder["queued"] == 1, "the steering entry was not left queued"
        assert holder["cancel_set"] is False
    finally:
        # The human approves -- after the turn, like the survivor-card pin.
        answered.set()
    _wait_until(coordinator.all_finished, "the approved child never finished")
    assert review_finished.is_set()
    # The gate released for real, and the steering arrived only at the
    # post-approval boundary: tool result first, steering last.
    child_turns = chat.child_calls["gated task"]
    assert len(child_turns) == 2
    payload = child_turns[1]["messages_payload"]
    labeled = format_steering_message(STEERING_SOURCE_SUPERVISOR, "steer at the card")
    assert payload[-1] == {"role": "user", "content": labeled}
    assert str(payload[-2]["content"]).startswith(
        f"{FENCE_TOOL_RESULT_PREFIX}calculator:"
    )
    calc = _tool_results(db.get_run(_child_row(db)["id"]), "calculator")
    assert calc and "2" in calc[0]
    hid = holder["handle_id"]
    assert coordinator.get(hid).status == RUN_DONE
    assert coordinator.get(hid).queued_steering == 0


# -- whole-coordinator reach: foreign live survivors are steerable --------


def test_a_foreign_live_survivor_is_steerable(db):
    """A survivor spawned by an EARLIER turn's service is steerable from
    the current turn: the mailbox lives on the conversation-lifetime
    coordinator and needs no per-service state -- deliberately unlike
    `cancel_subagent`, whose ownership walk exists because cancel Events
    are service-local."""
    entered = threading.Event()
    release = threading.Event()
    holder: dict = {}

    def survivor_first_turn():
        entered.set()
        assert release.wait(30.0), "the survivor was never released"
        return fence("calculator", {"expression": "2+2"})

    def turn_1_parent_answer():
        assert entered.wait(_JOIN_TIMEOUT), "the survivor never started"
        return "turn 1 done"

    def steer_foreign():
        survivor = _live_running_handle(holder["coordinator"])
        holder["handle_id"] = survivor.handle_id
        return fence(
            SEND_TO_AGENT_TOOL_NAME,
            {"id": survivor.handle_id, "message": "focus on the tests"},
        )

    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    chat = FleetChat(
        [
            fence(SPAWN_TOOL_NAME, {"task": "survivor"}),
            turn_1_parent_answer,
            steer_foreign,
            "turn 2 done",
        ],
        {"survivor": [survivor_first_turn, "released"]},
    )
    coordinator = FleetCoordinator(max_live=3, clock=time.monotonic)
    holder["coordinator"] = coordinator
    service_1 = AgentService(
        db=db, registry=registry, chat_call=chat, fleet_coordinator=coordinator
    )
    service_2 = AgentService(
        db=db, registry=registry, chat_call=chat, fleet_coordinator=coordinator
    )
    try:
        _run_id_1, outcome_1 = service_1.run_turn(
            conversation_id="c",
            messages=[{"role": "user", "content": "go 1"}],
            config=FLEET_CFG,
            api_endpoint="llama_cpp",
        )
        assert outcome_1.status == RUN_DONE
        # Turn 2, on a FRESH service sharing the coordinator, steers the
        # survivor it never spawned.
        run_id_2, outcome_2 = service_2.run_turn(
            conversation_id="c",
            messages=[{"role": "user", "content": "go 2"}],
            config=FLEET_CFG,
            api_endpoint="llama_cpp",
        )
        assert outcome_2.status == RUN_DONE
        sends = _sends(db, run_id_2)
        assert sends and "ERROR" not in sends[0] and "queued" in sends[0]
        survivor = coordinator.get(holder["handle_id"])
        assert survivor.run_id
        assert holder["handle_id"] not in sends[0]
        assert f"run:{survivor.run_id}" in sends[0]
        assert survivor.queued_steering == 1
        # And the post cancelled nothing: the survivor is still running.
        assert coordinator.get(holder["handle_id"]).status == RUN_RUNNING
    finally:
        release.set()
    _wait_until(coordinator.all_finished, "the survivor never completed")
    assert coordinator.get(holder["handle_id"]).status == RUN_DONE
    assert coordinator.get(holder["handle_id"]).result == "released"
    labeled = format_steering_message(
        STEERING_SOURCE_SUPERVISOR, "focus on the tests"
    )
    second_payload = chat.child_calls["survivor"][1]["messages_payload"]
    assert second_payload[-1] == {"role": "user", "content": labeled}
