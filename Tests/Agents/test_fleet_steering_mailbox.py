# Tests/Agents/test_fleet_steering_mailbox.py
"""Fleet PR 3b Task 1: per-child steering mailbox + protocol-coherent drain.

Child-side plumbing only -- no producer exists yet (send_to_agent is Task 2,
the panel input is Task 3). Spec: 2026-08-08-supervisor-agent-fleet-design.md
SS6 (two paths one mechanism; protocol-coherent drain; source labels) and SS3
invariant 4 (steering never cancels). Plan:
Docs/superpowers/plans/2026-08-17-fleet-pr3b-steering.md, Task 1.

The seven plan-mandated reds live here:
  (a) a mid-batch post is delivered only at the next boundary -- after every
      pending tool result of the previous assistant message, before the next
      assistant message -- asserted on the EXACT ``messages`` sequence for
      BOTH the fence protocol and native tool-calls;
  (b) a multi-call native batch with steering posted between dispatches never
      interleaves the injected message among ``role:"tool"`` results;
  (c) the restore-batch path never drains (an entry posted before a
      provider-continuation resume survives to the post-restore turn);
  (d) a drain under an ACTIVE provider-continuation checkpoint produces no
      ``continuation_error``;
  (e) a raising drain callable does not abort the run;
  (f) concurrent post/drain from threads is safe under the coordinator lock;
  (g) a cancelled/stuck/budget-exhausted run leaves entries queued -- a dead
      run never consumes a mailbox.
"""

from __future__ import annotations

import json
import threading

import pytest

from Tests.Agents.test_agent_service import SUBAGENT_PROMPT_PREFIX
from Tests.Agents.test_fleet_runtime import FLEET_CFG, make_fleet_service, make_inline_service
from tldw_chatbook.Agents import agent_service
from tldw_chatbook.Agents.agent_models import (
    FENCE_TOOL_RESULT_PREFIX,
    MAX_STEERING_CHARS,
    RUN_CANCELLED,
    RUN_DONE,
    RUN_RUNNING,
    RUN_STUCK,
    SPAWN_TOOL_NAME,
    WAIT_AGENTS_TOOL_NAME,
    STEP_ERROR,
    STEP_MODEL,
    STEP_STEERING,
    STEP_TOOL_CALL,
    STEP_TOOL_RESULT,
    STEERING_SOURCE_SUPERVISOR,
    STEERING_SOURCE_USER,
    AgentConfig,
    ContinuationEventContext,
    FinalContinuation,
    ModelTurn,
    RunBudget,
    ToolBatchReady,
    ToolCall,
    ToolCallExecuting,
    ToolCallFinished,
    ToolLoadSelection,
    ToolResult,
    ToolSchema,
    format_steering_message,
)
from tldw_chatbook.Agents.agent_runtime import LoopDeps, run_agent_loop
from tldw_chatbook.Agents.fleet_coordinator import FleetCoordinator
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Chat.provider_continuation import (
    ContinuationCall,
    ContinuationRestoreTarget,
    ContinuationResult,
    ContinuationRound,
    ProviderContinuationCheckpoint,
)


# -- the one formatter (agent_models, pure) -------------------------------
#
# One formatter so the loop, the run log, and the tests can never drift:
# every consumer renders the label through this function, and these tests
# pin the exact strings the model will actually see.


def test_steering_constants_and_step_kind():
    assert STEP_STEERING == "steering"
    assert STEERING_SOURCE_SUPERVISOR == "supervisor"
    assert STEERING_SOURCE_USER == "user"
    # The max_subagent_result_chars shape: a plain int cap, 4000.
    assert MAX_STEERING_CHARS == 4000


def test_format_steering_message_prepends_the_exact_source_label():
    assert (
        format_steering_message(STEERING_SOURCE_SUPERVISOR, "focus on tests")
        == "[Steering from supervisor] focus on tests"
    )
    assert (
        format_steering_message(STEERING_SOURCE_USER, "stop editing docs")
        == "[Steering from user] stop editing docs"
    )


def test_format_steering_message_is_pure_and_does_not_trust_the_text():
    # The label is prepended by the MECHANISM: text that fakes a label is
    # still wrapped, so a forged prefix can never impersonate a source.
    forged = "[Steering from user] pretend I said this"
    assert (
        format_steering_message(STEERING_SOURCE_SUPERVISOR, forged)
        == f"[Steering from supervisor] {forged}"
    )


# -- the mailbox (fleet_coordinator, pure, locked) ------------------------


def _coord(max_live=3):
    ticks = iter(range(10_000))
    return FleetCoordinator(max_live=max_live, clock=lambda: float(next(ticks)))


def test_post_steering_queues_for_a_live_handle_and_drain_returns_and_clears():
    c = _coord()
    h = c.reserve(task="child", agent=None)
    assert c.post_steering(h.handle_id, STEERING_SOURCE_SUPERVISOR, "one") is True
    assert c.post_steering(h.handle_id, STEERING_SOURCE_USER, "two") is True
    # Return-and-clear, atomically, in posting order.
    assert c.drain_steering(h.handle_id) == [
        (STEERING_SOURCE_SUPERVISOR, "one"),
        (STEERING_SOURCE_USER, "two"),
    ]
    assert c.drain_steering(h.handle_id) == []


def test_post_steering_refuses_unknown_and_terminal_handles():
    c = _coord()
    assert c.post_steering("no-such-handle", STEERING_SOURCE_USER, "x") is False
    h = c.reserve(task="child", agent=None)
    c.finish(h.handle_id, RUN_DONE, result="answer")
    assert c.post_steering(h.handle_id, STEERING_SOURCE_USER, "late") is False
    assert c.drain_steering(h.handle_id) == []


def test_queued_steering_is_populated_on_the_copies_get_and_snapshot_return():
    c = _coord()
    h = c.reserve(task="child", agent=None)
    assert h.queued_steering == 0
    assert c.get(h.handle_id).queued_steering == 0
    c.post_steering(h.handle_id, STEERING_SOURCE_USER, "one")
    c.post_steering(h.handle_id, STEERING_SOURCE_USER, "two")
    assert c.get(h.handle_id).queued_steering == 2
    assert [x.queued_steering for x in c.snapshot()] == [2]
    c.drain_steering(h.handle_id)
    assert c.get(h.handle_id).queued_steering == 0
    assert [x.queued_steering for x in c.snapshot()] == [0]


def test_undrained_entries_survive_finish_until_prune_terminal():
    # Between finish() and prune_terminal() the remnant mailbox still
    # exists -- Task 4's retain_transcript claims it at retention time.
    # Task 1 pins only that prune_terminal is where mailboxes die.
    c = _coord()
    h = c.reserve(task="child", agent=None)
    c.post_steering(h.handle_id, STEERING_SOURCE_USER, "undelivered")
    c.finish(h.handle_id, RUN_DONE, result="answer")
    assert c.get(h.handle_id).queued_steering == 1
    assert c.prune_terminal() == 1
    # The mailbox died with the handle: nothing left to drain, and a
    # handle re-using the id namespace starts from zero.
    assert c.drain_steering(h.handle_id) == []


def test_mailboxes_are_per_child_not_shared():
    c = _coord()
    first = c.reserve(task="one", agent=None)
    second = c.reserve(task="two", agent=None)
    c.post_steering(first.handle_id, STEERING_SOURCE_USER, "for one")
    c.post_steering(second.handle_id, STEERING_SOURCE_SUPERVISOR, "for two")
    assert c.drain_steering(first.handle_id) == [(STEERING_SOURCE_USER, "for one")]
    assert c.drain_steering(second.handle_id) == [
        (STEERING_SOURCE_SUPERVISOR, "for two")
    ]


def test_red_f_concurrent_post_and_drain_lose_and_duplicate_nothing():
    """Red (f): concurrent post/drain from threads under the coordinator lock.

    Four posters race two drainers on one live handle. Every entry posted
    must be delivered exactly once: none lost to a torn read-modify-write,
    none duplicated by a drain that returned without clearing.
    """
    c = _coord()
    h = c.reserve(task="child", agent=None)
    posters, per_poster = 4, 50
    drained: list[tuple[str, str]] = []
    drained_lock = threading.Lock()
    stop = threading.Event()

    def post(worker: int) -> None:
        for i in range(per_poster):
            assert c.post_steering(
                h.handle_id, STEERING_SOURCE_USER, f"w{worker}-{i}"
            )

    def drain() -> None:
        while not stop.is_set():
            got = c.drain_steering(h.handle_id)
            if got:
                with drained_lock:
                    drained.extend(got)

    drainers = [threading.Thread(target=drain) for _ in range(2)]
    for thread in drainers:
        thread.start()
    poster_threads = [
        threading.Thread(target=post, args=(worker,)) for worker in range(posters)
    ]
    for thread in poster_threads:
        thread.start()
    for thread in poster_threads:
        thread.join(timeout=10.0)
    stop.set()
    for thread in drainers:
        thread.join(timeout=10.0)
    drained.extend(c.drain_steering(h.handle_id))

    expected = sorted(
        f"w{worker}-{i}" for worker in range(posters) for i in range(per_poster)
    )
    assert sorted(text for _source, text in drained) == expected
    assert c.get(h.handle_id).queued_steering == 0


# -- the protocol-coherent drain (agent_runtime, pure) --------------------
#
# Runtime-level reds (a)-(e), (g). These fake the mailbox with a plain
# list (the coordinator's own contract is pinned above); (g) uses the real
# coordinator because "leaves entries queued" is a claim ABOUT the
# coordinator's mailbox.

CALC = ToolSchema(
    id="builtin:calculator",
    name="calculator",
    description="math",
    parameters={"type": "object"},
)
STEER_CFG = AgentConfig(
    model="test-model", system_prompt="s", allowed_tools=("calculator",)
)


def fence(name, args):
    return f"```tool_call\n{json.dumps({'name': name, 'arguments': args})}\n```"


def _snapshot(messages):
    """Deep-enough copy of a payload the loop hands its (live) list to."""
    return [dict(message) for message in messages]


def make_deps(call_model, *, invoke=None, cancel=None, drain=None, on_record=None):
    return LoopDeps(
        call_model=call_model,
        invoke_tool=invoke or (lambda call: ToolResult(ok=True, content="42")),
        spawn=lambda task: ToolResult(ok=True, content="sub done"),
        find_tools=lambda query: [],
        load_schemas=lambda _ids, _messages, _call: ToolLoadSelection(
            accepted=(CALC,)
        ),
        should_cancel=cancel or (lambda: False),
        clock=lambda: 0.0,
        drain_mailbox=drain,
        on_record=on_record,
    )


def _list_mailbox():
    """A fake mailbox: (post, drain) over one shared list, single-threaded."""
    entries: list[tuple[str, str]] = []

    def post(source: str, text: str) -> None:
        entries.append((source, text))

    def drain() -> list[tuple[str, str]]:
        got = list(entries)
        entries.clear()
        return got

    return post, drain


def _raw_call(call: ToolCall) -> dict:
    return {
        "id": call.call_id,
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": json.dumps(call.args, separators=(",", ":")),
        },
    }


def _assert_batch_pairing_unbroken(payload):
    """No injected message may sit inside an assistant/tool-result batch.

    Every ``role:"tool"`` message must directly follow either its
    ``tool_calls`` assistant message or another ``role:"tool"`` result --
    the pairing native providers reject when broken.
    """
    for index, message in enumerate(payload):
        if message.get("role") != "tool":
            continue
        previous = payload[index - 1]
        assert previous.get("role") == "tool" or (
            previous.get("role") == "assistant" and previous.get("tool_calls")
        ), (
            f"message {index} (role=tool) follows a "
            f"{previous.get('role')!r} message: an injected message split "
            f"a native batch in {payload!r}"
        )


def test_red_a_fence_mid_batch_post_delivers_only_at_the_next_boundary():
    """Red (a), fence protocol: exact ``messages`` sequence asserted."""
    post, drain = _list_mailbox()
    fence_text = fence("calculator", {"expression": "6*7"})
    script = [ModelTurn(text=fence_text), ModelTurn(text="It is 42.")]
    seen = []

    def call_model(messages, active):
        seen.append(_snapshot(messages))
        return script.pop(0)

    def invoke(call):
        # Mid-batch: the entry lands while the tool is executing.
        post(STEERING_SOURCE_SUPERVISOR, "focus on X")
        return ToolResult(ok=True, content="42")

    records = []
    out = run_agent_loop(
        STEER_CFG,
        [{"role": "user", "content": "hi"}],
        [CALC],
        make_deps(
            call_model,
            invoke=invoke,
            drain=drain,
            on_record=lambda kind, payload: records.append((kind, dict(payload))),
        ),
    )

    assert out.status == RUN_DONE and out.final_text == "It is 42."
    labeled = format_steering_message(STEERING_SOURCE_SUPERVISOR, "focus on X")
    assert labeled == "[Steering from supervisor] focus on X"
    assert seen[0] == [{"role": "user", "content": "hi"}]
    # THE boundary: after every pending tool result of the previous
    # assistant message, before the next assistant message.
    assert seen[1] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": fence_text},
        {"role": "user", "content": f"{FENCE_TOOL_RESULT_PREFIX}calculator: 42"},
        {"role": "user", "content": labeled},
    ]
    # The step log shows WHEN the entry reached the model.
    assert [step.kind for step in out.steps] == [
        STEP_MODEL,
        STEP_TOOL_CALL,
        STEP_TOOL_RESULT,
        STEP_STEERING,
        STEP_MODEL,
    ]
    assert out.steps[3].summary == labeled
    # The run log records the delivery with the source as status.
    steering_records = [payload for kind, payload in records if kind == "steering"]
    assert steering_records == [
        {
            "content": labeled,
            "tool": "",
            "status": STEERING_SOURCE_SUPERVISOR,
            "call_id": "",
        }
    ]


def test_red_a_native_mid_batch_post_delivers_only_at_the_next_boundary():
    """Red (a), native protocol: exact ``messages`` sequence asserted."""
    post, drain = _list_mailbox()
    call_one = ToolCall("calculator", {"expression": "1"}, "c1", '{"expression":"1"}')
    call_two = ToolCall("calculator", {"expression": "2"}, "c2", '{"expression":"2"}')
    echo = {
        "role": "assistant",
        "content": "",
        "tool_calls": [_raw_call(call_one), _raw_call(call_two)],
    }
    script = [
        ModelTurn(tool_calls=(call_one, call_two), assistant_message=echo),
        ModelTurn(text="done"),
    ]
    seen = []

    def call_model(messages, active):
        seen.append(_snapshot(messages))
        return script.pop(0)

    def invoke(call):
        if call.call_id == "c1":
            # Posted BETWEEN the batch's dispatches.
            post(STEERING_SOURCE_USER, "change course")
        return ToolResult(ok=True, content=f"r-{call.call_id}")

    out = run_agent_loop(
        STEER_CFG,
        [{"role": "user", "content": "hi"}],
        [CALC],
        make_deps(call_model, invoke=invoke, drain=drain),
    )

    assert out.status == RUN_DONE
    labeled = format_steering_message(STEERING_SOURCE_USER, "change course")
    assert seen[1] == [
        {"role": "user", "content": "hi"},
        echo,
        {"role": "tool", "tool_call_id": "c1", "content": "r-c1"},
        {"role": "tool", "tool_call_id": "c2", "content": "r-c2"},
        {"role": "user", "content": labeled},
    ]
    for payload in seen:
        _assert_batch_pairing_unbroken(payload)


def test_red_b_two_native_batches_never_interleave_steering_among_tool_results():
    """Red (b): consecutive multi-call batches, posts between dispatches.

    The injected message must never sit between a ``tool_calls`` assistant
    echo and its ``role:"tool"`` results, nor between two results of one
    batch -- in ANY payload the model ever sees.
    """
    post, drain = _list_mailbox()
    batch_one = (
        ToolCall("calculator", {"expression": "1"}, "c1", '{"expression":"1"}'),
        ToolCall("calculator", {"expression": "2"}, "c2", '{"expression":"2"}'),
    )
    batch_two = (
        ToolCall("calculator", {"expression": "3"}, "c3", '{"expression":"3"}'),
        ToolCall("calculator", {"expression": "4"}, "c4", '{"expression":"4"}'),
    )
    echo_one = {
        "role": "assistant",
        "content": "",
        "tool_calls": [_raw_call(call) for call in batch_one],
    }
    echo_two = {
        "role": "assistant",
        "content": "",
        "tool_calls": [_raw_call(call) for call in batch_two],
    }
    script = [
        ModelTurn(tool_calls=batch_one, assistant_message=echo_one),
        ModelTurn(tool_calls=batch_two, assistant_message=echo_two),
        ModelTurn(text="done"),
    ]
    seen = []

    def call_model(messages, active):
        seen.append(_snapshot(messages))
        return script.pop(0)

    def invoke(call):
        if call.call_id in ("c1", "c3"):
            post(STEERING_SOURCE_SUPERVISOR, f"steer after {call.call_id}")
        return ToolResult(ok=True, content=f"r-{call.call_id}")

    out = run_agent_loop(
        # Two 2-call batches spend 11 steps before the final turn; the
        # default max_steps=8 would end this RUN_STUCK before turn 3.
        AgentConfig(
            model="test-model",
            system_prompt="s",
            allowed_tools=("calculator",),
            budget=RunBudget(max_steps=40, max_model_turns=40),
        ),
        [{"role": "user", "content": "hi"}],
        [CALC],
        make_deps(call_model, invoke=invoke, drain=drain),
    )

    assert out.status == RUN_DONE
    steer_one = format_steering_message(STEERING_SOURCE_SUPERVISOR, "steer after c1")
    steer_two = format_steering_message(STEERING_SOURCE_SUPERVISOR, "steer after c3")
    assert seen[2] == [
        {"role": "user", "content": "hi"},
        echo_one,
        {"role": "tool", "tool_call_id": "c1", "content": "r-c1"},
        {"role": "tool", "tool_call_id": "c2", "content": "r-c2"},
        {"role": "user", "content": steer_one},
        echo_two,
        {"role": "tool", "tool_call_id": "c3", "content": "r-c3"},
        {"role": "tool", "tool_call_id": "c4", "content": "r-c4"},
        {"role": "user", "content": steer_two},
    ]
    for payload in seen:
        _assert_batch_pairing_unbroken(payload)


# -- continuation fixtures (the deepseek shape the 4a/4c suites pin) ------


def _checkpoint(
    *calls: ContinuationCall,
    revision: int = 1,
    state: str = "active",
    assistant_content: str = "",
    reasoning: tuple[str, ...] = ("private",),
) -> ProviderContinuationCheckpoint:
    return ProviderContinuationCheckpoint(
        schema_version=1,
        checkpoint_revision=revision,
        provider="deepseek",
        protocol="responses",
        model="deepseek-v4-flash",
        api_base_url="https://api.deepseek.com/v1",
        state=state,  # type: ignore[arg-type]
        rounds=(
            ContinuationRound(
                assistant_content=assistant_content,
                reasoning_blocks=reasoning,
                calls=tuple(calls),
            ),
        ),
    )


def _pending_call(call_id: str = "call-1") -> ContinuationCall:
    return ContinuationCall(
        call_id=call_id,
        name="calculator",
        arguments=json.dumps({"expression": "2+2"}, separators=(",", ":")),
        state="pending",
    )


_RESTORE_TARGET = ContinuationRestoreTarget(
    "deepseek", "deepseek-v4-flash", "responses", "https://api.deepseek.com/v1"
)
_CONTEXT = ContinuationEventContext(
    owner_message_id="assistant-owner",
    run_id="run-1",
    agent_kind="primary",
    durability="persistent",
)


def _final_checkpoint() -> ProviderContinuationCheckpoint:
    return _checkpoint(
        ContinuationCall(
            call_id="call-1",
            name="calculator",
            arguments=json.dumps({"expression": "2+2"}, separators=(",", ":")),
            state="completed",
            result=ContinuationResult("4"),
        ),
        revision=4,
        state="complete",
    )


def test_red_c_restore_batch_path_never_drains():
    """Red (c): an entry posted before a resume survives to the
    post-restore turn -- the restoring branch itself never drains (a drain
    there would be wiped by ``expand_restore_history``'s slice-rewrite)."""
    order = []
    mailbox = [(STEERING_SOURCE_USER, "posted before resume")]

    def drain():
        order.append("drain")
        got = list(mailbox)
        mailbox.clear()
        return got

    def invoke(call):
        order.append("invoke")
        return ToolResult(ok=True, content="4")

    seen = []

    def call_model(messages, active):
        order.append("model")
        seen.append(_snapshot(messages))
        return ModelTurn(text="final", provider_continuation=_final_checkpoint())

    def expand(actual):
        return [
            {
                "role": "tool",
                "tool_call_id": call.call_id,
                "content": call.result.value if call.result else "…pending…",
            }
            for round_ in actual.rounds
            for call in round_.calls
        ]

    deps = make_deps(call_model, invoke=invoke, drain=drain)
    deps.continuation_context = _CONTEXT
    deps.persist_provider_continuation = lambda event: None
    deps.expand_provider_continuation = expand

    out = run_agent_loop(
        STEER_CFG,
        [{"role": "user", "content": "go"}],
        [CALC],
        deps,
        restore_provider_continuation=_checkpoint(_pending_call()),
        restore_provider_target=_RESTORE_TARGET,
        resume_provider_continuation=True,
    )

    assert out.status == RUN_DONE and out.final_text == "final"
    # The restored batch executed FIRST, undrained; the one drain happened
    # at the next (non-restoring) boundary, before the model call.
    assert order == ["invoke", "drain", "model"]
    labeled = format_steering_message(STEERING_SOURCE_USER, "posted before resume")
    assert seen == [
        [
            {"role": "user", "content": "go"},
            {"role": "tool", "tool_call_id": "call-1", "content": "4"},
            {"role": "user", "content": labeled},
        ]
    ]


def test_red_d_drain_under_an_active_checkpoint_produces_no_continuation_error():
    """Red (d): the full 4a barrier cycle with a mid-batch post stays
    RUN_DONE -- the injected user message never trips a continuation
    barrier, because no barrier validates ``messages``."""
    post, drain = _list_mailbox()
    call = ToolCall(
        "calculator", {"expression": "2+2"}, "call-1", '{"expression":"2+2"}'
    )
    events = []
    script = [
        ModelTurn(
            tool_calls=(call,),
            assistant_message={
                "role": "assistant",
                "content": "",
                "tool_calls": [_raw_call(call)],
            },
            provider_continuation=_checkpoint(_pending_call()),
        ),
        ModelTurn(text="4", provider_continuation=_final_checkpoint()),
    ]
    seen = []

    def call_model(messages, active):
        seen.append(_snapshot(messages))
        return script.pop(0)

    def invoke(actual):
        post(STEERING_SOURCE_SUPERVISOR, "keep it brief")
        return ToolResult(ok=True, content="4")

    deps = make_deps(call_model, invoke=invoke, drain=drain)
    deps.continuation_context = _CONTEXT
    deps.persist_provider_continuation = lambda event: events.append(type(event))

    out = run_agent_loop(
        STEER_CFG,
        [{"role": "user", "content": "2+2?"}],
        [CALC],
        deps,
    )

    assert out.status == RUN_DONE and out.final_text == "4"
    assert not [step for step in out.steps if step.kind == STEP_ERROR]
    # The full barrier sequence ran -- no barrier was skipped or repeated.
    assert events == [
        ToolBatchReady,
        ToolCallExecuting,
        ToolCallFinished,
        FinalContinuation,
    ]
    labeled = format_steering_message(STEERING_SOURCE_SUPERVISOR, "keep it brief")
    # Delivered under the ACTIVE checkpoint, at the coherent boundary.
    assert seen[1][-2:] == [
        {"role": "tool", "tool_call_id": "call-1", "content": "4"},
        {"role": "user", "content": labeled},
    ]


def test_red_e_a_raising_drain_does_not_abort_the_run():
    """Red (e): the on_step containment rule -- a broken drain callable
    costs the delivery, never the run."""
    drain_calls = []

    def drain():
        drain_calls.append(True)
        raise RuntimeError("mailbox exploded")

    script = [
        ModelTurn(text=fence("calculator", {"expression": "6*7"})),
        ModelTurn(text="recovered"),
    ]
    seen = []

    def call_model(messages, active):
        seen.append(_snapshot(messages))
        return script.pop(0)

    out = run_agent_loop(
        STEER_CFG,
        [{"role": "user", "content": "hi"}],
        [CALC],
        make_deps(call_model, drain=drain),
    )

    assert out.status == RUN_DONE and out.final_text == "recovered"
    assert len(drain_calls) == 2  # tried at every boundary, kept failing
    assert not [step for step in out.steps if step.kind == STEP_ERROR]
    for payload in seen:
        assert not [
            message
            for message in payload
            if "[Steering from" in str(message.get("content", ""))
        ]


# -- red (g): a dead run never consumes a mailbox -------------------------


def _coordinator_drain(coordinator, handle_id, drain_calls):
    def drain():
        drain_calls.append(True)
        return coordinator.drain_steering(handle_id)

    return drain


def test_red_g_a_cancelled_run_leaves_entries_queued():
    coordinator = _coord()
    handle = coordinator.reserve(task="child", agent=None)
    coordinator.post_steering(
        handle.handle_id, STEERING_SOURCE_USER, "still queued"
    )
    drain_calls = []

    def call_model(messages, active):
        raise AssertionError("a cancelled run must never call the model")

    out = run_agent_loop(
        STEER_CFG,
        [{"role": "user", "content": "hi"}],
        [CALC],
        make_deps(
            call_model,
            cancel=lambda: True,
            drain=_coordinator_drain(coordinator, handle.handle_id, drain_calls),
        ),
    )

    assert out.status == RUN_CANCELLED
    assert drain_calls == []
    assert coordinator.get(handle.handle_id).queued_steering == 1
    assert coordinator.drain_steering(handle.handle_id) == [
        (STEERING_SOURCE_USER, "still queued")
    ]


def test_red_g_a_mid_batch_cancel_leaves_a_late_post_queued():
    """A Stop landing DURING a model turn kills the batch before dispatch
    (:1068-1069); an entry posted after that turn's drain stays queued."""
    coordinator = _coord()
    handle = coordinator.reserve(task="child", agent=None)
    drain_calls = []
    cancelled = []

    def call_model(messages, active):
        coordinator.post_steering(
            handle.handle_id, STEERING_SOURCE_SUPERVISOR, "too late"
        )
        cancelled.append(True)
        return ModelTurn(text=fence("calculator", {"expression": "6*7"}))

    out = run_agent_loop(
        STEER_CFG,
        [{"role": "user", "content": "hi"}],
        [CALC],
        make_deps(
            call_model,
            invoke=lambda call: (_ for _ in ()).throw(
                AssertionError("cancel must precede dispatch")
            ),
            cancel=lambda: bool(cancelled),
            drain=_coordinator_drain(coordinator, handle.handle_id, drain_calls),
        ),
    )

    assert out.status == RUN_CANCELLED
    assert len(drain_calls) == 1  # the one boundary before the model call
    assert coordinator.get(handle.handle_id).queued_steering == 1


def test_red_g_a_budget_exhausted_run_leaves_entries_queued():
    coordinator = _coord()
    handle = coordinator.reserve(task="child", agent=None)
    coordinator.post_steering(
        handle.handle_id, STEERING_SOURCE_SUPERVISOR, "never consumed"
    )
    drain_calls = []

    def call_model(messages, active):
        raise AssertionError("an exhausted run must never call the model")

    out = run_agent_loop(
        AgentConfig(
            model="test-model",
            system_prompt="s",
            allowed_tools=("calculator",),
            budget=RunBudget(max_model_turns=0),
        ),
        [{"role": "user", "content": "hi"}],
        [CALC],
        make_deps(
            call_model,
            drain=_coordinator_drain(coordinator, handle.handle_id, drain_calls),
        ),
    )

    assert out.status == RUN_STUCK
    assert "model-turn budget exhausted" in out.steps[-1].summary
    assert drain_calls == []
    assert coordinator.get(handle.handle_id).queued_steering == 1


def test_red_g_a_cycle_stuck_run_leaves_a_late_post_queued():
    coordinator = _coord()
    handle = coordinator.reserve(task="child", agent=None)
    drain_calls = []
    turns = []

    def call_model(messages, active):
        turns.append(True)
        if len(turns) == 3:
            # After this turn's drain; the cycle detector kills the run
            # before any further boundary exists.
            coordinator.post_steering(
                handle.handle_id, STEERING_SOURCE_USER, "posted at the end"
            )
        return ModelTurn(text=fence("calculator", {"expression": "9"}))

    out = run_agent_loop(
        STEER_CFG,
        [{"role": "user", "content": "hi"}],
        [CALC],
        make_deps(
            call_model,
            drain=_coordinator_drain(coordinator, handle.handle_id, drain_calls),
        ),
    )

    assert out.status == RUN_STUCK
    assert "calculator" in out.steps[-1].summary
    assert len(drain_calls) == 3  # one per boundary, all before the post
    assert coordinator.get(handle.handle_id).queued_steering == 1


# -- service threading (agent_service: _run_one + spawn's fleet branch) ---
#
# The impure seam: a THREADED fleet child's LoopDeps.drain_mailbox is the
# service-built closure over that child's own coordinator mailbox;
# primaries and inline children stay unwired (None).


@pytest.fixture()
def db(tmp_path):
    return AgentRunsDB(tmp_path / "runs.db", client_id="test")


def test_fleet_child_drain_is_wired_to_its_own_coordinator_mailbox(db):
    """End-to-end: a post to the child's handle reaches the child's NEXT
    provider payload as the labeled user-role message, at the boundary."""
    holder = {}

    def steer_then_call():
        [handle] = [
            h for h in holder["coordinator"].snapshot() if h.status == RUN_RUNNING
        ]
        assert holder["coordinator"].post_steering(
            handle.handle_id, STEERING_SOURCE_SUPERVISOR, "wrap up quickly"
        )
        holder["handle_id"] = handle.handle_id
        return fence("calculator", {"expression": "6*7"})

    service, chat, coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "task one"}),
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "combined answer",
        ],
        {"task one": [steer_then_call, "child answer"]},
    )
    holder["coordinator"] = coordinator

    _run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=FLEET_CFG,
        api_endpoint="llama_cpp",
    )

    assert outcome.status == RUN_DONE
    labeled = format_steering_message(STEERING_SOURCE_SUPERVISOR, "wrap up quickly")
    second_payload = chat.child_calls["task one"][1]["messages_payload"]
    # Delivered at the coherent boundary: after the batch's tool result,
    # as the final message before the child's next assistant turn.
    assert second_payload[-1] == {"role": "user", "content": labeled}
    assert str(second_payload[-2]["content"]).startswith(
        f"{FENCE_TOOL_RESULT_PREFIX}calculator:"
    )
    # Consumed from THAT child's mailbox, not merely copied.
    assert coordinator.drain_steering(holder["handle_id"]) == []
    # The PRIMARY's payloads never carry the steering message -- the wiring
    # is per-child, not per-service.
    for call in chat.parent_calls:
        assert not [
            message
            for message in call["messages_payload"]
            if labeled in str(message.get("content", ""))
        ]


def test_only_the_threaded_fleet_child_is_wired_for_drain(db, monkeypatch):
    """The primary's LoopDeps.drain_mailbox is None; the fleet child's is
    the service-built closure."""
    recorded = []
    real_loop = agent_service.run_agent_loop

    def spy(config, messages, active, deps, **kwargs):
        recorded.append((config.system_prompt, deps.drain_mailbox))
        return real_loop(config, messages, active, deps, **kwargs)

    monkeypatch.setattr(agent_service, "run_agent_loop", spy)
    service, chat, coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "task one"}),
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "combined answer",
        ],
        {"task one": ["child answer"]},
    )
    _run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=FLEET_CFG,
        api_endpoint="llama_cpp",
    )

    assert outcome.status == RUN_DONE
    primary_drains = [
        drain
        for prompt, drain in recorded
        if not prompt.startswith(SUBAGENT_PROMPT_PREFIX)
    ]
    child_drains = [
        drain
        for prompt, drain in recorded
        if prompt.startswith(SUBAGENT_PROMPT_PREFIX)
    ]
    # TASK-25903 updated this contract: the primary is now wired too -- to
    # its USER-steering mailbox (steer_primary), a different producer from
    # the child's coordinator mailbox. What this test still pins is that the
    # fleet child gets its own drain and that they are distinct objects.
    assert len(primary_drains) == 1 and primary_drains[0] is not None
    assert len(child_drains) == 1 and child_drains[0] is not None
    assert primary_drains[0] is not child_drains[0]


def test_inline_children_and_their_primary_stay_unwired(db, monkeypatch):
    """CHARACTERIZATION PIN (not a red -- current behavior is already
    correct): the inline path has no handle and so no mailbox; wiring a
    drain there would be the regression this test exists to catch.

    TASK-25903: the PRIMARY half of the old assertion is superseded -- a
    primary now carries its user-steering drain -- so this pins only that
    the INLINE CHILD stays unwired."""
    recorded = []
    real_loop = agent_service.run_agent_loop

    def spy(config, messages, active, deps, **kwargs):
        recorded.append(deps.drain_mailbox)
        return real_loop(config, messages, active, deps, **kwargs)

    monkeypatch.setattr(agent_service, "run_agent_loop", spy)
    service, _chat = make_inline_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "inline task"}),
            "inline child answer",
            "final answer",
        ],
        monkeypatch,
    )
    _run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=FLEET_CFG,
        api_endpoint="llama_cpp",
    )

    assert outcome.status == RUN_DONE and outcome.final_text == "final answer"
    assert len(recorded) == 2  # primary + one inline child
    primary_drain, inline_child_drain = recorded
    assert primary_drain is not None, "the primary's user-steering drain"
    assert inline_child_drain is None, (
        "an inline child has no handle and must stay unwired"
    )
