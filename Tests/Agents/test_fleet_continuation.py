# Tests/Agents/test_fleet_continuation.py
"""Fleet PR 3b Task 4: finished-agent retention + continuation.

Spec SS6: ``send_to_agent`` to a finished child starts a NEW run seeded
with the retained transcript (+ undelivered queued steering) + the new
message, linked via ``resumed_from_run_id``; after a restart the
transcript is gone and the error says so. Plan:
Docs/superpowers/plans/2026-08-17-fleet-pr3b-steering.md, Task 4, and the
three coordinator rulings recorded there:

  1. a still-existing definition re-resolves to its CURRENT form (the new
     row's ``definition_fingerprint`` records the change); a deleted or
     disabled one refuses clearly suggesting a fresh spawn;
  2. an oversize transcript is NOT retained (truncation could split native
     pairs and silently change the child's memory);
  3. the same migration folds the task-15669 constant-vs-version-table
     drift fix (the DB-side reds live in Tests/DB/test_agent_runs_db.py).

The plan-mandated reds live here:
  - the coherent-boundary property: for EVERY terminal path,
    ``RunOutcome.final_messages`` never ends inside an unpaired native
    batch (Hypothesis, plus targeted exact-sequence pins per path);
  - retention caps + oldest-first eviction + the oversize refusal +
    cancelled/superseded never retained + the mailbox remnant claimed;
  - the prune window: a finished child remains continuable after
    ``prune_terminal`` (Task 2's concern (a), closed -- continuation
    resolves against the retention store, which survives the prune);
  - the resumed row carries ``resumed_from_run_id``, CURRENT-turn lineage
    (``parent_run_id`` = the resuming primary), and a FRESH
    ``definition_fingerprint``;
  - undelivered queued steering rides the seed with its ORIGINAL labels;
  - live-cap and spawn-budget refusals for the resume path (a resume
    consumes a spawn slot; a cap refusal unwinds it);
  - the post-restart error copy (fresh coordinator, empty retention: the
    transcript is gone, the error says so and suggests a fresh spawn).
"""

from __future__ import annotations

import json
import threading

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from Tests.Agents.test_agent_service import fence
from Tests.Agents.test_fleet_runtime import (
    _JOIN_TIMEOUT,
    _tool_results,
    _wait_until,
    make_fleet_service,
)
from tldw_chatbook.Agents.agent_models import (
    AGENT_KIND_SUBAGENT,
    FENCE_TOOL_RESULT_PREFIX,
    RUN_CANCELLED,
    RUN_DONE,
    RUN_ERROR,
    RUN_STUCK,
    RUN_SUPERSEDED,
    SEND_TO_AGENT_TOOL_NAME,
    SPAWN_TOOL_NAME,
    STEERING_SOURCE_SUPERVISOR,
    STEERING_SOURCE_USER,
    TERMINAL_RUN_STATUSES,
    WAIT_AGENTS_TOOL_NAME,
    AgentConfig,
    AgentDefinition,
    ModelTurn,
    RunBudget,
    ToolCall,
    ToolResult,
    definition_fingerprint,
    format_steering_message,
)
from tldw_chatbook.Agents.agent_runtime import run_agent_loop
from tldw_chatbook.Agents.fleet_coordinator import (
    DEFAULT_RETAINED_TRANSCRIPT_MAX_CHARS,
    DEFAULT_RETAINED_TRANSCRIPTS,
    FleetCoordinator,
)
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

# LoopDeps plumbing shared with the Task 1 suite (one `make_deps`, one
# pairing checker idiom) so the two coherence surfaces can never drift.
from Tests.Agents.test_fleet_steering_mailbox import (
    CALC,
    STEER_CFG,
    _list_mailbox,
    _raw_call,
    _snapshot,
    make_deps,
)


@pytest.fixture()
def db(tmp_path):
    return AgentRunsDB(tmp_path / "runs.db", client_id="test")


# =========================================================================
# 1. The coherent boundary (agent_runtime, pure): RunOutcome.final_messages
# =========================================================================


def _native_turn(calls):
    echo = {
        "role": "assistant",
        "content": "",
        "tool_calls": [_raw_call(call) for call in calls],
    }
    return ModelTurn(tool_calls=tuple(calls), assistant_message=echo)


def _assert_coherent(final_messages):
    """No unpaired native batch, no orphan fence call, anywhere.

    Stronger than the Task 1 pairing scan: every assistant message with
    ``tool_calls`` must be followed by EXACTLY its results (COMPLETE, in
    order) -- a transcript that merely never interleaves but ends with a
    half-answered batch would still poison a resumed run's first provider
    call.
    """
    assert final_messages is not None
    index = 0
    while index < len(final_messages):
        message = final_messages[index]
        if message.get("role") == "assistant" and message.get("tool_calls"):
            wanted = [c["id"] for c in message["tool_calls"]]
            got = []
            cursor = index + 1
            while (
                cursor < len(final_messages)
                and final_messages[cursor].get("role") == "tool"
            ):
                got.append(final_messages[cursor].get("tool_call_id"))
                cursor += 1
            assert got == wanted, (
                f"assistant batch at {index} wants results for {wanted} "
                f"but the transcript carries {got}: {final_messages!r}"
            )
            index = cursor
            continue
        if message.get("role") == "tool":
            pytest.fail(
                f"orphan role=tool message at {index}: {final_messages!r}"
            )
        index += 1


@st.composite
def _terminal_scripts(draw):
    """A random run shape plus a random way for it to die.

    ``rounds`` is a list of (native, size) tool batches; after them a
    plain-text turn ends the run DONE. ``cancel_after`` (when not None)
    flips ``should_cancel`` after that many tool invocations -- values on
    a batch boundary produce loop-top cancels, values inside a batch
    produce the mid-batch cancel return; ``identical`` makes every call
    byte-identical so the cycle detector's mid-batch RUN_STUCK return
    fires; ``tiny_steps`` exhausts the step budget instead; a steering
    post at a random invocation exercises the drain-boundary messages
    riding the transcript.
    """
    rounds = draw(
        st.lists(
            st.tuples(st.booleans(), st.integers(min_value=1, max_value=3)),
            min_size=0,
            max_size=4,
        )
    )
    # A fence turn carries exactly one call.
    rounds = [(native, size if native else 1) for native, size in rounds]
    total_invokes = sum(size for _native, size in rounds)
    cancel_after = draw(
        st.one_of(st.none(), st.integers(min_value=0, max_value=total_invokes))
    )
    identical = draw(st.booleans())
    tiny_steps = draw(st.booleans())
    post_at = (
        draw(
            st.one_of(
                st.none(), st.integers(min_value=1, max_value=total_invokes)
            )
        )
        if total_invokes
        else None
    )
    return rounds, cancel_after, identical, tiny_steps, post_at


def _run_scripted(rounds, cancel_after, identical, tiny_steps, post_at):
    """Drive the pure loop over one generated shape; return (outcome, seen)."""
    post, drain = _list_mailbox()
    script = []
    for r, (native, size) in enumerate(rounds):
        expr = "1+1" if identical else None
        if native:
            calls = [
                ToolCall(
                    "calculator",
                    {"expression": expr or f"{r}+{j}"},
                    f"c{r}-{j}",
                    json.dumps({"expression": expr or f"{r}+{j}"}),
                )
                for j in range(size)
            ]
            script.append(_native_turn(calls))
        else:
            script.append(
                ModelTurn(
                    text=fence(
                        "calculator", {"expression": expr or f"{r}+0"}
                    )
                )
            )
    script.append(ModelTurn(text="final answer"))

    seen = []
    invoked = [0]

    def call_model(messages, active):
        seen.append(_snapshot(messages))
        return script.pop(0) if script else ModelTurn(text="over-asked")

    def invoke(call):
        invoked[0] += 1
        if post_at is not None and invoked[0] == post_at:
            post(STEERING_SOURCE_USER, "mid-run note")
        return ToolResult(ok=True, content=f"r-{invoked[0]}")

    def should_cancel():
        return cancel_after is not None and invoked[0] >= cancel_after

    config = AgentConfig(
        model="test-model",
        system_prompt="s",
        allowed_tools=("calculator",),
        budget=RunBudget(
            max_steps=3 if tiny_steps else 200,
            max_model_turns=200,
            max_wall_seconds=240.0,
        ),
    )
    outcome = run_agent_loop(
        config,
        [{"role": "user", "content": "hi"}],
        [CALC],
        make_deps(call_model, invoke=invoke, cancel=should_cancel, drain=drain),
    )
    return outcome, seen


@settings(max_examples=80, deadline=None)
@given(_terminal_scripts())
def test_property_final_messages_end_at_a_coherent_boundary(script_shape):
    """THE property: for every terminal path the retained transcript is the
    last drain-boundary prefix -- exactly what the model saw at its last
    call (plus, on RUN_DONE only, the final assistant text) -- and never
    ends inside an unpaired native batch."""
    rounds, cancel_after, identical, tiny_steps, post_at = script_shape
    outcome, seen = _run_scripted(
        rounds, cancel_after, identical, tiny_steps, post_at
    )
    assert outcome.status in {RUN_DONE, RUN_CANCELLED, RUN_STUCK}
    final = outcome.final_messages
    _assert_coherent(final)
    boundary = seen[-1] if seen else [{"role": "user", "content": "hi"}]
    if outcome.status == RUN_DONE:
        assert final == boundary + [
            {"role": "assistant", "content": outcome.final_text}
        ]
    else:
        assert final == boundary


def test_run_done_appends_the_final_assistant_text():
    """RUN_DONE returns BEFORE the loop's own assistant append, so the
    retained transcript must append ``final_text`` itself -- exact list."""
    post, drain = _list_mailbox()
    fence_text = fence("calculator", {"expression": "6*7"})
    script = [ModelTurn(text=fence_text), ModelTurn(text="It is 42.")]

    def call_model(messages, active):
        return script.pop(0)

    out = run_agent_loop(
        STEER_CFG,
        [{"role": "user", "content": "hi"}],
        [CALC],
        make_deps(call_model, drain=drain),
    )
    assert out.status == RUN_DONE
    assert out.final_messages == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": fence_text},
        {"role": "user", "content": f"{FENCE_TOOL_RESULT_PREFIX}calculator: 42"},
        {"role": "assistant", "content": "It is 42."},
    ]


def test_mid_batch_cancel_excludes_the_split_batch():
    """The cancel return inside a 2-call native batch: the transcript is
    the LAST boundary -- the split batch (assistant echo + first result)
    is excluded wholesale, never half-kept."""
    call_one = ToolCall("calculator", {"expression": "1"}, "c1", '{"expression":"1"}')
    call_two = ToolCall("calculator", {"expression": "2"}, "c2", '{"expression":"2"}')
    script = [_native_turn([call_one, call_two])]
    invoked = [0]

    def call_model(messages, active):
        return script.pop(0)

    def invoke(call):
        invoked[0] += 1
        return ToolResult(ok=True, content=f"r-{call.call_id}")

    out = run_agent_loop(
        STEER_CFG,
        [{"role": "user", "content": "hi"}],
        [CALC],
        make_deps(
            call_model, invoke=invoke, cancel=lambda: invoked[0] >= 1
        ),
    )
    assert out.status == RUN_CANCELLED
    assert out.final_messages == [{"role": "user", "content": "hi"}]


def test_mid_batch_cycle_stuck_excludes_the_split_batch():
    """The cycle-stuck return mid-batch: one COMPLETE fence round survives
    into the transcript; the tripping batch does not."""
    fence_text = fence("calculator", {"expression": "6*7"})
    same = {"expression": "1+1"}
    calls = [
        ToolCall("calculator", same, f"c{j}", json.dumps(same)) for j in range(3)
    ]
    script = [ModelTurn(text=fence_text), _native_turn(calls)]

    def call_model(messages, active):
        return script.pop(0)

    out = run_agent_loop(
        STEER_CFG,
        [{"role": "user", "content": "hi"}],
        [CALC],
        make_deps(call_model),
    )
    assert out.status == RUN_STUCK
    assert out.final_messages == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": fence_text},
        {"role": "user", "content": f"{FENCE_TOOL_RESULT_PREFIX}calculator: 42"},
    ]


def test_no_calls_cancel_keeps_the_streamed_text_out_of_the_transcript():
    """A Stop landing while a tool-call-free turn streamed: RUN_CANCELLED
    with ``final_text`` set, but the never-delivered text does NOT ride
    ``final_messages`` (only RUN_DONE appends the assistant line)."""
    cancelled = [False]

    def call_model(messages, active):
        cancelled[0] = True  # flips DURING the (final, text-only) turn
        return ModelTurn(text="half-streamed answer")

    out = run_agent_loop(
        STEER_CFG,
        [{"role": "user", "content": "hi"}],
        [CALC],
        make_deps(call_model, cancel=lambda: cancelled[0]),
    )
    assert out.status == RUN_CANCELLED
    assert out.final_text == "half-streamed answer"
    assert out.final_messages == [{"role": "user", "content": "hi"}]


def test_delivered_steering_rides_the_coherent_transcript():
    """A steering entry drained at the boundary is part of what the child
    saw -- so it must be part of what a resumed child remembers."""
    post, drain = _list_mailbox()
    fence_text = fence("calculator", {"expression": "6*7"})
    script = [ModelTurn(text=fence_text), ModelTurn(text="done.")]

    def call_model(messages, active):
        return script.pop(0)

    def invoke(call):
        post(STEERING_SOURCE_USER, "remember this")
        return ToolResult(ok=True, content="42")

    out = run_agent_loop(
        STEER_CFG,
        [{"role": "user", "content": "hi"}],
        [CALC],
        make_deps(call_model, invoke=invoke, drain=drain),
    )
    labeled = format_steering_message(STEERING_SOURCE_USER, "remember this")
    assert out.status == RUN_DONE
    assert out.final_messages == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": fence_text},
        {"role": "user", "content": f"{FENCE_TOOL_RESULT_PREFIX}calculator: 42"},
        {"role": "user", "content": labeled},
        {"role": "assistant", "content": "done."},
    ]


def test_budget_exhausted_at_loop_top_yields_the_last_boundary():
    fence_text = fence("calculator", {"expression": "6*7"})
    script = [ModelTurn(text=fence_text), ModelTurn(text="never reached")]

    def call_model(messages, active):
        return script.pop(0)

    cfg = AgentConfig(
        model="test-model",
        system_prompt="s",
        allowed_tools=("calculator",),
        budget=RunBudget(max_steps=3, max_model_turns=40),
    )
    out = run_agent_loop(
        cfg,
        [{"role": "user", "content": "hi"}],
        [CALC],
        make_deps(call_model),
    )
    assert out.status == RUN_STUCK
    # The plan's semantics, pinned deliberately: `coherent_len` advances at
    # the DRAIN BOUNDARY (after the loop-top budget/cancel checks, before
    # the model call), so a loop-top terminal return carries the boundary
    # BEFORE the final completed round -- at most one fully-appended round
    # is dropped. Conservative by design: the one blessed capture point is
    # the same protocol-coherent line the steering drain earned, rather
    # than a second capture point that would need its own restore-machinery
    # proof. The transcript is exactly what the model saw at its last call.
    assert out.final_messages == [{"role": "user", "content": "hi"}]


# =========================================================================
# 2. Retention (fleet_coordinator, pure, locked)
# =========================================================================


def _coord(max_live=3, **caps):
    ticks = iter(range(100_000))
    return FleetCoordinator(
        max_live=max_live, clock=lambda: float(next(ticks)), **caps
    )


_TRANSCRIPT = (
    {"role": "user", "content": "study the logs"},
    {"role": "assistant", "content": "on it"},
)


def _finished_handle(c, status=RUN_DONE, task="child"):
    h = c.reserve(task=task, agent=None)
    c.finish(h.handle_id, status, result="r", error="")
    return h


def test_retention_defaults_and_caps_are_exposed():
    c = _coord()
    assert DEFAULT_RETAINED_TRANSCRIPTS == 5
    assert DEFAULT_RETAINED_TRANSCRIPT_MAX_CHARS == 200_000
    assert c.retained_transcripts == DEFAULT_RETAINED_TRANSCRIPTS
    assert c.retained_transcript_max_chars == DEFAULT_RETAINED_TRANSCRIPT_MAX_CHARS


@pytest.mark.parametrize("status", [RUN_DONE, RUN_STUCK, RUN_ERROR])
def test_done_stuck_and_error_children_are_retained(status):
    c = _coord()
    h = _finished_handle(c, status=status)
    assert c.retain_transcript(h.handle_id, list(_TRANSCRIPT)) is True
    entry = c.get_retained(h.handle_id)
    assert entry is not None
    assert entry.status == status
    assert list(entry.messages) == list(_TRANSCRIPT)


@pytest.mark.parametrize("status", [RUN_CANCELLED, RUN_SUPERSEDED])
def test_cancelled_and_superseded_children_are_never_retained(status):
    """The user killed it / it was replaced -- resuming either would undo
    an explicit human decision."""
    c = _coord()
    h = _finished_handle(c, status=status)
    assert c.retain_transcript(h.handle_id, list(_TRANSCRIPT)) is False
    assert c.get_retained(h.handle_id) is None


def test_a_live_handle_and_a_missing_transcript_are_refused():
    c = _coord()
    live = c.reserve(task="still running", agent=None)
    assert c.retain_transcript(live.handle_id, list(_TRANSCRIPT)) is False
    done = _finished_handle(c)
    assert c.retain_transcript(done.handle_id, None) is False
    assert c.get_retained(done.handle_id) is None


def test_retain_claims_the_undelivered_mailbox_remnant():
    """Task 1 pinned that remnants survive finish() until prune; Task 4 is
    the claimant -- the entries move INTO the retained entry, the mailbox
    reads empty, and the queued count on copies drops to 0."""
    c = _coord()
    h = c.reserve(task="child", agent=None)
    c.post_steering(h.handle_id, STEERING_SOURCE_USER, "one")
    c.post_steering(h.handle_id, STEERING_SOURCE_SUPERVISOR, "two")
    c.finish(h.handle_id, RUN_DONE, result="r")
    assert c.retain_transcript(h.handle_id, list(_TRANSCRIPT)) is True
    entry = c.get_retained(h.handle_id)
    assert list(entry.steering) == [
        (STEERING_SOURCE_USER, "one"),
        (STEERING_SOURCE_SUPERVISOR, "two"),
    ]
    assert c.drain_steering(h.handle_id) == []
    assert c.get(h.handle_id).queued_steering == 0


def test_retention_survives_prune_terminal_by_both_ids():
    """THE prune window (Task 2 concern (a)): retention lives in its own
    store, so the turn-start prune that drops terminal handles cannot
    drop a finished child's continuability."""
    c = _coord()
    h = c.reserve(task="child", agent=None)
    c.attach_run(h.handle_id, "run-xyz")
    c.finish(h.handle_id, RUN_DONE, result="r")
    assert c.retain_transcript(h.handle_id, list(_TRANSCRIPT)) is True
    assert c.prune_terminal() == 1
    assert c.get(h.handle_id) is None  # the handle really is gone
    assert c.get_retained(h.handle_id) is not None
    assert c.get_retained("run-xyz") is not None
    assert c.get_retained("run-xyz").run_id == "run-xyz"


def test_get_retained_resolves_handle_id_before_a_colliding_run_id():
    """Same vocabulary order as live resolution (Task 2's pin): a forged
    collision -- one child's run id equals another's handle id -- lands on
    the handle-id owner."""
    c = _coord()
    a = c.reserve(task="task a", agent=None)
    b = c.reserve(task="task b", agent=None)
    # Forge: A's run id becomes B's handle id.
    c.attach_run(a.handle_id, b.handle_id)
    c.finish(a.handle_id, RUN_DONE, result="a")
    c.finish(b.handle_id, RUN_DONE, result="b")
    assert c.retain_transcript(a.handle_id, [{"role": "user", "content": "a"}])
    assert c.retain_transcript(b.handle_id, [{"role": "user", "content": "b"}])
    assert c.get_retained(b.handle_id).handle_id == b.handle_id


def test_oversize_transcripts_are_not_retained():
    """Ruling #2: refuse, never truncate -- a cut transcript could split
    native pairs and silently change the child's memory."""
    c = _coord(retained_transcript_max_chars=80)
    h = _finished_handle(c)
    big = [{"role": "user", "content": "x" * 200}]
    assert c.retain_transcript(h.handle_id, big) is False
    assert c.get_retained(h.handle_id) is None
    small = _finished_handle(c, task="small")
    assert c.retain_transcript(
        small.handle_id, [{"role": "user", "content": "ok"}]
    ) is True


def test_oldest_is_evicted_first_at_the_count_cap():
    c = _coord(retained_transcripts=2)
    handles = []
    for index in range(3):
        h = c.reserve(task=f"t{index}", agent=None)
        c.attach_run(h.handle_id, f"run-{index}")
        c.finish(h.handle_id, RUN_DONE, result="r")
        assert c.retain_transcript(
            h.handle_id, [{"role": "user", "content": f"t{index}"}]
        )
        handles.append(h)
    assert c.get_retained(handles[0].handle_id) is None
    assert c.get_retained("run-0") is None
    assert c.get_retained(handles[1].handle_id) is not None
    assert c.get_retained(handles[2].handle_id) is not None


def test_set_retention_caps_resizes_in_place_and_evicts_oldest():
    """The set_max_live shape: a cross-turn owner re-reads config every
    turn and re-sizes the SAME store rather than replacing it."""
    c = _coord()
    handles = [
        _finished_handle(c, task=f"t{index}") for index in range(3)
    ]
    for h in handles:
        assert c.retain_transcript(h.handle_id, list(_TRANSCRIPT))
    c.set_retention_caps(1, 50)
    assert c.retained_transcripts == 1
    assert c.retained_transcript_max_chars == 50
    assert c.get_retained(handles[0].handle_id) is None
    assert c.get_retained(handles[1].handle_id) is None
    assert c.get_retained(handles[2].handle_id) is not None
    # The lowered char cap governs the NEXT retention.
    big = _finished_handle(c, task="big")
    assert c.retain_transcript(
        big.handle_id, [{"role": "user", "content": "y" * 100}]
    ) is False


def test_a_zero_count_cap_retains_nothing():
    c = _coord(retained_transcripts=0)
    h = _finished_handle(c)
    assert c.retain_transcript(h.handle_id, list(_TRANSCRIPT)) is False
    assert c.get_retained(h.handle_id) is None


class _CountingLock:
    """A drop-in for the coordinator's Lock that counts critical sections."""

    def __init__(self):
        self._inner = threading.Lock()
        self.acquisitions = 0

    def __enter__(self):
        self._inner.acquire()
        self.acquisitions += 1
        return self

    def __exit__(self, *_exc):
        self._inner.release()
        return False


def test_finish_with_transcript_retains_atomically_in_one_critical_section():
    """The retention race (Qodo finding on plan PR #1773), closed BY
    CONSTRUCTION: `finish(..., transcript=...)` performs the terminal
    transition AND the retention inside ONE critical section, so no
    observer -- a `send_to_agent` continuation racing the child's teardown
    -- can ever see a retainable child terminal-but-unretained. A
    finish-then-retain two-step (two lock acquisitions) is exactly the
    mutant this pin kills."""
    c = _coord()
    counting = _CountingLock()
    c._lock = counting
    h = c.reserve(task="child", agent=None)
    before = counting.acquisitions
    c.finish(h.handle_id, RUN_DONE, result="r", transcript=list(_TRANSCRIPT))
    assert counting.acquisitions == before + 1, (
        "finish-with-transcript took more than one critical section: the "
        "terminal status and the retention are separately observable"
    )
    assert c.get_retained(h.handle_id) is not None


def test_finish_with_transcript_respects_first_writer_wins():
    """A user cancel that wins the race must veto retention: the child's
    own straggling finish-with-transcript on an already-cancelled handle
    is wholly ignored -- this is WHY retention cannot simply run before
    finish (retainability depends on the first-writer-wins status)."""
    c = _coord()
    h = c.reserve(task="child", agent=None)
    c.finish(h.handle_id, RUN_CANCELLED, error="user cancelled")
    c.finish(h.handle_id, RUN_DONE, result="r", transcript=list(_TRANSCRIPT))
    assert c.get(h.handle_id).status == RUN_CANCELLED
    assert c.get_retained(h.handle_id) is None


def test_finish_without_transcript_retains_nothing_and_leaves_the_mailbox():
    """Every pre-existing finish caller (abandonment, thread-start
    failure, plain finishes) passes no transcript: nothing is retained
    and the undrained remnant keeps Task 1's survive-until-prune window."""
    c = _coord()
    h = c.reserve(task="child", agent=None)
    c.post_steering(h.handle_id, STEERING_SOURCE_USER, "undelivered")
    c.finish(h.handle_id, RUN_DONE, result="r")
    assert c.get_retained(h.handle_id) is None
    assert c.get(h.handle_id).queued_steering == 1


def test_retained_messages_are_copies_not_aliases():
    """The coordinator's copy discipline: neither the caller's later
    mutation of the passed list nor a reader's mutation of a returned
    entry can corrupt the stored transcript."""
    c = _coord()
    h = _finished_handle(c)
    passed = [{"role": "user", "content": "original"}]
    assert c.retain_transcript(h.handle_id, passed)
    passed[0]["content"] = "mutated by the caller"
    first_read = c.get_retained(h.handle_id)
    assert first_read.messages[0]["content"] == "original"
    first_read.messages[0]["content"] = "mutated by a reader"
    assert c.get_retained(h.handle_id).messages[0]["content"] == "original"


# =========================================================================
# 3. Continuation (agent_service): the send_to_agent terminal branch
# =========================================================================

#: Enough spawn budget for a spawn + a resume in one turn.
RESUME_CFG = AgentConfig(
    model="test-model",
    system_prompt="You are helpful.",
    allowed_tools=("calculator", "get_current_datetime", SPAWN_TOOL_NAME),
    budget=RunBudget(max_steps=60, max_model_turns=60, max_subagents=4),
)

#: Exactly ONE spawn slot per turn -- the resume-budget refusal fixture.
ONE_SLOT_CFG = AgentConfig(
    model="test-model",
    system_prompt="You are helpful.",
    allowed_tools=("calculator", SPAWN_TOOL_NAME),
    budget=RunBudget(max_steps=60, max_model_turns=60, max_subagents=1),
)

#: TWO spawn slots -- the live-cap unwind fixture needs spawn B + spawn C
#: to both fit AFTER the refused resume unwound its own slot.
TWO_SLOT_CFG = AgentConfig(
    model="test-model",
    system_prompt="You are helpful.",
    allowed_tools=("calculator", SPAWN_TOOL_NAME),
    budget=RunBudget(max_steps=60, max_model_turns=60, max_subagents=2),
)


def _run(service, config=RESUME_CFG, conversation_id="c"):
    return service.run_turn(
        conversation_id=conversation_id,
        messages=[{"role": "user", "content": "go"}],
        config=config,
        api_endpoint="llama_cpp",
    )


def _subagent_rows(db, conversation_id="c"):
    return [
        row
        for row in db.list_runs(conversation_id, include_superseded=True)
        if row["agent_kind"] == AGENT_KIND_SUBAGENT
    ]


def _finished_child(coordinator):
    """The single finished child's handle after turn 1's wait collected it."""
    return next(
        h
        for h in coordinator.snapshot()
        if h.status in TERMINAL_RUN_STATUSES
    )


def _await_retained(coordinator, handle_id):
    _wait_until(
        lambda: coordinator.get_retained(handle_id) is not None,
        "the finished child's transcript was never retained",
    )


def test_send_to_agent_to_a_finished_child_starts_a_resumed_seeded_run(db):
    """The whole seam, across two turns: turn 1 spawns and collects a
    child; turn 2 steers its handle id -- a NEW run launches, seeded with
    the retained transcript + the supervisor-labeled message, linked via
    ``resumed_from_run_id``, parented to the CURRENT primary."""
    holder: dict = {}

    def resume():
        return fence(
            SEND_TO_AGENT_TOOL_NAME,
            {"id": holder["handle_id"], "message": "now double-check it"},
        )

    service, chat, coordinator = make_fleet_service(
        db,
        [
            # -- turn 1
            fence(SPAWN_TOOL_NAME, {"task": "study the logs"}),
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "turn one answer",
            # -- turn 2
            resume,
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "turn two answer",
        ],
        {
            "study the logs": [
                fence("calculator", {"expression": "6*7"}),
                "the answer is 42",
                # -- the RESUMED run re-enters under the same task text
                # (its seed's first user message IS the original task).
                "rechecked: still 42",
            ]
        },
    )
    run1, outcome1 = _run(service)
    assert outcome1.status == RUN_DONE
    finished = _finished_child(coordinator)
    holder["handle_id"] = finished.handle_id
    old_run_id = finished.run_id
    assert old_run_id
    _await_retained(coordinator, finished.handle_id)

    run2, outcome2 = _run(service)
    assert outcome2.status == RUN_DONE

    # The resumed child's FIRST provider call: [system] + the exact
    # transcript the original child ended with + the labeled new message.
    original_calls = chat.child_calls["study the logs"]
    assert len(original_calls) == 3
    retained_history = original_calls[1]["messages_payload"][1:] + [
        {"role": "assistant", "content": "the answer is 42"}
    ]
    labeled = format_steering_message(
        STEERING_SOURCE_SUPERVISOR, "now double-check it"
    )
    resumed_payload = original_calls[2]["messages_payload"]
    assert resumed_payload[0]["role"] == "system"
    assert resumed_payload[1:] == retained_history + [
        {"role": "user", "content": labeled}
    ]

    # Lineage: a NEW row, resumed_from the OLD run, parented to the
    # CURRENT (turn 2) primary.
    rows = _subagent_rows(db)
    assert len(rows) == 2
    resumed_row = next(r for r in rows if r["id"] != old_run_id)
    assert resumed_row["resumed_from_run_id"] == old_run_id
    assert resumed_row["parent_run_id"] == run2
    steering_step = next(
        step
        for step in db.get_run(run2)["steps"]
        if step["kind"] == "tool_call"
        and step["tool_name"] == SEND_TO_AGENT_TOOL_NAME
    )
    assert resumed_row["spawn_event_id"] == (
        f"agent-step:{run2}:{steering_step['index']}"
    )
    resumed_lifecycle = [
        step
        for step in resumed_row["steps"]
        if step["kind"].startswith("agent_run_")
    ]
    assert [step["kind"] for step in resumed_lifecycle] == [
        "agent_run_reserved",
        "agent_run_created",
        "agent_run_resumed",
        "agent_run_started",
        "agent_run_completed",
    ]
    resumed_event = next(
        step
        for step in resumed_lifecycle
        if step["kind"] == "agent_run_resumed"
    )
    assert resumed_event["source_event_id"] == f"agent-run:{old_run_id}"
    old_row = next(r for r in rows if r["id"] == old_run_id)
    assert old_row["resumed_from_run_id"] is None
    assert old_row["parent_run_id"] == run1
    _wait_until(
        lambda: db.get_run_fresh(resumed_row["id"])["status"] == RUN_DONE,
        "the resumed child's row never went terminal",
    )

    # The ok copy: honest about the mechanism (a NEW run, seeded).
    sends = _tool_results(db.get_run(run2), SEND_TO_AGENT_TOOL_NAME)
    assert sends and "ERROR" not in sends[0]
    assert "resumed" in sends[0] and "new run" in sends[0].lower()
    assert holder["handle_id"] not in sends[0]
    assert f"run:{old_run_id}" in sends[0]
    new_handle = next(
        h for h in coordinator.snapshot() if h.run_id == resumed_row["id"]
    )
    assert new_handle.handle_id not in sends[0]
    assert f"run:{resumed_row['id']}" in sends[0]


def test_a_resumed_run_re_resolves_the_definition_to_its_current_form(db):
    """Ruling #1: a still-existing definition re-resolves to its CURRENT
    form; the new row's fresh ``definition_fingerprint`` records the
    change (that audit column exists for exactly this)."""
    definition_id = db.create_agent_definition(
        AgentDefinition(
            name="helper",
            description="a helper",
            instructions="Original instructions.",
        )
    )
    holder: dict = {}

    def resume():
        return fence(
            SEND_TO_AGENT_TOOL_NAME,
            {"id": holder["handle_id"], "message": "keep going"},
        )

    service, chat, coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "helper task", "agent": "helper"}),
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "turn one answer",
            resume,
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "turn two answer",
        ],
        {"helper task": ["helper done", "resumed helper done"]},
    )
    run1, outcome1 = _run(service)
    assert outcome1.status == RUN_DONE
    finished = _finished_child(coordinator)
    holder["handle_id"] = finished.handle_id
    _await_retained(coordinator, finished.handle_id)

    updated = AgentDefinition(
        name="helper",
        description="a helper",
        instructions="Updated instructions.",
    )
    db.update_agent_definition(definition_id, updated)

    run2, outcome2 = _run(service)
    assert outcome2.status == RUN_DONE

    resumed_system = chat.child_calls["helper task"][1]["messages_payload"][0][
        "content"
    ]
    assert "Updated instructions." in resumed_system
    assert "Original instructions." not in resumed_system

    rows = _subagent_rows(db)
    assert len(rows) == 2
    old_row = next(r for r in rows if r["resumed_from_run_id"] is None)
    new_row = next(r for r in rows if r["resumed_from_run_id"] is not None)
    assert new_row["agent_definition"] == "helper"
    assert new_row["definition_fingerprint"] == definition_fingerprint(updated)
    assert new_row["definition_fingerprint"] != old_row["definition_fingerprint"]


def test_a_deleted_definition_refuses_the_resume_and_suggests_a_fresh_spawn(db):
    """Ruling #1's other half: a deleted/disabled definition refuses
    clearly -- silent downgrade to a generic child would be the only
    WRONG option."""
    definition_id = db.create_agent_definition(
        AgentDefinition(
            name="helper", description="a helper", instructions="Help."
        )
    )
    holder: dict = {}

    def resume():
        return fence(
            SEND_TO_AGENT_TOOL_NAME,
            {"id": holder["handle_id"], "message": "keep going"},
        )

    service, _chat, coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "helper task", "agent": "helper"}),
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "turn one answer",
            resume,
            "turn two answer",
        ],
        {"helper task": ["helper done"]},
    )
    run1, outcome1 = _run(service)
    assert outcome1.status == RUN_DONE
    finished = _finished_child(coordinator)
    holder["handle_id"] = finished.handle_id
    _await_retained(coordinator, finished.handle_id)

    db.soft_delete_agent_definition(definition_id)

    run2, outcome2 = _run(service)
    assert outcome2.status == RUN_DONE
    sends = _tool_results(db.get_run(run2), SEND_TO_AGENT_TOOL_NAME)
    assert sends and "ERROR" in sends[0]
    assert "helper" in sends[0]
    assert "no longer exists" in sends[0]
    assert "fresh sub-agent" in sends[0]
    # No new child was created; the refusal cost nothing.
    assert len(_subagent_rows(db)) == 1


def test_undelivered_queued_steering_rides_the_seed_with_original_labels(db):
    """An entry posted after the child's LAST drain boundary never reached
    it -- the retained entry claims it, and the seed replays it with its
    ORIGINAL source label, before the new supervisor message."""
    in_final_call = threading.Event()
    posted = threading.Event()
    holder: dict = {}

    def gated_final():
        in_final_call.set()
        assert posted.wait(_JOIN_TIMEOUT), "the steering post never happened"
        return "done under fire"

    def post_then_wait():
        assert in_final_call.wait(_JOIN_TIMEOUT), (
            "the child never reached its final model call"
        )
        coordinator = holder["coordinator"]
        handle = next(
            h for h in coordinator.snapshot() if h.status == "running"
        )
        holder["handle_id"] = handle.handle_id
        assert coordinator.post_steering(
            handle.handle_id, STEERING_SOURCE_USER, "late user note"
        )
        posted.set()
        return fence(WAIT_AGENTS_TOOL_NAME, {})

    def resume():
        return fence(
            SEND_TO_AGENT_TOOL_NAME,
            {"id": holder["handle_id"], "message": "and also recheck"},
        )

    service, chat, coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "steered task"}),
            post_then_wait,
            "turn one answer",
            resume,
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "turn two answer",
        ],
        {"steered task": [gated_final, "resumed done"]},
    )
    holder["coordinator"] = coordinator
    run1, outcome1 = _run(service)
    assert outcome1.status == RUN_DONE
    _await_retained(coordinator, holder["handle_id"])
    entry = coordinator.get_retained(holder["handle_id"])
    assert list(entry.steering) == [(STEERING_SOURCE_USER, "late user note")]

    run2, outcome2 = _run(service)
    assert outcome2.status == RUN_DONE
    resumed_payload = chat.child_calls["steered task"][1]["messages_payload"]
    user_labeled = format_steering_message(STEERING_SOURCE_USER, "late user note")
    supervisor_labeled = format_steering_message(
        STEERING_SOURCE_SUPERVISOR, "and also recheck"
    )
    # Original label, original position: queued remnant FIRST, the new
    # supervisor message LAST.
    assert resumed_payload[-2] == {"role": "user", "content": user_labeled}
    assert resumed_payload[-1] == {"role": "user", "content": supervisor_labeled}


def test_a_resume_consumes_a_spawn_slot_and_refuses_at_the_budget(db):
    """A resume starts a NEW run, so it costs a spawn slot -- with the one
    slot already spent this turn, the resume is refused in spawn's own
    budget-refusal shape and no resumed row exists."""
    holder: dict = {}

    def resume():
        return fence(
            SEND_TO_AGENT_TOOL_NAME,
            {"id": holder["handle_id"], "message": "keep going"},
        )

    service, _chat, coordinator = make_fleet_service(
        db,
        [
            # -- turn 1: spend nothing unusual; child A finishes.
            fence(SPAWN_TOOL_NAME, {"task": "task a"}),
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "turn one answer",
            # -- turn 2: spawn B (the only slot), then try to resume A.
            fence(SPAWN_TOOL_NAME, {"task": "task b"}),
            resume,
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "turn two answer",
        ],
        {"task a": ["a done"], "task b": ["b done"]},
    )
    run1, outcome1 = _run(service, config=ONE_SLOT_CFG)
    assert outcome1.status == RUN_DONE
    finished = _finished_child(coordinator)
    holder["handle_id"] = finished.handle_id
    _await_retained(coordinator, finished.handle_id)

    run2, outcome2 = _run(service, config=ONE_SLOT_CFG)
    assert outcome2.status == RUN_DONE
    sends = _tool_results(db.get_run(run2), SEND_TO_AGENT_TOOL_NAME)
    assert sends and "ERROR" in sends[0]
    assert "sub-agent budget exhausted" in sends[0]
    rows = _subagent_rows(db)
    assert len(rows) == 2  # A and B only -- no resumed third row
    assert all(row["resumed_from_run_id"] is None for row in rows)


def test_a_live_cap_refusal_unwinds_the_resumes_spawn_slot(db):
    """At the live cap the resume is refused with spawn's own retryable
    copy -- and, like spawn's cap refusal, it must NOT consume a slot: a
    later spawn in the same turn still fits the budget."""
    b_entered = threading.Event()
    release_b = threading.Event()
    holder: dict = {}

    def gated_b():
        b_entered.set()
        assert release_b.wait(_JOIN_TIMEOUT)
        return "b done"

    def resume_at_cap():
        assert b_entered.wait(_JOIN_TIMEOUT), "child B never started"
        return fence(
            SEND_TO_AGENT_TOOL_NAME,
            {"id": holder["handle_id"], "message": "keep going"},
        )

    def release_then_wait():
        release_b.set()
        return fence(WAIT_AGENTS_TOOL_NAME, {})

    service, _chat, coordinator = make_fleet_service(
        db,
        [
            # -- turn 1
            fence(SPAWN_TOOL_NAME, {"task": "task a"}),
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "turn one answer",
            # -- turn 2: B fills the single live slot; the resume must be
            # refused (cap) WITHOUT costing the turn's second spawn slot,
            # which C then uses.
            fence(SPAWN_TOOL_NAME, {"task": "task b"}),
            resume_at_cap,
            release_then_wait,
            fence(SPAWN_TOOL_NAME, {"task": "task c"}),
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "turn two answer",
        ],
        {
            "task a": ["a done"],
            "task b": [gated_b],
            "task c": ["c done"],
        },
        max_live=1,
    )
    run1, outcome1 = _run(service, config=TWO_SLOT_CFG)
    assert outcome1.status == RUN_DONE
    finished = _finished_child(coordinator)
    holder["handle_id"] = finished.handle_id
    _await_retained(coordinator, finished.handle_id)

    run2, outcome2 = _run(service, config=TWO_SLOT_CFG)
    assert outcome2.status == RUN_DONE
    sends = _tool_results(db.get_run(run2), SEND_TO_AGENT_TOOL_NAME)
    assert sends and "ERROR" in sends[0]
    assert "live sub-agent limit reached" in sends[0]
    rows = _subagent_rows(db)
    # A, B and C all exist; the refused resume created nothing.
    assert sorted(row["task"] for row in rows) == ["task a", "task b", "task c"]
    assert all(row["resumed_from_run_id"] is None for row in rows)


def test_a_finished_child_remains_continuable_after_prune_terminal(db):
    """THE prune-window red, at the service seam: `prune_terminal` (the
    bridge runs it at every turn start) drops the terminal handle, so the
    id must resolve against the RETENTION store -- by the RUN id here, the
    vocabulary a wake notice hands the supervisor."""
    holder: dict = {}

    def resume_by_run_id():
        return fence(
            SEND_TO_AGENT_TOOL_NAME,
            {"id": holder["run_id"], "message": "one more pass"},
        )

    service, chat, coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "pruned task"}),
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "turn one answer",
            resume_by_run_id,
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "turn two answer",
        ],
        {"pruned task": ["first answer", "second answer"]},
    )
    run1, outcome1 = _run(service)
    assert outcome1.status == RUN_DONE
    finished = _finished_child(coordinator)
    holder["run_id"] = finished.run_id
    _await_retained(coordinator, finished.handle_id)

    # The bridge's turn-start prune: terminal handles gone, retention not.
    assert coordinator.prune_terminal() >= 1
    assert coordinator.get(finished.handle_id) is None

    run2, outcome2 = _run(service)
    assert outcome2.status == RUN_DONE
    rows = _subagent_rows(db)
    assert len(rows) == 2
    resumed_row = next(
        r for r in rows if r["resumed_from_run_id"] is not None
    )
    assert resumed_row["resumed_from_run_id"] == holder["run_id"]
    # The seed really carried the transcript (second call under the task).
    resumed_payload = chat.child_calls["pruned task"][1]["messages_payload"]
    assert {"role": "assistant", "content": "first answer"} in resumed_payload


def test_a_cancelled_child_draws_the_honest_not_retained_refusal_not_unknown(db):
    """A REAL finished child with no retained transcript (here: cancelled
    -- the cancel won the finish race, so its own finish-with-transcript
    was ignored) must draw the honest not-retained refusal, NEVER the
    unknown-id copy (the Qodo race pin's refusal half)."""
    child_started = threading.Event()
    release_child = threading.Event()
    holder: dict = {}

    def gated_child():
        child_started.set()
        assert release_child.wait(_JOIN_TIMEOUT)
        return fence("calculator", {"expression": "1+1"})

    def cancel_then_steer():
        assert child_started.wait(_JOIN_TIMEOUT)
        coordinator = holder["coordinator"]
        handle = next(
            h for h in coordinator.snapshot() if h.status == "running"
        )
        holder["handle_id"] = handle.handle_id
        # The cancel path's coordinator-side effect: the handle goes
        # terminal CANCELLED first; the child's own later finish (with
        # transcript) loses first-writer-wins and retains nothing.
        coordinator.finish(handle.handle_id, RUN_CANCELLED, error="cancelled")
        return fence(
            SEND_TO_AGENT_TOOL_NAME,
            {"id": handle.handle_id, "message": "please continue"},
        )

    def release_then_answer():
        release_child.set()
        return "turn answer"

    service, _chat, coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "doomed task"}),
            cancel_then_steer,
            release_then_answer,
        ],
        {"doomed task": [gated_child, "never delivered"]},
        allow_unconsumed=True,
    )
    holder["coordinator"] = coordinator
    run1, outcome = _run(service)
    assert outcome.status == RUN_DONE
    sends = _tool_results(db.get_run(run1), SEND_TO_AGENT_TOOL_NAME)
    assert sends and "ERROR" in sends[0]
    cancelled = coordinator.get(holder["handle_id"])
    assert cancelled.run_id
    assert holder["handle_id"] not in sends[0]
    assert f"run:{cancelled.run_id}" in sends[0]
    assert "no retained transcript" in sends[0]
    assert "fresh sub-agent" in sends[0]
    # NEVER the unknown-id copy for a child that was real.
    assert "no sub-agent matches" not in sends[0]
    assert coordinator.get_retained(holder["handle_id"]) is None


def test_after_a_restart_the_error_says_the_transcript_is_gone(db):
    """The spec's honest limit: retention is in-memory. A fresh
    coordinator (a restart) cannot resume -- the error says the transcript
    is gone and suggests a fresh spawn, NOT the unknown-id refusal."""
    service, _chat, coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "old task"}),
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "turn one answer",
        ],
        {"old task": ["old answer"]},
    )
    run1, outcome1 = _run(service)
    assert outcome1.status == RUN_DONE
    old_run_id = _finished_child(coordinator).run_id

    def resume_after_restart():
        return fence(
            SEND_TO_AGENT_TOOL_NAME,
            {"id": old_run_id, "message": "are you still there"},
        )

    # A restart: fresh service, fresh (empty) coordinator, SAME database.
    restarted, _chat2, _fresh = make_fleet_service(
        db,
        [resume_after_restart, "turn two answer"],
        {},
    )
    run2, outcome2 = _run(restarted)
    assert outcome2.status == RUN_DONE
    sends = _tool_results(db.get_run(run2), SEND_TO_AGENT_TOOL_NAME)
    assert sends and "ERROR" in sends[0]
    assert "restart" in sends[0]
    assert "transcript" in sends[0]
    assert "fresh sub-agent" in sends[0]
    # And nothing was launched.
    assert len(_subagent_rows(db)) == 1


def test_final_messages_never_reach_the_db(db):
    """`_persist` is untouched: the run row carries steps + result only --
    the transcript exists on the OUTCOME (this is what retention reads)
    and nowhere in the database."""
    service, _chat, coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "quick task"}),
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "turn answer",
        ],
        {"quick task": ["quick answer"]},
    )
    run1, outcome = _run(service)
    assert outcome.status == RUN_DONE
    # Not vacuous: the outcome really carries a transcript...
    assert outcome.final_messages is not None
    finished = _finished_child(coordinator)
    _await_retained(coordinator, finished.handle_id)
    child_row = db.get_run(finished.run_id)
    # ...and the DB rows carry no such column and no embedded transcript.
    assert "final_messages" not in child_row
    assert "final_messages" not in db.get_run(run1)
    assert child_row["result"] == "quick answer"


# =========================================================================
# 4. Sec 8's 3b cost-ticker audit (PR 3b Task 6), executed not assumed
# =========================================================================


def test_a_resumed_childs_spend_reaches_the_fleet_rollup_at_finish(db):
    """Sec 8's 3b audit, positive half: a RESUMED child's `total_tokens`
    reaches the fleet rollup through the same `fleet.finish` call as any
    child (`run_child`'s finally, shared via `_launch_fleet_child`), onto
    a NEW handle the rollup source sums.

    The rollup source is `ConsoleAgentController._console_agent_fleet_
    token_total`: `sum(handle.total_tokens for handle in fleet_snapshot(
    conversation_id))`, and `bridge.fleet_snapshot` returns this
    coordinator's `snapshot()` copies -- asserted here at the coordinator
    seam with the exact same summation.

    The audit's negative half -- a finished survivor's spend LEAVES that
    sum at the next turn's `prune_terminal`, and a continued task's
    aggregate (old + resumed run) is derivable from no surface (the DB
    joins `resumed_from_run_id` lineage but persists no tokens) -- is
    characterized at the tail, filed as TASK-18311, deliberately not
    patched here (the plan's own scope pin).
    """
    holder: dict = {}

    def resume():
        return fence(
            SEND_TO_AGENT_TOOL_NAME,
            {"id": holder["handle_id"], "message": "now double-check it"},
        )

    service, _chat, coordinator = make_fleet_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "study the logs"}),
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "turn one answer",
            resume,
            fence(WAIT_AGENTS_TOOL_NAME, {}),
            "turn two answer",
        ],
        {
            "study the logs": [
                "a real first answer with enough text for a token count",
                "a rechecked second answer with enough text for a token count",
            ]
        },
    )
    run1, outcome1 = _run(service)
    assert outcome1.status == RUN_DONE
    finished = _finished_child(coordinator)
    holder["handle_id"] = finished.handle_id
    old_run_id = finished.run_id
    _await_retained(coordinator, finished.handle_id)
    old_spend = coordinator.get(finished.handle_id).total_tokens
    assert old_spend > 0

    run2, outcome2 = _run(service)
    assert outcome2.status == RUN_DONE

    rows = _subagent_rows(db)
    resumed_row = next(r for r in rows if r["resumed_from_run_id"] is not None)
    assert resumed_row["resumed_from_run_id"] == old_run_id
    resumed_handle = next(
        h for h in coordinator.snapshot() if h.run_id == resumed_row["id"]
    )
    # The audit's claim: the resumed run's measured spend was recorded onto
    # its handle at finish -- the SAME seam PR2b Task 5 pinned for ordinary
    # children (`test_finished_children_record_their_measured_token_spend_
    # on_the_handle`), inherited by the resume with zero new wiring.
    assert resumed_handle.status == RUN_DONE
    assert resumed_handle.total_tokens > 0
    # ...and the rollup summation (the exact `_console_agent_fleet_token_
    # total` expression over the snapshot copies) includes BOTH figures
    # while the handles live.
    rollup = sum(h.total_tokens for h in coordinator.snapshot())
    assert rollup == old_spend + resumed_handle.total_tokens

    # -- The honest gap, characterized (TASK-18311; do not "fix" this
    # assertion without that task): the next turn's prune drops both
    # terminal handles, so the rollup reads 0 -- the finished survivor's
    # spend has left `fleet_snapshot`, and NO surface can reconstruct the
    # continued task's old+new aggregate (the DB has the lineage join but
    # no token column).
    assert coordinator.prune_terminal() >= 2
    assert sum(h.total_tokens for h in coordinator.snapshot()) == 0
