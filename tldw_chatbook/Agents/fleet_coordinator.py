"""Pure state machine for concurrent sub-agent handles.

No DB, Textual, app, or I/O imports — stdlib only (threading, dataclasses, uuid,
typing) plus agent_models constants. The impure thread-launching lives in
agent_service. Thread-safe: every public method holds a Lock.
"""

from __future__ import annotations

import dataclasses
import json
import threading
import uuid
from typing import Callable

from tldw_chatbook.Agents.agent_models import (
    RUN_DONE,
    RUN_ERROR,
    RUN_STUCK,
)

FLEET_STARTED = "fleet_started"
FLEET_FINISHED = "fleet_finished"

#: PR3b Task 4 (spec SS6, finished-agent continuation). Which terminal
#: statuses leave a transcript worth retaining: done/stuck/error are all
#: states a supervisor may want to continue from ("you stalled -- try
#: this"). Cancelled and superseded are deliberately absent: the user
#: killed it, or it was replaced -- resuming either would undo an explicit
#: human decision.
RETAINED_TRANSCRIPT_STATUSES = frozenset({RUN_DONE, RUN_STUCK, RUN_ERROR})

#: Default retention caps ([agents] retained_transcripts /
#: retained_transcript_max_chars -- read by the BRIDGE's coordinator
#: factory beside max_live; this pure module never reads config itself).
#: Count cap: oldest-evicted-first. Char cap: an oversize transcript is
#: NOT retained at all (coordinator ruling #2 -- truncation could split
#: native pairs and silently change the child's memory). A cap of 0 (or
#: below) retains nothing.
DEFAULT_RETAINED_TRANSCRIPTS = 5
DEFAULT_RETAINED_TRANSCRIPT_MAX_CHARS = 200_000


@dataclasses.dataclass(frozen=True)
class FleetEvent:
    """Event emitted by FleetCoordinator on handle state changes.

    Attributes:
        kind: Event type (FLEET_STARTED or FLEET_FINISHED).
        handle_id: The handle this event concerns.
        run_id: The run_id attached to the handle, if any.
        agent: The agent name attached to the handle, if any.
        status: The status of the handle at event time.
    """

    kind: str
    handle_id: str
    run_id: str | None
    agent: str | None
    status: str


@dataclasses.dataclass
class FleetHandle:
    """Track state and metadata for a concurrent sub-agent task.

    Attributes:
        handle_id: Unique identifier for this handle.
        run_id: The agent run ID, attached later via attach_run().
        agent: The agent name, if specified at reserve time.
        task: Human-readable task description.
        status: Current status (initially "running").
        result: Result string on completion (empty until finish).
        error: Error message on failure (empty until finish).
        started_at: Timestamp when handle was created.
        finished_at: Timestamp when handle reached terminal status.
        total_tokens: PR2b Task 5 (cost rollup). This child's measured
            cumulative prompt+completion token spend, from its own
            ``RunOutcome.total_tokens`` -- 0 until ``finish()`` records it
            (a running child's spend is not final, so it is never reported
            mid-flight rather than showing a partial/misleading number).
            This is the ONLY place per-child spend is threaded to the
            live rail today -- it is not persisted to the ``agent_runs``
            DB row, so a resumed/historical fleet row (no live coordinator
            in THIS process) shows no token figure; see
            ``Console_Modules/agent.py``'s row builders for the consumer.
    """

    handle_id: str
    run_id: str | None
    agent: str | None
    task: str
    status: str
    result: str = ""
    error: str = ""
    started_at: float = 0.0
    finished_at: float | None = None
    total_tokens: int = 0
    # PR3b Task 1 (steering). How many posted steering entries are queued
    # for this child, still awaiting its next drain boundary -- the
    # panel's honest "queued (N)" figure (spec SS6 latency honesty).
    # Stored state lives in the coordinator's mailbox dict, never here:
    # this field is COMPUTED onto the copies ``get()``/``snapshot()``
    # return, so a stale handle copy can never disagree with the mailbox.
    queued_steering: int = 0
    # TASK-28238 phase 2 T7 Qodo round, finding 7. The isolation mode this
    # child was spawned/resumed with (``None`` or ``"worktree"``) --
    # recorded here so `_retain_locked` can copy it onto the
    # ``RetainedTranscript`` at retention time, letting a LATER resume
    # re-admit a fresh worktree instead of silently sharing the tree.
    isolation: str | None = None


@dataclasses.dataclass(frozen=True)
class RetainedTranscript:
    """A finished child's coherent transcript, kept for continuation.

    PR3b Task 4 (spec SS6). Captured at finish time from the child's own
    ``RunOutcome.final_messages`` (the last drain-boundary prefix, so it
    can never end inside a split native batch) plus whatever steering the
    child never got to drain. IN-MEMORY ONLY -- cross-restart resurrection
    is explicitly out of scope (spec SS6), and after a restart the
    supervisor is told the transcript is gone.

    Attributes:
        handle_id: The finished child's fleet handle id.
        run_id: Its agent-run id, if one was ever attached (the id a
            completion notice speaks; ``resumed_from_run_id`` records it
            on the resumed row).
        agent: The agent-definition NAME the child was spawned from, or
            ``None`` for a plain spawn. Continuation re-resolves it
            against the CURRENT roster (coordinator ruling #1) -- the
            definition itself is deliberately not snapshotted here.
        task: The original task text (the resumed handle reuses it).
        status: The terminal status at retention time (done/stuck/error).
        messages: The coherent transcript (copies; see ``get_retained``).
        steering: The undelivered ``(source, text)`` mailbox remnant,
            claimed at retention time -- a resume seeds these (original
            labels) before the new supervisor message.
        retained_at: Coordinator-clock timestamp of the retention.
        isolation: The original child's isolation mode (``None`` or
            ``"worktree"``), copied off its ``FleetHandle`` at retention
            time (TASK-28238 phase 2 T7 Qodo round, finding 7) -- a resume
            reads this to re-admit a fresh worktree instead of silently
            sharing the tree. Defaults to ``None`` for backward
            compatibility with any coordination state that predates this
            field.
    """

    handle_id: str
    run_id: str | None
    agent: str | None
    task: str
    status: str
    messages: tuple[dict, ...]
    steering: tuple[tuple[str, str], ...]
    retained_at: float
    steering_with_causes: tuple[tuple[str, str, str | None], ...] = ()
    isolation: str | None = None


class FleetCoordinator:
    """State machine managing concurrent sub-agent task handles.

    Enforces live-task cap, emits events on state changes, and ensures
    idempotent finish (first-writer-wins: late finishes to already-terminal
    handles are ignored).
    """

    def __init__(
        self,
        max_live: int,
        clock: Callable[[], float],
        *,
        retained_transcripts: int = DEFAULT_RETAINED_TRANSCRIPTS,
        retained_transcript_max_chars: int = DEFAULT_RETAINED_TRANSCRIPT_MAX_CHARS,
    ) -> None:
        """Initialize the coordinator.

        Args:
            max_live: Maximum number of concurrently live handles.
            clock: Callable returning a float timestamp (for testing).
            retained_transcripts: How many finished children's transcripts
                to keep for continuation (oldest evicted first); 0 keeps
                none.
            retained_transcript_max_chars: Serialized-size ceiling above
                which a transcript is NOT retained (ruling #2: refuse,
                never truncate); 0 retains none.
        """
        self._max_live = max_live
        self._clock = clock
        self._lock = threading.Lock()  # No reentrant calls, lock to catch bugs
        self._handles: dict[str, FleetHandle] = {}
        self._live_ids: set[str] = set()  # handle_ids with status != terminal
        self._events: list[FleetEvent] = []
        # PR3b Task 1: per-child steering mailboxes, keyed by handle id.
        # Guarded by the same lock as every other public method. A key
        # exists only while entries are queued (drain pops the whole
        # list), and dies with its handle in prune_terminal.
        self._steering: dict[str, list[tuple[str, str, str | None]]] = {}
        # PR3b Task 4: retained transcripts of finished children, keyed by
        # handle id, insertion-ordered (Python dicts) so eviction is
        # oldest-first. DELIBERATELY SEPARATE from `_handles`:
        # `prune_terminal` drops terminal handles at every turn start, and
        # a finished child must stay continuable past that (Task 2's
        # concern (a)). Run-id resolution scans the values -- the store is
        # capped at `retained_transcripts`, so a scan is O(cap).
        self._retained_transcripts_cap = retained_transcripts
        self._retained_transcript_max_chars = retained_transcript_max_chars
        self._retained: dict[str, RetainedTranscript] = {}

    def reserve(
        self, task: str, agent: str | None, *, isolation: str | None = None
    ) -> FleetHandle | None:
        """Reserve a slot for a new task, returning a handle or None if at cap.

        Emits FLEET_STARTED event on success.

        Args:
            task: Human-readable task description.
            agent: Agent name, if applicable.
            isolation: The isolation mode this child launches under
                (``None`` or ``"worktree"``) -- recorded on the handle so
                a later retention/resume can see it (finding 7).

        Returns:
            A copy of the new FleetHandle if a slot is available, else
            None. A copy -- not the live, internally-stored object -- for
            the same reason `get()`/`snapshot()` return copies: it stops
            a caller from racing a concurrent `finish()`/`attach_run()`
            by reading a mutating object instead of a point-in-time
            snapshot. Every current consumer reads only the immutable
            `.handle_id` off the returned handle, or re-fetches a fresh
            copy via `get()` when it needs live state (e.g. `run_id`
            attached later) -- so this is safe.
        """
        with self._lock:
            if len(self._live_ids) >= self._max_live:
                return None

            handle_id = uuid.uuid4().hex
            started_at = self._clock()
            handle = FleetHandle(
                handle_id=handle_id,
                run_id=None,
                agent=agent,
                task=task,
                status="running",
                result="",
                error="",
                started_at=started_at,
                finished_at=None,
                isolation=isolation,
            )
            self._handles[handle_id] = handle
            self._live_ids.add(handle_id)
            self._events.append(
                FleetEvent(
                    kind=FLEET_STARTED,
                    handle_id=handle_id,
                    run_id=None,
                    agent=agent,
                    status="running",
                )
            )
            return dataclasses.replace(handle)

    def attach_run(self, handle_id: str, run_id: str) -> None:
        """Attach a run ID to an existing handle.

        Args:
            handle_id: The handle's ID.
            run_id: The run ID to attach.
        """
        with self._lock:
            if handle_id in self._handles and handle_id in self._live_ids:
                self._handles[handle_id].run_id = run_id

    def post_steering(self, handle_id: str, source: str, text: str) -> bool:
        """Queue one steering entry for a LIVE child (PR3b Task 1, spec SS6).

        Steering never cancels and never restarts (spec SS3 invariant 4):
        this only appends to the child's mailbox; the child consumes it at
        its own next protocol-coherent drain boundary
        (``agent_runtime.run_agent_loop``'s pre-model-call drain).

        Text validation (non-empty, ``MAX_STEERING_CHARS``) is the
        PRODUCERS' job at their own boundaries -- ``send_to_agent`` (Task
        2) and the panel input (Task 3) each need their own user-facing
        refusal copy, which a silent bool here could not carry. The label
        is likewise not this method's concern: the drain point renders it
        via ``format_steering_message``, so raw ``(source, text)`` pairs
        are what the mailbox holds.

        Args:
            handle_id: The target child's handle id.
            source: ``STEERING_SOURCE_SUPERVISOR`` or
                ``STEERING_SOURCE_USER``.
            text: The steering message body (raw, unlabeled).

        Returns:
            True when queued. False for an unknown handle or a TERMINAL
            one -- a finished child has no next model turn to deliver at,
            and the caller must say so instead of queueing into a void
            (Task 2's terminal branch upgrades that refusal into
            finished-agent continuation).
        """
        with self._lock:
            if handle_id not in self._handles or handle_id not in self._live_ids:
                return False
            self._steering.setdefault(handle_id, []).append((source, text, None))
            return True

    def post_steering_with_cause(
        self,
        handle_id: str,
        source: str,
        text: str,
        source_event_id: str,
    ) -> bool:
        """Queue steering with its durable causal event identity."""
        with self._lock:
            if handle_id not in self._handles or handle_id not in self._live_ids:
                return False
            self._steering.setdefault(handle_id, []).append(
                (source, text, source_event_id)
            )
            return True

    def drain_steering(self, handle_id: str) -> list[tuple[str, str]]:
        """Return-and-clear this child's queued steering, atomically.

        One locked pop: a concurrent ``post_steering`` either lands before
        the pop (and is returned here) or after it (and waits for the next
        drain) -- an entry is never lost or delivered twice.

        Args:
            handle_id: The child's handle id.

        Returns:
            The queued ``(source, text)`` entries in posting order; empty
            for an unknown handle or an empty mailbox.
        """
        with self._lock:
            return [
                (source, text)
                for source, text, _cause in self._steering.pop(handle_id, [])
            ]

    def drain_steering_with_causes(
        self, handle_id: str
    ) -> list[tuple[str, str, str | None]]:
        """Return-and-clear causal steering entries for the runtime seam."""
        with self._lock:
            return self._steering.pop(handle_id, [])

    def finish(
        self,
        handle_id: str,
        status: str,
        result: str = "",
        error: str = "",
        total_tokens: int = 0,
        transcript: "list[dict] | None" = None,
    ) -> None:
        """Mark a handle as finished with terminal status.

        Idempotent: if the handle is already terminal, this call is ignored
        (first-writer-wins). This protects abandoned children from overwriting
        a coordinator-issued cancellation after a timeout.

        Emits FLEET_FINISHED event only if the handle transitions to terminal.

        Args:
            handle_id: The handle's ID.
            status: Terminal status to record.
            result: Result string (ignored if already terminal).
            error: Error message (ignored if already terminal).
            total_tokens: PR2b Task 5 -- this child's measured cumulative
                token spend (``RunOutcome.total_tokens``), recorded onto the
                handle for the rail's cost rollup. Ignored if already
                terminal, like ``result``/``error``. Defaults to 0 for
                every pre-existing caller (abandonment, thread-start
                failure) that has no outcome to report a real figure from.
            transcript: PR3b Task 4 -- the child's coherent
                ``RunOutcome.final_messages``, retained ATOMICALLY with
                this terminal transition (same critical section) when the
                status is retainable and the transcript fits the caps.
                Atomic on purpose, not merely convenient (Qodo finding on
                the plan PR #1773): a separate post-``finish`` retention
                call leaves a window where the child answers as terminal
                but ``get_retained`` still misses -- a ``send_to_agent``
                continuation racing that window would refuse a child that
                is genuinely continuable microseconds later. Under one
                lock, any observer that sees the terminal status also sees
                the retention. ``None`` (every pre-existing caller --
                abandonment, thread-start failure, plain finishes) retains
                nothing and leaves the mailbox remnant untouched (Task 1's
                pinned survive-until-prune window). First-writer-wins
                covers retention too: a late finish-with-transcript on an
                already-cancelled handle is wholly ignored, so a
                user-cancelled child can never be retained by its own
                straggling teardown.
        """
        with self._lock:
            if handle_id not in self._handles:
                return

            # First-writer-wins: ignore if handle is no longer live.
            # This check is based on liveness, not status vocabulary, so it
            # protects against any status (e.g., "timeout") that a Task 6
            # abandonment handler might use, not just the closed set in
            # TERMINAL_RUN_STATUSES.
            if handle_id not in self._live_ids:
                return

            handle = self._handles[handle_id]

            # Transition to terminal
            finished_at = self._clock()
            handle.status = status
            handle.result = result
            handle.error = error
            handle.finished_at = finished_at
            handle.total_tokens = total_tokens
            self._live_ids.discard(handle_id)

            # Emit event with current run_id and agent
            self._events.append(
                FleetEvent(
                    kind=FLEET_FINISHED,
                    handle_id=handle_id,
                    run_id=handle.run_id,
                    agent=handle.agent,
                    status=status,
                )
            )

            # PR3b Task 4: retention, in the SAME critical section as the
            # transition above -- see the `transcript` arg docstring.
            if transcript is not None:
                self._retain_locked(handle_id, transcript)

    def get(self, handle_id: str) -> FleetHandle | None:
        """Retrieve a handle by ID.

        Returns a copy to prevent external mutation.

        Args:
            handle_id: The handle's ID.

        Returns:
            A copy of the handle, or None if not found.
        """
        with self._lock:
            if handle_id not in self._handles:
                return None
            return self._copy_with_queued(self._handles[handle_id])

    def snapshot(self) -> list[FleetHandle]:
        """Return a snapshot of all handles.

        Returns copies to prevent external mutation.

        Returns:
            A list of copies of all FleetHandle objects.
        """
        with self._lock:
            return [
                self._copy_with_queued(h) for h in self._handles.values()
            ]

    def durable_handle_map(self) -> dict[str, str]:
        """Snapshot process handles that still have a durable run identity.

        Returns:
            A new handle-id to run-id mapping. No transcript or task content
            is exposed.
        """
        with self._lock:
            mapping = {
                handle_id: entry.run_id
                for handle_id, entry in self._retained.items()
                if entry.run_id
            }
            mapping.update(
                {
                    handle_id: handle.run_id
                    for handle_id, handle in self._handles.items()
                    if handle.run_id
                }
            )
            return mapping

    def _copy_with_queued(self, handle: FleetHandle) -> FleetHandle:
        """A point-in-time copy carrying the CURRENT mailbox depth.

        Caller must hold ``self._lock``. ``queued_steering`` is computed
        from the mailbox here rather than stored on the live handle, so
        the copies ``get()``/``snapshot()`` return can never disagree with
        what ``drain_steering`` would actually deliver.
        """
        return dataclasses.replace(
            handle,
            queued_steering=len(self._steering.get(handle.handle_id, ())),
        )

    @property
    def max_live(self) -> int:
        """The configured live-handle cap.

        Exposed (read-only) so an owner holding this coordinator across
        turns -- ``ConsoleAgentBridge``, since PR3a-1 Task 6a -- can tell
        whether ``[agents] max_live_subagents`` still matches the cap this
        instance was built with, without rebuilding it blindly and
        orphaning the live children it is currently accounting for.
        """
        return self._max_live

    def set_max_live(self, max_live: int) -> None:
        """Re-size the live cap in place (PR3a-1 Task 6a).

        A cross-turn owner re-reads ``[agents] max_live_subagents`` every
        turn, and a user can change it mid-conversation. Re-sizing beats
        replacing the coordinator: a replacement would drop every live
        handle from the only surface that can see or stop it -- a silent
        loss of exactly the survivors this PR exists to keep visible --
        whereas re-sizing keeps them and applies the new cap to the NEXT
        ``reserve()``. Lowering the cap below the current live count never
        cancels anything: it simply refuses new reservations until enough
        children finish, which is the same back-pressure a full fleet
        already applies.

        Args:
            max_live: The new cap.
        """
        with self._lock:
            self._max_live = max_live

    @property
    def retained_transcripts(self) -> int:
        """The configured retention COUNT cap (read-only, like max_live)."""
        return self._retained_transcripts_cap

    @property
    def retained_transcript_max_chars(self) -> int:
        """The configured retention SIZE cap (read-only, like max_live)."""
        return self._retained_transcript_max_chars

    def set_retention_caps(
        self, retained_transcripts: int, retained_transcript_max_chars: int
    ) -> None:
        """Re-size the retention caps in place (the ``set_max_live`` shape).

        The cross-turn owner (the bridge's coordinator factory) re-reads
        ``[agents] retained_transcripts`` / ``retained_transcript_max_chars``
        every turn; re-sizing beats replacing the coordinator for the same
        reason ``set_max_live`` does -- a replacement would drop the
        retained transcripts along with every live handle. Lowering the
        count cap evicts oldest-first immediately; the char cap governs
        only FUTURE retentions (already-retained transcripts were measured
        against the cap in force when they were captured).

        Args:
            retained_transcripts: The new count cap; 0 keeps nothing new
                and evicts everything currently held.
            retained_transcript_max_chars: The new size ceiling.
        """
        with self._lock:
            self._retained_transcripts_cap = retained_transcripts
            self._retained_transcript_max_chars = retained_transcript_max_chars
            self._evict_over_cap_locked()

    def _evict_over_cap_locked(self) -> None:
        """Drop oldest retained entries until within the count cap.

        Caller must hold ``self._lock``. Insertion order IS age order
        (retain appends; nothing reorders), so the dict's first key is
        always the oldest entry.
        """
        cap = max(self._retained_transcripts_cap, 0)
        while len(self._retained) > cap:
            oldest = next(iter(self._retained))
            del self._retained[oldest]

    def retain_transcript(
        self, handle_id: str, messages: "list[dict] | None"
    ) -> bool:
        """Retain a finished child's coherent transcript for continuation.

        The standalone seam over ``_retain_locked``. Production retention
        rides ``finish(..., transcript=...)`` instead -- ATOMIC with the
        terminal transition, closing the terminal-but-unretained window a
        separate post-finish call would open (see ``finish``'s
        ``transcript`` docstring). This method exists for callers that
        already hold a terminal handle (and for tests of the retention
        rules themselves); it runs the exact same checks and claim.

        The mailbox claim is Task 1's pinned window
        (``test_undrained_entries_survive_finish_until_prune_terminal``):
        the remnant still exists between ``finish()`` and
        ``prune_terminal()``, and retention CLAIMS it into the entry, so
        the fleet row's "queued (N)" honestly reads 0 for a finished
        child and a resume can replay the entries the child never saw.

        Args:
            handle_id: The finished child's handle id.
            messages: The child's ``RunOutcome.final_messages`` -- the
                coherent transcript -- or ``None`` when the run died
                without producing one (a raise before the loop returned).

        Returns:
            True when retained. False -- and nothing stored, mailbox
            untouched -- for: an unknown or still-live handle; a terminal
            status outside ``RETAINED_TRANSCRIPT_STATUSES`` (cancelled/
            superseded: the user killed it or it was replaced); a missing
            transcript; an oversize one (ruling #2: refuse, never
            truncate); or caps of 0.
        """
        with self._lock:
            return self._retain_locked(handle_id, messages)

    def _retain_locked(
        self, handle_id: str, messages: "list[dict] | None"
    ) -> bool:
        """The retention rules + claim. Caller must hold ``self._lock``."""
        handle = self._handles.get(handle_id)
        if handle is None or handle_id in self._live_ids:
            return False
        if handle.status not in RETAINED_TRANSCRIPT_STATUSES:
            return False
        if messages is None:
            return False
        if self._retained_transcripts_cap <= 0:
            return False
        if self._retained_transcript_max_chars <= 0:
            return False
        try:
            size = len(json.dumps(messages, default=str))
        except Exception:  # noqa: BLE001 — an unmeasurable transcript
            return False  # cannot be size-bounded, so it is not kept
        if size > self._retained_transcript_max_chars:
            return False
        steering_with_causes = tuple(self._steering.pop(handle_id, []))
        steering = tuple(
            (source, text) for source, text, _cause in steering_with_causes
        )
        self._retained[handle_id] = RetainedTranscript(
            handle_id=handle_id,
            run_id=handle.run_id,
            agent=handle.agent,
            task=handle.task,
            status=handle.status,
            messages=tuple(dict(m) for m in messages),
            steering=steering,
            retained_at=self._clock(),
            steering_with_causes=steering_with_causes,
            isolation=handle.isolation,
        )
        self._evict_over_cap_locked()
        return True

    def get_retained(self, target_id: str) -> RetainedTranscript | None:
        """Resolve a retained transcript by handle id, then by run id.

        Same vocabulary order as live resolution (Task 2's pin): the
        handle id is the primary vocabulary, so a pathological collision
        (one child's run id equals another's handle id) lands on the
        handle-id owner.

        Args:
            target_id: A retained child's handle id, or its run id.

        Returns:
            A copy of the entry (fresh message dicts, so a reader's
            mutation can never corrupt the store), or ``None``.
        """
        with self._lock:
            entry = self._retained.get(target_id)
            if entry is None:
                entry = next(
                    (
                        candidate
                        for candidate in self._retained.values()
                        if candidate.run_id == target_id
                    ),
                    None,
                )
            if entry is None:
                return None
            return dataclasses.replace(
                entry, messages=tuple(dict(m) for m in entry.messages)
            )

    def prune_terminal(self) -> int:
        """Forget every already-terminal handle. Returns how many went.

        PR3a-1 Task 6a. Until this PR a coordinator lived for exactly one
        turn, so "never forget a handle" cost nothing and bought
        ``_pending_handles``' "a vanished handle counts as finished"
        safety. A per-CONVERSATION coordinator lives for the whole
        process, so without pruning ``_handles`` would grow without bound
        across a long conversation and ``snapshot()`` would hand the fleet
        panel every child the conversation has ever run.

        Call it only BETWEEN turns, from the owner, never mid-turn: a
        handle this turn still holds an id for (``my_handle_ids``,
        ``_fleet_cancels``) must stay resolvable for the whole turn --
        ``_settle_fleet``/``wait_agents``/``check_agents`` all resolve
        ids through ``get()``. Live handles are never pruned, so a
        survivor of an earlier turn is untouched by construction.

        Returns:
            The number of handles dropped.
        """
        with self._lock:
            terminal = [
                handle_id
                for handle_id in self._handles
                if handle_id not in self._live_ids
            ]
            for handle_id in terminal:
                del self._handles[handle_id]
                # PR3b Task 1: the mailbox dies with its handle. An
                # undelivered remnant is claimed BEFORE this point by Task
                # 4's retention (retain_transcript runs at finish time,
                # from run_child's finally); by prune time it is garbage.
                self._steering.pop(handle_id, None)
            return len(terminal)

    def live_count(self) -> int:
        """Return the number of live (non-terminal) handles.

        Returns:
            Count of handles not in terminal status.
        """
        with self._lock:
            return len(self._live_ids)

    def drain_events(self) -> list[FleetEvent]:
        """Return and clear all pending events.

        Returns:
            A list of FleetEvent objects, clearing the internal queue.
        """
        with self._lock:
            events = self._events
            self._events = []
            return events

    def all_finished(self) -> bool:
        """Check if all handles are in terminal status.

        Returns:
            True if no live (non-terminal) handles exist.
        """
        with self._lock:
            return len(self._live_ids) == 0
