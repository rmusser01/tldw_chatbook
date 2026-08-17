"""Pure state machine for concurrent sub-agent handles.

No DB, Textual, app, or I/O imports — stdlib only (threading, dataclasses, uuid,
typing) plus agent_models constants. The impure thread-launching lives in
agent_service. Thread-safe: every public method holds a Lock.
"""

from __future__ import annotations

import dataclasses
import threading
import uuid
from typing import Callable

from tldw_chatbook.Agents.agent_models import TERMINAL_RUN_STATUSES

FLEET_STARTED = "fleet_started"
FLEET_FINISHED = "fleet_finished"


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


class FleetCoordinator:
    """State machine managing concurrent sub-agent task handles.

    Enforces live-task cap, emits events on state changes, and ensures
    idempotent finish (first-writer-wins: late finishes to already-terminal
    handles are ignored).
    """

    def __init__(self, max_live: int, clock: Callable[[], float]) -> None:
        """Initialize the coordinator.

        Args:
            max_live: Maximum number of concurrently live handles.
            clock: Callable returning a float timestamp (for testing).
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
        self._steering: dict[str, list[tuple[str, str]]] = {}

    def reserve(self, task: str, agent: str | None) -> FleetHandle | None:
        """Reserve a slot for a new task, returning a handle or None if at cap.

        Emits FLEET_STARTED event on success.

        Args:
            task: Human-readable task description.
            agent: Agent name, if applicable.

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
            self._steering.setdefault(handle_id, []).append((source, text))
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
            return self._steering.pop(handle_id, [])

    def finish(
        self,
        handle_id: str,
        status: str,
        result: str = "",
        error: str = "",
        total_tokens: int = 0,
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
