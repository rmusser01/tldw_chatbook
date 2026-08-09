"""Pure state machine for concurrent sub-agent handles.

No DB, Textual, app, or I/O imports — stdlib only (threading, dataclasses, uuid,
typing) plus agent_models constants. The impure thread-launching lives in
agent_service. Thread-safe: every public method holds an RLock.
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
        self._lock = threading.RLock()
        self._handles: dict[str, FleetHandle] = {}
        self._live_ids: set[str] = set()  # handle_ids with status != terminal
        self._events: list[FleetEvent] = []

    def reserve(self, task: str, agent: str | None) -> FleetHandle | None:
        """Reserve a slot for a new task, returning a handle or None if at cap.

        Emits FLEET_STARTED event on success.

        Args:
            task: Human-readable task description.
            agent: Agent name, if applicable.

        Returns:
            A new FleetHandle if a slot is available, else None.
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
            return handle

    def attach_run(self, handle_id: str, run_id: str) -> None:
        """Attach a run ID to an existing handle.

        Args:
            handle_id: The handle's ID.
            run_id: The run ID to attach.
        """
        with self._lock:
            if handle_id in self._handles:
                self._handles[handle_id].run_id = run_id

    def finish(
        self, handle_id: str, status: str, result: str = "", error: str = ""
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
        """
        with self._lock:
            if handle_id not in self._handles:
                return

            handle = self._handles[handle_id]

            # First-writer-wins: ignore if already terminal
            if handle.status in TERMINAL_RUN_STATUSES:
                return

            # Transition to terminal
            finished_at = self._clock()
            handle.status = status
            handle.result = result
            handle.error = error
            handle.finished_at = finished_at
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
            return dataclasses.replace(self._handles[handle_id])

    def snapshot(self) -> list[FleetHandle]:
        """Return a snapshot of all handles.

        Returns copies to prevent external mutation.

        Returns:
            A list of copies of all FleetHandle objects.
        """
        with self._lock:
            return [
                dataclasses.replace(h) for h in self._handles.values()
            ]

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
