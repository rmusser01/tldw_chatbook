"""Exact received-turn observations, separate from acceptance and presentation.

These records never authorize work. Original control sources, native readers,
permission owners and ordinary trees continue to own admission and execution.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import threading
from contextlib import contextmanager
from typing import Any


_native = threading.local()


def _creator():
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    return os.getpid(), threading.current_thread(), storage._task_identity()


class _ReadResources:
    def __init__(self, invocation, read):
        self.invocation = invocation
        self.read = read
        self.effect = None
        self.creator = None
        self.entered = False
        self.closed = False
        self.invalid = False
        self.observations = {
            name: [] for name in ("acquisition", "lease", "raw", "descriptor", "file")
        }
        self.connections = []

    def require_creator(self):
        if self.creator != _creator():
            self.invalid = True
            raise RuntimeError("Console read observation creator changed")

    @contextmanager
    def observe(self):
        import time
        from loguru import logger

        # Do not replace either the permission recorder or the core connection
        # receiver. Both original owners may be active inside this observation.
        if getattr(_native, "read", None) is not None:
            self.invalid = True
            raise RuntimeError("Nested Console read observation")
        self.creator = _creator()
        self.entered = True
        _native.read = self
        started, cpu_started = time.monotonic(), time.thread_time()
        try:
            yield
        finally:
            self.require_creator()
            self.closed = True
            _native.read = None
            logger.warning(
                "Console received native read settled; elapsed={} thread_cpu={}",
                time.monotonic() - started,
                time.thread_time() - cpu_started,
            )

    def confirmed(self):
        from tldw_chatbook.Backup_Recovery import raw_participants as raw
        from tldw_chatbook.Backup_Recovery import storage_admission as storage
        from tldw_chatbook.Backup_Recovery.participants import retired_core_connections
        from tldw_chatbook.MCP.local_permission_operation import (
            _hold_retired_or_shared,
            _state_resources_retired,
        )

        if (
            not _definitions_current()
            or self.invalid
            or (self.entered and not self.closed)
        ):
            return False
        if not self.read.retired.done() or self.read.retired.cancelled():
            return False
        producer = self.read._producer
        if producer is not None and not producer.done():
            return False
        with storage._lock:
            valid = all(
                item not in storage._pending_acquisitions
                for item in self.observations["acquisition"]
            )
            valid &= all(
                handle.closed and native.closed
                for handle, native in self.observations["file"]
            )
            for operation, state in self.observations["raw"]:
                valid &= (
                    operation not in raw._states
                    and operation not in storage._raw_operations
                    and _state_resources_retired(state, storage)
                )
            for state, descriptor in self.observations["descriptor"]:
                valid &= descriptor not in state.descriptors and not state.uncertain
            for lease, hold in self.observations["lease"]:
                valid &= lease not in storage._live_leases and lease._key is None
                valid &= _hold_retired_or_shared(hold, storage)
            for database, connection, participant, lease in self.connections:
                try:
                    sqlite3.Connection.in_transaction.__get__(connection)
                except sqlite3.ProgrammingError:
                    closed = True
                except BaseException:
                    closed = False
                else:
                    closed = False
                valid &= (
                    closed
                    and database._maintenance_participant is participant
                    and connection in retired_core_connections
                    and connection not in participant.connections
                    and lease not in storage._live_leases
                    and lease._key is None
                )
        return bool(valid)


def observe_enrollment(kind: str, value: object) -> None:
    """Retain original enrollment only on the exact active read's creator."""
    observation = getattr(_native, "read", None)
    if observation is None:
        return
    observation.require_creator()
    if kind not in observation.observations:
        observation.invalid = True
        raise RuntimeError("Unknown Console native enrollment")
    if kind == "lease":
        from tldw_chatbook.Backup_Recovery import storage_admission as storage

        value = value, storage._holds.get(value._key)
    observation.observations[kind].append(value)


def observe_core_connection(database: object, connection: object) -> None:
    """Record the actual installed core connection, including cached borrowing."""
    observation = getattr(_native, "read", None)
    if observation is None or connection is None:
        return
    observation.require_creator()
    participant = getattr(database, "_maintenance_participant", None)
    if participant is None:
        observation.invalid = True
        return
    lease = participant.connections.get(connection)
    if lease is None:
        observation.invalid = True
        return
    if not any(item[1] is connection for item in observation.connections):
        observation.connections.append((database, connection, participant, lease))


def _independent_display_read(read: object) -> bool:
    """Recognize only the already-admitted original display native phase."""
    from types import CodeType

    from tldw_chatbook.Backup_Recovery.runtime_producer_lifetime import (
        ConsolePreparationRead,
    )
    from .console_domain_display import (
        ConsoleDomainDisplayOwner,
        ConsoleDisplayFull,
        _require_definitions,
    )
    from .console_display_phases import (
        ConsoleDisplayPhysicalOwner,
        require_display_phase_definitions,
    )
    from .console_native_commit import _ConsoleNativeBinding

    if (
        type(read) is not ConsolePreparationRead
        or type(read.creator) is not ConsoleDomainDisplayOwner
    ):
        return False
    full = read.creator._full
    if type(full) is not ConsoleDisplayFull:
        return False
    _require_definitions(full)
    require_display_phase_definitions(full)
    full.require_current()
    full.require_phase_task()
    original = next(
        item
        for item in ConsoleDisplayPhysicalOwner.native_phase.__code__.co_consts
        if type(item) is CodeType and item.co_name == "invoke"
    )
    callback = read.callback
    if type(callback) is not FunctionType or callback.__code__ is not original:
        return False
    captured = dict(
        zip(
            original.co_freevars,
            (cell.cell_contents for cell in callback.__closure__),
            strict=True,
        )
    )
    phase, binding = captured["phase"], captured["binding"]
    if (
        read.task is not asyncio.current_task()
        or read._producer is not None
        or captured["self"] is not full
        or not full.phase_pending
        or not full.retirements
        or full.retirements[-1] is not phase
        or phase.joined
        or type(binding) is not _ConsoleNativeBinding
        or binding.turn_owner is not full
        or binding.retirement is not phase.retirement
        or binding.store is not full.store
        or binding.persistence is not full.store.persistence
        or binding.database is not full.database
    ):
        raise RuntimeError("Independent Console display read association changed")
    return True


def register_preparation_read(read: object) -> _ReadResources | None:
    """Associate a live handoff with its already-bound original received source."""
    from .console_control_dispatch import _CommandSource, _command_source

    source = _command_source.get()
    if type(source) is not _CommandSource:
        return None
    if source.turn_retirement is None:
        from .console_backend_presentation import ConsoleBackendPresentation
        from .console_backend_run_history import ConsoleBackendRunHistoryRead

        if type(read.creator) is ConsoleBackendRunHistoryRead:
            read.creator.require_current()
            return read.creator.register(read, source)

        presentation = source.controls._domain.presentation
        if (
            type(presentation) is ConsoleBackendPresentation
            and read.creator is presentation._marker_projection
        ):
            presentation.require_association(presentation.association)
            return presentation.register_marker_read(source, read)
        return None
    if _independent_display_read(read):
        # Its actual private display effect owns retirement. An inherited old
        # turn context neither authorizes this phase nor owns its resources.
        return None
    invocation = source.turn_retirement
    invocation.require_source(source)
    observation = _ReadResources(invocation, read)
    if read.task is not invocation.task:
        # Inherited context does not certify a delegated Task's ownership.
        # Keep the uncertainty visible instead of calling it no-entry.
        invocation.unqualified_read = True
    observation.effect = invocation.controls._begin_effect(
        "received_preparation_read", observation, source
    )
    invocation.reads.append(observation)
    return observation


def finish_preparation_read(
    observation: _ReadResources | None, original_error: BaseException | None = None
) -> None:
    """Settle the registered read effect from exact physical facts only."""
    if observation is None:
        return
    confirmed = observation.confirmed()
    observation.invocation.controls._finish_effect(
        observation.effect, retired=confirmed
    )
    if not confirmed and original_error is None:
        raise RuntimeError("Console preparation native retirement is unproven")


def retain_ordinary_tree(controller: object, tree: object) -> None:
    """Keep an actual admitted tree before the original submission removes it."""
    from .console_control_dispatch import _CommandSource, _command_source

    source = _command_source.get()
    if type(source) is not _CommandSource or source.turn_retirement is None:
        return
    invocation = source.turn_retirement
    invocation.require_source(source)
    if (
        controller is not invocation.controls._controller
        or tree.controller is not controller
        or tree.runtime is not invocation.controls._runtime
        or asyncio.current_task() is not invocation.task
        or tree not in tree.runtime._ordinary_turns
    ):
        raise RuntimeError("Console ordinary retirement owner changed")
    if not any(item is tree for item in invocation.trees):
        invocation.trees.append(tree)


class ConsoleTurnRetirement:
    """Retain one original turn until logical and physical owners both settle."""

    def __init__(self, controls: Any, identity: Any, turn_id: str) -> None:
        if not _definitions_current():
            raise RuntimeError("Console turn retirement definitions changed")
        self.controls = controls
        self.identity = identity
        self.turn_id = turn_id
        self.record = None
        self.task = None
        self.source = None
        self.source_closed = False
        self.logical_closed = False
        self.accepted = False
        self.unqualified_read = False
        self.reads = []
        self.trees = []
        self.retired = controls._loop.create_future()
        self.settlement = None
        self.counts = None
        self.plan_keys = []

    def require_source(self, source: object) -> None:
        entry = self.controls._turns.get(self.turn_id)
        if (
            self.source is not source
            or source.turn_retirement is not self
            or source.controls is not self.controls
            or entry is None
            or entry[0] is not self.identity
            or entry[1] is not self.record
            or entry[2] is not self.task
            or self.source_closed
        ):
            raise RuntimeError("Console turn retirement source changed")

    def terminal(self, accepted: bool) -> None:
        """Observe original custody release; acceptance is not retirement."""
        if self.controls._runtime._turn_custody.get(self.turn_id) is self.record:
            raise RuntimeError("Console logical custody has not been released")
        self.logical_closed = True
        self.accepted = accepted

    def bind(self, record: object) -> None:
        """Bind the actual original accepted task after its lazy registration."""
        self.record, self.task = record, record.task
        self.task.add_done_callback(self._finished)

    def _finished(self, task: asyncio.Task) -> None:
        if task is not self.task or self.settlement is not None:
            return
        self.settlement = self.controls._loop.create_task(self._settle())
        self.controls._turn_retirement_tasks.add(self.settlement)
        self.settlement.add_done_callback(self.controls._turn_retirement_tasks.discard)

    async def _settle(self) -> None:
        confirmed = False
        last_check = None
        last_effect = None
        error_type = None

        def checked(name, result):
            nonlocal last_check
            last_check = (name, result)
            return result

        def checked_effect(index, effect):
            nonlocal last_effect
            done = effect.completed.done()
            physical = effect.physically_retired if done else None
            outcome = effect.completed.result() if done and physical else None
            last_effect = (done, physical, outcome)
            return checked(f"effect_{index}_{effect.kind}", done and physical and outcome is True)

        try:
            for tree in self.trees:
                await asyncio.shield(asyncio.wrap_future(tree.retired))
            source = self.source
            if source is not None:
                if not _definitions_current():
                    raise RuntimeError("Console turn retirement definitions changed")
                await _control_trace.drain_source_trace_workers(source)
            confirmed = (
                checked("definitions", _definitions_current())
                and checked("task", self.task.done())
                and checked("logical_closed", self.logical_closed)
                and checked("qualified_reads", not self.unqualified_read)
                and checked("source_closed", self.source_closed or source is None)
                and all(
                    checked(f"read_{index}", read.confirmed())
                    for index, read in enumerate(self.reads)
                )
                and all(
                    checked_effect(index, effect)
                    for index, effect in enumerate(
                        () if source is None else source.effects
                    )
                )
            )
            if confirmed:
                from dataclasses import replace

                from .console_control_dispatch import _PendingPlan

                controls = self.controls
                with controls._lock:
                    entry = controls._turns.get(self.turn_id)
                    if (
                        controls._turn_retirements.get(self.turn_id) is not self
                        or entry is None
                        or entry[0] is not self.identity
                        or entry[1] is not self.record
                        or entry[2] is not self.task
                    ):
                        raise RuntimeError("Console retired control owner changed")
                    replacements = []
                    for ordinal, key in enumerate(self.plan_keys, 1):
                        pending = controls._plans.get(key)
                        if (
                            key != (self.turn_id, ordinal)
                            or type(pending) is not _PendingPlan
                            or pending.task is not self.task
                            or pending.projection.turn_id != self.turn_id
                            or pending.projection.ordinal != ordinal
                            or pending.projection.received_request_id
                            != self.identity.request_id
                            or pending.state not in {"finished", "abandoned", "conflict"}
                        ):
                            raise RuntimeError("Console retired plan owner changed")
                        replacements.append(
                            (pending, replace(pending.projection, recovery=None))
                        )
                    # Only the original positive physical proof reaches here.
                    # Validate the whole turn before releasing any heavy owner.
                    for pending, projection in replacements:
                        pending.projection = projection
                        pending.plan = pending.task = pending.acknowledged = None
                    controls._turns[self.turn_id] = (self.identity, None, None, None)
                    controls._turn_retirements.pop(self.turn_id)
                    self.plan_keys.clear()
        except BaseException as error:
            error_type = type(error).__name__
            confirmed = False
        finally:
            if not confirmed:
                # Temporary body-free diagnostic; preserve short-circuit checks.
                from loguru import logger

                logger.warning(
                    "Console turn retirement unconfirmed; last_check={} effect_facts={} error_type={} reads={} trees={}",
                    last_check, last_effect, error_type, len(self.reads), len(self.trees),
                )
            if not self.retired.done():
                self.retired.set_result(bool(confirmed))
            if confirmed:
                self.counts = (len(self.reads), len(self.trees))
                self.reads.clear()
                self.trees.clear()
                if self.source is not None:
                    self.source.effects.clear()
                self.source = self.record = self.task = None


from types import FunctionType
from tldw_chatbook.Utils.definition_records import capture_definitions
from tldw_chatbook.Backup_Recovery import participants as _participants
from tldw_chatbook.MCP import permission_process as _permission_process
from . import console_control_trace as _control_trace
from .console_domain_services import _definition

_DEFINITIONS = capture_definitions(
    (
        *(
            (globals(), name, globals()[name])
            for name in (
                "_creator",
                "observe_enrollment",
                "observe_core_connection",
                "_independent_display_read",
                "register_preparation_read",
                "finish_preparation_read",
                "retain_ordinary_tree",
            )
        ),
        *(
            (owner, name, value)
            for owner in (
                _ReadResources,
                ConsoleTurnRetirement,
                _control_trace._TraceWriteOwner,
            )
            for name, value in vars(owner).items()
            if type(value) is FunctionType
        ),
        *(
            (_control_trace, name, getattr(_control_trace, name))
            for name in (
                "retain_trace_worker_future",
                "drain_source_trace_workers",
            )
        ),
        (
            _participants,
            "register_core_connection",
            _participants.register_core_connection,
        ),
        (_participants, "core_cached_connection", _participants.core_cached_connection),
        (_permission_process, "_enroll", _permission_process._enroll),
    )
)


def _definitions_current(_records=_DEFINITIONS) -> bool:
    from tldw_chatbook.Utils.definition_records import definitions_current

    return definitions_current(_records)


_DOMAIN_SOURCE_DEFINITIONS = tuple(
    (name, _definition(globals()[name]))
    for name in (
        "_creator",
        "_definitions_current",
        "observe_enrollment",
        "observe_core_connection",
        "_independent_display_read",
        "register_preparation_read",
        "finish_preparation_read",
        "retain_ordinary_tree",
    )
)
