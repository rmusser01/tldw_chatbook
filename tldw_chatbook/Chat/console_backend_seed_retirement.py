"""Observe the original saved-view seed independently of its optional result."""

from __future__ import annotations

import asyncio
import inspect
import os
from types import FunctionType, MethodType

from tldw_chatbook.DB import base_db
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

from .console_domain_services import _definition, _definition_current
from .console_execution_retirement import _ReadResources
from .console_fleet_wake import ConsoleFleetWakeCoordinator
from .console_preparation_reads import preparation_read_for, run_preparation_read


class ConsoleBackendSeedRead:
    """Exact issued seed, its original native callback and independent receipt."""

    def __init__(self, lifecycle, seed) -> None:
        from .console_backend_view_lifecycle import ConsoleBackendViewLifecycle

        if type(lifecycle) is not ConsoleBackendViewLifecycle:
            raise TypeError("Original backend view lifecycle is required")
        owners = lifecycle._require_checkpoint(lifecycle._current)
        if (
            lifecycle._current.action == "hidden"
            or lifecycle._seeds.get(lifecycle._attachment) is not seed
            or seed.task is not None
            or seed.receipt is not None
            or any(a is not b for a, b in zip(seed.owners, owners, strict=True))
        ):
            raise RuntimeError("Original backend view seed is required")
        self.lifecycle, self.seed = lifecycle, seed
        self.owners = owners
        self.wake = owners[2]
        self.database = self.wake._runs_db()
        if type(self.database) is not AgentRunsDB or self.database.is_memory_db:
            raise RuntimeError("Saved backend seed requires its original file database")
        self.loop, self.pid = asyncio.get_running_loop(), os.getpid()
        self.task = None
        self.reads = set()
        self.observation = None
        self.handoff = None
        self.producer = None
        self.issued = False
        self.settled = False
        self.native_retired = False
        self._registration = self, lifecycle, seed, self.database

    def require_sources(self) -> None:
        """Recheck the captured graph on its loop or original finite worker."""
        lifecycle, seed, database = self.lifecycle, self.seed, self.database
        binding = _BINDING.get()
        if (
            type(self) is not ConsoleBackendSeedRead
            or self._registration != (self, lifecycle, seed, database)
            or self._registration[0] is not self
            or seed.receipt is not self
            or self.pid != os.getpid()
            or self.task is None
            or seed.task is not self.task
            or binding is None
            or binding[0] is not self.task
            or binding[1] is not self
            or not _definition_current(_CURRENT)
            or not _CURRENT[0]()
        ):
            raise RuntimeError("Backend seed receipt owner changed")
        controller, store, wake, bridge = self.owners
        backend = lifecycle.presentation.backend
        backend.require_sources()
        backend.domain.require_current()
        if (
            lifecycle.presentation.view_lifecycle is not lifecycle
            or backend.controller is not controller
            or backend.store is not store
            or controller.store is not store
            or controller._fleet_wake is not wake
            or controller._agent_bridge is not bridge
            or backend.runtime._agent_bridge is not bridge
            or wake._controller is not controller
            or wake._disposed
            or wake._runs_db() is not database
            or any(
                inspect.getattr_static(wake, name, None) is not record[0]
                or not _definition_current(record)
                for name, record in _WAKE
            )
        ):
            raise RuntimeError("Backend seed native source changed")

    def confirmed(self) -> bool:
        """Return only this original receipt's sticky positive native outcome."""
        return bool(
            type(self) is ConsoleBackendSeedRead
            and self._registration[0] is self
            and self.seed.receipt is self
            and self.seed.task is self.task
            and self.task is not None
            and self.task.done()
            and self.pid == os.getpid()
            and _definition_current(_CURRENT)
            and _CURRENT[0]()
            and self.issued
            and self.settled
            and self.native_retired
            and self.handoff is not None
            and type(self.observation) is _OBSERVATION
            and self.observation.read is self.handoff
            and self.handoff.creator is self
            and self.handoff.task is self.task
            and self.producer is not None
            and self.handoff._producer is self.producer
            and self.producer.done()
            and not self.producer.cancelled()
            and self.observation.confirmed()
        )


async def observe_seed_owned_db_call(owner, database, operation, invoke, args, kwargs):
    """Observe only the exact original seed callback inside run_owned_db_call."""
    if type(owner) is not ConsoleBackendSeedRead:
        raise TypeError("Original backend seed read owner is required")
    owner.require_sources()
    if (
        asyncio.get_running_loop() is not owner.loop
        or asyncio.current_task() is not owner.task
        or owner.issued
        or database is not owner.database
        or type(operation) is not MethodType
        or operation.__self__ is not owner.wake
        or operation.__func__ is not dict(_WAKE)["seed_from_marks"][0]
        or args != ()
        or set(kwargs) != {"database"}
        or kwargs["database"] is not database
        or type(invoke) is not FunctionType
        or invoke.__code__ is not _INVOKE
        or invoke.__globals__ is not vars(base_db)
        or inspect.getclosurevars(invoke).nonlocals
        != {
            "database": database,
            "operation": operation,
            "args": args,
            "kwargs": kwargs,
        }
    ):
        raise RuntimeError("Backend seed original invocation changed")
    owner.issued = True

    def read():
        handoff = preparation_read_for(owner)
        if (
            handoff is None
            or handoff.creator is not owner
            or handoff.task is not owner.task
            or handoff.source is not owner
            or owner.handoff is not None
        ):
            raise RuntimeError("Backend seed physical handoff changed")
        owner.handoff = handoff
        owner.producer = handoff._producer
        observation = owner.observation = _ReadResources(None, handoff)
        with observation.observe():
            return invoke()

    try:
        return await run_preparation_read(
            read,
            creator=owner,
            session_id=None,
            reads=owner.reads,
            observers=(
                owner.lifecycle.presentation.backend.runtime._preparation_reads,
            ),
            require_current=owner.require_sources,
            source=owner,
        )
    finally:
        # Preserve the original optional read error; it is independent of native
        # settlement. A later registry cleanup cannot change this sticky result.
        owner.native_retired = bool(
            owner.observation is not None and owner.observation.confirmed()
        )
        owner.settled = True


def start_backend_view_seed(lifecycle, seed) -> asyncio.Task[int]:
    """Retain the real seed Task before its unchanged producer can enter."""
    owner = ConsoleBackendSeedRead(lifecycle, seed)
    seed.receipt = owner

    async def original_seed():
        # Keep original task ownership exact even with an eager task factory.
        await asyncio.sleep(0)
        token = _BINDING.set((owner.task, owner))
        try:
            owner.require_sources()
            return await _SEED[0](owner.wake)
        finally:
            _BINDING.reset(token)

    task = asyncio.create_task(original_seed(), name="console_backend_view_seed")
    seed.task = owner.task = task
    lifecycle._tasks.append((task, owner))
    return task


_WAKE = tuple(
    (name, _definition(getattr(ConsoleFleetWakeCoordinator, name)))
    for name in ("_seed_owned_history", "seed_from_marks", "retry_soon", "_runs_db")
)
_SEED = dict(_WAKE)["_seed_owned_history"]
_DB_CALL = _definition(base_db.run_owned_db_call)
_BINDING = base_db._CONSOLE_SEED_READ
_INVOKE = next(
    code
    for code in base_db.run_owned_db_call.__code__.co_consts
    if inspect.iscode(code) and code.co_name == "invoke"
)
_READ = _definition(run_preparation_read)
_HANDOFF = _definition(preparation_read_for)
_OBSERVATION = _ReadResources
_OBSERVATION_METHODS = tuple(
    (name, _definition(getattr(_ReadResources, name)))
    for name in ("__init__", "observe", "require_creator", "confirmed")
)
_METHODS = tuple(
    (name, _definition(value))
    for name, value in vars(ConsoleBackendSeedRead).items()
    if type(value) is FunctionType
)


def _definitions_current():
    return (
        base_db.run_owned_db_call is _DB_CALL[0]
        and base_db._CONSOLE_SEED_READ is _BINDING
        and _definition_current(_DB_CALL)
        and run_preparation_read is _READ[0]
        and _definition_current(_READ)
        and preparation_read_for is _HANDOFF[0]
        and _definition_current(_HANDOFF)
        and _ReadResources is _OBSERVATION
        and all(
            inspect.getattr_static(_ReadResources, name, None) is record[0]
            and _definition_current(record)
            for name, record in _OBSERVATION_METHODS
        )
        and all(
            inspect.getattr_static(ConsoleBackendSeedRead, name, None) is record[0]
            and _definition_current(record)
            for name, record in _METHODS
        )
        and all(
            globals().get(name) is record[0] and _definition_current(record)
            for name, record in _HELPERS
        )
    )


_CURRENT = _definition(_definitions_current)
_HELPERS = tuple(
    (name, _definition(globals()[name]))
    for name in (
        "observe_seed_owned_db_call",
        "start_backend_view_seed",
        "_definitions_current",
    )
)
