"""Finite native ownership of existing Change Review jobs (ADR-285)."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock

from tldw_chatbook.Agents.activation import _guard, execution
from tldw_chatbook.Agents.execution_capacity import CapacityRefused, WorkOrigin
from tldw_chatbook.DB.base_db import operation_owned_connection
from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain
from tldw_chatbook.Workspaces.worktree_activity import WorktreeFootprint

_current_snapshot: ContextVar[_SnapshotSource | None] = ContextVar(
    "native_change_review_snapshot", default=None
)


def native_snapshot_source() -> _SnapshotSource | None:
    """Return only the original source active on this physical callback thread."""
    return _current_snapshot.get()


class _NativeSnapshotIssuer:
    """One runtime's issuer; execution stays in the existing coordinator."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.capacity = runtime.execution_capacity
        self.activity = runtime.worktree_activity
        self.coordinator = None

    def capture(
        self,
        bridge,
        *,
        session_id,
        conversation_id,
        origin,
        roots,
        owner=None,
        claim=None,
    ):
        if (
            self.runtime._agent_bridge is not bridge
            or self.runtime._change_review_coordinator is not self.coordinator
            or bridge._change_finalization_coordinator is not self.coordinator
            or self.runtime._disposed
            or not isinstance(origin, WorkOrigin)
            or (owner is None) != (claim is None)
        ):
            raise ValueError("native change-review source unavailable")
        session = next(
            (s for s in bridge._store.sessions() if s.id == session_id), None
        )
        if session is None:
            raise ValueError("native change-review session unavailable")
        chains = tuple(capture_directory_chain(Path(root)) for root in roots)
        if tuple(str(c.canonical_root) for c in chains) != tuple(map(str, roots)):
            raise ValueError("native change-review root changed")
        persistence = getattr(bridge._store, "persistence", None)
        registry = getattr(persistence, "workspace_registry", None)
        databases = tuple(
            db
            for db in (
                bridge._db,
                getattr(persistence, "db", None),
                getattr(registry, "db", None),
            )
            if db is not None
        )
        return _SnapshotSource(
            self,
            bridge,
            self.runtime._chat_controller,
            bridge._store,
            session,
            conversation_id,
            origin,
            chains,
            _guard.captured_sources(bridge),
            databases,
            owner,
            claim,
        )


@dataclass(frozen=True)
class _SnapshotSource:
    issuer: _NativeSnapshotIssuer
    bridge: object
    controller: object
    store: object
    session: object
    conversation_id: str
    origin: WorkOrigin
    chains: tuple
    sources: tuple
    databases: tuple
    owner: object = None
    claim: object = None

    def check_source(self):
        runtime = self.issuer.runtime
        if (
            runtime._agent_bridge is not self.bridge
            or runtime._change_review_coordinator is not self.issuer.coordinator
            or self.bridge._change_finalization_coordinator
            is not self.issuer.coordinator
            or self.bridge._store is not self.store
            or runtime._chat_store is not self.store
            or runtime._chat_controller is not self.controller
            or not any(session is self.session for session in self.store.sessions())
        ):
            raise ValueError("native change-review source changed")

    def check(self):
        self.check_source()
        for chain in self.chains:
            if capture_directory_chain(chain.canonical_root) != chain:
                raise ValueError("native change-review root changed")

    def validate_root(self, root):
        self.check()
        canonical = Path(root).resolve(strict=True)
        if not any(canonical.is_relative_to(c.canonical_root) for c in self.chains):
            raise ValueError("native change-review discovered root outside admission")
        return canonical

    @contextmanager
    def activate(self, owner):
        with ExitStack() as stack:
            stack.enter_context(owner.activate())
            stack.enter_context(execution(sources=self.sources))
            for db in dict.fromkeys(self.databases):
                stack.enter_context(operation_owned_connection(db))
            token = _current_snapshot.set(self)
            try:
                yield
            finally:
                _current_snapshot.reset(token)

    def admit(self):
        self.check_source()
        capacity, activity = self.issuer.capacity, self.issuer.activity
        if self.owner is not None:
            if (
                self.owner.capacity is not capacity
                or self.owner.origin is not self.origin
                or self.owner.conversation_id != self.conversation_id
            ):
                raise ValueError("native change-review owner changed")
            activity.validate_opaque_claim(self.claim, owner=self.owner)
            try:
                operation = self.owner.reserve_tool()
            except CapacityRefused as error:
                if str(error) in {
                    "tool_capacity",
                    "automatic_tool_capacity",
                    "previous_tool_still_running",
                }:
                    return None, "capacity"
                raise
            return _SnapshotTicket(self, self.owner, operation, None), ""
        with activity.prelaunch(footprint=WorktreeFootprint(opaque=True)) as claim:
            if claim is None:
                return None, "activity"
            owner = capacity.begin_execution(
                origin=self.origin, conversation_id=self.conversation_id
            )
            try:
                operation = owner.reserve_tool()
            except CapacityRefused as error:
                owner.finish_root()
                if str(error) in {"tool_capacity", "automatic_tool_capacity"}:
                    return None, "capacity"
                raise
            try:
                claim.bind_owner(owner)
            except Exception:
                operation.finish()
                owner.finish_root()
                raise
            return _SnapshotTicket(self, owner, operation, claim), ""


class _SnapshotTicket:
    """Original operation survives both actual callback and handoff return."""

    def __init__(self, source, owner, operation, claim):
        self.source, self.owner = source, owner
        self.publish = source.issuer.coordinator._publish
        self.operation, self.claim = operation, claim
        self._lock = Lock()
        self.retired = Event()
        self._worker_returned = self._handoff_returned = self._retired = False

    def returned(self, *, worker=False, handoff=False):
        with self._lock:
            self._worker_returned |= worker
            self._handoff_returned |= handoff
            if self._retired or not (self._worker_returned and self._handoff_returned):
                return
            self._retired = True
        if self.claim is not None:
            # Pure observation's conclusive handoff is not task/phase settlement.
            self.claim.settle(durable=True)
            self.owner.finish_root()
        self.operation.finish()
        self.retired.set()
