"""Finite native ownership of existing Change Review jobs (ADR-285)."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock

from tldw_chatbook.Agents.activation import execution
from tldw_chatbook.Agents.execution_capacity import CapacityRefused, WorkOrigin
from tldw_chatbook.DB.base_db import operation_owned_connection
from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain
from tldw_chatbook.Workspaces.change_tracking import (
    NativeUndoIssuer as NativeUndoIssuer,
    _NativeInitializationIssuer as _NativeInitializationIssuer,
    _NativeSnapshotIssuer as _NativeSnapshotIssuer,
)
from tldw_chatbook.Workspaces.worktree_activity import WorktreeFootprint

_current_snapshot: ContextVar[_SnapshotSource | _InitializationSource | None] = (
    ContextVar("native_change_review_snapshot", default=None)
)

_current_ticket: ContextVar[_SnapshotTicket | None] = ContextVar(
    "native_change_review_ticket", default=None
)


def native_snapshot_ticket() -> _SnapshotTicket | None:
    """Return the exact admitted ticket on the original physical callback."""
    return _current_ticket.get()


def native_snapshot_source() -> _SnapshotSource | _InitializationSource | None:
    """Return only the original source active on this physical callback thread."""
    return _current_snapshot.get()


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
    ordinary_tree: object = None
    workflow_wave: object = None
    _holds: dict = field(default_factory=dict, compare=False, repr=False)
    _holds_lock: object = field(default_factory=Lock, compare=False, repr=False)
    _closed: object = field(default_factory=Event, compare=False, repr=False)

    def prepare_ordinary(self):
        """Reserve the actual discover/B/E bundle before capture publication."""
        tickets = {}
        try:
            for kind in ("discover", "baseline", "finalize"):
                tickets[kind] = _SnapshotTicket(
                    self,
                    self.owner,
                    self.owner.reserve_tool(),
                    None,
                    publish=self.issuer.coordinator._publish,
                )
        except BaseException:
            for ticket in tickets.values():
                ticket.returned(worker=True, handoff=True)
            raise
        self._holds.update(tickets)

    def prepare_survivor(self):
        """A real child makes one originating survivor window possible."""
        if self.ordinary_tree is None:
            return
        with self._holds_lock:
            if self._closed.is_set() or "survivor_finalize" in self._holds:
                return
        ticket = _SnapshotTicket(
            self,
            self.owner,
            self.owner.reserve_tool(),
            None,
            publish=self.issuer.coordinator._publish,
        )
        with self._holds_lock:
            if not self._closed.is_set() and "survivor_finalize" not in self._holds:
                self._holds["survivor_finalize"] = ticket
                return
        ticket.returned(worker=True, handoff=True)

    def restore_unstarted(self, kind, ticket):
        if self.ordinary_tree is None:
            ticket.returned(worker=True, handoff=True)
            return
        key = "finalize" if kind == "failure" else kind
        with self._holds_lock:
            if not self._closed.is_set():
                self._holds[key] = ticket
                return
        ticket.returned(worker=True, handoff=True)

    def close_unused(self):
        with self._holds_lock:
            self._closed.set()
            tickets = tuple(self._holds.values())
            self._holds.clear()
        for ticket in tickets:
            ticket.returned(worker=True, handoff=True)

    def check_source(self):
        if self.workflow_wave is not None:
            self.workflow_wave.check_snapshot_source(self)
        if self.ordinary_tree is not None:
            self.ordinary_tree.check()
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
            if self.ordinary_tree is not None and self.ordinary_tree.source is not None:
                stack.enter_context(self.ordinary_tree.source.activate())
            stack.enter_context(owner.activate())
            stack.enter_context(execution(sources=self.sources))
            for db in dict.fromkeys(self.databases):
                stack.enter_context(operation_owned_connection(db))
            token = _current_snapshot.set(self)
            try:
                yield
            finally:
                _current_snapshot.reset(token)

    def admit(self, kind=None):
        self.check_source()
        if self.ordinary_tree is not None:
            key = "finalize" if kind == "failure" else kind
            with self._holds_lock:
                ticket = self._holds.pop(key, None)
            if ticket is None:
                raise ValueError("original preheld snapshot unavailable")
            return ticket, ""
        # Cold control-plane settings retain their original admission on this
        # existing publisher callback, never during inert bridge construction.
        with execution(sources=self.sources):
            return self._admit_owned()

    def _admit_owned(self):
        capacity, activity = self.issuer.capacity, self.issuer.activity
        self.issuer.coordinator._subscribe_native(capacity, activity)
        if self.owner is not None:
            if (
                self.owner.capacity is not capacity
                or self.owner.origin is not self.origin
                or self.owner.conversation_id
                != (
                    self.ordinary_tree.launch_key
                    if self.ordinary_tree is not None
                    else self.conversation_id
                )
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
            return _SnapshotTicket(
                self,
                self.owner,
                operation,
                None,
                publish=self.issuer.coordinator._publish,
            ), ""
        with activity._snapshot_prelaunch(self) as claim:
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
            return _SnapshotTicket(
                self, owner, operation, claim, publish=self.issuer.coordinator._publish
            ), ""


class _SnapshotTicket:
    """Original operation survives both actual callback and handoff return."""

    def __init__(self, source, owner, operation, claim, *, publish=None):
        self.source, self.owner = source, owner
        self.publish = publish
        self.operation, self.claim = operation, claim
        self._lock = Lock()
        self.retired = Event()
        self._worker_returned = self._handoff_returned = self._retired = False
        self._failed_write = False
        self._workflow_completion = None
        self._observation_returned = False

    @contextmanager
    def activate(self):
        with self.source.activate(self.owner):
            token = _current_ticket.set(self)
            try:
                yield
            finally:
                _current_ticket.reset(token)

    def returned(
        self, *, worker=False, handoff=False, durable=True, observation_returned=False
    ):
        with self._lock:
            self._worker_returned |= worker
            self._handoff_returned |= handoff
            self._observation_returned |= observation_returned
            self._failed_write |= not durable
            if (
                self._retired
                or (
                    self._failed_write
                    and not (
                        self._observation_returned
                        and getattr(self.source, "workflow_wave", None) is not None
                    )
                )
                or not (self._worker_returned and self._handoff_returned)
            ):
                return
            self._retired = True
        with self.activate():
            self.operation.finish()
        if self.claim is not None:
            # Source/storage exits and availability callbacks inside finish
            # must physically return before the finite claim can retire.
            self.owner.finish_root()
            if self._failed_write and self.source.workflow_wave is not None:
                self.source.issuer.activity._release_workflow_observation(self)
            else:
                self.claim.settle(durable=not self._failed_write)
        self.retired.set()
        if self._workflow_completion is not None:
            public, kind = self._workflow_completion
            self.source.workflow_wave.snapshot_returned(public, kind, self)


@dataclass(frozen=True)
class _InitializationSource:
    issuer: _NativeInitializationIssuer
    work: object
    selector: Path
    data: Path
    shadow: object
    chain: object
    initializer: object
    git_executable: str | None
    ordinary_tree: object = None

    def check(self):
        from tldw_chatbook.Backup_Recovery import bootstrap

        self.issuer.check()
        if bootstrap.effective_config_path() != self.selector:
            raise ValueError("native initialization profile changed")
        if (
            self.shadow._data_dir != self.data / "change_review"
            or self.shadow._git != self.git_executable
        ):
            raise ValueError("native initialization shadow selection changed")
        if capture_directory_chain(self.chain.canonical_root) != self.chain:
            raise ValueError("native initialization root changed")
        # Root observation may block after the first original-source check.
        self.issuer.check()

    def validate_root(self, root):
        self.check()
        canonical = Path(root).resolve(strict=True)
        if not canonical.is_relative_to(self.chain.canonical_root):
            raise ValueError("native initialization root outside admission")
        return canonical

    @contextmanager
    def storage_scope(self):
        from tldw_chatbook.Backup_Recovery.activation import execution_scope

        self.check()
        with ExitStack() as stack:
            for owner, path in (
                ("config", self.selector),
                ("db.workspaces", self.issuer.database_path),
                ("workspaces.change_tracking", self.shadow._data_dir),
            ):
                if not stack.enter_context(execution_scope((owner,), path)):
                    raise ValueError("native initialization storage unavailable")
            stack.enter_context(operation_owned_connection(self.issuer.database))
            self.check()
            yield

    def admit(self):
        with self.storage_scope():
            self.check()
            if self.issuer.runtime._disposed or not self.issuer.service._work_current(
                self.work
            ):
                raise ValueError("native initialization admission changed")
            capacity, activity = self.issuer.capacity, self.issuer.activity
            if self.ordinary_tree is not None:
                owner = self.ordinary_tree.borrow(self.issuer.runtime)
                return _SnapshotTicket(self, owner, owner.reserve_tool(), None)
            with activity.prelaunch(footprint=WorktreeFootprint(opaque=True)) as claim:
                if claim is None:
                    raise CapacityRefused("initialization_activity_busy")
                owner = capacity.begin_execution(
                    origin=self.work.origin, conversation_id=""
                )
                try:
                    operation = owner.reserve_tool()
                except BaseException:
                    owner.finish_root()
                    raise
                try:
                    claim.bind_owner(owner)
                except BaseException:
                    operation.finish()
                    owner.finish_root()
                    raise
                return _SnapshotTicket(self, owner, operation, claim)

    @contextmanager
    def activate(self, owner):
        with self.storage_scope(), owner.activate():
            token = _current_snapshot.set(self)
            try:
                yield
            finally:
                _current_snapshot.reset(token)

    def initialize(self):
        from tldw_chatbook.Backup_Recovery import profile_paths
        from tldw_chatbook.config import get_user_data_dir, load_settings
        from tldw_chatbook.Workspaces.change_review_consent import (
            _default_root_initializer,
        )

        self.check()
        if profile_paths.user_data_dir(load_settings()) != self.data:
            raise ValueError("native initialization storage selection changed")
        if get_user_data_dir() != self.data:
            raise ValueError("native initialization secured storage changed")
        self.check()
        if self.initializer is _default_root_initializer:
            self.initializer(self.work.root, shadow_service=self.shadow)
        else:
            self.initializer(self.work.root)
