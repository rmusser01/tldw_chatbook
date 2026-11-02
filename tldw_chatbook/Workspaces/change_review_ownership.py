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

_current_snapshot: ContextVar[_SnapshotSource | _InitializationSource | None] = (
    ContextVar("native_change_review_snapshot", default=None)
)


def native_snapshot_source() -> _SnapshotSource | _InitializationSource | None:
    """Return only the original source active on this physical callback thread."""
    return _current_snapshot.get()


class _NativeReviewResources:
    """Pin the original runtime resources without invoking cold getters."""

    def __init__(self, runtime):
        self.runtime = runtime
        self._capacity = runtime._execution_capacity
        self._activity = runtime._worktree_activity

    @property
    def capacity(self):
        if self._capacity is None:
            self._capacity = self.runtime.execution_capacity
        elif self.runtime._execution_capacity is not self._capacity:
            raise ValueError("native change-review capacity changed")
        return self._capacity

    @property
    def activity(self):
        if self._activity is None:
            self._activity = self.runtime.worktree_activity
        elif self.runtime._worktree_activity is not self._activity:
            raise ValueError("native change-review activity changed")
        return self._activity


class _NativeSnapshotIssuer(_NativeReviewResources):
    """One runtime's issuer; execution stays in the existing coordinator."""

    def __init__(self, runtime):
        super().__init__(runtime)
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
        # Observe resources created since issuer construction without loading them.
        if self._capacity is None:
            self._capacity = self.runtime._execution_capacity
        if self._activity is None:
            self._activity = self.runtime._worktree_activity
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
            return _SnapshotTicket(
                self,
                self.owner,
                operation,
                None,
                publish=self.issuer.coordinator._publish,
            ), ""
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


class _NativeInitializationIssuer(_NativeReviewResources):
    """Original consent feature source; never impersonates a Chat or bridge."""

    def __init__(self, runtime, service, registry):
        super().__init__(runtime)
        self.app, self.service, self.registry = runtime._app, service, registry
        self.database = registry.db
        self.database_path = Path(registry.db.db_path)
        self.initializer = service._initialize_root
        self.capability_reader = service._capability_reader

    def identity_current(self) -> bool:
        """Compare only original object references, safe under the consent lock."""
        return (
            self.runtime._app is self.app
            and self.app.console_runtime is self.runtime
            and self.runtime._change_review_consent_service is self.service
            and self.app.change_review_consent_service is self.service
            and self.app.workspace_registry_service is self.registry
            and self.registry.db is self.database
            and self.registry._change_review_binding_owner is self.service
            and self.service._registry is self.registry
            and self.service._native is self
            and self.service._initialize_root is self.initializer
            and self.service._capability_reader is self.capability_reader
        )

    def check(self):
        if (
            not self.identity_current()
            or Path(self.database.db_path) != self.database_path
        ):
            raise ValueError("native initialization source changed")

    def capture(self, work):
        from tldw_chatbook.Backup_Recovery import bootstrap, profile_paths
        from tldw_chatbook.Backup_Recovery.activation import execution_scope
        from tldw_chatbook.config import load_settings
        from tldw_chatbook.Workspaces.change_tracking import ShadowRepoService

        self.check()
        if self.runtime._disposed:
            raise ValueError("native initialization is closed")
        if self._capacity is None:
            self._capacity = self.runtime._execution_capacity
        if self._activity is None:
            self._activity = self.runtime._worktree_activity
        selector = bootstrap.effective_config_path()
        with execution_scope(("config",), selector) as allowed:
            if not allowed:
                raise ValueError("native initialization source unavailable")
            data = profile_paths.user_data_dir(load_settings())
        # Explicit metadata selection avoids get_user_data_dir's mkdir/hardening.
        shadow = ShadowRepoService(data_dir=data / "change_review")
        chain = capture_directory_chain(Path(work.root))
        if str(chain.canonical_root) != work.root:
            raise ValueError("native initialization root changed")
        return _InitializationSource(
            self,
            work,
            selector,
            data,
            shadow,
            chain,
            self.initializer,
            shadow._git,
        )


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
