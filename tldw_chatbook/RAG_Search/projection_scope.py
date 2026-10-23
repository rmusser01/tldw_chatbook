"""Code-issued, finite project admission for the installed embedding wrapper."""

from __future__ import annotations

import threading
import weakref
from contextlib import contextmanager
from contextvars import ContextVar

from tldw_chatbook.Backup_Recovery.admission_runtime import execution_identity


class ProjectProjectionRequired(PermissionError):
    """The selected project embedding operation has no current admission."""

    def __init__(self, reason: str = "project_projection_scope_required"):
        super().__init__(reason)


_issued: weakref.WeakSet[ProjectionExecutionScope] = weakref.WeakSet()
_active: ContextVar[_ProjectExecution | None] = ContextVar(
    "project_projection_execution", default=None
)
_SEAL = object()


class _ProjectExecution:
    def __init__(self, carrier: ProjectionExecutionScope, parent=None):
        self.carrier = carrier
        self.identity = execution_identity()
        self.parent = parent
        self.live = True

    def check(self):
        self.carrier._check_live()
        if not self.live or self.identity[0] != execution_identity()[0]:
            raise ProjectProjectionRequired()
        if self.parent is not None:
            self.parent.check()


class ProjectionExecutionScope:
    """One exact, accepted operation; close after wrapper/native retirement.

    The carrier is an observation, not a saved grant. A later build or query
    obtains a new carrier under a fresh project parent. Native work retires`r`n    before each carrier closes; the wrapper may keep a compatible idle model.`r`n    Copies are absent from the private issue set.
    """

    def __init__(self, *, service, scope, settings, revision, target, parent, _seal):
        if _seal is not _SEAL:
            raise ProjectProjectionRequired()
        self._service = service
        self.scope = scope
        self.settings = settings
        self.semantic_revision = revision
        self.model_name = target.model_name
        self.api_key = target.api_key
        self.base_url = target.base_url
        self.local_files_only = target.local_files_only
        self.artifact_fingerprint = target.artifact_fingerprint
        self.__config = dict(target.config)
        self._parent = parent
        self._identity = execution_identity()
        self._lock = threading.Lock()
        self._children = 0
        self._closed = False
        self._sealed = True
        _issued.add(self)

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False) and name not in ("_closed", "_children"):
            raise AttributeError("issued_project_scope_immutable")
        object.__setattr__(self, name, value)

    @property
    def _config(self):
        return dict(self.__config)

    def __repr__(self):
        return "ProjectionExecutionScope(<issued project operation>)"

    @classmethod
    def require(cls, value):
        if type(value) is not cls or value not in _issued:
            raise ProjectProjectionRequired()
        value._check_live()
        return value

    def _check_live(self):
        if (
            self not in _issued
            or self._closed
            or self._parent.pin.closed
            or self._parent.operation.done.is_set()
            or self._parent.operation not in self._service._operations
        ):
            raise ProjectProjectionRequired()

    def validate_constructor(
        self,
        *,
        model_name,
        api_key,
        base_url,
        device,
        cache_dir,
        embedding_cache_db,
        local_files_only,
    ):
        self.require(self)
        if execution_identity() != self._identity:
            raise ProjectProjectionRequired()
        if (
            model_name != self.model_name
            or api_key != self.api_key
            or base_url != self.base_url
            or (device is not None and device != self._config.get("device"))
            or cache_dir is not None
            or embedding_cache_db is not None
            or local_files_only is not self.local_files_only
        ):
            raise ProjectProjectionRequired("project_projection_target_changed")

    def _current_state(self):
        service = self._service
        with service._store(self.scope.key, cleanup_only=True) as store:
            with service._read_fenced(store, self.scope.key) as authority:
                if authority is None or authority.key != self.scope.key:
                    raise ProjectProjectionRequired()
                status = store.load_status(self.scope.key)
                state = store.load_semantic_state(self.scope)
                if (
                    status.scope != self.scope
                    or not state.enabled
                    or state.state not in ("Pending", "Building", "Ready")
                    or state.semantic_revision != self.semantic_revision
                    or state.settings != self.settings
                    or self.settings.approved_destination_fingerprint
                    != self.settings.destination_fingerprint
                ):
                    raise ProjectProjectionRequired()

    @contextmanager
    def execution(self, *, transfer=None):
        """Retain a fresh child pin and verify consent before any model effect."""
        self.require(self)
        current = _active.get() if transfer is None else transfer
        if current is None:
            if execution_identity() != self._identity:
                raise ProjectProjectionRequired()
        else:
            current.check()
            if current.carrier is not self or (
                transfer is None and current.identity != execution_identity()
            ):
                raise ProjectProjectionRequired()
        with self._lock:
            self._check_live()
            self._children += 1
        try:
            service = self._service
            with service.registry_connection_scope(), service._factory.retain(
                self.scope.key
            ) as pin, pin.bind():
                self._current_state()
                state = _ProjectExecution(self, current)
                token = _active.set(state)
                try:
                    yield
                    state.check()
                finally:
                    state.live = False
                    _active.reset(token)
        finally:
            with self._lock:
                self._children -= 1

    def accepted(self):
        """Return only this task's live accepted state for explicit transfer."""
        self.require(self)
        state = _active.get()
        if state is not None:
            state.check()
            if state.carrier is not self or state.identity != execution_identity():
                raise ProjectProjectionRequired()
        elif execution_identity() != self._identity:
            raise ProjectProjectionRequired()
        return state

    def accepts_successor(self, successor):
        """Permit one compatible fresh issuer after this parent has retired."""
        if self not in _issued or not self._closed or self._children:
            raise ProjectProjectionRequired()
        successor = self.require(successor)
        if (
            successor._service is not self._service
            or successor.scope.key != self.scope.key
            or successor.settings != self.settings
            or successor.artifact_fingerprint != self.artifact_fingerprint
            or successor.model_name != self.model_name
            or successor.api_key != self.api_key
            or successor.base_url != self.base_url
            or successor.local_files_only is not self.local_files_only
        ):
            raise ProjectProjectionRequired("project_projection_target_changed")
        if execution_identity() != successor._identity:
            raise ProjectProjectionRequired()
        return successor

    def close(self):
        self.require(self)
        if execution_identity() != self._identity:
            raise ProjectProjectionRequired()
        with self._lock:
            if self._children:
                raise ProjectProjectionRequired("project_projection_native_work_active")
            self._parent.stack.close()
            self._closed = True

    def __enter__(self):
        self.require(self)
        return self

    def __exit__(self, *_):
        self.close()


def _issue_carrier(*, service, scope, settings, revision, target, parent):
    return ProjectionExecutionScope(
        service=service,
        scope=scope,
        settings=settings,
        revision=revision,
        target=target,
        parent=parent,
        _seal=_SEAL,
    )
