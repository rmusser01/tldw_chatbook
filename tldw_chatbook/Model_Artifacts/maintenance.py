"""Native lifetime accounting for the installed model store and its leases."""

import sys
import threading
from contextlib import contextmanager
from functools import partial, wraps
from pathlib import Path

from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery import storage_admission as storage

_local = threading.local()


class _Lifetime:
    def __init__(self, root):
        self.root = root.resolve()
        self.lease = storage.acquire_storage(self.root)
        self.references = 1
        with storage._changed:
            storage._raw_operations.add(self)

    def retain(self):
        with storage._changed:
            if (
                self not in storage._raw_operations
                or self.lease not in storage._live_leases
                or self.references <= 0
            ):
                raise bootstrap.RecoveryRequired("model_lifetime_inactive")
            self.references += 1
        return self

    def release(self):
        with storage._changed:
            self.references -= 1
            last = self.references == 0
        if last:
            self.lease.close()
            with storage._changed:
                storage._raw_operations.discard(self)
                storage._changed.notify_all()


def _installed_caller(service):
    from .service import ModelArtifactService

    if type(service) is not ModelArtifactService:
        return False
    frame = sys._getframe(1)
    while frame and frame.f_globals.get("__name__") in {
        __name__,
        "tldw_chatbook.Model_Artifacts.leases",
    }:
        frame = frame.f_back
    return (
        frame is not None
        and frame.f_locals.get("self") is service
        and any(
            frame.f_code
            is getattr(getattr(method, "__wrapped__", method), "__code__", None)
            for method in vars(ModelArtifactService).values()
        )
    )


def model_call(function=None, *, execution=False):
    """Count finite original service calls, including constructors and readers."""

    if function is None:
        return partial(model_call, execution=execution)

    @wraps(function)
    def call(self, *args, **kwargs):
        root = (
            (args[0] if args else kwargs.get("root"))
            if function.__name__ == "__init__"
            else self._root
        )
        if not isinstance(root, Path):
            return function(self, *args, **kwargs)
        previous = getattr(_local, "active", None)
        entry = getattr(_local, "entry", None)
        _local.entry = None
        if (
            previous is not None
            and previous[0] is self
            and (entry == (self, function) or _installed_caller(self))
        ):
            lifetime = previous[1].retain()
        else:
            lifetime = _Lifetime(root)
        _local.active = (self, lifetime)
        try:
            if execution:
                from tldw_chatbook.Backup_Recovery.activation import execution_scope

                from .service import ArtifactStateError

                with execution_scope(
                    ("config", "models.artifacts"), root, retained=lifetime.lease
                ) as allowed:
                    if not allowed:
                        raise ArtifactStateError("model_activation_required")
                    return function(self, *args, **kwargs)
            return function(self, *args, **kwargs)
        finally:
            _local.active = previous
            lifetime.release()

    return call


def acquisition_call(function):
    """Gate network acquisition through its existing native settlement boundary."""

    @wraps(function)
    async def call(self, *args, **kwargs):
        from tldw_chatbook.Backup_Recovery.activation import execution_scope

        from .acquisition import AcquisitionError

        with execution_scope(
            ("config", "models.artifacts"), self._core.locks_path.parent
        ) as allowed:
            if not allowed:
                raise AcquisitionError("model_activation_required")
            return await function(self, *args, **kwargs)

    return call


def lease_lifetime(lock_root):
    """Keep admission until the existing artifact lock's actual close succeeds."""
    active = getattr(_local, "active", None)
    if (
        active is not None
        and lock_root == active[1].root / "locks"
        and _installed_caller(active[0])
    ):
        return active[1].retain()
    return _Lifetime(lock_root.parent)


@contextmanager
def provision_continuation(core, lease, function):
    """Use only the exact provision session's held store in its native core hop."""
    from .acquisition import ACQUISITION_SESSION_LEASE_KEY
    from .leases import ArtifactOperationLease
    from .service import ModelArtifactService

    if (
        type(core) is not ModelArtifactService
        or type(lease) is not ArtifactOperationLease
        or not lease.acquired
        or lease.key != ACQUISITION_SESSION_LEASE_KEY
        or lease._lock_root != core.locks_path
        or lease._maintenance_lifetime is None
    ):
        raise bootstrap.RecoveryRequired("model_provision_continuation_invalid")
    allowed = {
        ModelArtifactService._download_stage_for,
        ModelArtifactService._finalize_download_stage,
        ModelArtifactService.activate,
        ModelArtifactService.list_installed,
        ModelArtifactService.disk_usage,
    }
    method = function.func if isinstance(function, partial) else function
    if (
        getattr(method, "__self__", None) is not core
        or getattr(method, "__func__", None) not in allowed
        or not hasattr(method.__func__, "__wrapped__")
    ):
        # Custom methods retain their ordinary route, without inherited authority.
        yield
        return
    lifetime = lease._maintenance_lifetime.retain()
    previous = getattr(_local, "active", None)
    previous_entry = getattr(_local, "entry", None)
    _local.active = (core, lifetime)
    _local.entry = (core, method.__func__.__wrapped__)
    try:
        yield
    finally:
        _local.active = previous
        _local.entry = previous_entry
        lifetime.release()


def staged_read(function):
    """Count the acquisition preflight's finite sidecar read."""

    @wraps(function)
    def call(self, *args, **kwargs):
        from .service import ModelArtifactService

        if type(self._core) is not ModelArtifactService:
            return function(self, *args, **kwargs)
        lease = self._maintenance_lease()
        lifetime = (
            lease._maintenance_lifetime.retain()
            if lease is not None
            else _Lifetime(self._core._root)
        )
        previous = getattr(_local, "active", None)
        previous_entry = getattr(_local, "entry", None)
        _local.active = (self._core, lifetime)
        _local.entry = (
            self._core,
            ModelArtifactService._download_stage_for.__wrapped__,
        )
        try:
            return function(self, *args, **kwargs)
        finally:
            _local.active = previous
            _local.entry = previous_entry
            lifetime.release()

    return call
