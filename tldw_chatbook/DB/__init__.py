"""DB layer package.

It also hosts the backup-participant registry (ADR-126, TASK-36948.1). The registry
is stdlib only and lives here because every store import already loads this package,
so the registry adds no boot module, and a store in DB or a feature package declares
itself without an upward import. ``Backup_Recovery.participants`` reads the registry
and keeps no store list of its own. Each store declares its exact type once, at
class definition, with ``@declare_store``. Subclasses confer no authority.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable


@dataclass(frozen=True)
class StoreDeclaration:
    """One store's registration with the backup participants."""

    owner_id: str
    #: Owner method that closes the calling thread's cache (``None``: no closer).
    closer: str | None = None
    #: For a store without a closer: returns the calling thread's cached
    #: connection, or ``None``. Participants closes it under ``core_closing``.
    thread_cache: Callable[[object], object] | None = None
    #: Resource-policy id of its connections (``None``: the owner id).
    policy: str | None = None
    #: Resource-policy id of a read-only instance's connections, when it differs.
    read_only_policy: str | None = None
    #: False: ``run_finite_local_worker`` leaves the caches it opened to the
    #: store's owner instead of closing them.
    finite_worker: bool = True
    #: One native connection serves every thread, so a retirement on any
    #: thread refuses access and a live operation on any thread refuses close.
    shared_connection: bool = False

    def policy_id(self, read_only: bool) -> str:
        if read_only and self.read_only_policy is not None:
            return self.read_only_policy
        return self.owner_id if self.policy is None else self.policy


_store_declarations: dict[str, StoreDeclaration] = {}
#: Owner id -> declaration (read-only view).
store_declarations = MappingProxyType(_store_declarations)
#: Exact type -> owner id. Each declaration replaces the table instead of
#: mutating it, so readers share one table and never see it change mid-iteration.
store_types = MappingProxyType({})
_store_lock = threading.Lock()


def declare_store(owner_id: str, **fields):
    """Class decorator registering ``cls`` as the exact type of ``owner_id``.

    Raises:
        TypeError: ``closer`` does not name a callable attribute of ``cls``.
        ValueError: ``owner_id`` is already declared.
    """
    declaration = StoreDeclaration(owner_id, **fields)

    def register(cls):
        global store_types
        closer = declaration.closer
        if closer is not None and not callable(getattr(cls, closer, None)):
            raise TypeError(f"{cls.__qualname__} has no callable closer {closer!r}")
        with _store_lock:  # Store modules can be imported on two threads at once.
            if owner_id in _store_declarations:
                raise ValueError(f"backup participant {owner_id!r} is already declared")
            _store_declarations[owner_id] = declaration
            store_types = MappingProxyType({**store_types, cls: owner_id})
        return cls

    return register


# Import datetime fix to register adapters on module load
from . import sqlite_datetime_fix  # noqa: E402,F401
