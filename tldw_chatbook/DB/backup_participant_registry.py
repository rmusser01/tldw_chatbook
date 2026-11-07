"""Where stores declare their backup-participant registration (ADR-126).

Stdlib only and in the DB layer, so a store in DB or in a feature package imports
it without an upward edge. ``Backup_Recovery.participants`` reads it and keeps no
store list of its own. Each store declares its exact type once, at class
definition, with ``@declare_store``; subclasses confer no authority.
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


#: Owner id -> declaration.
declarations: dict[str, StoreDeclaration] = {}
#: Exact type -> owner id. Each declaration replaces the table instead of
#: mutating it, so readers share one table and never see it change mid-iteration.
types = MappingProxyType({})
_lock = threading.Lock()


def declare_store(owner_id: str, **fields):
    """Class decorator registering ``cls`` as the exact type of ``owner_id``."""
    declaration = StoreDeclaration(owner_id, **fields)

    def register(cls):
        global types
        with _lock:  # Store modules can be imported on two threads at once.
            declarations[owner_id] = declaration
            types = MappingProxyType({**types, cls: owner_id})
        return cls

    return register
