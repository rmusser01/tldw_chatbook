"""Shared runtime for per-family recovery-admission guards (ADR-126).

The Agents, MCP, and RAG_Search activation modules each own their observed
sources, family exception, and guard policies. This module owns the one
execution skeleton they previously carried as three hand-maintained copies:
identity derivation, lease reuse across nesting, observed-source dedup,
per-source admission through ``execution_scope``, and ContextVar state
handoff into native workers.
"""

import asyncio
import os
import threading
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from .activation import execution_scope
from .storage_admission import acquire_storage


def execution_identity():
    """Identify the current execution by PID, thread, and asyncio task."""
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return os.getpid(), threading.get_ident(), task


@dataclass
class ExecutionState:
    """One accepted admission: reusable in nesting, transferable to workers."""

    identity: tuple
    leases: dict
    sources: tuple
    parent: "ExecutionState | None" = None
    live: bool = True

    def check(self, error):
        """Refuse dead states, forked use, or a dead ancestor chain."""
        if not self.live or self.identity[0] != os.getpid():
            raise error()
        if self.parent is not None:
            self.parent.check(error)


class RecoveryAdmissionGuard:
    """Parameterized admission skeleton for one family's activation module.

    ``error`` builds the family exception; ``sources`` observes the family's
    installed paths; ``owners`` maps each observed owner to its
    ``execution_scope`` owners tuple; ``admit_source`` optionally replaces
    per-source admission (generation witnesses for plain config bytes);
    ``finalize`` optionally runs a whole-observation family review; and
    ``check_states`` adds parent-chain liveness verification when accepted
    scopes can be transferred into awaited workers.
    """

    def __init__(
        self,
        name,
        *,
        error,
        sources,
        owners=None,
        admit_source=None,
        finalize=None,
        check_states=False,
    ):
        self.context = ContextVar(name + "_execution", default=None)
        self.error = error
        self.sources = sources
        self.owners = owners or (lambda owner: (owner,))
        self.admit_source = admit_source
        self.finalize = finalize
        self.check_states = check_states

    def captured_sources(self, service):
        """Reuse this task's accepted observation, or observe afresh."""
        active = self.context.get()
        if active is not None and active.identity == execution_identity():
            return active.sources
        return self.sources(service)

    @contextmanager
    def execution(self, service=None, *, sources=(), transfer=None, resolve=None):
        """Retain admission through final persistence in this task/thread.

        Sources are observations, not permissions. Worker threads reacquire
        them; copied contexts cannot borrow another execution's accepted
        leases. ``transfer`` re-enters an accepted scope inside a worker
        without letting it outlive the original admission.
        """
        active = transfer if transfer is not None else self.context.get()
        if active is not None:
            if self.check_states:
                active.check(self.error)
            if transfer is None and active.identity != execution_identity():
                raise self.error()
        leases = dict(active.leases) if active else {}
        observe = resolve or (lambda service: tuple(sources) + self.sources(service))
        observed = tuple(
            dict.fromkeys((active.sources if active else ()) + observe(service))
        )
        with ExitStack() as stack:
            try:
                for owner, path in observed:
                    if path not in leases:
                        leases[path] = acquire_storage(path)
                        stack.callback(leases[path].close)
                    if self.admit_source is not None and self.admit_source(
                        stack, owner, path, leases[path]
                    ):
                        continue
                    if not stack.enter_context(
                        execution_scope(
                            self.owners(owner), path, retained=leases[path]
                        )
                    ):
                        raise self.error()
                if self.finalize is not None and not self.finalize(observed, leases):
                    raise self.error()
            except (OSError, ValueError, TypeError, RuntimeError, AttributeError):
                raise self.error() from None
            state = ExecutionState(execution_identity(), leases, observed, active)
            token = self.context.set(state)
            try:
                yield
            finally:
                state.live = False
                self.context.reset(token)

    @contextmanager
    def worker_isolation(self):
        """Enter native workers with no borrowed state; they reacquire."""
        token = self.context.set(None)
        try:
            yield
        finally:
            self.context.reset(token)
