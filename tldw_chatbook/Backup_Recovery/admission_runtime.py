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
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from .activation import execution_scope
from .storage_admission import acquire_storage


def execution_identity() -> "tuple[int, int, asyncio.Task[object] | None]":
    """Identify the current execution by PID, thread, and asyncio task.

    Returns:
        The ``(pid, thread ident, asyncio task)`` triple; the task is None
        outside a running event loop. Copied contexts cannot counterfeit a
        match because the task object differs.
    """
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return os.getpid(), threading.get_ident(), task


@dataclass
class ExecutionState:
    """One accepted admission: reusable in nesting, transferable to workers.

    Args:
        identity: The :func:`execution_identity` triple that earned admission.
        leases: Storage leases held for this execution, keyed by path; nested
            executions copy and extend this mapping.
        sources: Deduplicated ``(owner, path)`` observations admitted so far.
        parent: The state this one was entered from, if any; the ancestor
            chain is re-verified on reuse.
        live: Cleared when the owning scope exits; dead states refuse reuse.
    """

    identity: tuple
    leases: dict
    sources: tuple
    parent: "ExecutionState | None" = None
    live: bool = True

    def check(self, error: Callable[[], Exception]) -> None:
        """Refuse dead states, forked use, or a dead ancestor chain.

        Args:
            error: Zero-argument factory building the family exception.

        Raises:
            Exception: Whatever ``error`` builds, when this state (or any
                ancestor) is dead or was created in another process.
        """
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
        name: str,
        *,
        error: Callable[[], PermissionError],
        sources: Callable[..., tuple],
        owners: Callable[[str], tuple] | None = None,
        admit_source: Callable[..., bool] | None = None,
        finalize: Callable[..., bool] | None = None,
        check_states: bool = False,
    ) -> None:
        """Build one family's admission guard.

        Args:
            name: Family name; names the guard's ContextVar for debugging.
            error: Zero-argument factory building the family exception.
            sources: ``(service) -> tuple[(owner, path)]`` observing the
                family's installed paths without constructing them.
            owners: Optional ``(owner) -> owners tuple`` mapping for
                ``execution_scope``; defaults to the owner itself.
            admit_source: Optional ``(stack, owner, path, lease) -> bool``
                replacing per-source admission (return True when handled).
            finalize: Optional ``(observed, leases) -> bool`` whole-
                observation family review after every source is admitted.
            check_states: Verify the full ancestor chain (live/PID) when
                reusing accepted states, for families that transfer scopes
                into awaited workers.
        """
        self.context = ContextVar(name + "_execution", default=None)
        self.error = error
        self.sources = sources
        self.owners = owners or (lambda owner: (owner,))
        self.admit_source = admit_source
        self.finalize = finalize
        self.check_states = check_states

    def captured_sources(self, service: object) -> tuple:
        """Reuse this task's accepted observation, or observe afresh.

        Args:
            service: The family service whose paths ``sources`` observes.

        Returns:
            The accepted ``(owner, path)`` tuple when this task already holds
            admission, otherwise a fresh observation.
        """
        active = self.context.get()
        if active is not None and active.identity == execution_identity():
            return active.sources
        return self.sources(service)

    @contextmanager
    def execution(
        self,
        service: object = None,
        *,
        sources: tuple = (),
        transfer: "ExecutionState | None" = None,
        resolve: Callable[..., tuple] | None = None,
    ) -> Iterator[None]:
        """Retain admission through final persistence in this task/thread.

        Sources are observations, not permissions. Worker threads reacquire
        them; copied contexts cannot borrow another execution's accepted
        leases. ``transfer`` re-enters an accepted scope inside a worker
        without letting it outlive the original admission.

        Args:
            service: The family service whose paths are observed.
            sources: Extra caller-supplied ``(owner, path)`` observations.
            transfer: An accepted state to re-enter inside a worker.
            resolve: Optional ``(service) -> tuple`` replacing the default
                ``sources + family sources`` observation order.

        Raises:
            Exception: Whatever ``error`` builds, on a foreign state, a
                failed admission, or a storage failure during admission.
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
    def worker_isolation(self) -> Iterator[None]:
        """Enter native workers with no borrowed state; they reacquire.

        Yields:
            Nothing; the family ContextVar is cleared for the duration so a
            worker's ``execution`` call acquires its own leases, then
            restored on exit.
        """
        token = self.context.set(None)
        try:
            yield
        finally:
            self.context.reset(token)
