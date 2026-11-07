"""Process-local completion callbacks for managed SQLite transactions, and the
semantic-revision coordination port CharactersRAGDB calls inside them."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import logging
import sqlite3
from threading import RLock
from typing import Any


TransactionCompletion = Callable[[bool | None], None]


@dataclass(slots=True)
class _TransactionState:
    connection: sqlite3.Connection
    token: object
    callbacks: list[TransactionCompletion] = field(default_factory=list)


_LOCK = RLock()
_ACTIVE: dict[int, _TransactionState] = {}
_LOGGER = logging.getLogger(__name__)


def begin_managed_transaction(connection: sqlite3.Connection) -> object:
    """Publish one manager-owned transaction and return its opaque identity."""

    key = id(connection)
    with _LOCK:
        if key in _ACTIVE:
            raise RuntimeError("managed_transaction_already_active")
        token = object()
        _ACTIVE[key] = _TransactionState(connection, token)
        return token


def current_managed_transaction(connection: sqlite3.Connection) -> object | None:
    """Return the active manager-issued transaction identity, if any."""

    with _LOCK:
        state = _ACTIVE.get(id(connection))
        if state is None or state.connection is not connection:
            return None
        return state.token


def register_transaction_completion(
    connection: sqlite3.Connection,
    token: object,
    callback: TransactionCompletion,
) -> None:
    """Register work to run after the exact managed transaction completes."""

    with _LOCK:
        state = _ACTIVE.get(id(connection))
        if (
            state is None
            or state.connection is not connection
            or state.token is not token
        ):
            raise RuntimeError("managed_transaction_required")
        state.callbacks.append(callback)


def complete_managed_transaction(
    connection: sqlite3.Connection,
    token: object,
    *,
    committed: bool | None,
) -> None:
    """Publish commit, rollback, or an ambiguous outcome and release state."""

    with _LOCK:
        state = _ACTIVE.get(id(connection))
        if (
            state is None
            or state.connection is not connection
            or state.token is not token
        ):
            raise RuntimeError("managed_transaction_identity")
        _ACTIVE.pop(id(connection))
        callbacks = tuple(state.callbacks)
    for callback in callbacks:
        try:
            callback(committed)
        except Exception:
            # Completion is already durable (or definitively ambiguous).  A
            # process-local observer must never rewrite that transaction result.
            _LOGGER.error("managed transaction completion callback failed")


def active_managed_transaction_count() -> int:
    """Return active observer state count for boundedness assertions."""

    with _LOCK:
        return len(_ACTIVE)


# --- Semantic-revision coordination port (TASK-36949.3) ---------------------
# CharactersRAGDB must run semantic-revision coordination inside the write
# transaction, but the coordinator lives in the Console domain layer, which
# imports the DB. The DB holds only this process-wide slot; each composition
# root installs the real factory once
# (``Chat.console_semantic_revision.install_semantic_coordination``). Fail
# closed: a coordinated write with nothing installed raises instead of silently
# skipping coordination. Stdlib only: this module sits in the DB import closure.

_factory: Callable[[Any], Any] | None = None


class SemanticCoordinationNotInstalledError(RuntimeError):
    """A coordinated message write ran before any composition root installed the port."""


def install_semantic_coordinator_factory(factory: Callable[[Any], Any]) -> None:
    """Install ``factory(db) -> coordinator`` for this process. Idempotent."""
    global _factory
    _factory = factory


def semantic_coordinator_for(db: Any) -> Any:
    """Return the coordinator for ``db`` or raise if nothing was installed."""
    if _factory is None:
        raise SemanticCoordinationNotInstalledError(
            "Semantic-revision coordination is not installed. Call "
            "tldw_chatbook.Chat.console_semantic_revision.install_semantic_coordination() "
            "once at process startup (the composition root) before writing messages."
        )
    return _factory(db)
