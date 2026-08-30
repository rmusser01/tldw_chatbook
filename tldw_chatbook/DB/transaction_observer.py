"""Process-local completion callbacks for managed SQLite transactions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import logging
import sqlite3
from threading import RLock


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
