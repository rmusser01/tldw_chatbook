"""Process-local completion callbacks for managed SQLite transactions, and the
semantic-revision coordination, chat-record hashing and Safe-capture trimming
ports CharactersRAGDB calls inside them."""

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
    factory = _factory  # read once: a concurrent clear must not become a TypeError
    if factory is None:
        raise SemanticCoordinationNotInstalledError(
            "Semantic-revision coordination is not installed. Call "
            "tldw_chatbook.Chat.console_semantic_revision.install_semantic_coordination() "
            "once at process startup (the composition root) before writing messages."
        )
    return factory(db)


# --- Chat-record hashing port (TASK-36949.4) --------------------------------
# A tombstone write must stamp the deleted message's Sync v2 record hash on its
# delete intent before the row's content is cleared, inside the same write
# transaction. The hash rules belong to Sync_Interop, which imports the DB, so
# the DB holds only this slot. ``Sync_Interop.chat_sync_source`` installs it on
# import, and every composition root installs it at startup. Fail closed: a
# tombstone write with nothing installed raises inside its transaction, which
# rolls back, instead of committing a delete Sync can never project.

_chat_record_hasher: Callable[[Any], str] | None = None


class ChatRecordHashingNotInstalledError(RuntimeError):
    """A message tombstone write ran before any root installed the port."""


def install_chat_record_hasher(hasher: Callable[[Any], str]) -> None:
    """Install the process-wide chat-record hasher. Idempotent.

    Args:
        hasher: ``hasher(row) -> "sha256:<hex>"`` over one live message row.
    """
    global _chat_record_hasher
    _chat_record_hasher = hasher


def chat_record_hash(row: Any) -> str:
    """Hash one live message row with the installed hasher.

    Args:
        row: The message row a tombstone writer is about to delete.

    Returns:
        The row's Sync record hash.

    Raises:
        ChatRecordHashingNotInstalledError: No composition root installed the
            hasher; raised inside the caller's transaction, which rolls back.
    """
    hasher = _chat_record_hasher  # read once, like semantic_coordinator_for
    if hasher is None:
        raise ChatRecordHashingNotInstalledError(
            "Chat-record hashing is not installed. Call "
            "tldw_chatbook.Sync_Interop.chat_sync_source.install_chat_sync_hashing() "
            "once at process startup (the composition root) before deleting messages."
        )
    return hasher(row)


# --- Safe-capture trimming port (TASK-36949.6) ------------------------------
# The v52->v53 migration rewrites each stored Safe exchange capture with the
# Console capture codec (decode, compact the history copy, re-encode through
# the credential sanitizer). The codec belongs to Chat, which imports the DB,
# so the DB holds only this slot. ``Chat.console_exchange_capture`` installs it
# on import, and every composition root installs it at startup. Fail closed:
# the migration reaches the slot only for a stored Safe blob, and with nothing
# installed it raises inside its transaction, which rolls the step back and
# leaves the database at v52 for a composed process to upgrade.

_safe_capture_trimmer: Callable[[bytes], bytes | None] | None = None


class SafeCaptureTrimmingNotInstalledError(RuntimeError):
    """The v52->v53 migration met a Safe capture before any root installed the port."""


class UnreadableCaptureBlobError(ValueError):
    """A stored capture blob the trimmer cannot decode; the migration leaves it as-is."""


def install_safe_capture_trimmer(trimmer: Callable[[bytes], bytes | None]) -> None:
    """Install ``trimmer(blob) -> compacted blob or None`` for this process. Idempotent."""
    global _safe_capture_trimmer
    _safe_capture_trimmer = trimmer


def trim_stored_safe_capture(blob: bytes) -> bytes | None:
    """Return the compacted blob, or None when unchanged; raise if not installed.

    Raises:
        UnreadableCaptureBlobError: The installed trimmer cannot decode ``blob``.
        SafeCaptureTrimmingNotInstalledError: Nothing installed the port.
    """
    trimmer = _safe_capture_trimmer  # read once, like semantic_coordinator_for
    if trimmer is None:
        raise SafeCaptureTrimmingNotInstalledError(
            "Safe-capture trimming is not installed. Call "
            "tldw_chatbook.Chat.console_exchange_capture.install_safe_capture_trimming() "
            "once at process startup (the composition root) before opening a "
            "database older than schema v53."
        )
    return trimmer(blob)
