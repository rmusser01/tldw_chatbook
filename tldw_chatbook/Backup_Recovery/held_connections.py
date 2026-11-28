"""Console SQLite handles that stay open between owned calls (TASK-40608 D1).

On Windows opening a private SQLite connection re-verifies the folder chain and
both sidecars with native handles, and an owned call
(``DB.base_db.operation_owned_connection``) used to pay that on every call. A
Console store's connection opened by an owned call now stays in its worker
thread's cache. While it is open Windows refuses to rename, replace or delete
the database file and its folders, so the handle goes on reading the file that
was checked. Permissions can still change: a NEW connect keeps every check.

A local pause (backup, restore), app quit and an owner's explicit close retire
the idle held handles through ``retire_held_connections``; a handle in use is
left for the next call, so ``pause.drain()`` still waits for it. POSIX keeps
the per-call close: its checks are cheap and an open handle pins nothing there.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import weakref

from . import storage_admission as storage

#: Pins whether an owned call's new handle is held: True or False decides it
#: outright (tests of one rule or the other); None asks ``_holding``.
_HOLD_OPERATION_CONNECTIONS: bool | None = None
#: Environment override for ``_holding``, in either direction, on Windows.
HOLD_ENVIRONMENT_VARIABLE = "TLDW_HOLD_DB_CONNECTIONS"
#: Above this many held handles an owned call closes its handle as before.
_HELD_CONNECTIONS_MAX = 96
#: Guards the two tables below. Taken before the storage coordinator lock,
#: never while holding it.
_HELD_LOCK = threading.RLock()
#: id(connection) -> (connection, opening thread, weak store reference).
_held_connections: dict[
    int, tuple[sqlite3.Connection, threading.Thread, weakref.ReferenceType]
] = {}
#: (thread ident, id(store)) -> depth of owned calls running there now.
_owned_calls: dict[tuple[int, int], int] = {}


def _holding() -> bool:
    """Whether an owned call's new handle is held in this process now.

    On by default on Windows; never on POSIX. Off under pytest
    (``PYTEST_CURRENT_TEST``, the signal the screen pre-importer,
    ``Utils/optional_deps.py`` and ``Metrics/metrics_logger.py`` already gate
    on): a large part of the suite pins that a finite operation closes the
    handle it opened and releases its storage lease, which stays the contract
    on POSIX and, on Windows, for every other store, a paused profile and the
    cap. With handles held those tests fail on Windows only, by design and
    not by defect (68 of them in the first 324 files compared with dev on
    2026-11-22). ``TLDW_HOLD_DB_CONNECTIONS`` overrides in either direction:
    ``"1"`` turns holding on under pytest (this feature's own tests and the
    TASK-40608 journey), ``"0"`` turns it off in the app.
    """
    pinned = _HOLD_OPERATION_CONNECTIONS
    if pinned is not None:
        return pinned
    if os.name != "nt":
        return False
    override = os.environ.get(HOLD_ENVIRONMENT_VARIABLE)
    if override is not None:
        return override.strip().lower() not in ("", "0", "false", "no")
    return "PYTEST_CURRENT_TEST" not in os.environ


def _held_store_types() -> tuple[type, ...]:
    """The Console stores whose owned-call handles may stay open."""
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB

    return (AgentRunsDB, CharactersRAGDB, WorkspaceDB)


def _connection_closed(connection: sqlite3.Connection) -> bool:
    try:
        sqlite3.Connection.in_transaction.__get__(connection)
    except sqlite3.ProgrammingError:
        return True
    return False


def _close_held(connection: sqlite3.Connection, database: object | None) -> bool:
    """Close one held handle natively and forget it in its store's registry."""
    try:
        connection.close()
    except Exception:  # noqa: BLE001 - a failed close keeps its lease visible
        return _connection_closed(connection)
    registry = getattr(database, "_connection_quiescence", None)
    if registry is not None:
        registry.unregister(connection)
    return True


def owned_call_entered(database: object) -> tuple[int, int] | None:
    """Record that an owned call is running on this thread and store.

    Returns:
        The token for ``owned_call_left``, or None where handles are not held.
    """
    if not _holding():
        return None
    token = (threading.get_ident(), id(database))
    with _HELD_LOCK:
        _owned_calls[token] = _owned_calls.get(token, 0) + 1
    return token


def owned_call_left(token: tuple[int, int] | None) -> None:
    """Forget one ``owned_call_entered``."""
    if token is None:
        return
    with _HELD_LOCK:
        depth = _owned_calls.get(token, 0) - 1
        if depth > 0:
            _owned_calls[token] = depth
        else:
            _owned_calls.pop(token, None)


def hold_operation_connection(database: object, connection: sqlite3.Connection) -> bool:
    """Keep an owned call's new handle in its thread cache instead of closing it.

    Returns:
        True when the handle is now held; False when the caller must close it
        as before (other platform, other store, paused storage, or the cap).
    """
    if (
        not _holding()
        or type(database) not in _held_store_types()
        or getattr(database, "is_memory_db", True)
    ):
        return False
    participant = getattr(database, "_maintenance_participant", None)
    with storage._lock:
        if storage._pause is not None or getattr(participant, "closed", True):
            return False
    stale = []
    with _HELD_LOCK:
        if id(connection) in _held_connections:
            return True
        for key, (known, thread, reference) in tuple(_held_connections.items()):
            if not thread.is_alive() or _connection_closed(known):
                # Its worker is gone, so nothing can use or close it any more.
                stale.append((known, reference()))
                del _held_connections[key]
        held = len(_held_connections) < _HELD_CONNECTIONS_MAX
        if held:
            _held_connections[id(connection)] = (
                connection,
                threading.current_thread(),
                weakref.ref(database),
            )
    for known, owner in stale:
        _close_held(known, owner)
    return held


def retire_held_connections(database: object | None = None) -> bool:
    """Close held owned-call handles that nothing is using.

    A handle is left open while an owned call runs on its thread and store,
    while a counted core operation runs on its thread and file, inside a
    transaction, or while a cursor reservation is open. Callers that need
    every handle closed (a pause drain, quit) call again until this is True.

    Args:
        database: Limit the close to this store; None closes for every store.

    Returns:
        True when no held handle remains for the selection.
    """
    if not _held_connections:
        return True
    remaining = False
    with _HELD_LOCK:
        for key, (connection, thread, reference) in tuple(_held_connections.items()):
            owner = reference()
            if database is not None and owner is not database:
                continue
            if _connection_closed(connection):
                del _held_connections[key]
                continue
            if owner is not None and thread.is_alive():
                path = getattr(owner, "db_path", None)
                with storage._lock:
                    counted = any(
                        operation.thread is thread and operation.participant.path == path
                        for operation in storage._operations
                    )
                tokens = getattr(connection, "_quiescence_tokens", None)
                try:
                    busy = (
                        counted
                        or bool(_owned_calls.get((thread.ident, id(owner))))
                        or bool(tokens)
                        or connection.in_transaction
                    )
                except sqlite3.ProgrammingError:
                    busy = False
                if busy:
                    remaining = True
                    continue
            # Closed under _HELD_LOCK: an owned call entering on that thread
            # waits here and then sees the retired handle, never a closing one.
            if _close_held(connection, owner):
                del _held_connections[key]
            else:
                remaining = True
    return not remaining


def is_held_connection(connection: sqlite3.Connection | None) -> bool:
    """Whether an owned call left this open handle held.

    A caller that used to prove its own handle closed after a finite operation
    asks this instead: a held handle is open on purpose, and a pause, quit or
    owner close retires it.
    """
    if connection is None or not _held_connections:
        return False
    with _HELD_LOCK:
        entry = _held_connections.get(id(connection))
    return entry is not None and entry[0] is connection and not _connection_closed(connection)


def held_connection_count() -> int:
    """Number of owned-call handles currently held open (diagnostics, tests)."""
    with _HELD_LOCK:
        return len(_held_connections)
