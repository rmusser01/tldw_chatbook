"""Explicit app-versus-process native SQLite retirement (ADR225)."""

import sqlite3
import threading
from typing import Any


def capture_chat_borrower(notes_service: Any) -> tuple[Any, sqlite3.Connection | None]:
    """Read the exact injected cache before app composition can replace it."""
    owner = getattr(notes_service, "db", None)
    return owner, getattr(getattr(owner, "_local", None), "conn", None)


async def retire_app_sqlite_handles(app: Any) -> None:
    """Retire settled app caches, preserving native borrowers (ADR225)."""
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Backup_Recovery.participants import (
        _close_settled_core_cache,
        _core_cached_connection,
        _repository_participant,
    )
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.Evals_DB import EvalsDB
    from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
    from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Evals.eval_orchestrator import EvaluationOrchestrator

    evaluations = getattr(app, "evaluation_orchestrator", None)
    owners = (
        (
            "Chat",
            getattr(app, "chachanotes_db", None),
            CharactersRAGDB,
            "_local",
            "conn",
        ),
        (
            "Evaluations",
            getattr(evaluations, "db", None),
            EvalsDB,
            "_local",
            "connection",
        ),
        (
            "Collections",
            getattr(app, "local_library_collections_db", None),
            LibraryCollectionsDB,
            "_thread_local",
            "conn",
        ),
        (
            "Subscriptions",
            getattr(app, "subscriptions_db", None),
            SubscriptionsDB,
            "_local",
            "conn",
        ),
        (
            "Workspace",
            getattr(app, "local_workspace_db", None),
            WorkspaceDB,
            "_thread_local",
            "conn",
        ),
    )
    borrowed_owner, borrowed_connection = getattr(
        app, "_chat_sqlite_borrowed", (None, None)
    )
    incomplete = []
    for name, owner, source_type, local_attribute, connection_attribute in owners:
        if type(owner) is not source_type or owner.is_memory_db:
            continue  # Custom/memory owners retain their existing contract.
        if name == "Evaluations" and type(evaluations) is EvaluationOrchestrator:
            evaluations._maintenance_close_admission()
        cache = getattr(owner, local_attribute)
        connection = _core_cached_connection(
            owner, getattr(cache, connection_attribute, None)
        )
        if (
            name == "Chat"
            and owner is borrowed_owner
            and connection is borrowed_connection
            and connection is not None
        ):
            continue
        participant = _repository_participant(owner)
        current_thread = threading.current_thread()
        with storage._lock:
            busy = (
                storage._pause is not None
                or participant.closed
                or (
                    current_thread in participant.retiring_threads
                    if name == "Chat"
                    else bool(participant.retiring_threads)
                )
                or any(
                    operation.participant is participant
                    and (name != "Chat" or operation.thread is current_thread)
                    for operation in storage._operations
                )
                or (
                    name not in ("Chat", "Evaluations")
                    and any(
                        lease.resource_thread is not current_thread
                        for lease in participant.connections.values()
                    )
                )
            )
        if connection is not None:
            try:
                busy = busy or connection.in_transaction
            except sqlite3.ProgrammingError:
                pass  # The source close can clear an already-retired cache.
            if isinstance(connection, sqlite3.Connection) and hasattr(
                connection, "_quiescence_tokens_lock"
            ):
                with connection._quiescence_tokens_lock:
                    busy = busy or bool(connection._quiescence_tokens)
        if busy:
            incomplete.append(name)
            continue
        try:
            if name == "Evaluations":
                if type(evaluations) is not EvaluationOrchestrator:
                    incomplete.append(name)
                    continue
                await evaluations.aclose()
            elif name == "Chat":
                owner.close_connection()  # Current-thread handle, not shared owner admission.
            elif not _close_settled_core_cache(owner):
                incomplete.append(name)
                continue
            if (
                _core_cached_connection(
                    owner, getattr(cache, connection_attribute, None)
                )
                is not None
            ):
                incomplete.append(name)
            elif name == "Evaluations":
                with storage._lock:
                    if participant.connections:
                        incomplete.append(name)
        except Exception as error:  # noqa: BLE001 - settle independent owners; report categories only.
            incomplete.append(f"{name}:{type(error).__name__}")
    if incomplete:
        raise RuntimeError("sqlite_retirement_incomplete:" + ",".join(incomplete))
