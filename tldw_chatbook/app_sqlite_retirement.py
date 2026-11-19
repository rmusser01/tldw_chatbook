"""Explicit app-versus-process native SQLite retirement (ADR225)."""

import sqlite3
import threading
from typing import Any


async def retire_app_sqlite_handles(app: Any) -> None:
    """Retire settled app caches; a handle still in use is left open (ADR225)."""
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Backup_Recovery.participants import (
        close_settled_core_cache,
        core_cached_connection,
        repository_participant,
    )
    from tldw_chatbook.DB.base_db import SqliteRetirementIncomplete
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.Evals_DB import EvalsDB
    from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
    from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Evals.eval_orchestrator import EvaluationOrchestrator

    def read(attribute):
        try:
            return attribute()
        except AttributeError:
            return None  # An app (or double) that never wired this owner.

    evaluations = read(lambda: app.evaluation_orchestrator)
    owners = (
        ("Chat", read(lambda: app.chachanotes_db), CharactersRAGDB, "_local", "conn"),
        (
            "Evaluations",
            getattr(evaluations, "db", None),
            EvalsDB,
            "_local",
            "connection",
        ),
        (
            "Collections",
            read(lambda: app.local_library_collections_db),
            LibraryCollectionsDB,
            "_thread_local",
            "conn",
        ),
        (
            "Subscriptions",
            read(lambda: app.subscriptions_db),
            SubscriptionsDB,
            "_local",
            "conn",
        ),
        (
            "Workspace",
            read(lambda: app.local_workspace_db),
            WorkspaceDB,
            "_thread_local",
            "conn",
        ),
    )
    incomplete = []
    for name, owner, source_type, local_attribute, connection_attribute in owners:
        if type(owner) is not source_type or owner.is_memory_db:
            continue  # Custom/memory owners retain their existing contract.
        try:
            if name == "Evaluations" and type(evaluations) is EvaluationOrchestrator:
                evaluations._maintenance_close_admission()
            cache = getattr(owner, local_attribute)
            connection = core_cached_connection(
                owner, getattr(cache, connection_attribute, None)
            )
            participant = repository_participant(owner)
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
            if name == "Evaluations":
                if type(evaluations) is not EvaluationOrchestrator:
                    incomplete.append(name)
                    continue
                await evaluations.aclose()
            elif name == "Chat":
                # Current-thread handle, not shared owner admission. No explicit
                # TRUNCATE: while other handles hold the WAL it would backfill on
                # the loop (the last close checkpoints anyway).
                owner.close_connection(checkpoint=False)
            elif not close_settled_core_cache(owner):
                incomplete.append(name)
                continue
            if (
                core_cached_connection(
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
        raise SqliteRetirementIncomplete(
            "sqlite_retirement_incomplete:" + ",".join(incomplete)
        )
