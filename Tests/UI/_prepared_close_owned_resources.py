"""Exact resources declared by the isolated prepared-Close fixture."""

import asyncio
import json
import sqlite3
from contextlib import nullcontext
from pathlib import Path
import threading


class PreparedCloseOwnedResources:
    """Retire a fresh fixture runtime before deleting its contained databases."""

    def __init__(self, app, directory, characters_db, explicit_runs=None):
        from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
        from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
        from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

        self.app = app
        self.directory = Path(directory).absolute()
        self.creator = threading.current_thread()
        self.characters_db = characters_db
        self.runtime = getattr(app, "console_runtime", None)
        if (
            type(self.runtime) is not ConsoleRuntime
            or self.runtime.app is not app
            or self.runtime._disposed
            or self.runtime.chat_store is not None
            or self.runtime.chat_controller is not None
            or self.runtime.provider_gateway is not None
            or self.runtime._agent_runs_db is not None
        ):
            raise RuntimeError("prepared_close_runtime_not_fresh_owned")
        if (
            type(characters_db) is not CharactersRAGDB
            or characters_db.is_memory_db
            or Path(characters_db.db_path).absolute() != self.directory / "chats.sqlite"
        ):
            raise RuntimeError("prepared_close_characters_not_declared")
        self.runs = []
        if explicit_runs is not None:
            if (
                type(explicit_runs) is not AgentRunsDB
                or explicit_runs.is_memory_db
                or Path(explicit_runs.db_path).absolute()
                != self.directory / "runs.sqlite"
            ):
                raise RuntimeError("prepared_close_runs_not_declared")
            self.runs.append(explicit_runs)
        self.declarations = [
            (characters_db, CharactersRAGDB, self.directory / "chats.sqlite")
        ]
        if explicit_runs is not None:
            self.declarations.append(
                (explicit_runs, AgentRunsDB, self.directory / "runs.sqlite")
            )
        self.runtime_runs = None
        self.dispose_task = None
        self.runtime_terminal = False

    def adopt_runtime_runs(self):
        """Capture the naturally created sibling during existing preparation."""
        from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

        if threading.current_thread() is not self.creator:
            raise RuntimeError("prepared_close_adoption_wrong_thread")
        current = self.runtime._agent_runs_db
        if current is None:
            return
        if self.runtime_runs is not None:
            if current is not self.runtime_runs:
                raise RuntimeError("prepared_close_runtime_runs_replaced")
            return
        if (
            type(current) is not AgentRunsDB
            or current.is_memory_db
            or Path(current.db_path).absolute() != self.directory / "agent_runs.db"
        ):
            raise RuntimeError("prepared_close_runtime_runs_not_contained")
        self.runtime_runs = current
        self.runs.append(current)
        self.declarations.append(
            (current, AgentRunsDB, self.directory / "agent_runs.db")
        )

    def request_connection_scope(self, runs_db):
        """Retire only a new callback handle belonging to this declared scope."""
        if not any(runs_db is owned for owned in self.runs):
            return nullcontext()
        from tldw_chatbook.DB.base_db import operation_owned_connection

        return operation_owned_connection(runs_db)

    async def dispose_runtime(self):
        """Keep actual disposal owned if its awaiting fixture is cancelled."""
        if threading.current_thread() is not self.creator:
            raise RuntimeError("prepared_close_disposal_wrong_thread")
        if self.runtime._agent_runs_db is not self.runtime_runs:
            raise RuntimeError("prepared_close_runtime_runs_not_retained")
        if self.dispose_task is None:
            self.dispose_task = asyncio.create_task(
                self.runtime.dispose(), name="prepared-close-runtime-dispose"
            )
        try:
            try:
                await asyncio.shield(self.dispose_task)
            except asyncio.CancelledError:
                # The existing API owns its unchanged grace; do not cancel it
                # merely because the fixture's awaiting task was cancelled.
                await asyncio.shield(self.dispose_task)
                raise
        finally:
            self.runtime_terminal = (
                self.dispose_task.done()
                and not self.dispose_task.cancelled()
                and self.dispose_task.exception() is None
                and self.runtime._disposed
            )

    def close_creators(self):
        """Permanently fence only settled exact owners before sandbox deletion."""
        if threading.current_thread() is not self.creator:
            raise RuntimeError("prepared_close_finalization_wrong_thread")
        if not self.runtime_terminal or not self.runtime._disposed:
            raise RuntimeError("prepared_close_runtime_not_disposed")
        from tldw_chatbook.Backup_Recovery import storage_admission as storage
        from tldw_chatbook.Backup_Recovery.participants import (
            _close_settled_core_cache,
            _repository_participant,
        )

        def settled(participant, expected_path):
            # Only this exact declared participant/path is examined under lock.
            return (
                storage._pause is None
                and not participant.connections
                and not participant.retiring_threads
                and not any(
                    operation.participant is participant
                    for operation in storage._operations
                )
                and not any(
                    lease.resource_path == expected_path
                    for lease in storage._live_leases
                )
                and not any(
                    getattr(attempt.operation, "participant", None) is participant
                    for attempt in storage._pending_acquisitions
                )
            )

        def thread_facts(thread):
            # Original stdlib Thread fields only; opaque/custom owners stay unknown.
            stock = type(thread) in {
                threading.Thread,
                threading._MainThread,
                threading._DummyThread,
            }
            values = vars(thread) if stock else {}
            return {
                "object_id": None if thread is None else id(thread),
                "stock_thread": stock,
                "ident": values.get("_ident") if stock else None,
                "native_id": values.get("_native_id") if stock else None,
                "creator_same": thread is self.creator,
                "current_same": thread is threading.current_thread(),
            }

        def census(participant, expected_path, *, detailed=False):
            # Caller already holds the original storage lock. Pre-close is scalar;
            # native descriptors are examined only after the original helper fails.
            path_leases = tuple(
                lease
                for lease in storage._live_leases
                if lease.resource_path == expected_path
            )
            facts = {
                "participant_closed": participant.closed,
                "pause_present": storage._pause is not None,
                "connection_count": len(participant.connections),
                "retiring_thread_count": len(participant.retiring_threads),
                "operation_count": sum(
                    operation.participant is participant
                    for operation in storage._operations
                ),
                "pending_count": sum(
                    getattr(attempt.operation, "participant", None) is participant
                    for attempt in storage._pending_acquisitions
                ),
                "path_lease_count": len(path_leases),
            }
            if detailed:
                rows = []
                for connection, lease in tuple(participant.connections.items())[:16]:
                    # The base C descriptor bypasses every subclass/opaque getter.
                    # Non-SQLite proxies are explicitly unknown, never probed.
                    native = isinstance(connection, sqlite3.Connection)
                    status, transaction = "unknown", None
                    if native:
                        try:
                            transaction = sqlite3.Connection.in_transaction.__get__(
                                connection
                            )
                        except sqlite3.ProgrammingError as error:
                            status = (
                                "closed"
                                if "closed" in str(error).lower()
                                else "unknown"
                            )
                        except Exception:
                            status = "unknown"
                        else:
                            status = "open"
                    rows.append(
                        {
                            "connection_id": id(connection),
                            "lease_id": None if lease is None else id(lease),
                            "lease_live": lease in storage._live_leases,
                            "native_SQLite_descriptor": native,
                            "native_status": status,
                            "in_transaction": transaction,
                            "resource_thread": thread_facts(
                                lease.resource_thread
                                if type(lease) is storage.StorageLease
                                else None
                            ),
                        }
                    )
                facts.update(
                    cached_connections=rows,
                    cached_connection_rows_truncated=len(participant.connections) > 16,
                    retiring_threads=[
                        thread_facts(thread)
                        for thread in tuple(participant.retiring_threads)[:16]
                    ],
                    path_leases=[
                        {
                            "lease_id": id(lease),
                            "registered_connection_ids": [
                                id(connection)
                                for connection, registered in participant.connections.items()
                                if registered is lease
                            ][:16],
                            "resource_thread": thread_facts(
                                lease.resource_thread
                                if type(lease) is storage.StorageLease
                                else None
                            ),
                        }
                        for lease in path_leases[:16]
                    ],
                    path_lease_rows_truncated=len(path_leases) > 16,
                )
            return facts

        participants = []
        for database, expected_type, expected_path in self.declarations:
            if (
                type(database) is not expected_type
                or database.is_memory_db
                or Path(database.db_path).absolute() != expected_path
            ):
                raise RuntimeError("prepared_close_database_owner_changed")
            participant = _repository_participant(database)
            participants.append(participant)
            with storage._lock:
                already_retired = participant.closed and settled(
                    participant, expected_path
                )
                try:
                    pre_close = census(participant, expected_path)
                except Exception:
                    pre_close = (
                        None  # Optional metadata must not prevent the original close.
                    )
            if already_retired:
                continue
            if not _close_settled_core_cache(database):
                failure = RuntimeError("prepared_close_database_not_retired")
                # Failure-only evidence; neither census nor print grants retirement.
                try:
                    with storage._lock:
                        post_close = census(participant, expected_path, detailed=True)
                    failure.add_note(
                        "prepared_close_resource_census: "
                        + json.dumps(
                            {
                                "database_type": expected_type.__name__,
                                "resources": post_close,
                            },
                            sort_keys=True,
                        )
                    )
                    print(
                        json.dumps(
                            {
                                "diagnostic_only": True,
                                "stage": "prepared_close_original_helper_false",
                                "declared_index": len(participants) - 1,
                                "declared_type": expected_type.__name__,
                                "private_path_leaf": expected_path.name,
                                "database_id": id(database),
                                "participant_id": id(participant),
                                "pre_close": pre_close,
                                "post_close": post_close,
                            },
                            sort_keys=True,
                        )
                    )
                except Exception:
                    # Optional facts must not replace the original negative oracle.
                    pass
                raise failure
            with storage._lock:
                participant.close_admission()
                if not settled(participant, expected_path):
                    raise RuntimeError("prepared_close_database_not_retired")
        with storage._lock:
            if (
                storage._pause is not None
                or any(
                    participant.connections or participant.retiring_threads
                    for participant in participants
                )
                or any(
                    operation.participant in participants
                    for operation in storage._operations
                )
                or any(
                    getattr(attempt.operation, "participant", None) in participants
                    for attempt in storage._pending_acquisitions
                )
                or any(
                    isinstance(getattr(lease, "resource_path", None), Path)
                    and lease.resource_path.is_relative_to(self.directory)
                    for lease in storage._live_leases
                )
            ):
                raise RuntimeError("prepared_close_directory_has_live_storage")
