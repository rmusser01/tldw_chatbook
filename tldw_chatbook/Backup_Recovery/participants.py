"""Installed persistence maintenance participants (ADR-126)."""

from contextlib import contextmanager
from functools import wraps
import sqlite3
import sys
import threading
import time
import weakref
from typing import Protocol

from .models import Inventory


class Participant(Protocol):
    owner_id: str

    def close_admission(self) -> None: ...

    def drain(self, deadline: float) -> bool: ...

    def resume(self) -> None: ...


def require_participant_coverage(
    inventory: Inventory, participants: tuple[Participant, ...]
) -> None:
    """Refuse uncovered paths; a capture adapter ID is not a participant.

    Passive sources currently have no exemption. Installed passive-source
    classifications require separate source evidence before relaxing this guard.
    """
    covered = {participant.owner_id for participant in participants}
    required = {item.owner for item in inventory.items if item.path is not None}
    if required - covered:
        raise ValueError("participant_missing")


_installed_repositories = weakref.WeakSet()


class _RepositoryParticipant:
    def __init__(self):
        raise TypeError("repository_participant_is_installed")

    def operation(self):
        from .storage_admission import _repository_operation

        return _repository_operation(self)

    def close_admission(self) -> None:
        from . import storage_admission as storage

        with storage._lock:
            self.closed = True

    def drain(self, deadline: float) -> bool:
        from . import storage_admission as storage

        with storage._changed:
            if not self.closed:
                raise RuntimeError("participant_admission_not_closed")
            while (
                self.connections
                or self.retiring_threads
                or any(
                    operation.participant is self for operation in storage._operations
                )
                or storage._pending_acquisitions
                or storage._retiring_holds
                or any(
                    lease.resource_path == self.path for lease in storage._live_leases
                )
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                storage._changed.wait(min(remaining, 0.05))
            # Unknown pending/retiring scopes conservatively block this cohort.
            # Neither a finished method nor zero operations retires a handle.
            return True

    def resume(self) -> None:
        from . import storage_admission as storage

        with storage._lock:
            if storage._pause is not None:
                raise RuntimeError("process_pause_still_active")
            self.closed = False


def _repository_types():
    """Exact source-backed runtime declarations; subclasses confer no authority."""
    from tldw_chatbook.Notifications.event_state_repository import EventStateRepository
    from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository

    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
    from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
    from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
    from tldw_chatbook.DB.Library_Ingest_Jobs_DB import LibraryIngestJobsDB

    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.Notifications.client_notifications_db import (
        ClientNotificationsDB,
    )
    from tldw_chatbook.Scheduling.db.scheduled_tasks_db import ScheduledTasksDB
    from tldw_chatbook.Research_Interop.local_research_service import (
        LocalResearchService,
    )
    from tldw_chatbook.Writing_Interop.local_writing_service import LocalWritingService

    # These source modules are discovered only after their own caller loads
    # them. Native getter lookup must not import configuration/encryption or
    # optional runtime stacks just to compare an unrelated source's exact type.
    optional = {}
    for module_name, class_name, owner_id in (
        ("tldw_chatbook.DB.Evals_DB", "EvalsDB", "db.evals"),
        ("tldw_chatbook.DB.Subscriptions_DB", "SubscriptionsDB", "db.subscriptions"),
        (
            "tldw_chatbook.Notes.file_notes_replica",
            "FileNotesReplica",
            "notes.file_notes",
        ),
        (
            "tldw_chatbook.Notes.note_import_receipts",
            "NoteImportReceiptRepository",
            "notes.sync_state",
        ),
        (
            "tldw_chatbook.Kanban_Interop.local_kanban_service",
            "LocalKanbanService",
            "kanban.local",
        ),
        (
            "tldw_chatbook.Subscriptions.site_config_manager",
            "SiteConfigManager",
            "db.subscriptions.site_configs",
        ),
    ):
        source_type = getattr(sys.modules.get(module_name), class_name, None)
        if source_type is not None:
            optional[source_type] = owner_id
    return {
        **optional,
        WorkspaceDB: "db.workspaces",
        AgentRunsDB: "db.agent_runs",
        ClientNotificationsDB: "notifications.client",
        ScheduledTasksDB: "db.scheduled_tasks",
        LocalResearchService: "research.local",
        LocalWritingService: "writing.local",
        CharactersRAGDB: "db.chachanotes.primary",
        MediaDatabase: "db.media.primary",
        PromptsDatabase: "db.prompts.primary",
        LibraryCollectionsDB: "db.library_collections",
        LibraryIngestJobsDB: "db.library_ingest_jobs",
        EventStateRepository: "runtime.event_state",
        SyncStateRepository: "runtime.sync_state",
    }


def _repository_participant(repository):
    """Bind exact installed repository types to their actual selected paths."""
    from . import storage_admission as storage

    types = _repository_types()
    if (
        type(repository) not in types
        or repository.is_memory_db
        or repository.db_path is None
    ):
        raise ValueError("repository_participant_not_installed")
    with storage._lock:
        participant = getattr(repository, "_maintenance_participant", None)
        if participant is None:
            participant = object.__new__(_RepositoryParticipant)
            participant.repository = weakref.ref(repository)
            participant.owner_id = types[type(repository)]
            participant.path = repository.db_path
            participant.read_only = getattr(repository, "_read_only", False)
            participant.closed = False
            participant.connections = {}
            participant.retiring_threads = set()
            _installed_repositories.add(participant)
            repository._maintenance_participant = participant
        if (
            participant not in _installed_repositories
            or participant.repository() is not repository
            or participant.path != repository.db_path
            or participant.read_only != getattr(repository, "_read_only", False)
        ):
            raise ValueError("repository_participant_not_installed")
        return participant


def _core_access(repository):
    """Refuse a new cached-handle borrower while local admission is closed."""
    if repository.is_memory_db:
        return
    from . import storage_admission as storage
    from .bootstrap import RecoveryRequired

    participant = (
        _repository_participant(repository)
        if type(repository) in _repository_types()
        else None
    )
    with storage._lock:
        if participant is not None:
            _check_core_retirement(participant)
        operation = getattr(storage._operation_local, "operation", None)
        if operation is not None:
            if participant is None:
                raise RecoveryRequired("repository_participant_not_installed")
            storage._check_operation(operation, participant.path)
            if operation.participant is not participant:
                raise RecoveryRequired("operation_provenance_invalid")
        elif (
            participant is not None and participant.closed
        ) or storage._pause is not None:
            raise RecoveryRequired("storage_locally_paused")


@contextmanager
def _core_operation(repository):
    """Count an installed core scope without revoking its native borrowers."""
    if repository.is_memory_db or type(repository) not in _repository_types():
        _core_access(repository)
        yield
    else:
        with _repository_participant(repository).operation():
            yield


def _core_transaction(function):
    @wraps(function)
    @contextmanager
    def counted(repository, *args, **kwargs):
        with _core_operation(repository):
            with function(repository, *args, **kwargs) as native:
                yield native

    return counted


def _check_core_retirement(participant):
    from .bootstrap import RecoveryRequired

    if threading.current_thread() in participant.retiring_threads or (
        participant.owner_id in {"db.library_ingest_jobs", "notes.file_notes"}
        and participant.retiring_threads
    ):
        raise RecoveryRequired("core_connection_retiring")


@contextmanager
def _core_closing(repository, connection):
    """Reserve a source cache before explicit native close, outside coordinator locks.

    This is not permission to revoke raw borrowers: the actual owner must call it
    only at its explicit close boundary. Live managed work or paused borrowed
    transactions cannot be discarded, and foreign-thread close is refused.
    """
    if repository.is_memory_db or type(repository) not in _repository_types():
        yield True
        return
    from . import storage_admission as storage

    participant = _repository_participant(repository)
    current = threading.current_thread()
    with storage._lock:
        lease = participant.connections.get(connection)
        allowed = not (
            current in participant.retiring_threads
            or (lease is not None and lease.resource_thread is not current)
            or any(
                operation.participant is participant
                and (
                    participant.owner_id
                    in {"db.library_ingest_jobs", "notes.file_notes"}
                    or lease is None
                    or operation.thread is lease.resource_thread
                )
                for operation in storage._operations
            )
        )
        reserved = allowed
        if reserved:
            participant.retiring_threads.add(current)
    try:
        if allowed and (storage._pause is not None or participant.closed):
            try:
                allowed = not connection.in_transaction
            except sqlite3.ProgrammingError:
                pass  # An already-closed cache may be explicitly cleared.
        yield allowed
    finally:
        with storage._changed:
            # Only the caller that reserved this close can retire the reservation.
            if reserved:
                participant.retiring_threads.discard(current)
            storage._changed.notify_all()


def _register_core_connection(repository, connection):
    """Retain actual core native handles; caller names never confer authority."""
    if repository.is_memory_db or type(repository) not in _repository_types():
        return connection
    from . import storage_admission as storage
    from .bootstrap import RecoveryRequired
    from tldw_chatbook.DB.private_sqlite import (
        _ordinary_connections,
        _validated_owner_policy,
    )

    participant = _repository_participant(repository)
    with storage._lock:
        # Unwrapped native objects (including non-weakrefable Connection) are
        # never installed core provenance. Their ordinary leases still count.
        try:
            lease = _ordinary_connections.get(connection)
        except TypeError:
            lease = None
        policy_id = (
            "db.base"
            if participant.owner_id
            in {
                "db.library_collections",
                "db.workspaces",
                "db.agent_runs",
                "notifications.client",
                "db.scheduled_tasks",
                "runtime.event_state",
                "runtime.sync_state",
            }
            else participant.owner_id
        )
        if participant.owner_id == "notes.file_notes":
            policy_id = "notes.file_notes_replica"
        if participant.owner_id == "db.subscriptions":
            policy_id = (
                "db.subscriptions.agent_read" if participant.read_only else "db.base"
            )
        if (
            lease is None
            or lease not in storage._live_leases
            or lease.resource_policy is not _validated_owner_policy(policy_id)
            or lease.resource_path != participant.path
            or lease.resource_thread is not threading.current_thread()
            or getattr(lease, "resource_participant", participant) is not participant
        ):
            raise RecoveryRequired("core_connection_provenance_invalid")
        participant.connections[connection] = lease
        lease.resource_participant = participant
    return connection


_retired_core_connections = weakref.WeakSet()


def _core_cached_connection(repository, connection):
    """Invalidate a cache only after the wrapper observed successful native close."""
    if repository.is_memory_db or type(repository) not in _repository_types():
        return connection
    from . import storage_admission as storage

    with storage._lock:
        if connection in _retired_core_connections:
            return None
    return connection


def _core_getter(function):
    """A cross-instance raw getter cannot inherit another owner's operation."""

    @wraps(function)
    def accessed(repository, *args, **kwargs):
        from . import storage_admission as storage

        previous = getattr(storage._operation_local, "operation", None)
        if previous is not None and not repository.is_memory_db:
            with storage._lock:
                storage._check_operation(previous, previous.path)
                participant = _repository_participant(repository)
                independent = previous.participant is not participant
            if independent:
                with participant.operation():
                    return function(repository, *args, **kwargs)
        return function(repository, *args, **kwargs)

    return accessed
