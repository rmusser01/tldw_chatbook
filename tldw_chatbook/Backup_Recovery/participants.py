"""Installed persistence maintenance participants (ADR-126)."""

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
                any(operation.participant is self for operation in storage._operations)
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


def _repository_participant(repository):
    """Bind only the actual first cohort's installed types and selected paths."""
    from tldw_chatbook.Notifications.event_state_repository import EventStateRepository
    from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository
    from . import storage_admission as storage

    types = {
        EventStateRepository: "runtime.event_state",
        SyncStateRepository: "runtime.sync_state",
    }
    if type(repository) not in types or repository.is_memory_db:
        raise ValueError("repository_participant_not_installed")
    with storage._lock:
        participant = getattr(repository, "_maintenance_participant", None)
        if participant is None:
            participant = object.__new__(_RepositoryParticipant)
            participant.repository = weakref.ref(repository)
            participant.owner_id = types[type(repository)]
            participant.path = repository.db_path
            participant.closed = False
            _installed_repositories.add(participant)
            repository._maintenance_participant = participant
        if (
            participant not in _installed_repositories
            or participant.repository() is not repository
        ):
            raise ValueError("repository_participant_not_installed")
        return participant
