"""Bounded lifecycle API for the encrypted Personal Context Sync outbox."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .repository import PersonalContextRepository


@dataclass(frozen=True, slots=True, repr=False)
class ProfileSyncOutboxEntry:
    """Content-free routing metadata for one encrypted pending body."""

    outbox_id: str
    object_type: str
    object_id: str
    version_id: str
    status: str
    created_at: str

    def __repr__(self) -> str:
        return f"ProfileSyncOutboxEntry(outbox_id={self.outbox_id!r}, status={self.status!r})"


class ProfileSyncOutbox:
    """Read, acknowledge, or quarantine encrypted profile journal entries."""

    def __init__(self, repository: PersonalContextRepository) -> None:
        self.repository = repository

    def list_pending(self, *, limit: int = 100) -> tuple[ProfileSyncOutboxEntry, ...]:
        return tuple(
            ProfileSyncOutboxEntry(**row)
            for row in self.repository.list_pending_outbox(limit=limit)
        )

    def list_dispatchable(
        self,
        *,
        limit: int = 100,
    ) -> tuple[ProfileSyncOutboxEntry, ...]:
        """List entries eligible under current global/workspace bindings."""

        return tuple(
            ProfileSyncOutboxEntry(**row)
            for row in self.repository.list_dispatchable_outbox(limit=limit)
        )

    def read_body(self, outbox_id: str) -> dict[str, Any] | None:
        return self.repository.get_outbox_body(outbox_id)

    def acknowledge(self, outbox_id: str, destination_envelope_id: str) -> None:
        self.repository.acknowledge_outbox(outbox_id, destination_envelope_id)

    def quarantine(
        self,
        outbox_id: str,
        reason_code: str,
        *,
        preserve_body: bool = False,
    ) -> None:
        self.repository.quarantine_outbox(
            outbox_id,
            reason_code,
            preserve_body=preserve_body,
        )

    def get_receipt(self, outbox_id: str) -> str | None:
        return self.repository.get_outbox_receipt(outbox_id)

    def get_quarantine_reason(self, outbox_id: str) -> str | None:
        return self.repository.get_outbox_quarantine_reason(outbox_id)


__all__ = ["ProfileSyncOutbox", "ProfileSyncOutboxEntry"]
