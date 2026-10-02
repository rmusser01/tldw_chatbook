"""Review-bound revision drain using the shared lifecycle owner's work records."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from uuid import uuid4

from .review import OperationReceipt
from .revocation import RevocationOperation, RevocationTarget


def preserve_selection(
    old_selected: frozenset[str], available: frozenset[str]
) -> frozenset[str]:
    """New or newly supported capabilities never enter the existing maximum."""
    return old_selected & available


@dataclass
class RevisionTicket:
    installation_id: str
    revision_digest: str
    token: str
    expires_at: float
    review_token: str | None = None
    phase: str = "reserved"
    records: tuple = ()
    cleanup: RevocationOperation | None = None
    task: asyncio.Task | None = None
    operation_id: str | None = None
    unresolved: tuple[str, ...] = ()
    ownership_pending: bool = True


class RevisionDrain:
    """Close fresh admission without revoking the authority of admitted work.

    All tickets and run custody live under the existing LivePluginFences lock.
    The storage owner supplies exact unsettled-token inventory before commitment.
    Idle transport adapters may register close callbacks through ordinary retained
    work ownership; absence of active requests alone never proves process cleanup.
    """

    def __init__(self, fences, inventory=None):
        self.fences = fences
        self._inventory = inventory

    def reserve(
        self,
        installation_id: str,
        revision_digest: str,
        *,
        review_token: str,
        expires_at: float,
    ) -> str:
        with self.fences.live_lock:
            token = uuid4().hex
            self.fences.drains[token] = RevisionTicket(
                installation_id, revision_digest, token, expires_at, review_token
            )
            return token

    def begin(self, installation_id: str, revision_digest: str) -> str:
        token = self.reserve(
            installation_id,
            revision_digest,
            review_token=None,
            expires_at=time.monotonic() + 900,
        )
        self.activate(token)
        return token

    def ticket(self, token: str) -> RevisionTicket:
        try:
            return self.fences.drains[token]
        except KeyError:
            raise ValueError("plugin_drain_unavailable") from None

    def activate(self, token: str, *, review=None, operation_id=None) -> RevisionTicket:
        with self.fences.live_lock:
            ticket = self.ticket(token)
            if ticket.review_token is not None and (
                review is None
                or ticket.review_token != review.token
                or ticket.installation_id != review.installation_id
                or ticket.revision_digest != review.previous_revision
            ):
                raise ValueError("plugin_drain_review_changed")
            if ticket.operation_id is not None and operation_id != ticket.operation_id:
                raise ValueError("plugin_drain_operation_changed")
            if ticket.phase in {"cancelled", "expired"}:
                raise ValueError("plugin_drain_cancelled_or_expired")
            if ticket.phase != "reserved":
                return ticket
            if time.monotonic() >= ticket.expires_at:
                ticket.phase = "expired"
                raise ValueError("plugin_drain_expired")
            ticket.operation_id = operation_id
            ticket.records = tuple(
                record
                for record in self.fences.runs.values()
                if record.installation_id == ticket.installation_id
                and record.revision_digest == ticket.revision_digest
            )
            ticket.phase = "waiting"
            return ticket

    def blockers(self, token: str) -> tuple[dict, ...]:
        with self.fences.live_lock:
            ticket = self.ticket(token)
            rows = [
                {
                    "run_id": record.run_id,
                    "handle_id": record.handle_id,
                    "workspace_id": record.workspace_id,
                    "lease_token": record.lease_token,
                    "kind": "active_work",
                }
                for record in ticket.records
                if not record.completed.is_set()
            ]
            rows.extend(
                {"lease_token": value, "kind": "unresolved_cleanup"}
                for value in ticket.unresolved
            )
            if ticket.ownership_pending and ticket.phase in {"waiting", "committing"}:
                rows.append({"kind": "ownership_inventory_pending"})
            if ticket.cleanup and ticket.cleanup.cleanup_tasks:
                rows.append({"kind": "cleanup_callback"})
            return tuple(rows)

    def record_inventory(self, token: str, unsettled: tuple[str, ...]) -> None:
        """Publish a worker-observed inventory without performing I/O under the fence."""
        with self.fences.live_lock:
            ticket = self.ticket(token)
            active = {
                record.lease_token
                for record in ticket.records
                if not record.completed.is_set()
            }
            ticket.unresolved = tuple(
                value for value in unsettled if value not in active
            )
            ticket.ownership_pending = False

    def cancel(self, token: str) -> None:
        """Cancel this proposal only; never cancel work or release other tickets."""
        with self.fences.live_lock:
            ticket = self.ticket(token)
            if ticket.phase in {"committing", "complete"}:
                raise ValueError("plugin_drain_already_committing")
            ticket.phase = "cancelled"

    def cancel_work(self, token: str) -> None:
        """Explicitly transfer the captured callbacks; terminal evidence stays owned."""
        with self.fences.live_lock:
            ticket = self.ticket(token)
            if ticket.phase != "waiting":
                raise ValueError("plugin_drain_not_waiting")
            if ticket.cleanup is None:
                ticket.cleanup = RevocationOperation(
                    RevocationTarget(ticket.installation_id, None, True),
                    "revoke",
                    token,
                    ticket.records,
                    OperationReceipt(token, "session_only", False),
                )
                ticket.cleanup.cancel_owned()

    async def wait(self, token: str) -> None:
        """Wait without holding the live lock; waiter cancellation owns no cleanup."""
        while True:
            with self.fences.live_lock:
                ticket = self.ticket(token)
                if ticket.phase in {"cancelled", "expired", "reserved"}:
                    raise ValueError("plugin_drain_not_waiting")
            if self._inventory is None:
                raise RuntimeError("plugin_drain_owner_inventory_unavailable")
            # The synchronous fence is already closed. Storage stalls cannot
            # prevent another live disable/cancel from acquiring its lock.
            unsettled = await self._inventory(ticket.installation_id)
            self.record_inventory(token, unsettled)
            with self.fences.live_lock:
                ticket = self.ticket(token)
                if ticket.phase in {"cancelled", "expired"}:
                    raise ValueError("plugin_drain_cancelled_or_expired")
                if ticket.phase == "reserved":
                    raise ValueError("plugin_drain_not_started")
                if not self.blockers(token):
                    return
            await asyncio.sleep(0.01)
