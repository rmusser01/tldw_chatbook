"""User-facing Sync v2 conflict review contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


RECOVERY_ACTIONS: tuple[str, ...] = (
    "retry",
    "keep-local",
    "accept-remote",
    "duplicate-fork",
    "defer-later",
)


@dataclass(frozen=True, slots=True)
class SyncV2ConflictReviewItem:
    """Safe conflict row shown to users without exposing encrypted payload text."""

    domain: str
    item_label: str
    cause: str
    local_summary: str
    remote_summary: str
    recovery_options: dict[str, str]
    conflict_review_id: int | None = None
    resolution_status: str = "open"


class SyncV2ConflictReviewService:
    """Build actionable conflict and retained-failure rows for Sync v2 users."""

    def __init__(self, *, state_repository: Any) -> None:
        self.state_repository = state_repository

    def build_review_items(
        self,
        *,
        server_profile_id: str,
        authenticated_principal_id: str | None,
        workspace_scope: str | None,
        dataset_id: str,
        domains: Sequence[str] | None = None,
    ) -> tuple[SyncV2ConflictReviewItem, ...]:
        """Return durable conflict rows plus retained outbox failure rows."""

        domain_filter = {str(domain) for domain in domains or () if str(domain)}
        rows = self.state_repository.list_sync_v2_conflict_reviews(
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
            dataset_id=dataset_id,
            domains=list(domain_filter) or None,
            resolution_status="open",
        )
        review_items = [self._from_review_row(row) for row in rows]
        retained = self.state_repository.list_pending_sync_v2_outbox_envelopes(
            server_profile_id=server_profile_id,
            authenticated_principal_id=authenticated_principal_id,
            workspace_scope=workspace_scope,
            dataset_id=dataset_id,
            domains=list(domain_filter) or None,
        )
        review_items.extend(
            self._from_retained_outbox_entry(entry)
            for entry in retained
            if entry.get("last_error")
        )
        return tuple(review_items)

    @staticmethod
    def _from_review_row(row: Mapping[str, Any]) -> SyncV2ConflictReviewItem:
        return SyncV2ConflictReviewItem(
            conflict_review_id=int(row["conflict_review_id"]),
            domain=str(row["domain"]),
            item_label=str(row["item_label"]),
            cause=str(row["cause"]),
            local_summary=str(row["local_summary"]),
            remote_summary=str(row["remote_summary"]),
            recovery_options=_normalize_recovery_options(row.get("recovery_options")),
            resolution_status=str(row.get("resolution_status") or "open"),
        )

    @staticmethod
    def _from_retained_outbox_entry(entry: Mapping[str, Any]) -> SyncV2ConflictReviewItem:
        envelope = entry.get("envelope") if isinstance(entry.get("envelope"), Mapping) else {}
        last_error = entry.get("last_error") if isinstance(entry.get("last_error"), Mapping) else {}
        error_code = str(last_error.get("error_code") or "push_failed")
        message = str(last_error.get("message") or "Outgoing change was retained.")
        domain = str(entry.get("domain") or envelope.get("domain") or "sync")
        entity_id = str(envelope.get("entity_id") or entry.get("client_envelope_id") or "pending")
        return SyncV2ConflictReviewItem(
            domain=domain,
            item_label=f"{domain} {entity_id}",
            cause=f"{error_code}: {message}",
            local_summary=f"Local pending {domain} change retained for retry.",
            remote_summary="Remote state unavailable until retry or conflict review.",
            recovery_options={
                "retry": "available",
                "keep-local": "unavailable",
                "accept-remote": "unavailable",
                "duplicate-fork": "unavailable",
                "defer-later": "available",
            },
        )


def _normalize_recovery_options(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {action: "unavailable" for action in RECOVERY_ACTIONS}
    return {
        action: str(value.get(action) or "unavailable")
        for action in RECOVERY_ACTIONS
    }
