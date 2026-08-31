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
    conflict_review_id: int | str | None = None
    resolution_status: str = "open"


class SyncV2ConflictReviewService:
    """Build actionable conflict and retained-failure rows for Sync v2 users."""

    def __init__(
        self,
        *,
        state_repository: Any,
        notes_repository: Any | None = None,
        notes_organization_sync_service: Any | None = None,
    ) -> None:
        self.state_repository = state_repository
        self.notes_repository = notes_repository
        self.notes_organization_sync_service = notes_organization_sync_service

    def build_notes_organization_adoption_items(
        self, *, dataset_id: str
    ) -> tuple[SyncV2ConflictReviewItem, ...]:
        """Return content-free Notes-owned adoption rows through this review seam."""

        if self.notes_repository is None:
            return ()
        rows = (
            self.notes_repository.db.get_connection()
            .execute(
                "SELECT review_id, domain, display_name, portable_path, state "
                "FROM notes_organization_adoption_reviews WHERE server_profile_id = ? "
                "AND dataset_id = ? AND state = 'open' ORDER BY created_at, review_id",
                (self.notes_repository.server_profile_id, dataset_id),
            )
            .fetchall()
        )
        return tuple(
            SyncV2ConflictReviewItem(
                conflict_review_id=str(row["review_id"]),
                domain=str(row["domain"]),
                item_label=str(row["display_name"]),
                cause="A local organization item collides with a different server identity.",
                local_summary=(
                    f"Local path: {row['portable_path']}"
                    if row["portable_path"]
                    else "Local organization item requires an adoption decision."
                ),
                remote_summary="Server identity remains separate until review.",
                recovery_options={
                    "merge": "available",
                    "rename_local": "available",
                    "keep_local": "available",
                },
                resolution_status=str(row["state"]),
            )
            for row in rows
        )

    def resolve_notes_organization_adoption(
        self, *, review_id: str, action: str, new_name: str | None = None
    ) -> bool:
        """Resolve one Notes-owned adoption row once without content exposure."""

        if self.notes_repository is None:
            raise ValueError("Notes organization repository is required")
        if action not in {"merge", "rename_local", "keep_local"}:
            raise ValueError("Unsupported Notes organization adoption action")
        linked_receipt = (
            self.notes_repository.db.get_connection()
            .execute(
                "SELECT 1 FROM note_organization_receipts WHERE review_id = ? "
                "AND state = 'placement_review'",
                (review_id,),
            )
            .fetchone()
        )
        if linked_receipt is not None:
            if self.notes_organization_sync_service is None:
                raise ValueError("Notes organization finalizer is required")
            return bool(
                self.notes_organization_sync_service.resolve_placement_review(
                    review_id=review_id,
                    action=action,
                    new_name=new_name,
                )
            )
        from datetime import UTC, datetime

        now = datetime.now(UTC).isoformat()
        with self.notes_repository.db.transaction() as cursor:
            row = cursor.execute(
                "SELECT domain, local_object_id, remote_object_id "
                "FROM notes_organization_adoption_reviews WHERE review_id = ? "
                "AND server_profile_id = ? AND state = 'open'",
                (review_id, self.notes_repository.server_profile_id),
            ).fetchone()
            if row is None:
                return False
            if action == "merge":
                self._merge_notes_organization_identity(cursor, row)
            elif action == "rename_local":
                self._rename_notes_organization_resource(cursor, row, new_name)
            result = cursor.execute(
                "UPDATE notes_organization_adoption_reviews SET state = 'resolved', "
                "resolution = ?, resolved_at = ?, updated_at = ? WHERE review_id = ? "
                "AND server_profile_id = ? AND state = 'open'",
                (
                    action,
                    now,
                    now,
                    review_id,
                    self.notes_repository.server_profile_id,
                ),
            )
        return result.rowcount == 1

    def _merge_notes_organization_identity(self, cursor: Any, review: Any) -> None:
        table = {
            "notes.keyword": "keywords",
            "notes.keyword_collection": "keyword_collections",
            "notes.folder": "note_folders",
        }.get(str(review["domain"]))
        if table is None:
            raise ValueError("Unsupported Notes organization resource domain")
        cursor.execute(
            f"UPDATE {table} SET sync_id = ? WHERE id = ?",
            (str(review["remote_object_id"]), review["local_object_id"]),
        )

    def _rename_notes_organization_resource(
        self, cursor: Any, review: Any, new_name: str | None
    ) -> None:
        from tldw_chatbook.Notes.notes_organization_repository import (
            portable_collision_key,
        )

        domain = str(review["domain"])
        maximum = {
            "notes.keyword": 100,
            "notes.keyword_collection": 255,
            "notes.folder": 500,
        }.get(domain)
        if maximum is None:
            raise ValueError("Unsupported Notes organization resource domain")
        if new_name is None:
            raise ValueError("rename_local requires a new_name")
        portable_collision_key(new_name, maximum=maximum)
        if self.notes_repository is None:  # pragma: no cover - guarded by caller
            raise ValueError("Notes organization repository is required")
        notes_repository = self.notes_repository
        local_id = review["local_object_id"]
        if domain == "notes.keyword":
            row = cursor.execute(
                "SELECT version FROM keywords WHERE id = ?", (local_id,)
            ).fetchone()
            if row is None:
                raise ValueError("Notes organization local resource is missing")
            notes_repository.db.update_keyword(
                int(local_id), new_name, int(row["version"]), cursor=cursor
            )
            return
        if domain == "notes.keyword_collection":
            row = cursor.execute(
                "SELECT version FROM keyword_collections WHERE id = ?", (local_id,)
            ).fetchone()
            if row is None:
                raise ValueError("Notes organization local resource is missing")
            notes_repository.db.update_keyword_collection(
                int(local_id), {"name": new_name}, int(row["version"]), cursor=cursor
            )
            return
        from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository

        row = cursor.execute(
            "SELECT version FROM note_folders WHERE id = ?", (local_id,)
        ).fetchone()
        if row is None:
            raise ValueError("Notes organization local resource is missing")
        LocalNoteFolderRepository(notes_repository.db).rename_folder(
            str(local_id),
            name=new_name,
            expected_version=int(row["version"]),
            cursor=cursor,
        )

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
        durable_source_keys = {
            str(row.get("source_conflict_key"))
            for row in rows
            if row.get("source_conflict_key")
        }
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
            for entry in sorted(retained, key=_retained_outbox_sort_key, reverse=True)
            if entry.get("last_error")
            and str(entry.get("client_envelope_id") or "") not in durable_source_keys
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
    def _from_retained_outbox_entry(
        entry: Mapping[str, Any],
    ) -> SyncV2ConflictReviewItem:
        envelope = (
            entry.get("envelope") if isinstance(entry.get("envelope"), Mapping) else {}
        )
        last_error = (
            entry.get("last_error")
            if isinstance(entry.get("last_error"), Mapping)
            else {}
        )
        error_code = str(last_error.get("error_code") or "push_failed")
        message = str(last_error.get("message") or "Outgoing change was retained.")
        domain = str(entry.get("domain") or envelope.get("domain") or "sync")
        entity_id = str(
            envelope.get("entity_id") or entry.get("client_envelope_id") or "pending"
        )
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
        action: str(value.get(action) or "unavailable") for action in RECOVERY_ACTIONS
    }


def _retained_outbox_sort_key(entry: Mapping[str, Any]) -> str:
    return str(entry.get("updated_at") or entry.get("created_at") or "")
