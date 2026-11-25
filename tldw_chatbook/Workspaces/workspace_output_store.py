"""Canonical workspace membership queries for Research output provenance."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from dataclasses import replace
from datetime import timedelta
from typing import TYPE_CHECKING

from pydantic import ValidationError

from .models import WorkspaceMembership
from .output_membership import (
    OutputCopyProvenance,
    encode_output_membership_metadata,
    output_membership_metadata,
)

if TYPE_CHECKING:
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB


class WorkspaceOutputMemberships:
    """Persist output links through the workspace registry's existing SQLite connection."""

    def __init__(
        self,
        db: WorkspaceDB,
        now_factory: Callable[[], str],
        bump_mutation_generation: Callable[[], None],
    ) -> None:
        self.db = db
        self._now_factory = now_factory
        self._bump_mutation_generation = bump_mutation_generation

    def upsert_membership_metadata(
        self,
        membership: WorkspaceMembership,
        *,
        expected_updated_at: str | None = None,
    ) -> WorkspaceMembership:
        """Write provenance to an existing output link using its captured timestamp."""
        from .registry_service import (
            WorkspaceMembershipConflict,
            WorkspaceRegistryServiceError,
            _STORAGE_FAILURE_MESSAGE,
            _membership_from_row,
            _parse_utc_timestamp,
        )

        metadata = output_membership_metadata(membership.metadata, membership.role)
        if not membership.membership_id or not metadata:
            raise ValueError(
                "Link the canonical output before saving generation provenance"
            )
        try:
            with self.db.transaction(immediate=True) as conn:
                row = conn.execute(
                    """
                    SELECT * FROM workspace_memberships
                    WHERE membership_id = ? AND workspace_id = ? AND item_type = ?
                      AND item_id = ? AND role = 'output'
                    """,
                    (
                        membership.membership_id,
                        membership.workspace_id,
                        membership.item_type,
                        membership.item_id,
                    ),
                ).fetchone()
                if row is None:
                    raise WorkspaceMembershipConflict()
                current = _membership_from_row(row)
                if current.metadata == metadata:
                    return current
                if current.updated_at != expected_updated_at:
                    raise WorkspaceMembershipConflict()
                timestamp = _parse_utc_timestamp(self._now_factory())
                if current.updated_at:
                    timestamp = max(
                        timestamp,
                        _parse_utc_timestamp(current.updated_at)
                        + timedelta(microseconds=1),
                    )
                updated_at = timestamp.isoformat()
                conn.execute(
                    "UPDATE workspace_memberships SET metadata_json = ?, updated_at = ? WHERE membership_id = ?",
                    (
                        encode_output_membership_metadata(metadata),
                        updated_at,
                        membership.membership_id,
                    ),
                )
                saved = replace(current, metadata=metadata, updated_at=updated_at)
        except sqlite3.Error as exc:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        self._bump_mutation_generation()
        return saved

    def list_workspace_artifact_copy_proofs(
        self, workspace_id: str, *, owner_id: str | None = None
    ) -> dict[tuple[str, str], tuple[int, OutputCopyProvenance]]:
        """Read verified incoming Copy receipts for linked Local Chatbooks."""
        from .registry_service import _normalize_required_text

        safe_id = _normalize_required_text(workspace_id, "workspace_id")
        with self.db.transaction() as conn:
            rows = conn.execute(
                """SELECT links.item_id, copies.item_key, copies.destination_owner_version,
                          json_extract(operations.manifest_json,
                              '$.items[' || copies.ordinal || '].source.version') AS source_revision
                   FROM research_copy_operations AS operations
                   JOIN research_copy_items AS copies USING (transfer_id)
                   JOIN workspace_memberships AS links
                     ON (links.workspace_id, links.item_id) =
                        (operations.destination_workspace_id, copies.destination_owner_id)
                   WHERE (operations.destination_data_source, operations.destination_server_profile_id,
                          operations.destination_principal_id, operations.destination_workspace_id)
                         = ('local', '', '', ?)
                     AND links.item_type = 'chatbook' AND links.role = 'output'
                     AND links.metadata_json IS NULL AND (? IS NULL OR links.item_id = ?)
                     AND copies.stage = 'verified' AND copies.rolled_back = 0
                     AND copies.destination_owner_version > 0
                     AND json_extract(operations.manifest_json,
                         '$.items[' || copies.ordinal || '].source.kind') = 'artifact'
                     AND json_extract(operations.manifest_json,
                         '$.items[' || copies.ordinal || '].treatment') = 'copy_content'""",
                (safe_id, owner_id, owner_id),
            ).fetchall()
        proofs = {}
        for row in rows:
            try:
                proof = OutputCopyProvenance(
                    item_key=row["item_key"], source_revision=row["source_revision"]
                )
            except ValidationError:
                continue
            proofs[row["item_id"], proof.item_key] = (
                row["destination_owner_version"],
                proof,
            )
        return proofs

    def list_workspace_output_memberships(
        self,
        workspace_id: str,
        *,
        limit: int = 20,
        offset: int = 0,
        operation_id: str | None = None,
        copied_owner_ids: tuple[str, ...] = (),
    ) -> tuple[tuple[WorkspaceMembership, ...], int]:
        """Page the workspace's output links, optionally by exact generation operation."""
        from .registry_service import _membership_from_row, _normalize_required_text

        safe_id = _normalize_required_text(workspace_id, "workspace_id")
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        if type(offset) is not int or not 0 <= offset <= 10_000:
            raise ValueError("offset must be between 0 and 10000")
        with self.db.transaction() as conn:
            params = (safe_id, json.dumps(copied_owner_ids), operation_id, operation_id)
            total = conn.execute(
                """SELECT COUNT(*) FROM workspace_memberships
                   WHERE workspace_id = ? AND role = 'output'
                     AND (metadata_json IS NOT NULL OR
                          (item_type = 'chatbook' AND item_id IN (SELECT value FROM json_each(?))))
                     AND (? IS NULL OR json_extract(metadata_json, '$.research_generation.operation_id') = ?)""",
                params,
            ).fetchone()[0]
            rows = conn.execute(
                """SELECT * FROM workspace_memberships
                   WHERE workspace_id = ? AND role = 'output'
                     AND (metadata_json IS NOT NULL OR
                          (item_type = 'chatbook' AND item_id IN (SELECT value FROM json_each(?))))
                     AND (? IS NULL OR json_extract(metadata_json, '$.research_generation.operation_id') = ?)
                   ORDER BY COALESCE(updated_at, created_at) DESC, membership_id ASC LIMIT ? OFFSET ?""",
                (*params, limit, offset),
            ).fetchall()
        return tuple(_membership_from_row(row) for row in rows), int(total)
