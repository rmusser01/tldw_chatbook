"""Canonical workspace membership queries for Research output provenance."""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from datetime import timedelta

from .models import WorkspaceMembership
from .output_membership import (
    encode_output_membership_metadata,
    output_membership_metadata,
)


class WorkspaceOutputMemberships:
    """Persist output links through the workspace registry's existing SQLite connection."""

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

    def list_workspace_output_memberships(
        self,
        workspace_id: str,
        *,
        limit: int = 20,
        offset: int = 0,
        operation_id: str | None = None,
    ) -> tuple[tuple[WorkspaceMembership, ...], int]:
        """Page the workspace's output links, optionally by exact generation operation."""
        from .registry_service import _membership_from_row, _normalize_required_text

        safe_id = _normalize_required_text(workspace_id, "workspace_id")
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        if type(offset) is not int or not 0 <= offset <= 10_000:
            raise ValueError("offset must be between 0 and 10000")
        with self.db.transaction() as conn:
            params = (safe_id, operation_id, operation_id)
            total = conn.execute(
                """SELECT COUNT(*) FROM workspace_memberships
                   WHERE workspace_id = ? AND role = 'output' AND metadata_json IS NOT NULL
                     AND (? IS NULL OR json_extract(metadata_json, '$.research_generation.operation_id') = ?)""",
                params,
            ).fetchone()[0]
            rows = conn.execute(
                """SELECT * FROM workspace_memberships
                   WHERE workspace_id = ? AND role = 'output' AND metadata_json IS NOT NULL
                     AND (? IS NULL OR json_extract(metadata_json, '$.research_generation.operation_id') = ?)
                   ORDER BY updated_at DESC, membership_id ASC LIMIT ? OFFSET ?""",
                (*params, limit, offset),
            ).fetchall()
        return tuple(_membership_from_row(row) for row in rows), int(total)
