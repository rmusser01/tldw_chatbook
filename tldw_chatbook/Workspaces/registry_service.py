"""Local workspace registry service."""

from __future__ import annotations

from collections.abc import Callable
import json
import sqlite3
from uuid import uuid4

from tldw_chatbook.DB.Workspace_DB import WorkspaceDB

from .models import (
    RuntimeBindingKind,
    RuntimeBindingStatus,
    WorkspaceAuthority,
    WorkspaceMembership,
    WorkspaceRecord,
    WorkspaceRuntimeBinding,
    WorkspaceSyncStatus,
    WorkspaceTransferPolicy,
    scrub_secret_metadata,
    utc_now_iso,
)


_STORAGE_FAILURE_MESSAGE = "Workspace registry storage failed."


class WorkspaceRegistryServiceError(Exception):
    """Base exception for workspace registry failures."""


class WorkspaceNotFound(WorkspaceRegistryServiceError):
    """Raised when a workspace operation targets a missing workspace."""


class DuplicateWorkspace(WorkspaceRegistryServiceError):
    """Raised when a workspace id already exists."""


class LocalWorkspaceRegistryService:
    """SQLite-backed local workspace registry."""

    def __init__(
        self,
        db: WorkspaceDB,
        *,
        id_factory: Callable[[], str] | None = None,
        now_factory: Callable[[], str] | None = None,
    ) -> None:
        self.db = db
        self._id_factory = id_factory or (lambda: f"workspace-link-{uuid4().hex}")
        self._now_factory = now_factory or utc_now_iso

    def create_workspace(
        self,
        *,
        workspace_id: str,
        name: str,
        description: str = "",
        authority: WorkspaceAuthority | str = WorkspaceAuthority.LOCAL_ONLY,
        sync_status: WorkspaceSyncStatus | str = WorkspaceSyncStatus.NOT_CONFIGURED,
    ) -> WorkspaceRecord:
        """Create a local workspace record."""

        now = self._now_factory()
        record = WorkspaceRecord(
            workspace_id=workspace_id,
            name=name,
            description=description,
            authority=authority,
            sync_status=sync_status,
            created_at=now,
            updated_at=now,
        )
        try:
            with self.db.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO workspace_records (
                        workspace_id,
                        name,
                        description,
                        authority,
                        sync_status,
                        active,
                        archived,
                        created_at,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.workspace_id,
                        record.name,
                        record.description,
                        record.authority.value,
                        record.sync_status.value,
                        int(record.active),
                        int(record.archived),
                        record.created_at,
                        record.updated_at,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise DuplicateWorkspace(record.workspace_id) from exc
        except sqlite3.Error as exc:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        created = self.get_workspace(record.workspace_id)
        if created is None:
            raise WorkspaceRegistryServiceError("Workspace creation failed.")
        return created

    def list_workspaces(self, *, include_archived: bool = False) -> tuple[WorkspaceRecord, ...]:
        """List local workspaces in stable creation order."""

        where_clause = "" if include_archived else "WHERE archived = 0"
        try:
            with self.db.connection() as conn:
                rows = conn.execute(
                    f"""
                    SELECT *
                    FROM workspace_records
                    {where_clause}
                    ORDER BY created_at ASC, workspace_id ASC
                    """
                ).fetchall()
        except sqlite3.Error as exc:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        return tuple(_workspace_from_row(row) for row in rows)

    def get_workspace(self, workspace_id: str) -> WorkspaceRecord | None:
        """Return one workspace record if it exists."""

        try:
            with self.db.connection() as conn:
                row = conn.execute(
                    """
                    SELECT *
                    FROM workspace_records
                    WHERE workspace_id = ?
                    """,
                    (workspace_id,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        return _workspace_from_row(row) if row is not None else None

    def set_active_workspace(self, workspace_id: str) -> WorkspaceRecord:
        """Set exactly one active workspace."""

        if self.get_workspace(workspace_id) is None:
            raise WorkspaceNotFound(workspace_id)
        now = self._now_factory()
        try:
            with self.db.transaction() as conn:
                conn.execute("UPDATE workspace_records SET active = 0")
                conn.execute(
                    """
                    UPDATE workspace_records
                    SET active = 1,
                        updated_at = ?
                    WHERE workspace_id = ?
                    """,
                    (now, workspace_id),
                )
        except sqlite3.Error as exc:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        active = self.get_active_workspace()
        if active is None:
            raise WorkspaceRegistryServiceError("Active workspace update failed.")
        return active

    def get_active_workspace(self) -> WorkspaceRecord | None:
        """Return the active workspace if one is selected."""

        try:
            with self.db.connection() as conn:
                row = conn.execute(
                    """
                    SELECT *
                    FROM workspace_records
                    WHERE active = 1
                        AND archived = 0
                    ORDER BY updated_at DESC, workspace_id ASC
                    LIMIT 1
                    """
                ).fetchone()
        except sqlite3.Error as exc:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        return _workspace_from_row(row) if row is not None else None

    def link_membership(
        self,
        workspace_id: str,
        *,
        item_type: str,
        item_id: str,
        role: str = "source",
        transfer_policy: WorkspaceTransferPolicy | str = WorkspaceTransferPolicy.REFERENCE,
        title: str = "",
    ) -> WorkspaceMembership:
        """Link a visible item to a workspace without hiding other memberships."""

        if self.get_workspace(workspace_id) is None:
            raise WorkspaceNotFound(workspace_id)
        membership = WorkspaceMembership(
            membership_id=self._id_factory(),
            workspace_id=workspace_id,
            item_type=item_type,
            item_id=item_id,
            role=role,
            transfer_policy=transfer_policy,
            title=title,
            created_at=self._now_factory(),
        )
        try:
            with self.db.transaction() as conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO workspace_memberships (
                        membership_id,
                        workspace_id,
                        item_type,
                        item_id,
                        role,
                        transfer_policy,
                        title,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        membership.membership_id,
                        membership.workspace_id,
                        membership.item_type,
                        membership.item_id,
                        membership.role,
                        membership.transfer_policy.value,
                        membership.title,
                        membership.created_at,
                    ),
                )
        except sqlite3.Error as exc:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        memberships = self.get_item_memberships(item_type, item_id)
        for existing in memberships:
            if existing.workspace_id == workspace_id and existing.role == role:
                return existing
        raise WorkspaceRegistryServiceError("Workspace membership link failed.")

    def get_item_memberships(
        self,
        item_type: str,
        item_id: str,
    ) -> tuple[WorkspaceMembership, ...]:
        """Return all workspace memberships for one visible item."""

        try:
            with self.db.connection() as conn:
                rows = conn.execute(
                    """
                    SELECT *
                    FROM workspace_memberships
                    WHERE item_type = ?
                        AND item_id = ?
                    ORDER BY created_at ASC, workspace_id ASC, role ASC
                    """,
                    (item_type, item_id),
                ).fetchall()
        except sqlite3.Error as exc:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        return tuple(_membership_from_row(row) for row in rows)

    def list_workspace_memberships(
        self,
        workspace_id: str,
    ) -> tuple[WorkspaceMembership, ...]:
        """Return item memberships for a workspace."""

        try:
            with self.db.connection() as conn:
                rows = conn.execute(
                    """
                    SELECT *
                    FROM workspace_memberships
                    WHERE workspace_id = ?
                    ORDER BY created_at ASC, item_type ASC, item_id ASC, role ASC
                    """,
                    (workspace_id,),
                ).fetchall()
        except sqlite3.Error as exc:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        return tuple(_membership_from_row(row) for row in rows)

    def save_runtime_binding(
        self,
        binding: WorkspaceRuntimeBinding,
    ) -> WorkspaceRuntimeBinding:
        """Create or update a workspace runtime binding."""

        if self.get_workspace(binding.workspace_id) is None:
            raise WorkspaceNotFound(binding.workspace_id)
        safe_binding = WorkspaceRuntimeBinding(
            workspace_id=binding.workspace_id,
            binding_id=binding.binding_id,
            binding_kind=binding.binding_kind,
            label=binding.label,
            locator=binding.locator,
            status=binding.status,
            metadata=scrub_secret_metadata(binding.metadata),
            created_at=binding.created_at,
            updated_at=self._now_factory(),
        )
        try:
            with self.db.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO workspace_runtime_bindings (
                        binding_id,
                        workspace_id,
                        binding_kind,
                        label,
                        locator,
                        status,
                        metadata_json,
                        created_at,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(binding_id) DO UPDATE SET
                        workspace_id = excluded.workspace_id,
                        binding_kind = excluded.binding_kind,
                        label = excluded.label,
                        locator = excluded.locator,
                        status = excluded.status,
                        metadata_json = excluded.metadata_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        safe_binding.binding_id,
                        safe_binding.workspace_id,
                        safe_binding.binding_kind.value,
                        safe_binding.label,
                        safe_binding.locator,
                        safe_binding.status.value,
                        json.dumps(safe_binding.metadata, sort_keys=True),
                        safe_binding.created_at,
                        safe_binding.updated_at,
                    ),
                )
        except sqlite3.Error as exc:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        stored = self.get_runtime_binding(safe_binding.binding_id)
        if stored is None:
            raise WorkspaceRegistryServiceError("Runtime binding save failed.")
        return stored

    def get_runtime_binding(self, binding_id: str) -> WorkspaceRuntimeBinding | None:
        """Return one runtime binding if it exists."""

        try:
            with self.db.connection() as conn:
                row = conn.execute(
                    """
                    SELECT *
                    FROM workspace_runtime_bindings
                    WHERE binding_id = ?
                    """,
                    (binding_id,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        return _runtime_binding_from_row(row) if row is not None else None

    def list_runtime_bindings(
        self,
        workspace_id: str,
    ) -> tuple[WorkspaceRuntimeBinding, ...]:
        """Return runtime bindings for a workspace."""

        try:
            with self.db.connection() as conn:
                rows = conn.execute(
                    """
                    SELECT *
                    FROM workspace_runtime_bindings
                    WHERE workspace_id = ?
                    ORDER BY created_at ASC, binding_id ASC
                    """,
                    (workspace_id,),
                ).fetchall()
        except sqlite3.Error as exc:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        return tuple(_runtime_binding_from_row(row) for row in rows)


def _workspace_from_row(row: sqlite3.Row) -> WorkspaceRecord:
    return WorkspaceRecord(
        workspace_id=row["workspace_id"],
        name=row["name"],
        description=row["description"],
        authority=WorkspaceAuthority(row["authority"]),
        sync_status=WorkspaceSyncStatus(row["sync_status"]),
        active=bool(row["active"]),
        archived=bool(row["archived"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _membership_from_row(row: sqlite3.Row) -> WorkspaceMembership:
    return WorkspaceMembership(
        membership_id=row["membership_id"],
        workspace_id=row["workspace_id"],
        item_type=row["item_type"],
        item_id=row["item_id"],
        role=row["role"],
        transfer_policy=WorkspaceTransferPolicy(row["transfer_policy"]),
        title=row["title"],
        created_at=row["created_at"],
    )


def _runtime_binding_from_row(row: sqlite3.Row) -> WorkspaceRuntimeBinding:
    metadata = json.loads(row["metadata_json"] or "{}")
    return WorkspaceRuntimeBinding(
        workspace_id=row["workspace_id"],
        binding_id=row["binding_id"],
        binding_kind=RuntimeBindingKind(row["binding_kind"]),
        label=row["label"],
        locator=row["locator"],
        status=RuntimeBindingStatus(row["status"]),
        metadata=metadata,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
