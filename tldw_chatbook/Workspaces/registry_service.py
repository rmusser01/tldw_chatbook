"""Local workspace registry service."""

from __future__ import annotations

from collections.abc import Callable
import json
import sqlite3
from uuid import uuid4

from loguru import logger

from tldw_chatbook.DB.Workspace_DB import WorkspaceDB

from .models import (
    DEFAULT_WORKSPACE_DESCRIPTION,
    DEFAULT_WORKSPACE_ID,
    DEFAULT_WORKSPACE_NAME,
    RuntimeBindingKind,
    RuntimeBindingStatus,
    WorkspaceAuthority,
    WorkspaceMembership,
    WorkspaceRecord,
    WorkspaceRuntimeBinding,
    WorkspaceSyncStatus,
    WorkspaceTransferPolicy,
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

        if include_archived:
            query = """
                SELECT *
                FROM workspace_records
                ORDER BY created_at ASC, workspace_id ASC
                """
            params: tuple[object, ...] = ()
        else:
            query = """
                SELECT *
                FROM workspace_records
                WHERE archived = ?
                ORDER BY created_at ASC, workspace_id ASC
                """
            params = (0,)
        try:
            with self.db.connection() as conn:
                rows = conn.execute(query, params).fetchall()
        except sqlite3.Error as exc:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        return tuple(_workspace_from_row(row) for row in rows)

    def get_workspace(self, workspace_id: str) -> WorkspaceRecord | None:
        """Return one workspace record if it exists."""

        safe_workspace_id = _normalize_required_text(workspace_id, "workspace_id")
        try:
            with self.db.connection() as conn:
                row = conn.execute(
                    """
                    SELECT *
                    FROM workspace_records
                    WHERE workspace_id = ?
                    """,
                    (safe_workspace_id,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        return _workspace_from_row(row) if row is not None else None

    def set_active_workspace(self, workspace_id: str) -> WorkspaceRecord:
        """Set exactly one active workspace."""

        safe_workspace_id = _normalize_required_text(workspace_id, "workspace_id")
        target_workspace = self.get_workspace(safe_workspace_id)
        if target_workspace is None or target_workspace.archived:
            raise WorkspaceNotFound(safe_workspace_id)
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
                    (now, safe_workspace_id),
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

    def ensure_default_workspace(self) -> WorkspaceRecord:
        """Ensure there is an active safe workspace for normal Console chat.

        The built-in Default workspace gives users a concrete active workspace
        without granting filesystem/runtime capabilities. Users can still browse
        and chat normally, but tool/file access requires an explicit workspace.
        """

        active_workspace = self.get_active_workspace()
        if active_workspace is not None:
            if active_workspace.workspace_id == DEFAULT_WORKSPACE_ID:
                self._delete_default_runtime_bindings()
            return active_workspace

        default_workspace = self.get_workspace(DEFAULT_WORKSPACE_ID)
        if default_workspace is None:
            self.create_workspace(
                workspace_id=DEFAULT_WORKSPACE_ID,
                name=DEFAULT_WORKSPACE_NAME,
                description=DEFAULT_WORKSPACE_DESCRIPTION,
                authority=WorkspaceAuthority.LOCAL_ONLY,
                sync_status=WorkspaceSyncStatus.NOT_CONFIGURED,
            )
        elif default_workspace.archived:
            self._restore_default_workspace()

        self._delete_default_runtime_bindings()
        return self.set_active_workspace(DEFAULT_WORKSPACE_ID)

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

        safe_workspace_id = _normalize_required_text(workspace_id, "workspace_id")
        if self.get_workspace(safe_workspace_id) is None:
            raise WorkspaceNotFound(safe_workspace_id)
        membership = WorkspaceMembership(
            membership_id=self._id_factory(),
            workspace_id=safe_workspace_id,
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
        try:
            with self.db.connection() as conn:
                row = conn.execute(
                    """
                    SELECT *
                    FROM workspace_memberships
                    WHERE workspace_id = ?
                        AND item_type = ?
                        AND item_id = ?
                        AND role = ?
                    """,
                    (
                        membership.workspace_id,
                        membership.item_type,
                        membership.item_id,
                        membership.role,
                    ),
                ).fetchone()
        except sqlite3.Error as exc:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        if row is not None:
            return _membership_from_row(row)
        raise WorkspaceRegistryServiceError("Workspace membership link failed.")

    def get_item_memberships(
        self,
        item_type: str,
        item_id: str,
    ) -> tuple[WorkspaceMembership, ...]:
        """Return all workspace memberships for one visible item."""

        safe_item_type = _normalize_required_text(item_type, "item_type")
        safe_item_id = _normalize_required_text(item_id, "item_id")
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
                    (safe_item_type, safe_item_id),
                ).fetchall()
        except sqlite3.Error as exc:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        return tuple(_membership_from_row(row) for row in rows)

    def list_workspace_memberships(
        self,
        workspace_id: str,
    ) -> tuple[WorkspaceMembership, ...]:
        """Return item memberships for a workspace."""

        safe_workspace_id = _normalize_required_text(workspace_id, "workspace_id")
        try:
            with self.db.connection() as conn:
                rows = conn.execute(
                    """
                    SELECT *
                    FROM workspace_memberships
                    WHERE workspace_id = ?
                    ORDER BY created_at ASC, item_type ASC, item_id ASC, role ASC
                    """,
                    (safe_workspace_id,),
                ).fetchall()
        except sqlite3.Error as exc:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        return tuple(_membership_from_row(row) for row in rows)

    def list_workspace_conversations(
        self,
        workspace_id: str,
    ) -> tuple[WorkspaceMembership, ...]:
        """Return conversation memberships for one workspace."""

        safe_workspace_id = _normalize_required_text(workspace_id, "workspace_id")
        try:
            with self.db.transaction() as conn:
                rows = conn.execute(
                    """
                    SELECT *
                    FROM workspace_memberships
                    WHERE workspace_id = ?
                        AND item_type = ?
                    ORDER BY created_at ASC, item_id ASC, role ASC
                    """,
                    (safe_workspace_id, "conversation"),
                ).fetchall()
        except sqlite3.Error as exc:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        return tuple(_membership_from_row(row) for row in rows)

    def save_runtime_binding(
        self,
        binding: WorkspaceRuntimeBinding,
    ) -> WorkspaceRuntimeBinding:
        """Create or update a workspace runtime binding."""

        if binding.workspace_id == DEFAULT_WORKSPACE_ID:
            raise WorkspaceRegistryServiceError(
                "Default workspace does not allow runtime bindings."
            )
        if self.get_workspace(binding.workspace_id) is None:
            raise WorkspaceNotFound(binding.workspace_id)
        safe_binding = WorkspaceRuntimeBinding(
            workspace_id=binding.workspace_id,
            binding_id=binding.binding_id,
            binding_kind=binding.binding_kind,
            label=binding.label,
            locator=binding.locator,
            status=binding.status,
            metadata=binding.metadata,
            created_at=binding.created_at,
            updated_at=self._now_factory(),
        )
        metadata_json = _metadata_to_json(safe_binding.metadata)
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
                        metadata_json,
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

        safe_binding_id = _normalize_required_text(binding_id, "binding_id")
        try:
            with self.db.connection() as conn:
                row = conn.execute(
                    """
                    SELECT *
                    FROM workspace_runtime_bindings
                    WHERE binding_id = ?
                    """,
                    (safe_binding_id,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        if row is None:
            return None
        binding = _runtime_binding_from_row(row)
        if binding.workspace_id == DEFAULT_WORKSPACE_ID:
            self._delete_default_runtime_bindings()
            return None
        return binding

    def list_runtime_bindings(
        self,
        workspace_id: str,
    ) -> tuple[WorkspaceRuntimeBinding, ...]:
        """Return runtime bindings for a workspace."""

        safe_workspace_id = _normalize_required_text(workspace_id, "workspace_id")
        if safe_workspace_id == DEFAULT_WORKSPACE_ID:
            self._delete_default_runtime_bindings()
            return ()
        try:
            with self.db.connection() as conn:
                rows = conn.execute(
                    """
                    SELECT *
                    FROM workspace_runtime_bindings
                    WHERE workspace_id = ?
                    ORDER BY created_at ASC, binding_id ASC
                    """,
                    (safe_workspace_id,),
                ).fetchall()
        except sqlite3.Error as exc:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        return tuple(_runtime_binding_from_row(row) for row in rows)

    def _delete_default_runtime_bindings(self) -> None:
        """Remove stale runtime bindings from the safe built-in Default workspace."""

        try:
            with self.db.connection() as conn:
                has_bindings = (
                    conn.execute(
                        """
                        SELECT 1
                        FROM workspace_runtime_bindings
                        WHERE workspace_id = ?
                        LIMIT 1
                        """,
                        (DEFAULT_WORKSPACE_ID,),
                    ).fetchone()
                    is not None
                )

            if not has_bindings:
                return

            with self.db.transaction() as conn:
                conn.execute(
                    """
                    DELETE FROM workspace_runtime_bindings
                    WHERE workspace_id = ?
                    """,
                    (DEFAULT_WORKSPACE_ID,),
                )
        except sqlite3.Error as exc:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE) from exc

    def _restore_default_workspace(self) -> None:
        """Restore the built-in Default workspace when it is the only safe active fallback."""

        now = self._now_factory()
        try:
            with self.db.transaction() as conn:
                conn.execute(
                    """
                    UPDATE workspace_records
                    SET name = ?,
                        description = ?,
                        authority = ?,
                        sync_status = ?,
                        archived = 0,
                        updated_at = ?
                    WHERE workspace_id = ?
                    """,
                    (
                        DEFAULT_WORKSPACE_NAME,
                        DEFAULT_WORKSPACE_DESCRIPTION,
                        WorkspaceAuthority.LOCAL_ONLY.value,
                        WorkspaceSyncStatus.NOT_CONFIGURED.value,
                        now,
                        DEFAULT_WORKSPACE_ID,
                    ),
                )
        except sqlite3.Error as exc:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE) from exc


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
    metadata = _metadata_from_json(row["metadata_json"], binding_id=row["binding_id"])
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


def _metadata_to_json(metadata: dict[str, object]) -> str:
    try:
        return json.dumps(metadata, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise WorkspaceRegistryServiceError(
            "Runtime binding metadata must be JSON-serializable."
        ) from exc


def _metadata_from_json(value: str, *, binding_id: str) -> dict[str, object]:
    try:
        decoded = json.loads(value or "{}")
    except json.JSONDecodeError:
        logger.warning(
            "Invalid workspace runtime binding metadata JSON; using empty metadata",
            binding_id=binding_id,
        )
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _normalize_required_text(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be text")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    return normalized
