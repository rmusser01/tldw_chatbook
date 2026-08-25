"""Local workspace registry service."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import base64
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from secrets import token_urlsafe
import sqlite3
from typing import Protocol
from uuid import NAMESPACE_URL, RFC_4122, UUID, uuid4, uuid5

from loguru import logger

from tldw_chatbook.Chat.rag_scope import (
    RagScope,
    ScopeItem,
    parse_scope,
    serialize_scope,
)
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Utils.sensitive_paths import find_root_binding_conflict

from .models import (
    DEFAULT_WORKSPACE_DESCRIPTION,
    DEFAULT_WORKSPACE_ID,
    DEFAULT_WORKSPACE_NAME,
    RuntimeBindingKind,
    RuntimeBindingStatus,
    ResearchQuickNoteReceipt,
    WorkspaceAuthority,
    WorkspaceMembership,
    WorkspaceRecord,
    WorkspaceRuntimeBinding,
    WorkspaceSyncStatus,
    WorkspaceTransferPolicy,
    utc_now_iso,
)
from .change_review_consent import (
    ChangeReviewConsent,
    ChangeReviewState,
    ChangeReviewStateConflict,
    MISSING_CHANGE_REVIEW_REVISION,
)


_STORAGE_FAILURE_MESSAGE = "Workspace registry storage failed."
_QUICK_NOTE_LEASE_SECONDS = 30
_QUICK_NOTE_ABANDON_SECONDS = 7 * 24 * 60 * 60
_QUICK_NOTE_MAX_FAILURES = 3
_QUICK_NOTE_FAILURE_REASONS = {
    "proof_mismatch",
    "owner_conflict",
    "owner_missing",
    "owner_unavailable",
    "registry_failure",
}


def _parse_utc_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _future_timestamp(value: str, seconds: int) -> str:
    return (_parse_utc_timestamp(value) + timedelta(seconds=seconds)).isoformat()


def _monotonic_timestamp(previous: str, candidate: str) -> str:
    return previous if _parse_utc_timestamp(previous) >= _parse_utc_timestamp(candidate) else candidate


def _length_prefixed_identity(*axes: str) -> str:
    payload = bytearray()
    for axis in axes:
        encoded = axis.encode("utf-8")
        payload.extend(len(encoded).to_bytes(4, "big"))
        payload.extend(encoded)
    return base64.urlsafe_b64encode(bytes(payload)).decode("ascii")


def _validate_quick_note_operation_token(value: str) -> None:
    if not value.startswith("research-note-") or len(value) != 46:
        raise ValueError("operation_token is not an app-minted Quick Note token")
    try:
        operation_uuid = UUID(hex=value.removeprefix("research-note-"))
    except ValueError as exc:
        raise ValueError(
            "operation_token is not an app-minted Quick Note token"
        ) from exc
    if (
        operation_uuid.version != 4
        or operation_uuid.variant != RFC_4122
        or value != f"research-note-{operation_uuid.hex}"
    ):
        raise ValueError("operation_token is not an app-minted Quick Note token")


def _opaque_receipt_token(factory: Callable[[], str], field_name: str) -> str:
    raw = _normalize_required_text(factory(), field_name)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class _ChangeReviewBindingOwner(Protocol):
    def binding_added(
        self,
        workspace_id: str,
        binding: WorkspaceRuntimeBinding,
    ) -> None: ...


def _filesystem_binding_missing(locator: str) -> bool:
    """Whether a local-filesystem binding's locator no longer resolves safely.

    A bound folder counts as missing not only once it no longer exists, but
    also once the path has been replaced by a symlink or its resolved form
    no longer matches the stored locator. ``add_folder_binding`` always
    stores an already-resolved path, so any drift here means a symlink or
    mount trick appeared along the path after binding -- trusting it could
    silently widen the sandboxed root at enforcement time (ADR-028).

    Args:
        locator: The binding's stored (already-resolved) folder path.

    Returns:
        True if the locator no longer points at a real, non-symlinked
        directory matching its own resolved form; False otherwise.
    """
    folder = Path(locator)
    if folder.is_symlink():
        return True
    if not folder.is_dir():
        return True
    try:
        return folder.resolve() != folder
    except OSError:
        return True


class WorkspaceRegistryServiceError(Exception):
    """Base exception for workspace registry failures."""


class WorkspaceNotFound(WorkspaceRegistryServiceError):
    """Raised when a workspace operation targets a missing workspace."""


class DuplicateWorkspace(WorkspaceRegistryServiceError):
    """Raised when a workspace id already exists."""


class BindingNotFound(WorkspaceRegistryServiceError):
    """Raised when a runtime binding id does not exist."""

    def __init__(self, binding_id: str) -> None:
        super().__init__(f"Runtime binding not found: {binding_id}")
        self.binding_id = binding_id


def _validate_folder_path_rules(path: str | Path) -> Path:
    """Validate filesystem and sensitive-path rules for a candidate folder.

    Private helper; does not check existing bindings. Resolves and vets the
    path itself: existence, root/home rejection, and sensitive-path denylist.
    Raises WorkspaceRegistryServiceError on failure.
    """
    candidate = Path(path).expanduser()
    try:
        resolved = candidate.resolve()
    except OSError as exc:
        raise WorkspaceRegistryServiceError(
            f"Folder path could not be resolved: {candidate}"
        ) from exc
    if not resolved.is_dir():
        raise WorkspaceRegistryServiceError(
            f"Folder does not exist or is not a directory: {resolved}"
        )
    if resolved == Path(resolved.anchor):
        raise WorkspaceRegistryServiceError(
            "The filesystem root cannot be bound to a workspace."
        )
    if resolved == Path.home().resolve():
        raise WorkspaceRegistryServiceError(
            "Your home directory itself cannot be bound; choose a "
            "project folder inside it."
        )
    # TASK-857: the sensitive-path denylist used to only be consulted
    # at file-tool READ/WRITE time, never here at the binding gate --
    # so binding e.g. ~/.config/tldw_cli (this app's own config,
    # API keys included), get_user_data_dir() (every app database), or
    # ~/.ssh as a workspace folder root all passed this check, widening
    # what the agent file tools can reach up to the edge of whatever
    # the denylist enumerates. See ``find_root_binding_conflict`` for
    # the exact "is, or contains, or is contained by" rule.
    conflict = find_root_binding_conflict(resolved)
    if conflict is not None:
        raise WorkspaceRegistryServiceError(
            f"'{resolved}' cannot be bound: it is, or contains, the "
            f"protected path '{conflict}'. Choose a folder that does "
            f"not overlap this application's own data, configuration, "
            f"or credential directories."
        )
    return resolved


def _validate_folder_overlap(
    resolved: Path,
    existing_locators: Sequence[str],
) -> None:
    """Validate that a folder does not duplicate or nest existing bindings.

    Private helper; assumes path has already passed filesystem/sensitive checks.
    Raises WorkspaceRegistryServiceError if the path duplicates or nests
    any existing binding, or if any existing binding nests within the path.
    """
    for locator in existing_locators:
        existing_path = Path(locator)
        if resolved == existing_path:
            raise WorkspaceRegistryServiceError(
                f"{resolved} is already bound to this workspace."
            )
        if existing_path in resolved.parents:
            raise WorkspaceRegistryServiceError(
                f"{resolved} is inside the already-bound folder {existing_path}."
            )
        if resolved in existing_path.parents:
            raise WorkspaceRegistryServiceError(
                f"The already-bound folder {existing_path} is inside "
                f"{resolved}; remove it first."
            )


def validate_folder_binding_path(
    path: str | Path,
    existing_locators: Sequence[str] = (),
) -> Path:
    """Resolve and vet a candidate folder-binding path (spec 2026-08-17 §4.2).

    Pure with respect to the registry: consults only the filesystem and the
    sensitive-path denylist, so creation UIs can vet folders before any
    workspace exists. Raises WorkspaceRegistryServiceError with the same
    user-facing messages ``add_folder_binding`` raised before extraction.

    Args:
        path: Candidate folder path (str or Path; supports ~ expansion).
        existing_locators: Already-bound folder paths to check against for
            duplicates and nesting conflicts. Defaults to empty sequence.

    Returns:
        The resolved, canonical folder path as a Path object.

    Raises:
        WorkspaceRegistryServiceError: If the path does not exist, is not a
            directory, is the filesystem root, is the home directory, overlaps
            a sensitive path, or duplicates/nests any existing binding.
    """
    resolved = _validate_folder_path_rules(path)
    _validate_folder_overlap(resolved, existing_locators)
    return resolved


class LocalWorkspaceRegistryService:
    """SQLite-backed local workspace registry."""

    def __init__(
        self,
        db: WorkspaceDB,
        *,
        id_factory: Callable[[], str] | None = None,
        now_factory: Callable[[], str] | None = None,
        receipt_token_factory: Callable[[], str] | None = None,
    ) -> None:
        self.db = db
        self._id_factory = id_factory or (lambda: f"workspace-link-{uuid4().hex}")
        self._now_factory = now_factory or utc_now_iso
        self._mutation_generation = 0
        self._change_review_binding_owner: _ChangeReviewBindingOwner | None = None
        self._receipt_token_factory = receipt_token_factory or (
            lambda: token_urlsafe(32)
        )

    @property
    def mutation_generation(self) -> int:
        """Monotonic count of registry display-read mutations (TASK-21118).

        Bumped by every mutator that can change what a workspace-record
        read (``get_active_workspace``, ``get_workspace``,
        ``list_workspaces``) returns: create, rename, archive, unarchive,
        set-active, clear-active, and the built-in Default restore.
        ``ensure_default_workspace`` bumps through those same legs when
        (and only when) it actually changed something. TASK-22201 widened
        the contract to the Console context build's whole display read set:
        runtime-binding mutators (``save_runtime_binding`` — which
        ``add_folder_binding`` and ``set_folder_binding_access`` route
        through — ``remove_runtime_binding``, and the Default stale-binding
        repair) and ``link_membership`` now bump too, so
        ``list_runtime_bindings`` / ``list_workspace_memberships`` results
        are also safe to cache against this value.
        ``set_workspace_scope`` / ``set_change_review_enabled`` write
        tables outside that read set and deliberately do not bump.

        This is the invalidation subscription point for read caches: the
        Console keystroke path memoizes its active-workspace resolution
        against this value instead of re-reading SQLite ~1.25x per key,
        and the Console run tick serves its registry display reads from a
        generation-keyed view instead of ~80 SQLite round-trips per 0.2 s
        tick (TASK-22201).
        Every UI seam that changes the active workspace (Console switcher,
        browser-row open, session switch, Settings "Set active", Library's
        create modal, archive flows) funnels through these mutators on the
        one app-level service instance, so comparing generations is
        equivalent to subscribing to all of them. In-memory only: a
        different process writing the same registry file is out of scope,
        as for every other in-process cache over this database.
        """
        return self._mutation_generation

    def _bump_mutation_generation(self) -> None:
        """Record one committed workspace-record mutation."""
        self._mutation_generation += 1

    def attach_change_review_consent_service(
        self,
        service: _ChangeReviewBindingOwner,
    ) -> None:
        """Attach the app-owned Change Review binding lifecycle observer."""
        self._change_review_binding_owner = service

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
        self._reject_duplicate_name(record.name)
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
            if "idx_workspace_records_name_ci" in str(exc):
                raise WorkspaceRegistryServiceError(
                    f"A workspace named {record.name} already exists."
                ) from exc
            raise DuplicateWorkspace(record.workspace_id) from exc
        except sqlite3.Error as exc:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        self._bump_mutation_generation()
        created = self.get_workspace(record.workspace_id)
        if created is None:
            raise WorkspaceRegistryServiceError("Workspace creation failed.")
        return created

    def list_workspaces(
        self, *, include_archived: bool = False
    ) -> tuple[WorkspaceRecord, ...]:
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

    def rename_workspace(self, workspace_id: str, name: str) -> WorkspaceRecord:
        """Rename a workspace (TASK-714).

        The built-in Default workspace keeps its identity: rail copy and
        runtime rules reference it by name, so it is protected from rename.

        Args:
            workspace_id: Workspace to rename.
            name: New user-facing name (must be non-blank).

        Returns:
            The updated workspace record.

        Raises:
            WorkspaceNotFound: Unknown or archived workspace.
            WorkspaceRegistryServiceError: Blank name, Default workspace, or
                storage failure.
        """
        safe_workspace_id = _normalize_required_text(workspace_id, "workspace_id")
        safe_name = str(name or "").strip()
        if not safe_name:
            raise WorkspaceRegistryServiceError("Workspace name cannot be blank.")
        if safe_workspace_id == DEFAULT_WORKSPACE_ID:
            raise WorkspaceRegistryServiceError(
                "The Default workspace cannot be renamed."
            )
        record = self.get_workspace(safe_workspace_id)
        if record is None or record.archived:
            raise WorkspaceNotFound(safe_workspace_id)
        self._reject_duplicate_name(safe_name, exclude_workspace_id=safe_workspace_id)
        now = self._now_factory()
        try:
            with self.db.transaction() as conn:
                conn.execute(
                    """
                    UPDATE workspace_records
                    SET name = ?, updated_at = ?
                    WHERE workspace_id = ?
                    """,
                    (safe_name, now, safe_workspace_id),
                )
        except sqlite3.IntegrityError as exc:
            raise WorkspaceRegistryServiceError(
                f"A workspace named {safe_name} already exists."
            ) from exc
        except sqlite3.Error as exc:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        self._bump_mutation_generation()
        renamed = self.get_workspace(safe_workspace_id)
        if renamed is None:
            raise WorkspaceRegistryServiceError("Workspace rename failed.")
        return renamed

    def archive_workspace(self, workspace_id: str) -> WorkspaceRecord:
        """Archive a workspace, hiding it from listings (TASK-714).

        Conversations and memberships are untouched - archiving only removes
        the workspace from the switcher/browser. When the archived workspace
        was active, the built-in Default workspace becomes active so Console
        always has a real context.

        Args:
            workspace_id: Workspace to archive.

        Returns:
            The archived workspace record.

        Raises:
            WorkspaceNotFound: Unknown or already-archived workspace.
            WorkspaceRegistryServiceError: Default workspace or storage
                failure.
        """
        safe_workspace_id = _normalize_required_text(workspace_id, "workspace_id")
        if safe_workspace_id == DEFAULT_WORKSPACE_ID:
            raise WorkspaceRegistryServiceError(
                "The Default workspace cannot be archived."
            )
        record = self.get_workspace(safe_workspace_id)
        if record is None or record.archived:
            raise WorkspaceNotFound(safe_workspace_id)
        was_active = record.active
        now = self._now_factory()
        try:
            with self.db.transaction() as conn:
                conn.execute(
                    """
                    UPDATE workspace_records
                    SET archived = 1, active = 0, updated_at = ?
                    WHERE workspace_id = ?
                    """,
                    (now, safe_workspace_id),
                )
        except sqlite3.Error as exc:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        self._bump_mutation_generation()
        if was_active:
            self.ensure_default_workspace()
            self.set_active_workspace(DEFAULT_WORKSPACE_ID)
        archived = self.get_workspace(safe_workspace_id)
        if archived is None:
            raise WorkspaceRegistryServiceError("Workspace archive failed.")
        return archived

    def unarchive_workspace(self, workspace_id: str) -> WorkspaceRecord:
        """Restore an archived workspace to listings (spec §2).

        Never auto-activates: the user chooses when to switch.

        Raises:
            WorkspaceNotFound: Unknown or not-archived workspace.
            WorkspaceRegistryServiceError: Storage failure.
        """
        safe_workspace_id = _normalize_required_text(workspace_id, "workspace_id")
        record = self.get_workspace(safe_workspace_id)
        if record is None or not record.archived:
            raise WorkspaceNotFound(safe_workspace_id)
        now = self._now_factory()
        try:
            with self.db.transaction() as conn:
                conn.execute(
                    """
                    UPDATE workspace_records
                    SET archived = 0, updated_at = ?
                    WHERE workspace_id = ?
                    """,
                    (now, safe_workspace_id),
                )
        except sqlite3.IntegrityError as exc:
            raise WorkspaceRegistryServiceError(
                f"A workspace named {record.name} already exists; rename it before unarchiving."
            ) from exc
        except sqlite3.Error as exc:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        self._bump_mutation_generation()
        restored = self.get_workspace(safe_workspace_id)
        if restored is None:
            raise WorkspaceRegistryServiceError("Workspace unarchive failed.")
        return restored

    def _reject_duplicate_name(
        self, name: str, *, exclude_workspace_id: str | None = None
    ) -> None:
        """Raise when a non-archived workspace already uses ``name``.

        Case-insensitive; archived workspaces do not block reuse (spec §2).
        """
        needle = name.strip().casefold()
        for record in self.list_workspaces():
            if exclude_workspace_id and record.workspace_id == exclude_workspace_id:
                continue
            if str(record.name or "").strip().casefold() == needle:
                raise WorkspaceRegistryServiceError(
                    f"A workspace named {name} already exists."
                )

    def set_active_workspace(self, workspace_id: str) -> WorkspaceRecord:
        """Set exactly one active workspace.

        Activating the built-in Default workspace also strips any stale
        runtime bindings from it (TASK-21118): this switch seam took over
        that repair from the per-keystroke context read, which used to run
        it via ``ensure_default_workspace`` up to ~1.25x per key.
        """

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
        self._bump_mutation_generation()
        if safe_workspace_id == DEFAULT_WORKSPACE_ID:
            self._delete_default_runtime_bindings()
        active = self.get_active_workspace()
        if active is None:
            raise WorkspaceRegistryServiceError("Active workspace update failed.")
        return active

    def clear_active_workspace(self) -> None:
        """Deselect every workspace, leaving no active record.

        task-15120 (owner ruling): the workspace context follows the
        conversation being viewed, and a GLOBAL conversation's context is the
        global scope -- which this registry represents as "no active
        workspace" (`get_active_workspace()` -> None), the same state a fresh
        registry starts in. Capability gates that require an explicit active
        workspace read exactly that, so a global conversation carries global
        capabilities rather than borrowing the previous workspace's.

        Raises:
            WorkspaceRegistryServiceError: If workspace storage cannot be
                updated.
        """
        try:
            with self.db.transaction() as conn:
                conn.execute("UPDATE workspace_records SET active = 0")
        except sqlite3.Error as exc:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        self._bump_mutation_generation()

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

        Returns:
            Existing active workspace, or the restored/created built-in Default
            workspace when no active workspace exists.

        Raises:
            WorkspaceRegistryServiceError: If workspace storage cannot be read
                or updated.
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

        # `set_active_workspace(DEFAULT_WORKSPACE_ID)` performs the stale-
        # runtime-binding repair itself (TASK-21118), so no separate
        # `_delete_default_runtime_bindings()` call is needed here.
        return self.set_active_workspace(DEFAULT_WORKSPACE_ID)

    def link_membership(
        self,
        workspace_id: str,
        *,
        item_type: str,
        item_id: str,
        role: str = "source",
        transfer_policy: WorkspaceTransferPolicy
        | str = WorkspaceTransferPolicy.REFERENCE,
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
        # Membership rows feed the Console context build's display reads,
        # so generation-keyed caches (TASK-22201) must revalidate. INSERT OR
        # IGNORE may have been a no-op; the occasional spurious bump merely
        # costs one re-read.
        self._bump_mutation_generation()
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

    def unlink_membership(
        self,
        workspace_id: str,
        *,
        item_type: str,
        item_id: str,
        role: str = "source",
    ) -> bool:
        """Remove one association without deleting the item."""

        safe_workspace_id = _normalize_required_text(workspace_id, "workspace_id")
        safe_item_type = _normalize_required_text(item_type, "item_type")
        safe_item_id = _normalize_required_text(item_id, "item_id")
        safe_role = _normalize_required_text(role, "role")
        try:
            with self.db.transaction() as conn:
                cursor = conn.execute(
                    """
                    DELETE FROM workspace_memberships
                    WHERE workspace_id = ?
                        AND item_type = ?
                        AND item_id = ?
                        AND role = ?
                    """,
                    (safe_workspace_id, safe_item_type, safe_item_id, safe_role),
                )
                if cursor.rowcount == 0:
                    return False
                if safe_role != "source":
                    return True

                row = conn.execute(
                    """
                    SELECT payload
                    FROM workspace_rag_scopes
                    WHERE workspace_id = ?
                    """,
                    (safe_workspace_id,),
                ).fetchone()
                if row is None:
                    return True
                try:
                    raw_scope = json.loads(row["payload"])
                except (TypeError, ValueError):
                    return True
                scope = parse_scope(raw_scope)
                if scope is None:
                    return True
                remaining = tuple(
                    item
                    for item in scope.items
                    if not (
                        item.source_type == safe_item_type
                        and item.source_id == safe_item_id
                    )
                )
                if remaining == scope.items:
                    return True
                if not remaining and not scope.empty_is_scoped:
                    conn.execute(
                        "DELETE FROM workspace_rag_scopes WHERE workspace_id = ?",
                        (safe_workspace_id,),
                    )
                    return True
                updated_scope = RagScope(
                    items=remaining,
                    updated_at=self._now_factory(),
                    empty_is_scoped=scope.empty_is_scoped,
                )
                conn.execute(
                    """
                    UPDATE workspace_rag_scopes
                    SET payload = ?, updated_at = ?
                    WHERE workspace_id = ?
                    """,
                    (
                        json.dumps(serialize_scope(updated_scope)),
                        updated_scope.updated_at,
                        safe_workspace_id,
                    ),
                )
                return True
        except sqlite3.Error as exc:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE) from exc

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

    def list_workspace_source_memberships(
        self,
        workspace_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[tuple[WorkspaceMembership, ...], int]:
        """Return one bounded page of canonical Media source associations."""

        safe_workspace_id = _normalize_required_text(workspace_id, "workspace_id")
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        if type(offset) is not int or not 0 <= offset <= 10_000:
            raise ValueError("offset must be between 0 and 10000")
        try:
            with self.db.connection() as conn:
                total = int(
                    conn.execute(
                        """
                        SELECT COUNT(*)
                        FROM workspace_memberships
                        WHERE workspace_id = ?
                            AND item_type = 'media'
                            AND role = 'source'
                        """,
                        (safe_workspace_id,),
                    ).fetchone()[0]
                )
                rows = conn.execute(
                    """
                    SELECT *
                    FROM workspace_memberships
                    WHERE workspace_id = ?
                        AND item_type = 'media'
                        AND role = 'source'
                    ORDER BY created_at ASC, item_id ASC, membership_id ASC
                    LIMIT ? OFFSET ?
                    """,
                    (safe_workspace_id, limit, offset),
                ).fetchall()
        except sqlite3.Error as exc:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        return tuple(_membership_from_row(row) for row in rows), total

    def list_workspace_note_memberships(
        self,
        workspace_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
        role: str = "note",
    ) -> tuple[tuple[WorkspaceMembership, ...], int]:
        """Return one bounded page of canonical Notes associations."""

        safe_workspace_id = _normalize_required_text(workspace_id, "workspace_id")
        safe_role = _normalize_required_text(role, "role")
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        if type(offset) is not int or not 0 <= offset <= 10_000:
            raise ValueError("offset must be between 0 and 10000")
        try:
            with self.db.connection() as conn:
                total = int(
                    conn.execute(
                        """
                        SELECT COUNT(*)
                        FROM workspace_memberships
                        WHERE workspace_id = ?
                            AND item_type = 'note'
                            AND role = ?
                        """,
                        (safe_workspace_id, safe_role),
                    ).fetchone()[0]
                )
                rows = conn.execute(
                    """
                    SELECT *
                    FROM workspace_memberships
                    WHERE workspace_id = ?
                        AND item_type = 'note'
                        AND role = ?
                    ORDER BY created_at DESC, item_id ASC, membership_id ASC
                    LIMIT ? OFFSET ?
                    """,
                    (safe_workspace_id, safe_role, limit, offset),
                ).fetchall()
        except sqlite3.Error as exc:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        return tuple(_membership_from_row(row) for row in rows), total

    @staticmethod
    def _quick_note_identity(
        *,
        workspace_id: str,
        local_user_id: str,
        operation_token: str,
        kind: str,
        data_source: str = "local",
        server_profile_id: str = "",
        principal_id: str = "",
    ) -> tuple[str, str]:
        seed = _length_prefixed_identity(
            "chatbook-quick-note-v2",
            data_source,
            server_profile_id,
            principal_id,
            workspace_id,
            local_user_id,
            operation_token,
        )
        canonical_note_id = f"research-note-{uuid5(NAMESPACE_URL, seed).hex}"
        receipt_seed = _length_prefixed_identity(seed, kind)
        receipt_id = f"quick-note-receipt-{uuid5(NAMESPACE_URL, receipt_seed).hex}"
        return receipt_id, canonical_note_id

    def claim_quick_note_create(
        self,
        workspace_id: str,
        *,
        local_user_id: str,
        operation_token: str,
    ) -> ResearchQuickNoteReceipt:
        """Idempotently claim an owner-qualified Local note create intent."""

        safe_workspace_id = _normalize_required_text(workspace_id, "workspace_id")
        safe_user_id = _normalize_required_text(local_user_id, "local_user_id")
        safe_token = _normalize_required_text(operation_token, "operation_token")
        _validate_quick_note_operation_token(safe_token)
        if self.get_workspace(safe_workspace_id) is None:
            raise WorkspaceNotFound(safe_workspace_id)
        receipt_id, note_id = self._quick_note_identity(
            workspace_id=safe_workspace_id,
            local_user_id=safe_user_id,
            operation_token=safe_token,
            kind="create",
        )
        now = self._now_factory()
        owner_proof = _opaque_receipt_token(
            self._receipt_token_factory, "owner_proof"
        )
        lease_token = _opaque_receipt_token(
            self._receipt_token_factory, "lease_token"
        )
        lease_expires_at = _future_timestamp(now, _QUICK_NOTE_LEASE_SECONDS)
        abandon_after = _future_timestamp(now, _QUICK_NOTE_ABANDON_SECONDS)
        try:
            with self.db.transaction(immediate=True) as conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO research_quick_note_receipts (
                        receipt_id, data_source, server_profile_id, principal_id,
                        workspace_id, local_user_id,
                        operation_token, operation_kind, canonical_note_id,
                        owner_proof, lease_token, lease_expires_at, abandon_after,
                        expected_version, state, revision, failure_count,
                        next_retry_at, blocked_reason_code, created_at, updated_at
                    ) VALUES (?, 'local', '', '', ?, ?, ?, 'create', ?, ?, ?, ?, ?,
                              NULL, 'pending', 1, 0, ?, '', ?, ?)
                    """,
                    (
                        receipt_id,
                        safe_workspace_id,
                        safe_user_id,
                        safe_token,
                        note_id,
                        owner_proof,
                        lease_token,
                        lease_expires_at,
                        abandon_after,
                        now,
                        now,
                        now,
                    ),
                )
                row = conn.execute(
                    "SELECT * FROM research_quick_note_receipts WHERE receipt_id = ?",
                    (receipt_id,),
                ).fetchone()
                if (
                    row is not None
                    and row["state"] == "pending"
                    and _parse_utc_timestamp(str(row["lease_expires_at"]))
                    <= _parse_utc_timestamp(now)
                    and _parse_utc_timestamp(str(row["next_retry_at"]))
                    <= _parse_utc_timestamp(now)
                ):
                    updated_at = _monotonic_timestamp(str(row["updated_at"]), now)
                    renewed_until = _future_timestamp(
                        updated_at, _QUICK_NOTE_LEASE_SECONDS
                    )
                    conn.execute(
                        """
                        UPDATE research_quick_note_receipts
                        SET lease_token = ?, lease_expires_at = ?,
                            revision = revision + 1, updated_at = ?
                        WHERE receipt_id = ? AND local_user_id = ?
                          AND operation_token = ? AND operation_kind = 'create'
                          AND canonical_note_id = ? AND state = 'pending'
                          AND lease_token = ? AND revision = ?
                        """,
                        (
                            lease_token,
                            renewed_until,
                            updated_at,
                            receipt_id,
                            safe_user_id,
                            safe_token,
                            note_id,
                            str(row["lease_token"]),
                            int(row["revision"]),
                        ),
                    )
                    row = conn.execute(
                        "SELECT * FROM research_quick_note_receipts WHERE receipt_id = ?",
                        (receipt_id,),
                    ).fetchone()
        except sqlite3.Error as exc:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        if row is None:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE)
        receipt = _quick_note_receipt_from_row(row)
        if (
            receipt.workspace_id != safe_workspace_id
            or receipt.local_user_id != safe_user_id
            or receipt.operation_token != safe_token
            or receipt.canonical_note_id != note_id
            or receipt.operation_kind != "create"
        ):
            raise WorkspaceRegistryServiceError("Quick Note receipt identity mismatch.")
        return receipt

    def claim_quick_note_delete(
        self,
        workspace_id: str,
        *,
        local_user_id: str,
        canonical_note_id: str,
        expected_version: int,
    ) -> ResearchQuickNoteReceipt:
        """Durably record a delete before mutating its canonical Notes owner."""

        safe_workspace_id = _normalize_required_text(workspace_id, "workspace_id")
        safe_user_id = _normalize_required_text(local_user_id, "local_user_id")
        safe_note_id = _normalize_required_text(canonical_note_id, "canonical_note_id")
        if type(expected_version) is not int or expected_version < 1:
            raise ValueError("expected_version must be a positive integer")
        delete_seed = _length_prefixed_identity(
            "chatbook-quick-note-delete-v2",
            "local",
            "",
            "",
            safe_workspace_id,
            safe_user_id,
            safe_note_id,
            str(expected_version),
        )
        operation_token = "research-note-delete-" + uuid5(
            NAMESPACE_URL, delete_seed
        ).hex
        receipt_id, _ = self._quick_note_identity(
            workspace_id=safe_workspace_id,
            local_user_id=safe_user_id,
            operation_token=operation_token,
            kind="delete",
        )
        now = self._now_factory()
        owner_proof = _opaque_receipt_token(
            self._receipt_token_factory, "owner_proof"
        )
        lease_token = _opaque_receipt_token(
            self._receipt_token_factory, "lease_token"
        )
        lease_expires_at = _future_timestamp(now, _QUICK_NOTE_LEASE_SECONDS)
        abandon_after = _future_timestamp(now, _QUICK_NOTE_ABANDON_SECONDS)
        try:
            with self.db.transaction(immediate=True) as conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO research_quick_note_receipts (
                        receipt_id, data_source, server_profile_id, principal_id,
                        workspace_id, local_user_id,
                        operation_token, operation_kind, canonical_note_id,
                        owner_proof, lease_token, lease_expires_at, abandon_after,
                        expected_version, state, revision, failure_count,
                        next_retry_at, blocked_reason_code, created_at, updated_at
                    ) VALUES (?, 'local', '', '', ?, ?, ?, 'delete', ?, ?, ?, ?, ?,
                              ?, 'pending', 1, 0, ?, '', ?, ?)
                    """,
                    (
                        receipt_id,
                        safe_workspace_id,
                        safe_user_id,
                        operation_token,
                        safe_note_id,
                        owner_proof,
                        lease_token,
                        lease_expires_at,
                        abandon_after,
                        expected_version,
                        now,
                        now,
                        now,
                    ),
                )
                row = conn.execute(
                    "SELECT * FROM research_quick_note_receipts WHERE receipt_id = ?",
                    (receipt_id,),
                ).fetchone()
                if (
                    row is not None
                    and row["state"] != "blocked"
                    and _parse_utc_timestamp(str(row["lease_expires_at"]))
                    <= _parse_utc_timestamp(now)
                    and _parse_utc_timestamp(str(row["next_retry_at"]))
                    <= _parse_utc_timestamp(now)
                ):
                    updated_at = _monotonic_timestamp(str(row["updated_at"]), now)
                    renewed_until = _future_timestamp(
                        updated_at, _QUICK_NOTE_LEASE_SECONDS
                    )
                    conn.execute(
                        """
                        UPDATE research_quick_note_receipts
                        SET lease_token = ?, lease_expires_at = ?,
                            revision = revision + 1, updated_at = ?
                        WHERE receipt_id = ? AND local_user_id = ?
                          AND workspace_id = ? AND operation_kind = 'delete'
                          AND canonical_note_id = ? AND expected_version = ?
                          AND lease_token = ? AND revision = ?
                        """,
                        (
                            lease_token,
                            renewed_until,
                            updated_at,
                            receipt_id,
                            safe_user_id,
                            safe_workspace_id,
                            safe_note_id,
                            expected_version,
                            str(row["lease_token"]),
                            int(row["revision"]),
                        ),
                    )
                    row = conn.execute(
                        "SELECT * FROM research_quick_note_receipts WHERE receipt_id = ?",
                        (receipt_id,),
                    ).fetchone()
        except sqlite3.Error as exc:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        if row is None:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE)
        receipt = _quick_note_receipt_from_row(row)
        if (
            receipt.workspace_id != safe_workspace_id
            or receipt.local_user_id != safe_user_id
            or receipt.canonical_note_id != safe_note_id
            or receipt.expected_version != expected_version
            or receipt.operation_kind != "delete"
        ):
            raise WorkspaceRegistryServiceError("Quick Note receipt identity mismatch.")
        return receipt

    def list_quick_note_receipts(
        self,
        local_user_id: str,
        *,
        workspace_id: str | None = None,
        operation_kind: str | None = None,
        include_blocked: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[tuple[ResearchQuickNoteReceipt, ...], int]:
        """List a bounded, owner-filtered receipt page for reconciliation."""

        safe_user_id = _normalize_required_text(local_user_id, "local_user_id")
        safe_workspace_id = (
            _normalize_required_text(workspace_id, "workspace_id")
            if workspace_id is not None
            else None
        )
        if operation_kind is not None and operation_kind not in {"create", "delete"}:
            raise ValueError("operation_kind is invalid")
        if type(include_blocked) is not bool:
            raise TypeError("include_blocked must be a bool")
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        if type(offset) is not int or not 0 <= offset <= 10_000:
            raise ValueError("offset must be between 0 and 10000")
        now = self._now_factory()
        params: tuple[object, ...] = (
            safe_user_id,
            safe_workspace_id,
            safe_workspace_id,
            operation_kind,
            operation_kind,
        )
        actionable_sql = ""
        if not include_blocked:
            actionable_sql = """
              AND state <> 'blocked'
              AND julianday(next_retry_at) <= julianday(?)
              AND julianday(lease_expires_at) <= julianday(?)
            """
            params += (now, now)
        try:
            with self.db.connection() as conn:
                total = int(
                    conn.execute(
                        f"""
                        SELECT COUNT(*) FROM research_quick_note_receipts
                        WHERE local_user_id = ?
                          AND (? IS NULL OR workspace_id = ?)
                          AND (? IS NULL OR operation_kind = ?)
                          {actionable_sql}
                        """,
                        params,
                    ).fetchone()[0]
                )
                rows = conn.execute(
                    f"""
                    SELECT * FROM research_quick_note_receipts
                    WHERE local_user_id = ?
                      AND (? IS NULL OR workspace_id = ?)
                      AND (? IS NULL OR operation_kind = ?)
                      {actionable_sql}
                    ORDER BY
                        julianday(next_retry_at) ASC,
                        julianday(updated_at) ASC,
                        receipt_id ASC
                    LIMIT ? OFFSET ?
                    """,
                    params + (limit, offset),
                ).fetchall()
        except sqlite3.Error as exc:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        return tuple(_quick_note_receipt_from_row(row) for row in rows), total

    def claim_quick_note_recovery(
        self,
        receipt_id: str,
        local_user_id: str,
        *,
        expected_revision: int,
        expected_lease_token: str,
    ) -> ResearchQuickNoteReceipt:
        """Claim one expired receipt before inspecting either durable owner."""

        safe_receipt_id = _normalize_required_text(receipt_id, "receipt_id")
        safe_user_id = _normalize_required_text(local_user_id, "local_user_id")
        safe_lease_token = _normalize_required_text(
            expected_lease_token, "expected_lease_token"
        )
        if type(expected_revision) is not int or expected_revision < 1:
            raise ValueError("expected_revision must be a positive integer")
        now = self._now_factory()
        new_lease_token = _opaque_receipt_token(
            self._receipt_token_factory, "lease_token"
        )
        try:
            with self.db.transaction(immediate=True) as conn:
                current = conn.execute(
                    """
                    SELECT * FROM research_quick_note_receipts
                    WHERE receipt_id = ? AND local_user_id = ?
                      AND revision = ? AND lease_token = ?
                      AND state <> 'blocked'
                      AND julianday(next_retry_at) <= julianday(?)
                      AND julianday(lease_expires_at) <= julianday(?)
                    """,
                    (
                        safe_receipt_id,
                        safe_user_id,
                        expected_revision,
                        safe_lease_token,
                        now,
                        now,
                    ),
                ).fetchone()
                if current is None:
                    raise WorkspaceRegistryServiceError(
                        "Quick Note receipt changed; reload and retry."
                    )
                updated_at = _monotonic_timestamp(str(current["updated_at"]), now)
                lease_expires_at = _future_timestamp(
                    updated_at, _QUICK_NOTE_LEASE_SECONDS
                )
                cursor = conn.execute(
                    """
                    UPDATE research_quick_note_receipts
                    SET lease_token = ?, lease_expires_at = ?,
                        revision = revision + 1, updated_at = ?
                    WHERE receipt_id = ? AND local_user_id = ?
                      AND data_source = ? AND server_profile_id = ?
                      AND principal_id = ? AND workspace_id = ?
                      AND operation_token = ? AND operation_kind = ?
                      AND canonical_note_id = ? AND revision = ?
                      AND lease_token = ?
                    """,
                    (
                        new_lease_token,
                        lease_expires_at,
                        updated_at,
                        safe_receipt_id,
                        safe_user_id,
                        current["data_source"],
                        current["server_profile_id"],
                        current["principal_id"],
                        current["workspace_id"],
                        current["operation_token"],
                        current["operation_kind"],
                        current["canonical_note_id"],
                        expected_revision,
                        safe_lease_token,
                    ),
                )
                if cursor.rowcount != 1:
                    raise WorkspaceRegistryServiceError(
                        "Quick Note receipt changed; reload and retry."
                    )
                row = conn.execute(
                    "SELECT * FROM research_quick_note_receipts WHERE receipt_id = ?",
                    (safe_receipt_id,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        if row is None:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE)
        return _quick_note_receipt_from_row(row)

    def mark_quick_note_owner_committed(
        self,
        receipt_id: str,
        local_user_id: str,
        *,
        expected_revision: int,
        expected_lease_token: str,
    ) -> ResearchQuickNoteReceipt:
        """Advance one receipt from pending to owner_committed exactly once."""

        safe_receipt_id = _normalize_required_text(receipt_id, "receipt_id")
        safe_user_id = _normalize_required_text(local_user_id, "local_user_id")
        safe_lease_token = _normalize_required_text(
            expected_lease_token, "expected_lease_token"
        )
        if type(expected_revision) is not int or expected_revision < 1:
            raise ValueError("expected_revision must be a positive integer")
        now = self._now_factory()
        try:
            with self.db.transaction(immediate=True) as conn:
                current = conn.execute(
                    """
                    SELECT * FROM research_quick_note_receipts
                    WHERE receipt_id = ? AND local_user_id = ?
                      AND lease_token = ?
                    """,
                    (safe_receipt_id, safe_user_id, safe_lease_token),
                ).fetchone()
                if current is None:
                    raise WorkspaceRegistryServiceError(
                        "Quick Note receipt changed; reload and retry."
                    )
                updated_at = _monotonic_timestamp(str(current["updated_at"]), now)
                lease_expires_at = _future_timestamp(
                    updated_at, _QUICK_NOTE_LEASE_SECONDS
                )
                cursor = conn.execute(
                    """
                    UPDATE research_quick_note_receipts
                    SET state = 'owner_committed', revision = revision + 1,
                        lease_expires_at = ?, next_retry_at = ?,
                        blocked_reason_code = '', updated_at = ?
                    WHERE receipt_id = ? AND local_user_id = ?
                      AND operation_kind = ? AND canonical_note_id = ?
                      AND state = 'pending' AND revision = ? AND lease_token = ?
                    """,
                    (
                        lease_expires_at,
                        updated_at,
                        updated_at,
                        safe_receipt_id,
                        safe_user_id,
                        current["operation_kind"],
                        current["canonical_note_id"],
                        expected_revision,
                        safe_lease_token,
                    ),
                )
                row = conn.execute(
                    """
                    SELECT * FROM research_quick_note_receipts
                    WHERE receipt_id = ? AND local_user_id = ?
                      AND lease_token = ?
                    """,
                    (safe_receipt_id, safe_user_id, safe_lease_token),
                ).fetchone()
                if row is None or (
                    cursor.rowcount == 0
                    and not (
                        row["state"] == "owner_committed"
                        and int(row["revision"]) >= expected_revision + 1
                    )
                ):
                    raise WorkspaceRegistryServiceError(
                        "Quick Note receipt changed; reload and retry."
                    )
        except sqlite3.Error as exc:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        return _quick_note_receipt_from_row(row)

    def discard_quick_note_receipt(
        self,
        receipt_id: str,
        local_user_id: str,
        *,
        expected_revision: int,
        expected_lease_token: str,
    ) -> bool:
        """Discard a receipt after proving no canonical recovery work remains."""

        safe_receipt_id = _normalize_required_text(receipt_id, "receipt_id")
        safe_user_id = _normalize_required_text(local_user_id, "local_user_id")
        safe_lease_token = _normalize_required_text(
            expected_lease_token, "expected_lease_token"
        )
        try:
            with self.db.transaction(immediate=True) as conn:
                cursor = conn.execute(
                    """
                    DELETE FROM research_quick_note_receipts
                    WHERE receipt_id = ? AND local_user_id = ?
                      AND revision = ? AND lease_token = ?
                    """,
                    (
                        safe_receipt_id,
                        safe_user_id,
                        expected_revision,
                        safe_lease_token,
                    ),
                )
                return cursor.rowcount > 0
        except sqlite3.Error as exc:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE) from exc

    def discard_abandoned_quick_note_receipt(
        self,
        receipt_id: str,
        local_user_id: str,
        *,
        expected_revision: int,
        expected_lease_token: str,
    ) -> bool:
        """Discard missing-owner recovery only after its durable grace."""

        safe_receipt_id = _normalize_required_text(receipt_id, "receipt_id")
        safe_user_id = _normalize_required_text(local_user_id, "local_user_id")
        safe_lease_token = _normalize_required_text(
            expected_lease_token, "expected_lease_token"
        )
        now = self._now_factory()
        try:
            with self.db.transaction(immediate=True) as conn:
                cursor = conn.execute(
                    """
                    DELETE FROM research_quick_note_receipts
                    WHERE receipt_id = ? AND local_user_id = ?
                      AND revision = ? AND lease_token = ?
                      AND julianday(abandon_after) <= julianday(?)
                    """,
                    (
                        safe_receipt_id,
                        safe_user_id,
                        expected_revision,
                        safe_lease_token,
                        now,
                    ),
                )
                return cursor.rowcount == 1
        except sqlite3.Error as exc:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE) from exc

    def record_quick_note_failure(
        self,
        receipt_id: str,
        local_user_id: str,
        *,
        expected_revision: int,
        expected_lease_token: str,
        reason_code: str,
        permanent: bool = False,
    ) -> ResearchQuickNoteReceipt:
        """Record one sanitized failure with bounded backoff and CAS fencing."""

        safe_receipt_id = _normalize_required_text(receipt_id, "receipt_id")
        safe_user_id = _normalize_required_text(local_user_id, "local_user_id")
        safe_lease_token = _normalize_required_text(
            expected_lease_token, "expected_lease_token"
        )
        if type(expected_revision) is not int or expected_revision < 1:
            raise ValueError("expected_revision must be a positive integer")
        if reason_code not in _QUICK_NOTE_FAILURE_REASONS:
            raise ValueError("reason_code is invalid")
        if type(permanent) is not bool:
            raise TypeError("permanent must be a bool")
        now = self._now_factory()
        try:
            with self.db.transaction(immediate=True) as conn:
                current = conn.execute(
                    """
                    SELECT * FROM research_quick_note_receipts
                    WHERE receipt_id = ? AND local_user_id = ?
                      AND revision = ? AND lease_token = ?
                    """,
                    (
                        safe_receipt_id,
                        safe_user_id,
                        expected_revision,
                        safe_lease_token,
                    ),
                ).fetchone()
                if current is None:
                    raise WorkspaceRegistryServiceError(
                        "Quick Note receipt changed; reload and retry."
                    )
                failure_count = min(
                    _QUICK_NOTE_MAX_FAILURES, int(current["failure_count"]) + 1
                )
                blocked = permanent or (
                    reason_code != "owner_missing"
                    and failure_count >= _QUICK_NOTE_MAX_FAILURES
                )
                state = "blocked" if blocked else str(current["state"])
                updated_at = _monotonic_timestamp(str(current["updated_at"]), now)
                next_retry_at = _future_timestamp(
                    updated_at,
                    300 if reason_code == "owner_missing" else min(300, 2**failure_count),
                )
                conn.execute(
                    """
                    UPDATE research_quick_note_receipts
                    SET state = ?, revision = revision + 1,
                        failure_count = ?, next_retry_at = ?,
                        lease_expires_at = ?, blocked_reason_code = ?, updated_at = ?
                    WHERE receipt_id = ? AND local_user_id = ? AND revision = ?
                      AND operation_kind = ? AND canonical_note_id = ?
                      AND lease_token = ?
                    """,
                    (
                        state,
                        failure_count,
                        next_retry_at,
                        updated_at,
                        reason_code,
                        updated_at,
                        safe_receipt_id,
                        safe_user_id,
                        expected_revision,
                        current["operation_kind"],
                        current["canonical_note_id"],
                        safe_lease_token,
                    ),
                )
                row = conn.execute(
                    """
                    SELECT * FROM research_quick_note_receipts
                    WHERE receipt_id = ? AND local_user_id = ?
                    """,
                    (safe_receipt_id, safe_user_id),
                ).fetchone()
        except sqlite3.Error as exc:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        if row is None:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE)
        return _quick_note_receipt_from_row(row)

    def project_quick_note_create(
        self,
        receipt_id: str,
        local_user_id: str,
        *,
        expected_revision: int,
        expected_lease_token: str,
        title: str,
    ) -> ResearchQuickNoteReceipt:
        """Atomically link the owner and persist proof-cleanup-pending state."""

        safe_receipt_id = _normalize_required_text(receipt_id, "receipt_id")
        safe_user_id = _normalize_required_text(local_user_id, "local_user_id")
        safe_lease_token = _normalize_required_text(
            expected_lease_token, "expected_lease_token"
        )
        safe_title = str(title).strip()
        now = self._now_factory()
        try:
            with self.db.transaction(immediate=True) as conn:
                row = conn.execute(
                    """
                    SELECT * FROM research_quick_note_receipts
                    WHERE receipt_id = ? AND local_user_id = ?
                      AND operation_kind = 'create'
                      AND state = 'owner_committed' AND revision = ?
                      AND lease_token = ?
                    """,
                    (
                        safe_receipt_id,
                        safe_user_id,
                        expected_revision,
                        safe_lease_token,
                    ),
                ).fetchone()
                if row is None:
                    raise WorkspaceRegistryServiceError(
                        "Quick Note receipt changed; reload and retry."
                    )
                conn.execute(
                    """
                    INSERT OR IGNORE INTO workspace_memberships (
                        membership_id, workspace_id, item_type, item_id, role,
                        transfer_policy, title, created_at
                    ) VALUES (?, ?, 'note', ?, 'note', 'reference', ?, ?)
                    """,
                    (
                        self._id_factory(),
                        row["workspace_id"],
                        row["canonical_note_id"],
                        safe_title,
                        now,
                    ),
                )
                updated_at = _monotonic_timestamp(str(row["updated_at"]), now)
                lease_expires_at = _future_timestamp(
                    updated_at, _QUICK_NOTE_LEASE_SECONDS
                )
                cursor = conn.execute(
                    """
                    UPDATE research_quick_note_receipts
                    SET state = 'projection_committed', revision = revision + 1,
                        lease_expires_at = ?, next_retry_at = ?,
                        blocked_reason_code = '', updated_at = ?
                    WHERE receipt_id = ? AND local_user_id = ?
                      AND operation_kind = 'create'
                      AND canonical_note_id = ? AND state = 'owner_committed'
                      AND revision = ? AND lease_token = ?
                    """,
                    (
                        lease_expires_at,
                        updated_at,
                        updated_at,
                        safe_receipt_id,
                        safe_user_id,
                        row["canonical_note_id"],
                        expected_revision,
                        safe_lease_token,
                    ),
                )
                if cursor.rowcount != 1:
                    raise WorkspaceRegistryServiceError(
                        "Quick Note receipt changed; reload and retry."
                    )
                projected = conn.execute(
                    "SELECT * FROM research_quick_note_receipts WHERE receipt_id = ?",
                    (safe_receipt_id,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        if projected is None:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE)
        return _quick_note_receipt_from_row(projected)

    def complete_quick_note_create(
        self,
        receipt_id: str,
        local_user_id: str,
        *,
        expected_revision: int,
        expected_lease_token: str,
    ) -> bool:
        """Consume a projection only after the caller removed its owner proof."""

        safe_receipt_id = _normalize_required_text(receipt_id, "receipt_id")
        safe_user_id = _normalize_required_text(local_user_id, "local_user_id")
        safe_lease_token = _normalize_required_text(
            expected_lease_token, "expected_lease_token"
        )
        try:
            with self.db.transaction(immediate=True) as conn:
                cursor = conn.execute(
                    """
                    DELETE FROM research_quick_note_receipts
                    WHERE receipt_id = ? AND local_user_id = ?
                      AND operation_kind = 'create'
                      AND state = 'projection_committed' AND revision = ?
                      AND lease_token = ?
                    """,
                    (
                        safe_receipt_id,
                        safe_user_id,
                        expected_revision,
                        safe_lease_token,
                    ),
                )
                return cursor.rowcount == 1
        except sqlite3.Error as exc:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE) from exc

    def complete_quick_note_delete(
        self,
        receipt_id: str,
        local_user_id: str,
        *,
        expected_revision: int,
        expected_lease_token: str,
    ) -> bool:
        """Atomically remove every projection and consume a committed delete."""

        safe_receipt_id = _normalize_required_text(receipt_id, "receipt_id")
        safe_user_id = _normalize_required_text(local_user_id, "local_user_id")
        safe_lease_token = _normalize_required_text(
            expected_lease_token, "expected_lease_token"
        )
        try:
            with self.db.transaction(immediate=True) as conn:
                row = conn.execute(
                    """
                    SELECT * FROM research_quick_note_receipts
                    WHERE receipt_id = ? AND local_user_id = ?
                      AND operation_kind = 'delete'
                      AND state = 'owner_committed' AND revision = ?
                      AND lease_token = ?
                    """,
                    (
                        safe_receipt_id,
                        safe_user_id,
                        expected_revision,
                        safe_lease_token,
                    ),
                ).fetchone()
                if row is None:
                    return False
                note_id = str(row["canonical_note_id"])
                conn.execute(
                    "DELETE FROM workspace_memberships WHERE item_type = 'note' AND item_id = ?",
                    (note_id,),
                )
                scope_rows = conn.execute(
                    "SELECT workspace_id, payload FROM workspace_rag_scopes"
                ).fetchall()
                for scope_row in scope_rows:
                    try:
                        scope = parse_scope(json.loads(scope_row["payload"]))
                    except (TypeError, ValueError):
                        scope = None
                    if scope is None:
                        conn.execute(
                            "DELETE FROM workspace_rag_scopes WHERE workspace_id = ?",
                            (scope_row["workspace_id"],),
                        )
                        continue
                    remaining = tuple(
                        item
                        for item in scope.items
                        if not (
                            item.source_type == "note" and item.source_id == note_id
                        )
                    )
                    if remaining == scope.items:
                        continue
                    if not remaining and not scope.empty_is_scoped:
                        conn.execute(
                            "DELETE FROM workspace_rag_scopes WHERE workspace_id = ?",
                            (scope_row["workspace_id"],),
                        )
                        continue
                    updated = RagScope(
                        items=remaining,
                        updated_at=self._now_factory(),
                        empty_is_scoped=scope.empty_is_scoped,
                    )
                    conn.execute(
                        """
                        UPDATE workspace_rag_scopes SET payload = ?, updated_at = ?
                        WHERE workspace_id = ?
                        """,
                        (
                            json.dumps(serialize_scope(updated)),
                            updated.updated_at,
                            scope_row["workspace_id"],
                        ),
                    )
                conn.execute(
                    """
                    DELETE FROM research_quick_note_receipts
                    WHERE receipt_id = ? AND local_user_id = ?
                      AND canonical_note_id = ? AND revision = ?
                      AND lease_token = ?
                    """,
                    (
                        safe_receipt_id,
                        safe_user_id,
                        note_id,
                        expected_revision,
                        safe_lease_token,
                    ),
                )
                return True
        except sqlite3.Error as exc:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE) from exc

    def get_workspace_source_membership(
        self, workspace_id: str, membership_id: str
    ) -> WorkspaceMembership | None:
        """Return one source membership by its association identity."""

        safe_workspace_id = _normalize_required_text(workspace_id, "workspace_id")
        safe_membership_id = _normalize_required_text(
            membership_id, "membership_id"
        )
        try:
            with self.db.connection() as conn:
                row = conn.execute(
                    """
                    SELECT *
                    FROM workspace_memberships
                    WHERE workspace_id = ?
                        AND membership_id = ?
                        AND item_type = 'media'
                        AND role = 'source'
                    """,
                    (safe_workspace_id, safe_membership_id),
                ).fetchone()
        except sqlite3.Error as exc:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        return _membership_from_row(row) if row is not None else None

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
        self._bump_mutation_generation()
        stored = self.get_runtime_binding(safe_binding.binding_id)
        if stored is None:
            raise WorkspaceRegistryServiceError("Runtime binding save failed.")
        return stored

    def add_folder_binding(
        self,
        workspace_id: str,
        path: str | Path,
        *,
        allow_write: bool = False,
    ) -> WorkspaceRuntimeBinding:
        """Bind a folder as a file-tool access root (spec §2).

        Read-only by default; canonical (resolved) locator; denies the
        filesystem root, the home directory itself, non-directories,
        duplicate/nested roots within the same workspace, and any root
        that is or contains a path ``Utils.sensitive_paths`` protects
        (TASK-857). Default-workspace and unknown-workspace rejection is
        delegated to ``save_runtime_binding``.
        """
        resolved = _validate_folder_path_rules(path)
        _validate_folder_overlap(
            resolved,
            [b.locator for b in self.list_folder_bindings(workspace_id)],
        )
        binding = WorkspaceRuntimeBinding(
            workspace_id=workspace_id,
            binding_id=f"folder-{uuid4().hex[:12]}",
            binding_kind=RuntimeBindingKind.LOCAL_FILESYSTEM,
            label=resolved.name or str(resolved),
            locator=str(resolved),
            status=RuntimeBindingStatus.READY,
            metadata={"access": "rw" if allow_write else "ro"},
        )
        binding_result = self.save_runtime_binding(binding)
        try:
            if self._change_review_binding_owner is not None:
                self._change_review_binding_owner.binding_added(
                    workspace_id,
                    binding_result,
                )
        except Exception:  # noqa: BLE001 -- persistence must never fail on observer
            logger.opt(exception=True).debug(
                "change_review: binding observer failed after registration"
            )
        return binding_result

    def list_folder_bindings(
        self, workspace_id: str
    ) -> tuple[WorkspaceRuntimeBinding, ...]:
        """Local-filesystem bindings with status recomputed from disk."""
        bindings = self.list_runtime_bindings(workspace_id)
        # Filter to only local-filesystem bindings
        filtered = [
            b
            for b in bindings
            if str(b.binding_kind)
            in ("local-filesystem", str(RuntimeBindingKind.LOCAL_FILESYSTEM))
        ]
        refreshed: list[WorkspaceRuntimeBinding] = []
        for binding in filtered:
            actual = (
                RuntimeBindingStatus.MISSING
                if _filesystem_binding_missing(binding.locator)
                else RuntimeBindingStatus.READY
            )
            refreshed.append(
                WorkspaceRuntimeBinding(
                    workspace_id=binding.workspace_id,
                    binding_id=binding.binding_id,
                    binding_kind=binding.binding_kind,
                    label=binding.label,
                    locator=binding.locator,
                    status=actual,
                    metadata=binding.metadata,
                    created_at=binding.created_at,
                    updated_at=binding.updated_at,
                )
            )
        return tuple(refreshed)

    def remove_runtime_binding(self, binding_id: str) -> None:
        """Delete a runtime binding row (spec §2)."""
        safe_binding_id = _normalize_required_text(binding_id, "binding_id")
        try:
            with self.db.transaction() as conn:
                cursor = conn.execute(
                    "DELETE FROM workspace_runtime_bindings WHERE binding_id = ?",
                    (safe_binding_id,),
                )
        except sqlite3.Error as exc:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        if cursor.rowcount == 0:
            raise BindingNotFound(safe_binding_id)
        self._bump_mutation_generation()

    def set_folder_binding_access(
        self, binding_id: str, *, allow_write: bool
    ) -> WorkspaceRuntimeBinding:
        """Flip a folder binding's ro/rw access flag (spec §4 toggle)."""
        existing = self.get_runtime_binding(binding_id)
        if existing is None:
            raise BindingNotFound(binding_id)
        metadata = dict(existing.metadata)
        metadata["access"] = "rw" if allow_write else "ro"
        return self.save_runtime_binding(
            WorkspaceRuntimeBinding(
                workspace_id=existing.workspace_id,
                binding_id=existing.binding_id,
                binding_kind=existing.binding_kind,
                label=existing.label,
                locator=existing.locator,
                status=existing.status,
                metadata=metadata,
                created_at=existing.created_at,
            )
        )

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
        if row["workspace_id"] == DEFAULT_WORKSPACE_ID:
            self._delete_default_runtime_bindings()
            return None
        binding = _runtime_binding_from_row(row)
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

    def read_change_review_consent(self, workspace_id: str) -> ChangeReviewConsent:
        """Read explicit per-workspace Change Review consent and revision.

        Args:
            workspace_id: Workspace identifier.

        Returns:
            Enabled/disabled state with its opaque revision. A missing row is
            disabled with the missing sentinel; storage failure is unavailable.
        """
        safe_workspace_id = _normalize_required_text(workspace_id, "workspace_id")
        try:
            with self.db.connection() as conn:
                row = conn.execute(
                    """
                    SELECT enabled, updated_at FROM workspace_change_review
                    WHERE workspace_id = ?
                    """,
                    (safe_workspace_id,),
                ).fetchone()
        except sqlite3.Error:
            logger.debug("change_review: workspace consent read unavailable")
            return ChangeReviewConsent(ChangeReviewState.UNAVAILABLE)
        if row is None:
            return ChangeReviewConsent(
                ChangeReviewState.DISABLED,
                MISSING_CHANGE_REVIEW_REVISION,
            )
        return ChangeReviewConsent(
            ChangeReviewState.ENABLED
            if bool(row["enabled"])
            else ChangeReviewState.DISABLED,
            str(row["updated_at"]),
        )

    def change_review_enabled(self, workspace_id: str) -> bool:
        """Whether this workspace has explicit enabled Change Review consent."""
        return (
            self.read_change_review_consent(workspace_id).state
            is ChangeReviewState.ENABLED
        )

    def compare_and_set_change_review_consent(
        self,
        workspace_id: str,
        *,
        expected: ChangeReviewConsent,
        enabled: bool,
    ) -> ChangeReviewConsent:
        """Persist consent only when state and opaque revision still match.

        Args:
            workspace_id: Workspace identifier.
            expected: Exact state/revision previously observed by the caller.
            enabled: New explicit consent value.

        Returns:
            The newly committed consent observation.

        Raises:
            ChangeReviewStateConflict: If ``expected`` is stale.
            WorkspaceRegistryServiceError: If the transaction fails.
        """
        safe_workspace_id = _normalize_required_text(workspace_id, "workspace_id")
        new_revision = f"{self._now_factory()}:{uuid4().hex}"
        try:
            with self.db.transaction() as conn:
                row = conn.execute(
                    """
                    SELECT enabled, updated_at FROM workspace_change_review
                    WHERE workspace_id = ?
                    """,
                    (safe_workspace_id,),
                ).fetchone()
                current = (
                    ChangeReviewConsent(
                        ChangeReviewState.DISABLED,
                        MISSING_CHANGE_REVIEW_REVISION,
                    )
                    if row is None
                    else ChangeReviewConsent(
                        ChangeReviewState.ENABLED
                        if bool(row["enabled"])
                        else ChangeReviewState.DISABLED,
                        str(row["updated_at"]),
                    )
                )
                if current != expected:
                    raise ChangeReviewStateConflict(
                        "Change Review state changed; refresh and try again."
                    )
                conn.execute(
                    """
                    INSERT INTO workspace_change_review
                        (workspace_id, enabled, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(workspace_id) DO UPDATE SET
                        enabled = excluded.enabled,
                        updated_at = excluded.updated_at
                    """,
                    (safe_workspace_id, 1 if enabled else 0, new_revision),
                )
        except ChangeReviewStateConflict:
            raise
        except sqlite3.Error as exc:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        return ChangeReviewConsent(
            ChangeReviewState.ENABLED if enabled else ChangeReviewState.DISABLED,
            new_revision,
        )

    def set_change_review_enabled(self, workspace_id: str, enabled: bool) -> None:
        """Unconditionally persist explicit consent for administrative callers.

        Args:
            workspace_id: Workspace identifier.
            enabled: Whether the workspace's roots are tracked.

        Raises:
            WorkspaceRegistryServiceError: If the write fails.
        """
        safe_workspace_id = _normalize_required_text(workspace_id, "workspace_id")
        new_revision = f"{self._now_factory()}:{uuid4().hex}"
        try:
            with self.db.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO workspace_change_review
                        (workspace_id, enabled, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(workspace_id) DO UPDATE SET
                        enabled = excluded.enabled,
                        updated_at = excluded.updated_at
                    """,
                    (
                        safe_workspace_id,
                        1 if enabled else 0,
                        new_revision,
                    ),
                )
        except sqlite3.Error as exc:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE) from exc

    def get_workspace_scope(self, workspace_id: str) -> RagScope | None:
        """Return a workspace's stored RAG retrieval scope, guarded.

        Reads the ``workspace_rag_scopes`` table -- co-located with the
        workspace registry (this database), not ``ChaChaNotes_DB``, since a
        workspace has no row there to hang metadata off of (design spec
        section 2). Guarded end to end, mirroring
        ``Chat.rag_scope.read_conversation_scope``: a missing row, malformed
        payload JSON, a non-dict payload, or a malformed/forward-versioned
        scope payload (``parse_scope``'s own guards) all fail closed to an
        explicit empty scope. A missing row remains unscoped. An ordinary
        stored zero-item scope still reads as ``None``; only Research's
        explicit-empty encoding and corrupt existing rows narrow retrieval.

        Args:
            workspace_id: Workspace identifier.

        Returns:
            The stored ``RagScope``; ``None`` only when missing, cleared, or
            ordinarily empty. Malformed existing rows return explicit empty.

        Raises:
            WorkspaceRegistryServiceError: If the underlying read fails.
        """

        safe_workspace_id = _normalize_required_text(workspace_id, "workspace_id")
        try:
            with self.db.connection() as conn:
                row = conn.execute(
                    """
                    SELECT payload, updated_at
                    FROM workspace_rag_scopes
                    WHERE workspace_id = ?
                    """,
                    (safe_workspace_id,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE) from exc
        if row is None:
            return None
        try:
            raw = json.loads(row["payload"])
        except (TypeError, ValueError):
            logger.warning(
                "workspace rag_scope payload malformed for {}; failing closed",
                safe_workspace_id,
            )
            return RagScope(
                items=(),
                updated_at=str(row["updated_at"] or "corrupt"),
                empty_is_scoped=True,
            )
        scope = parse_scope(raw)
        if scope is None:
            safe_stamp = raw.get("updated_at") if isinstance(raw, dict) else None
            return RagScope(
                items=(),
                updated_at=(
                    safe_stamp
                    if isinstance(safe_stamp, str)
                    else str(row["updated_at"] or "corrupt")
                ),
                empty_is_scoped=True,
            )
        if scope is not None and not scope.items and not scope.empty_is_scoped:
            return None
        return scope

    def reconcile_research_source_selection(
        self,
        workspace_id: str,
        *,
        media_id: str,
        desired_selected: bool,
    ) -> RagScope | None:
        """Atomically reconcile one attached Media item into Research scope.

        A missing scope means every workspace source is implicitly selected.
        Selecting another source therefore leaves the row absent, while
        deselecting one materializes all other representable source
        memberships. Existing explicit scopes change only the target Media
        item. Malformed stored state starts from explicit empty so a repair
        cannot widen retrieval.

        Args:
            workspace_id: Workspace owning the source association.
            media_id: Canonical Local Media id already linked as a source.
            desired_selected: Whether that Media item should be desired.

        Returns:
            The persisted explicit scope, or ``None`` when implicit selection
            remains authoritative.

        Raises:
            ValueError: If an identity is invalid or selection is not boolean.
            WorkspaceRegistryServiceError: If the item is not attached or the
                storage update fails.
        """

        safe_workspace_id = _normalize_required_text(workspace_id, "workspace_id")
        safe_media_id = _normalize_required_text(media_id, "media_id")
        if type(desired_selected) is not bool:
            raise ValueError("desired_selected must be a bool")
        try:
            with self.db.transaction(immediate=True) as conn:
                memberships = conn.execute(
                    """
                    SELECT item_type, item_id
                    FROM workspace_memberships
                    WHERE workspace_id = ?
                        AND role = 'source'
                        AND item_type IN ('media', 'note')
                    ORDER BY created_at ASC, item_type ASC, item_id ASC,
                        membership_id ASC
                    """,
                    (safe_workspace_id,),
                ).fetchall()
                if not any(
                    row["item_type"] == "media" and row["item_id"] == safe_media_id
                    for row in memberships
                ):
                    raise WorkspaceRegistryServiceError(
                        "Research source is not attached to this workspace."
                    )

                row = conn.execute(
                    """
                    SELECT payload
                    FROM workspace_rag_scopes
                    WHERE workspace_id = ?
                    """,
                    (safe_workspace_id,),
                ).fetchone()
                if row is None and desired_selected:
                    return None

                if row is None:
                    existing = [
                        ScopeItem(item["item_type"], item["item_id"])
                        for item in memberships
                        if not (
                            item["item_type"] == "media"
                            and item["item_id"] == safe_media_id
                        )
                    ]
                else:
                    try:
                        raw_scope = json.loads(row["payload"])
                    except (TypeError, ValueError):
                        raw_scope = None
                    scope = parse_scope(raw_scope)
                    existing = list(scope.items if scope is not None else ())
                    existing = [
                        item
                        for item in existing
                        if not (
                            item.source_type == "media"
                            and item.source_id == safe_media_id
                        )
                    ]
                    if desired_selected:
                        existing.append(ScopeItem("media", safe_media_id))

                scope = RagScope(
                    items=tuple(existing),
                    updated_at=self._now_factory(),
                    empty_is_scoped=True,
                )
                conn.execute(
                    """
                    INSERT INTO workspace_rag_scopes (
                        workspace_id, payload, updated_at
                    )
                    VALUES (?, ?, ?)
                    ON CONFLICT(workspace_id) DO UPDATE SET
                        payload = excluded.payload,
                        updated_at = excluded.updated_at
                    """,
                    (
                        safe_workspace_id,
                        json.dumps(serialize_scope(scope)),
                        scope.updated_at,
                    ),
                )
                return scope
        except sqlite3.Error as exc:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE) from exc

    def set_workspace_scope(self, workspace_id: str, scope: RagScope | None) -> None:
        """Persist or clear a workspace's RAG retrieval scope.

        ``scope=None`` deletes the stored row entirely. A zero-item scope
        also normalizes to a delete, mirroring
        ``Chat.rag_scope.write_conversation_scope``'s "save with zero
        selected clears the scope" contract (design spec section 4). Research
        explicitly opts into a distinct zero-item scope with
        ``empty_is_scoped=True``; ordinary Console callers do not.

        Deleting a scope for a workspace id that has none (or that does not
        exist) is a harmless no-op. Setting a non-empty scope for a
        workspace id that does not exist in ``workspace_records`` raises
        ``WorkspaceNotFound`` (enforced by the table's foreign key, mirroring
        ``link_membership``'s existence check).

        Args:
            workspace_id: Workspace identifier.
            scope: The scope to persist, or ``None``/empty to clear it.

        Raises:
            WorkspaceNotFound: If ``scope`` is non-empty and ``workspace_id``
                does not reference an existing workspace.
            WorkspaceRegistryServiceError: If the underlying write fails.
        """

        safe_workspace_id = _normalize_required_text(workspace_id, "workspace_id")
        if scope is not None and not scope.items and not scope.empty_is_scoped:
            scope = None
        try:
            with self.db.transaction() as conn:
                if scope is None:
                    conn.execute(
                        "DELETE FROM workspace_rag_scopes WHERE workspace_id = ?",
                        (safe_workspace_id,),
                    )
                else:
                    payload = json.dumps(serialize_scope(scope))
                    conn.execute(
                        """
                        INSERT INTO workspace_rag_scopes (
                            workspace_id, payload, updated_at
                        )
                        VALUES (?, ?, ?)
                        ON CONFLICT(workspace_id) DO UPDATE SET
                            payload = excluded.payload,
                            updated_at = excluded.updated_at
                        """,
                        (safe_workspace_id, payload, scope.updated_at),
                    )
        except sqlite3.IntegrityError as exc:
            raise WorkspaceNotFound(safe_workspace_id) from exc
        except sqlite3.Error as exc:
            raise WorkspaceRegistryServiceError(_STORAGE_FAILURE_MESSAGE) from exc

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
        # A binding row changed, so generation-keyed display-read caches
        # (TASK-22201) must revalidate; only reached when rows were deleted.
        self._bump_mutation_generation()

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
        self._bump_mutation_generation()


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


def _quick_note_receipt_from_row(row: sqlite3.Row) -> ResearchQuickNoteReceipt:
    return ResearchQuickNoteReceipt(
        receipt_id=row["receipt_id"],
        data_source=row["data_source"],
        server_profile_id=row["server_profile_id"],
        principal_id=row["principal_id"],
        workspace_id=row["workspace_id"],
        local_user_id=row["local_user_id"],
        operation_token=row["operation_token"],
        operation_kind=row["operation_kind"],
        canonical_note_id=row["canonical_note_id"],
        owner_proof=row["owner_proof"],
        lease_token=row["lease_token"],
        lease_expires_at=row["lease_expires_at"],
        abandon_after=row["abandon_after"],
        expected_version=row["expected_version"],
        state=row["state"],
        revision=int(row["revision"]),
        failure_count=int(row["failure_count"]),
        next_retry_at=row["next_retry_at"],
        blocked_reason_code=row["blocked_reason_code"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
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


def next_local_workspace_identity(
    registry_service: LocalWorkspaceRegistryService,
) -> tuple[str, str]:
    """Return a collision-free local workspace id and display name.

    Scans the registry for existing workspace ids and names, then returns the
    first ``workspace-local-<n>`` / ``Workspace <n>`` pair that is not already
    in use. The helper is shared between Library and Console so both create
    local workspaces with the same naming scheme.

    Args:
        registry_service: The local workspace registry to check for collisions.

    Returns:
        A tuple of ``(workspace_id, workspace_name)``.
    """
    existing_workspaces = tuple(registry_service.list_workspaces(include_archived=True))
    existing_ids = {workspace.workspace_id for workspace in existing_workspaces}
    existing_names = {workspace.name for workspace in existing_workspaces}
    index = 1
    while True:
        workspace_id = f"workspace-local-{index}"
        workspace_name = f"Workspace {index}"
        if workspace_id not in existing_ids and workspace_name not in existing_names:
            return workspace_id, workspace_name
        index += 1
