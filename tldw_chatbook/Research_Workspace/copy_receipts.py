"""Persist and fence Copy progress without storing canonical item bodies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
import json
from time import time
from uuid import uuid4

from tldw_chatbook.DB.Workspace_DB import WorkspaceDB

from .contracts import QualifiedWorkspaceRef, WorkspaceDataSource
from .copy_models import (
    CopyItemKind,
    CopyOwnerRef,
    CopyTreatment,
    ResearchCopyConfirmation,
    ResearchCopyItem,
    ResearchCopyManifest,
    ResearchCopyStatus,
    copy_identifier,
    validate_copy_workspace,
)


class CopyItemStage(StrEnum):
    PLANNED = "planned"
    DESTINATION_SAVED = "destination_saved"
    ASSOCIATED = "associated"
    VERIFIED = "verified"


class CopyReceiptFailure(StrEnum):
    NONE = ""
    OWNER_UNAVAILABLE = "owner_unavailable"
    SOURCE_CHANGED = "source_changed"
    DESTINATION_CHANGED = "destination_changed"
    AUTHORITY_CHANGED = "authority_changed"
    ASSOCIATION_FAILED = "association_failed"
    VERIFICATION_FAILED = "verification_failed"
    SERVER_MIGRATION_FAILED = "server_migration_failed"


class CopyReceiptConflict(ValueError):
    """The requested transition no longer matches the durable Copy state."""


_STAGE_ORDER = {stage: index for index, stage in enumerate(CopyItemStage)}
_FINISH_STATES = frozenset(
    {
        ResearchCopyStatus.COMPLETED,
        ResearchCopyStatus.PARTIALLY_COMPLETED,
        ResearchCopyStatus.ROLLED_BACK,
        ResearchCopyStatus.FAILED_RETRYABLE,
        ResearchCopyStatus.FAILED_TERMINAL,
    }
)
_RECOVERABLE_STATES = frozenset(
    {
        ResearchCopyStatus.CONFIRMED,
        ResearchCopyStatus.IN_PROGRESS,
        ResearchCopyStatus.PARTIALLY_COMPLETED,
        ResearchCopyStatus.FAILED_RETRYABLE,
    }
)


@dataclass(frozen=True, slots=True)
class CopyItemReceipt:
    item_key: str
    item: ResearchCopyItem
    stage: CopyItemStage
    destination: CopyOwnerRef | None
    failure: CopyReceiptFailure
    retryable: bool
    rolled_back: bool


@dataclass(frozen=True, slots=True)
class ResearchCopyReceipt:
    manifest: ResearchCopyManifest
    status: ResearchCopyStatus
    revision: int
    items: tuple[CopyItemReceipt, ...]
    replacement_confirmed: bool
    server_migration_id: str
    server_migration_finalized: bool
    lease_token: str
    lease_expires_at: int | None


def _identity(ref: QualifiedWorkspaceRef) -> tuple[str, str, str, str]:
    return (
        ref.data_source.value,
        ref.server_profile_id,
        ref.principal_id,
        ref.workspace_id,
    )


def _decode_owner(payload: dict) -> CopyOwnerRef:
    fields = dict(payload)
    workspace = QualifiedWorkspaceRef(**fields.pop("workspace"))
    return CopyOwnerRef(workspace, **fields)


def _decode_manifest(row) -> ResearchCopyManifest:
    payload = json.loads(row["manifest_json"])
    items = []
    for entry in payload["items"]:
        fields = dict(entry)
        source = _decode_owner(fields.pop("source"))
        destination = fields.pop("destination")
        items.append(
            ResearchCopyItem(
                source,
                destination=_decode_owner(destination) if destination else None,
                **fields,
            )
        )
    manifest = ResearchCopyManifest(
        QualifiedWorkspaceRef(**payload["source"]),
        QualifiedWorkspaceRef(**payload["destination"]),
        tuple(items),
        payload["created_at"],
        payload.get("target_name", ""),
    )
    if (manifest.transfer_id, manifest.manifest_hash) != (
        row["transfer_id"],
        row["manifest_hash"],
    ):
        raise CopyReceiptConflict("The frozen Copy manifest could not be verified")
    for role, ref in (
        ("source", manifest.source),
        ("destination", manifest.destination),
    ):
        recorded = tuple(
            row[f"{role}_{field}"]
            for field in (
                "data_source",
                "server_profile_id",
                "principal_id",
                "workspace_id",
            )
        )
        if recorded != _identity(ref):
            raise CopyReceiptConflict("The Copy receipt workspace identity changed")
    return manifest


class ResearchCopyReceiptStore:
    """Store a confirmation, lease and monotonic per-item owner checkpoints."""

    def __init__(self, db: WorkspaceDB, *, clock: Callable[[], float] = time) -> None:
        self.db = db
        self._clock = clock

    def _now(self) -> int:
        return int(self._clock())

    @staticmethod
    def _read(conn, transfer_id: str) -> ResearchCopyReceipt | None:
        row = conn.execute(
            "SELECT * FROM research_copy_operations WHERE transfer_id = ?",
            (transfer_id,),
        ).fetchone()
        if row is None:
            return None
        manifest = _decode_manifest(row)
        items = conn.execute(
            "SELECT * FROM research_copy_items WHERE transfer_id = ? ORDER BY ordinal",
            (transfer_id,),
        ).fetchall()
        if len(items) != len(manifest.items):
            raise CopyReceiptConflict("The Copy receipt has missing item rows")
        receipts = []
        for position, (item, stored) in enumerate(
            zip(manifest.items, items, strict=True)
        ):
            if stored["ordinal"] != position or stored[
                "item_key"
            ] != manifest.item_idempotency_key(item):
                raise CopyReceiptConflict(
                    "A Copy receipt item does not match the frozen manifest"
                )
            destination = None
            if stored["destination_owner_id"]:
                destination = CopyOwnerRef(
                    manifest.destination,
                    item.source.kind,
                    stored["destination_owner_id"],
                    stored["destination_owner_version"],
                    stored["destination_snapshot_hash"],
                )
            receipts.append(
                CopyItemReceipt(
                    stored["item_key"],
                    item,
                    CopyItemStage(stored["stage"]),
                    destination,
                    CopyReceiptFailure(stored["failure"]),
                    bool(stored["retryable"]),
                    bool(stored["rolled_back"]),
                )
            )
        return ResearchCopyReceipt(
            manifest,
            ResearchCopyStatus(row["status"]),
            row["revision"],
            tuple(receipts),
            bool(row["replacement_confirmed"]),
            row["server_migration_id"],
            bool(row["server_migration_finalized"]),
            row["lease_token"],
            row["lease_expires_at"],
        )

    def create_from_manifest(
        self, manifest: ResearchCopyManifest, confirmation: ResearchCopyConfirmation
    ) -> ResearchCopyReceipt:
        """Persist the exact confirmation before either canonical owner can change."""
        confirmation.validate(manifest)
        with self.db.transaction(immediate=True) as conn:
            previous = self._read(conn, manifest.transfer_id)
            if previous is not None:
                return previous
            conn.execute(
                """INSERT INTO research_copy_operations (
                    transfer_id, manifest_hash, manifest_json,
                    source_data_source, source_server_profile_id, source_principal_id, source_workspace_id,
                    destination_data_source, destination_server_profile_id, destination_principal_id, destination_workspace_id,
                    replacement_confirmed, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    manifest.transfer_id,
                    manifest.manifest_hash,
                    json.dumps(
                        asdict(manifest), ensure_ascii=True, separators=(",", ":")
                    ),
                    *_identity(manifest.source),
                    *_identity(manifest.destination),
                    int(confirmation.replacement_confirmed is True),
                    ResearchCopyStatus.CONFIRMED,
                    manifest.created_at,
                    self._now(),
                ),
            )
            conn.executemany(
                """INSERT INTO research_copy_items (transfer_id, item_key, ordinal, stage)
                   VALUES (?, ?, ?, ?)""",
                (
                    (
                        manifest.transfer_id,
                        manifest.item_idempotency_key(item),
                        index,
                        CopyItemStage.PLANNED,
                    )
                    for index, item in enumerate(manifest.items)
                ),
            )
            return self._require(conn, manifest.transfer_id)

    def get(self, transfer_id: str) -> ResearchCopyReceipt | None:
        copy_identifier(transfer_id, "transfer ID")
        with self.db.transaction() as conn:
            return self._read(conn, transfer_id)

    def list_recoverable(
        self, ref: QualifiedWorkspaceRef, *, limit: int = 20
    ) -> tuple[ResearchCopyReceipt, ...]:
        return self._list_history(ref, limit=limit, recoverable_only=True)

    def list_history(
        self, ref: QualifiedWorkspaceRef, *, limit: int = 20
    ) -> tuple[ResearchCopyReceipt, ...]:
        """Show recent confirmed operations for this exact workspace."""
        return self._list_history(ref, limit=limit, recoverable_only=False)

    def _list_history(
        self, ref, *, limit, recoverable_only
    ) -> tuple[ResearchCopyReceipt, ...]:
        validate_copy_workspace(ref)
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("Choose a bounded Copy receipt page")
        identity = _identity(ref)
        with self.db.transaction() as conn:
            rows = conn.execute(
                """SELECT transfer_id FROM research_copy_operations
                   WHERE (? = 0 OR status IN ('confirmed', 'in_progress', 'partially_completed', 'failed_retryable'))
                     AND ((source_data_source, source_server_profile_id, source_principal_id, source_workspace_id) = (?, ?, ?, ?)
                        OR (destination_data_source, destination_server_profile_id, destination_principal_id, destination_workspace_id) = (?, ?, ?, ?))
                   ORDER BY updated_at DESC, transfer_id LIMIT ?""",
                (int(recoverable_only), *identity, *identity, limit),
            ).fetchall()
            return tuple(self._require(conn, row["transfer_id"]) for row in rows)

    def previous_destinations(
        self, source: QualifiedWorkspaceRef, destination: QualifiedWorkspaceRef
    ) -> dict[tuple[CopyItemKind, str], CopyOwnerRef]:
        """Find the last receipted owner for each item in this exact authority pair."""
        validate_copy_workspace(source)
        validate_copy_workspace(destination)
        targets = {}
        with self.db.transaction() as conn:
            rows = conn.execute(
                """SELECT transfer_id FROM research_copy_operations
                   WHERE (source_data_source, source_server_profile_id, source_principal_id, source_workspace_id) = (?, ?, ?, ?)
                     AND (destination_data_source, destination_server_profile_id, destination_principal_id, destination_workspace_id) = (?, ?, ?, ?)
                     AND status IN ('completed', 'partially_completed')
                   ORDER BY updated_at DESC, created_at DESC, transfer_id LIMIT 20""",
                (*_identity(source), *_identity(destination)),
            ).fetchall()
            for row in rows:
                for item in self._require(conn, row["transfer_id"]).items:
                    if (
                        item.destination
                        and item.stage is CopyItemStage.VERIFIED
                        and not item.rolled_back
                    ):
                        targets.setdefault(
                            item.item.canonical_identity, item.destination
                        )
        return targets

    def _require(
        self, conn, transfer_id: str, expected_revision: int | None = None
    ) -> ResearchCopyReceipt:
        receipt = self._read(conn, transfer_id)
        if receipt is None:
            raise CopyReceiptConflict("The Copy receipt is unavailable")
        if expected_revision is not None and receipt.revision != expected_revision:
            raise CopyReceiptConflict("The Copy receipt revision changed")
        return receipt

    def _owned(
        self, conn, transfer_id: str, expected_revision: int, lease_token: str
    ) -> ResearchCopyReceipt:
        self._check_revision(expected_revision)
        current = self._require(conn, transfer_id, expected_revision)
        if (
            not lease_token
            or current.lease_token != lease_token
            or current.lease_expires_at is None
            or current.lease_expires_at <= self._now()
        ):
            raise CopyReceiptConflict(
                "Another worker owns this Copy receipt, or its lease expired"
            )
        return current

    def _publish(
        self, conn, previous: ResearchCopyReceipt, updated: ResearchCopyReceipt
    ) -> ResearchCopyReceipt:
        transfer_id = previous.manifest.transfer_id
        changed = conn.execute(
            """UPDATE research_copy_operations
                 SET status = ?, revision = revision + 1, server_migration_id = ?,
                     server_migration_finalized = ?, lease_token = ?, lease_expires_at = ?, updated_at = ?
               WHERE transfer_id = ? AND revision = ?""",
            (
                updated.status,
                updated.server_migration_id,
                int(updated.server_migration_finalized),
                updated.lease_token,
                updated.lease_expires_at,
                self._now(),
                transfer_id,
                previous.revision,
            ),
        )
        if changed.rowcount != 1:
            raise CopyReceiptConflict("The Copy receipt revision changed")
        return self._require(conn, transfer_id)

    @staticmethod
    def _check_revision(expected_revision: int) -> None:
        if type(expected_revision) is not int or expected_revision < 1:
            raise CopyReceiptConflict("Use the captured Copy receipt revision")

    @staticmethod
    def _check_lease_seconds(lease_seconds: int) -> None:
        if type(lease_seconds) is not int or not 1 <= lease_seconds <= 300:
            raise ValueError("Copy lease duration must be between 1 and 300 seconds")

    def claim(
        self, transfer_id: str, *, expected_revision: int, lease_seconds: int = 120
    ) -> ResearchCopyReceipt:
        self._check_revision(expected_revision)
        self._check_lease_seconds(lease_seconds)
        with self.db.transaction(immediate=True) as conn:
            current = self._require(conn, transfer_id, expected_revision)
            available = current.status in _RECOVERABLE_STATES
            if not available or (
                current.lease_expires_at is not None
                and current.lease_expires_at > self._now()
            ):
                raise CopyReceiptConflict(
                    "The Copy receipt is settled or owned by another worker"
                )
            return self._publish(
                conn,
                current,
                replace(
                    current,
                    status=ResearchCopyStatus.IN_PROGRESS,
                    lease_token=uuid4().hex,
                    lease_expires_at=self._now() + lease_seconds,
                ),
            )

    def renew(
        self,
        transfer_id: str,
        *,
        expected_revision: int,
        lease_token: str,
        lease_seconds: int = 120,
    ) -> ResearchCopyReceipt:
        self._check_lease_seconds(lease_seconds)
        with self.db.transaction(immediate=True) as conn:
            current = self._owned(conn, transfer_id, expected_revision, lease_token)
            return self._publish(
                conn,
                current,
                replace(current, lease_expires_at=self._now() + lease_seconds),
            )

    def record_server_migration(
        self,
        transfer_id: str,
        migration_id: str,
        *,
        expected_revision: int,
        lease_token: str,
        finalized: bool = False,
    ) -> ResearchCopyReceipt:
        copy_identifier(migration_id, "server migration ID")
        if type(finalized) is not bool:
            raise TypeError("Server migration finalization must be confirmed")
        with self.db.transaction(immediate=True) as conn:
            current = self._owned(conn, transfer_id, expected_revision, lease_token)
            if (
                current.manifest.destination.data_source
                is not WorkspaceDataSource.SERVER
            ):
                raise ValueError("A Server migration requires a Server destination")
            if (
                current.server_migration_id
                and current.server_migration_id != migration_id
            ):
                raise CopyReceiptConflict(
                    "This transfer already has a different Server migration"
                )
            updated = replace(
                current,
                server_migration_id=migration_id,
                server_migration_finalized=current.server_migration_finalized
                or finalized,
            )
            return (
                current if updated == current else self._publish(conn, current, updated)
            )

    def record_item_stage(
        self,
        transfer_id: str,
        item_key: str,
        stage: CopyItemStage,
        *,
        expected_revision: int,
        lease_token: str,
        destination: CopyOwnerRef | None = None,
        failure: CopyReceiptFailure = CopyReceiptFailure.NONE,
        retryable: bool = False,
        rolled_back: bool = False,
    ) -> ResearchCopyReceipt:
        stage, failure = CopyItemStage(stage), CopyReceiptFailure(failure)
        if type(retryable) is not bool or type(rolled_back) is not bool:
            raise TypeError("Copy retry and rollback must be explicit")
        if retryable and failure is CopyReceiptFailure.NONE:
            raise ValueError("Only a recorded failure can be retried")
        with self.db.transaction(immediate=True) as conn:
            current = self._owned(conn, transfer_id, expected_revision, lease_token)
            item = next(
                (row for row in current.items if row.item_key == item_key), None
            )
            if item is None or not item.item.actionable or item.rolled_back:
                raise CopyReceiptConflict(
                    "This Copy receipt item is unavailable for mutation"
                )
            if destination is not None and (
                not isinstance(destination, CopyOwnerRef)
                or destination.workspace != current.manifest.destination
                or destination.kind is not item.item.source.kind
                or not destination.has_revision
            ):
                raise ValueError(
                    "The confirmed owner must match the exact destination item"
                )
            owner = destination or item.destination
            if item.destination is not None and owner != item.destination:
                raise CopyReceiptConflict(
                    "The destination owner changed after its first acknowledgement"
                )
            before, after = _STAGE_ORDER[item.stage], _STAGE_ORDER[stage]
            if not (before <= after <= before + 1) or (failure and before != after):
                raise CopyReceiptConflict(
                    "Record the next stage after acknowledging its owner action"
                )
            if bool(owner) != (stage is not CopyItemStage.PLANNED):
                raise ValueError(
                    "A completed item stage requires a confirmed destination owner"
                )
            if (stage is CopyItemStage.VERIFIED and failure) or (
                rolled_back and (item.destination is None or failure or before != after)
            ):
                raise CopyReceiptConflict(
                    "The receipt does not support this final item outcome"
                )
            if (
                item.stage == stage
                and item.destination == owner
                and item.failure == failure
                and item.retryable == retryable
                and not rolled_back
            ):
                return current
            conn.execute(
                """UPDATE research_copy_items
                     SET stage = ?, destination_owner_id = ?, destination_owner_version = ?,
                         destination_snapshot_hash = ?, failure = ?, retryable = ?, rolled_back = ?
                   WHERE transfer_id = ? AND item_key = ?""",
                (
                    stage,
                    owner.owner_id if owner else "",
                    owner.version if owner else None,
                    owner.snapshot_hash if owner else "",
                    failure,
                    int(retryable),
                    int(rolled_back),
                    transfer_id,
                    item_key,
                ),
            )
            return self._publish(conn, current, current)

    def finish(
        self,
        transfer_id: str,
        status: ResearchCopyStatus,
        *,
        expected_revision: int,
        lease_token: str,
    ) -> ResearchCopyReceipt:
        status = ResearchCopyStatus(status)
        if status not in _FINISH_STATES:
            raise ValueError("Choose a Copy completion status")
        with self.db.transaction(immediate=True) as conn:
            current = self._owned(conn, transfer_id, expected_revision, lease_token)
            owned = tuple(row for row in current.items if row.item.actionable)
            verified = sum(
                row.stage is CopyItemStage.VERIFIED and not row.rolled_back
                for row in owned
            )
            blocked = any(
                row.item.treatment is CopyTreatment.BLOCKED for row in current.items
            )
            if status is ResearchCopyStatus.COMPLETED and (
                verified != len(owned) or blocked
            ):
                raise CopyReceiptConflict(
                    "Complete Copy requires verified owner receipts for every selected item"
                )
            if (
                status is ResearchCopyStatus.COMPLETED
                and current.manifest.destination.data_source
                is WorkspaceDataSource.SERVER
                and not current.server_migration_finalized
            ):
                raise CopyReceiptConflict(
                    "Complete Server Copy requires a finalized Server migration"
                )
            if status is ResearchCopyStatus.PARTIALLY_COMPLETED and not verified:
                raise CopyReceiptConflict(
                    "Partial completion requires a verified destination item"
                )
            if status is ResearchCopyStatus.ROLLED_BACK and (
                not any(row.rolled_back for row in owned)
                or any(row.destination and not row.rolled_back for row in owned)
            ):
                raise CopyReceiptConflict(
                    "Rollback requires acknowledgement from every modified owner"
                )
            return self._publish(
                conn,
                current,
                replace(current, status=status, lease_token="", lease_expires_at=None),
            )
