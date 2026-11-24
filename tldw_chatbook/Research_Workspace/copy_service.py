"""Execute confirmed workspace Copy items under their durable owner checkpoints."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from loguru import logger

from tldw_chatbook.Backup_Recovery.participants import run_finite_local_worker
from tldw_chatbook.tldw_api.exceptions import APIConnectionError, APIResponseError

from .contracts import QualifiedWorkspaceRef, WorkspaceDataSource
from .copy_models import (
    CopyConflictChoice,
    CopyOwnerRef,
    CopyTreatment,
    ResearchCopyConfirmation,
    ResearchCopyItem,
    ResearchCopyManifest,
    ResearchCopyStatus,
)
from .copy_receipts import (
    CopyItemReceipt,
    CopyItemStage,
    CopyReceiptConflict,
    CopyReceiptFailure,
    ResearchCopyReceipt,
    ResearchCopyReceiptStore,
)
from .server_migration_receipts import ServerMigrationReceiptService


@dataclass(frozen=True, slots=True)
class CopySourceSnapshot:
    owner: CopyOwnerRef
    body: object = field(repr=False)


class CopyOwnerActionError(ValueError):
    def __init__(self, failure: CopyReceiptFailure, *, retryable: bool = False) -> None:
        super().__init__(failure.value)
        self.failure = failure
        self.retryable = retryable


class CopyItemAdapter(Protocol):
    async def inspect_source(self, ref: CopyOwnerRef) -> CopySourceSnapshot: ...

    async def create_destination(
        self,
        item: ResearchCopyItem,
        source: CopySourceSnapshot,
        workspace: QualifiedWorkspaceRef,
        item_key: str,
    ) -> CopyOwnerRef: ...

    async def replace_destination(
        self,
        item: ResearchCopyItem,
        source: CopySourceSnapshot,
        item_key: str,
    ) -> CopyOwnerRef: ...

    async def reference_destination(
        self,
        item: ResearchCopyItem,
        source: CopySourceSnapshot,
    ) -> CopyOwnerRef: ...

    async def associate_destination(
        self, item: ResearchCopyItem, destination: CopyOwnerRef
    ) -> None: ...

    async def verify_destination(
        self, item: ResearchCopyItem, destination: CopyOwnerRef
    ) -> bool: ...


def _retryable(error: Exception) -> bool:
    if isinstance(error, CopyOwnerActionError):
        return error.retryable
    if isinstance(error, APIResponseError):
        return error.status_code in {408, 425, 429} or error.status_code >= 500
    return isinstance(error, (APIConnectionError, OSError, TimeoutError))


def _notify_progress(
    observer: Callable[[ResearchCopyReceipt], None] | None, receipt: ResearchCopyReceipt
) -> None:
    if observer is None:
        return
    try:
        observer(receipt)
    except Exception as exc:
        logger.warning(
            "Copy progress display failed (exception_category={})", type(exc).__name__
        )


def _checkpoint(receipt: ResearchCopyReceipt) -> dict:
    return {"expected_revision": receipt.revision, "lease_token": receipt.lease_token}


class ResearchCopyService:
    """Call only registered canonical adapters for a confirmed, captured transfer."""

    def __init__(
        self,
        receipts: ResearchCopyReceiptStore,
        owner_for: Callable[
            [ResearchCopyItem, QualifiedWorkspaceRef], CopyItemAdapter | None
        ],
        *,
        assert_current: Callable[[ResearchCopyManifest], None],
        migrations: ServerMigrationReceiptService | None = None,
    ) -> None:
        self.receipts = receipts
        self.owner_for = owner_for
        self.assert_current = assert_current
        self.migrations = migrations

    @staticmethod
    async def _off_loop(function, *args, **kwargs):
        return await asyncio.to_thread(
            run_finite_local_worker, function, *args, **kwargs
        )

    def _require_migrations(
        self, manifest: ResearchCopyManifest
    ) -> ServerMigrationReceiptService | None:
        if manifest.destination.data_source is WorkspaceDataSource.LOCAL:
            return None
        if self.migrations is None:
            raise ValueError("Server Copy needs an authenticated migration service")
        return self.migrations

    async def execute_copy(
        self,
        manifest: ResearchCopyManifest,
        confirmation: ResearchCopyConfirmation,
        *,
        cancel_event: asyncio.Event | None = None,
        on_progress: Callable[[ResearchCopyReceipt], None] | None = None,
    ) -> ResearchCopyReceipt:
        confirmation.validate(manifest)
        self.assert_current(manifest)
        self._require_migrations(manifest)
        if any(
            item.actionable and self.owner_for(item, manifest.destination) is None
            for item in manifest.items
        ):
            raise ValueError(
                "A canonical Copy owner is unavailable. Refresh the preflight."
            )
        stored = await self._off_loop(
            self.receipts.create_from_manifest, manifest, confirmation
        )
        return await self._execute(
            stored, cancel_event=cancel_event, on_progress=on_progress
        )

    async def get_copy_receipt(
        self,
        transfer_id: str,
        *,
        workspace: QualifiedWorkspaceRef,
    ) -> ResearchCopyReceipt | None:
        receipt = await self._off_loop(self.receipts.get, transfer_id)
        if receipt is not None and workspace in (
            receipt.manifest.source,
            receipt.manifest.destination,
        ):
            return receipt
        return None

    async def resume_copy(
        self,
        transfer_id: str,
        *,
        workspace: QualifiedWorkspaceRef,
        cancel_event: asyncio.Event | None = None,
        on_progress: Callable[[ResearchCopyReceipt], None] | None = None,
    ) -> ResearchCopyReceipt:
        receipt = await self.get_copy_receipt(transfer_id, workspace=workspace)
        if receipt is None:
            raise ValueError(
                "This Copy receipt belongs to another workspace or is unavailable"
            )
        return await self._execute(
            receipt, cancel_event=cancel_event, on_progress=on_progress
        )

    async def _finish(
        self, receipt: ResearchCopyReceipt, status: ResearchCopyStatus
    ) -> ResearchCopyReceipt:
        return await self._off_loop(
            self.receipts.finish,
            receipt.manifest.transfer_id,
            status,
            **_checkpoint(receipt),
        )

    async def _execute(
        self,
        receipt: ResearchCopyReceipt,
        *,
        cancel_event: asyncio.Event | None = None,
        on_progress: Callable[[ResearchCopyReceipt], None] | None = None,
    ) -> ResearchCopyReceipt:
        manifest = receipt.manifest
        self.assert_current(manifest)
        migrations = self._require_migrations(manifest)
        if receipt.status in {
            ResearchCopyStatus.COMPLETED,
            ResearchCopyStatus.ROLLED_BACK,
            ResearchCopyStatus.FAILED_TERMINAL,
        }:
            return receipt
        receipt = await self._off_loop(
            self.receipts.claim,
            manifest.transfer_id,
            expected_revision=receipt.revision,
        )
        _notify_progress(on_progress, receipt)
        if cancel_event is not None and cancel_event.is_set():
            return await self._finish(receipt, ResearchCopyStatus.FAILED_RETRYABLE)
        if migrations is not None:
            try:
                state = await migrations.begin(
                    manifest,
                    target_name=manifest.target_name
                    or f"Research copy {manifest.manifest_hash[:8]}",
                )
                self.assert_current(manifest)
            except Exception as exc:
                return await self._finish(receipt, self._failure_status(receipt, exc))
            receipt = await self._off_loop(
                self.receipts.record_server_migration,
                manifest.transfer_id,
                state.id,
                finalized=state.status == "finalized",
                **_checkpoint(receipt),
            )
        all_acknowledged = True
        for item in receipt.items:
            if cancel_event is not None and cancel_event.is_set():
                outcome = self._failure_status(receipt, TimeoutError())
                return await self._finish(receipt, outcome)
            if not item.item.actionable:
                all_acknowledged &= item.item.treatment is CopyTreatment.OMIT
                continue
            if item.rolled_back or (
                item.failure is not CopyReceiptFailure.NONE and not item.retryable
            ):
                all_acknowledged = False
                continue
            receipt = await self._off_loop(
                self.receipts.renew,
                manifest.transfer_id,
                **_checkpoint(receipt),
            )
            receipt, acknowledged = await self._apply_item(receipt, item, migrations)
            all_acknowledged &= acknowledged
            _notify_progress(on_progress, receipt)
        if cancel_event is not None and cancel_event.is_set():
            return await self._finish(
                receipt, self._failure_status(receipt, TimeoutError())
            )
        if not all_acknowledged:
            return await self._finish(receipt, self._failure_status(receipt))
        if migrations is not None and not receipt.server_migration_finalized:
            try:
                self.assert_current(manifest)
                state = await migrations.finalize(manifest)
                self.assert_current(manifest)
            except Exception as exc:
                return await self._finish(receipt, self._failure_status(receipt, exc))
            receipt = await self._off_loop(
                self.receipts.record_server_migration,
                manifest.transfer_id,
                state.id,
                finalized=True,
                **_checkpoint(receipt),
            )
        return await self._finish(receipt, ResearchCopyStatus.COMPLETED)

    @staticmethod
    def _failure_status(
        receipt: ResearchCopyReceipt, error: Exception | None = None
    ) -> ResearchCopyStatus:
        if any(
            row.stage is CopyItemStage.VERIFIED and not row.rolled_back
            for row in receipt.items
        ):
            return ResearchCopyStatus.PARTIALLY_COMPLETED
        if any(row.retryable for row in receipt.items) or (
            error is not None and _retryable(error)
        ):
            return ResearchCopyStatus.FAILED_RETRYABLE
        return ResearchCopyStatus.FAILED_TERMINAL

    async def _record(
        self,
        receipt: ResearchCopyReceipt,
        item: CopyItemReceipt,
        stage: CopyItemStage,
        *,
        destination: CopyOwnerRef | None = None,
        failure: CopyReceiptFailure = CopyReceiptFailure.NONE,
        retryable: bool = False,
    ) -> tuple[ResearchCopyReceipt, CopyItemReceipt]:
        saved = await self._off_loop(
            self.receipts.record_item_stage,
            receipt.manifest.transfer_id,
            item.item_key,
            stage,
            destination=destination,
            failure=failure,
            retryable=retryable,
            **_checkpoint(receipt),
        )
        updated = next(row for row in saved.items if row.item_key == item.item_key)
        return saved, updated

    async def _create(
        self,
        manifest: ResearchCopyManifest,
        item: CopyItemReceipt,
        owner: CopyItemAdapter,
    ) -> CopyOwnerRef:
        source = await owner.inspect_source(item.item.source)
        self.assert_current(manifest)
        if source.owner != item.item.source:
            raise CopyOwnerActionError(CopyReceiptFailure.SOURCE_CHANGED)
        if item.item.conflict is CopyConflictChoice.REPLACE_DESTINATION:
            return await owner.replace_destination(item.item, source, item.item_key)
        if item.item.treatment is CopyTreatment.REFERENCE:
            return await owner.reference_destination(item.item, source)
        return await owner.create_destination(
            item.item, source, manifest.destination, item.item_key
        )

    async def _apply_item(
        self,
        receipt: ResearchCopyReceipt,
        item: CopyItemReceipt,
        migrations: ServerMigrationReceiptService | None,
    ) -> tuple[ResearchCopyReceipt, bool]:
        manifest = receipt.manifest
        failure = CopyReceiptFailure.OWNER_UNAVAILABLE
        try:
            self.assert_current(manifest)
            owner = self.owner_for(item.item, manifest.destination)
            if owner is None:
                raise CopyOwnerActionError(failure)
            if item.stage is CopyItemStage.PLANNED:
                destination = await self._create(manifest, item, owner)
                self.assert_current(manifest)
                receipt, item = await self._record(
                    receipt,
                    item,
                    CopyItemStage.DESTINATION_SAVED,
                    destination=destination,
                )
            if item.stage is CopyItemStage.DESTINATION_SAVED:
                failure = CopyReceiptFailure.ASSOCIATION_FAILED
                await owner.associate_destination(item.item, item.destination)
                self.assert_current(manifest)
                receipt, item = await self._record(
                    receipt, item, CopyItemStage.ASSOCIATED
                )
            failure = CopyReceiptFailure.VERIFICATION_FAILED
            verified = await owner.verify_destination(item.item, item.destination)
            self.assert_current(manifest)
            if verified is not True:
                raise CopyOwnerActionError(failure)
            if item.stage is not CopyItemStage.VERIFIED:
                receipt, item = await self._record(
                    receipt, item, CopyItemStage.VERIFIED
                )
            if migrations is not None:
                await migrations.ack_item(manifest, item)
                self.assert_current(manifest)
            return receipt, True
        except CopyReceiptConflict:
            raise
        except CopyOwnerActionError as exc:
            failure, retryable = exc.failure, exc.retryable
        except Exception as exc:
            logger.warning(
                "Research Copy item failed (exception_category={})", type(exc).__name__
            )
            retryable = _retryable(exc)
        if item.stage is not CopyItemStage.VERIFIED:
            receipt, _ = await self._record(
                receipt,
                item,
                item.stage,
                failure=failure,
                retryable=retryable,
            )
        return receipt, False
