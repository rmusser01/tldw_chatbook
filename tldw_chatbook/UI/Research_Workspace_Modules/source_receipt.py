"""Bounded, privacy-safe Research source operation receipts."""

from __future__ import annotations

from textual import on
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Static

from ...Research_Workspace.source_operations import (
    CanonicalItemType,
    ResearchSourceOperation,
    SourceOperationStage,
    SourceOperationStatus,
)


MAX_VISIBLE_SOURCE_RECEIPTS = 20


def _status_label(status: SourceOperationStatus) -> str:
    return status.value.replace("_", " ").title()


class _ResearchSourceReceiptSlot(Vertical):
    """One recyclable receipt row.

    Children are built at construction and kept as direct references so
    ``sync_operation`` stays synchronous even while the slot's subtree is
    still being mounted (TASK-23024) — slots mount on demand as operations
    arrive.
    """

    def __init__(self, index: int) -> None:
        owner = Static("", id=f"research-source-receipt-owner-{index}")
        stages = Static("", id=f"research-source-receipt-stages-{index}")
        error = Static("", id=f"research-source-receipt-error-{index}")
        retry = Button(
            "Retry",
            id=f"research-source-receipt-retry-{index}",
            compact=True,
            disabled=True,
        )
        super().__init__(
            owner,
            stages,
            error,
            Horizontal(retry, classes="research-source-receipt-actions"),
            id=f"research-source-receipt-{index}",
        )
        self.index = index
        self.operation: ResearchSourceOperation | None = None
        self._owner = owner
        self._stages = stages
        self._error = error
        self._retry = retry

    def sync_operation(self, operation: ResearchSourceOperation | None) -> None:
        self.operation = operation
        self.display = operation is not None
        if operation is None:
            return
        catalog_owner = (
            "Library"
            if operation.canonical_item_type is CanonicalItemType.LOCAL_LIBRARY
            else "Media"
        )
        self._owner.update(
            f"{operation.data_source.value.title()} workspace {operation.workspace_id} · "
            f"operation {operation.operation_id}"
        )
        self._stages.update(
            f"{catalog_owner}: {_status_label(operation.catalog_status)} | "
            f"Workspace association: {_status_label(operation.association_status)} | "
            f"Index/readiness: {_status_label(operation.readiness_status)}"
        )
        error = self._error
        error.update(operation.error_message)
        error.display = bool(operation.error_message)
        retry = self._retry
        labels = {
            SourceOperationStage.CATALOG: f"Retry {catalog_owner} ingest",
            SourceOperationStage.ASSOCIATION: "Retry workspace link",
            SourceOperationStage.READINESS: "Refresh / Recheck",
        }
        retry.label = labels.get(operation.error_stage, "Retry")
        retry.disabled = operation.error_stage is None


class ResearchSourceReceiptList(Vertical):
    """Demand-grown receipt slots plus a bounded-result disclosure.

    The pool starts empty (TASK-23024) and grows with the operations shown,
    capped at ``MAX_VISIBLE_SOURCE_RECEIPTS``; surplus slots recycle via
    ``display = False`` and are never unmounted. New slots always mount
    before the bounded-result disclosure so child order stays
    heading, receipts…, bound.
    """

    class RetryRequested(Message):
        def __init__(self, operation_id: str, stage: SourceOperationStage) -> None:
            super().__init__()
            self.operation_id = operation_id
            self.stage = stage

    def __init__(self, *args, **kwargs) -> None:
        heading = Static(
            "Receipts · Library/Media | Workspace association | Index/readiness",
            id="research-source-receipts-heading",
        )
        bound = Static("", id="research-source-receipts-bound")
        super().__init__(heading, bound, *args, **kwargs)
        self._bound = bound
        self._slots: list[_ResearchSourceReceiptSlot] = []

    def _ensure_slot_pool(self, needed: int) -> None:
        """Grow the mounted slot pool to ``needed`` receipts (never shrink)."""

        needed = min(needed, MAX_VISIBLE_SOURCE_RECEIPTS)
        if needed <= len(self._slots):
            return
        new_slots = [
            _ResearchSourceReceiptSlot(index)
            for index in range(len(self._slots), needed)
        ]
        self._slots.extend(new_slots)
        if self.is_attached:
            self._mount_slots(new_slots)

    def _mount_slots(self, slots: list[_ResearchSourceReceiptSlot]) -> None:
        if self._bound.parent is self:
            self.mount(*slots, before=self._bound)
        else:
            self.mount(*slots)

    def on_mount(self) -> None:
        detached = [slot for slot in self._slots if slot.parent is None]
        if detached:
            self._mount_slots(detached)
        self.sync_operations((), incomplete=False)

    def sync_operations(
        self,
        operations: tuple[ResearchSourceOperation, ...],
        *,
        incomplete: bool,
    ) -> None:
        self._ensure_slot_pool(len(operations))
        for index, slot in enumerate(self._slots):
            slot.sync_operation(operations[index] if index < len(operations) else None)
        self._bound.update(
            "More receipts may exist · open Library operation status for the full history."
            if incomplete or len(operations) > MAX_VISIBLE_SOURCE_RECEIPTS
            else f"{len(operations)} recent operation(s)."
        )

    @on(Button.Pressed, ".research-source-receipt-actions Button")
    def retry_stage(self, event: Button.Pressed) -> None:
        slot = event.button.parent
        while slot is not None and not isinstance(slot, _ResearchSourceReceiptSlot):
            slot = slot.parent
        operation = (
            slot.operation if isinstance(slot, _ResearchSourceReceiptSlot) else None
        )
        if operation is not None and operation.error_stage is not None:
            self.post_message(
                self.RetryRequested(operation.operation_id, operation.error_stage)
            )
