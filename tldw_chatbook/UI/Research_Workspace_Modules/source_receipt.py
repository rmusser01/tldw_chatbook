"""Bounded, privacy-safe Research source operation receipts."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
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
    def __init__(self, index: int) -> None:
        super().__init__(id=f"research-source-receipt-{index}")
        self.index = index
        self.operation: ResearchSourceOperation | None = None

    def compose(self) -> ComposeResult:
        yield Static("", id=f"research-source-receipt-owner-{self.index}")
        yield Static("", id=f"research-source-receipt-stages-{self.index}")
        yield Static("", id=f"research-source-receipt-error-{self.index}")
        with Horizontal(classes="research-source-receipt-actions"):
            yield Button(
                "Retry",
                id=f"research-source-receipt-retry-{self.index}",
                compact=True,
                disabled=True,
            )

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
        self.query_one(f"#research-source-receipt-owner-{self.index}", Static).update(
            f"{operation.data_source.value.title()} workspace {operation.workspace_id} · "
            f"operation {operation.operation_id}"
        )
        self.query_one(f"#research-source-receipt-stages-{self.index}", Static).update(
            f"{catalog_owner}: {_status_label(operation.catalog_status)} | "
            f"Workspace association: {_status_label(operation.association_status)} | "
            f"Index/readiness: {_status_label(operation.readiness_status)}"
        )
        error = self.query_one(f"#research-source-receipt-error-{self.index}", Static)
        error.update(operation.error_message)
        error.display = bool(operation.error_message)
        retry = self.query_one(f"#research-source-receipt-retry-{self.index}", Button)
        labels = {
            SourceOperationStage.CATALOG: f"Retry {catalog_owner} ingest",
            SourceOperationStage.ASSOCIATION: "Retry workspace link",
            SourceOperationStage.READINESS: "Refresh / Recheck",
        }
        retry.label = labels.get(operation.error_stage, "Retry")
        retry.disabled = operation.error_stage is None


class ResearchSourceReceiptList(Vertical):
    """Twenty stable receipt slots plus a bounded-result disclosure."""

    class RetryRequested(Message):
        def __init__(self, operation_id: str, stage: SourceOperationStage) -> None:
            super().__init__()
            self.operation_id = operation_id
            self.stage = stage

    def compose(self) -> ComposeResult:
        yield Static(
            "Receipts · Library/Media | Workspace association | Index/readiness",
            id="research-source-receipts-heading",
        )
        for index in range(MAX_VISIBLE_SOURCE_RECEIPTS):
            yield _ResearchSourceReceiptSlot(index)
        yield Static("", id="research-source-receipts-bound")

    def on_mount(self) -> None:
        self.sync_operations((), incomplete=False)

    def sync_operations(
        self,
        operations: tuple[ResearchSourceOperation, ...],
        *,
        incomplete: bool,
    ) -> None:
        for index, slot in enumerate(self.query(_ResearchSourceReceiptSlot)):
            slot.sync_operation(operations[index] if index < len(operations) else None)
        bound = self.query_one("#research-source-receipts-bound", Static)
        bound.update(
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
