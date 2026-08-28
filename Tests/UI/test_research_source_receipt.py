"""Mounted source-operation receipts expose independent durable stages."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Static

from tldw_chatbook.Research_Workspace.source_operations import (
    CanonicalItemType,
    ResearchSourceOperation,
    SourceOperationStage,
    SourceOperationStatus,
)
from tldw_chatbook.UI.Research_Workspace_Modules.source_receipt import (
    ResearchSourceReceiptList,
)


NOW = "2026-08-24T12:00:00Z"


def _operation(**changes) -> ResearchSourceOperation:
    values = {
        "operation_id": "operation-1",
        "idempotency_key": "task4:operation-1",
        "data_source": "local",
        "workspace_id": "workspace-local",
        "canonical_item_type": CanonicalItemType.LOCAL_LIBRARY,
        "desired_selected": True,
        "created_at": NOW,
        "updated_at": NOW,
        "ingest_job_id": "job-1",
        "canonical_item_id": "41",
        "workspace_source_id": "membership-41",
        "catalog_status": SourceOperationStatus.SUCCEEDED,
        "association_status": SourceOperationStatus.SUCCEEDED,
        "readiness_status": SourceOperationStatus.FAILED,
        "error_stage": SourceOperationStage.READINESS,
        "error_code": "readiness_refresh_failed",
        "error_message": "Readiness could not be refreshed.",
        "revision": 7,
    }
    values.update(changes)
    return ResearchSourceOperation(**values)


class _ReceiptHarness(App[None]):
    def compose(self) -> ComposeResult:
        yield ResearchSourceReceiptList(id="receipts")


@pytest.mark.asyncio
async def test_receipt_renders_three_stages_and_readiness_refresh_copy() -> None:
    app = _ReceiptHarness()
    async with app.run_test(size=(80, 20)) as pilot:
        await pilot.pause()
        receipts = app.query_one(ResearchSourceReceiptList)
        receipts.sync_operations((_operation(),), incomplete=False)
        await pilot.pause()

        painted = " ".join(
            str(widget.render()) for widget in receipts.query(Static) if widget.display
        )
        assert "Library: Succeeded" in painted
        assert "Workspace association: Succeeded" in painted
        assert "Index/readiness: Failed" in painted
        assert "Local workspace workspace-local" in painted
        assert "Readiness could not be refreshed." in painted
        retry = receipts.query_one("#research-source-receipt-retry-0", Button)
        assert retry.label.plain == "Refresh / Recheck"
        assert not retry.disabled


@pytest.mark.asyncio
async def test_receipt_catalog_failure_offers_stage_specific_retry() -> None:
    app = _ReceiptHarness()
    failed = _operation(
        canonical_item_id="",
        workspace_source_id="",
        catalog_status=SourceOperationStatus.FAILED,
        association_status=SourceOperationStatus.PENDING,
        readiness_status=SourceOperationStatus.PENDING,
        error_stage=SourceOperationStage.CATALOG,
        error_code="catalog_ingest_failed",
        error_message="Catalog ingest did not complete successfully.",
        revision=3,
    )
    async with app.run_test(size=(80, 20)) as pilot:
        await pilot.pause()
        receipts = app.query_one(ResearchSourceReceiptList)
        receipts.sync_operations((failed,), incomplete=True)
        await pilot.pause()

        assert (
            receipts.query_one("#research-source-receipt-retry-0", Button).label.plain
            == "Retry Library ingest"
        )
        assert "More receipts may exist" in str(
            receipts.query_one("#research-source-receipts-bound", Static).render()
        )


@pytest.mark.asyncio
async def test_enabled_receipt_retry_emits_exact_operation_and_failed_stage() -> None:
    messages = []
    app = _ReceiptHarness()
    async with app.run_test(size=(80, 20), message_hook=messages.append) as pilot:
        await pilot.pause()
        receipts = app.query_one(ResearchSourceReceiptList)
        receipts.sync_operations((_operation(),), incomplete=False)
        # TASK-23024: the first sync mounts the demand-grown slot; let it
        # settle before pressing the retry button inside it.
        await pilot.pause()
        receipts.query_one("#research-source-receipt-retry-0", Button).press()
        await pilot.pause()

    retries = [
        message
        for message in messages
        if isinstance(message, ResearchSourceReceiptList.RetryRequested)
    ]
    assert len({id(message) for message in retries}) == 1
    assert retries[0].operation_id == "operation-1"
    assert retries[0].stage is SourceOperationStage.READINESS
