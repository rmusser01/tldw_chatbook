"""Demand-grown Research slot pools: empty is free, max still recycles.

TASK-23024: `ResearchSourceList` / `ResearchSourceReceiptList` used to
compose their full 25/20 slot pools eagerly, so an empty Research profile
mounted ~470 widgets in `display=False` subtrees on every visit. The pools
now start empty and grow with content; these tests pin the growth contract
and — because the pool exists to avoid mount/unmount churn — that recycling
at the maximum row count still mounts and unmounts nothing.
"""

from __future__ import annotations

import pytest
import textual.widget as textual_widget_module
from textual.app import App, ComposeResult
from textual.widgets import Button, Static

from tldw_chatbook.Research_Workspace import (
    QualifiedWorkspaceRef,
    ResearchCapability,
    ResearchSourcePage,
    ResearchSourceSummary,
    WorkspaceDataSource,
)
from tldw_chatbook.Research_Workspace.source_operations import (
    CanonicalItemType,
    ResearchSourceOperation,
    SourceOperationStage,
    SourceOperationStatus,
)
from tldw_chatbook.UI.Research_Workspace_Modules.source_list import (
    MAX_VISIBLE_SOURCE_ROWS,
    ResearchSourceList,
    _ResearchSourceRowSlot,
)
from tldw_chatbook.UI.Research_Workspace_Modules.source_receipt import (
    MAX_VISIBLE_SOURCE_RECEIPTS,
    ResearchSourceReceiptList,
    _ResearchSourceReceiptSlot,
)
from tldw_chatbook.UI.Research_Workspace_Modules.sources_region import (
    ResearchSourcesRegion,
)


REF = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "workspace-local")
AVAILABLE = ResearchCapability(True, "available", "Available.", "local")


class _RegionHarness(App[None]):
    def compose(self) -> ComposeResult:
        yield ResearchSourcesRegion(id="research-sources-pane")


def _page(tag: str, rows: int, *, desired: tuple[str, ...] = ()) -> ResearchSourcePage:
    items = tuple(
        ResearchSourceSummary(
            ref=REF,
            source_id=f"membership-{tag}-{index}",
            catalog_item_id=f"{tag}-{index}",
            title=f"Source {tag} {index}",
            source_type="pdf",
            updated_at="2026-08-24T00:00:00Z",
        )
        for index in range(rows)
    )
    return ResearchSourcePage(
        items=items, limit=rows or 1, total=rows, desired_source_ids=desired
    )


def _operations(tag: str, count: int) -> tuple[ResearchSourceOperation, ...]:
    return tuple(
        ResearchSourceOperation(
            operation_id=f"operation-{tag}-{index}",
            idempotency_key=f"probe:{tag}:{index}",
            data_source=WorkspaceDataSource.LOCAL,
            workspace_id="workspace-local",
            canonical_item_type=CanonicalItemType.LOCAL_LIBRARY,
            desired_selected=True,
            created_at="2026-08-24T12:00:00Z",
            updated_at="2026-08-24T12:00:00Z",
            ingest_job_id=f"job-{tag}-{index}",
            canonical_item_id=f"{tag}-{index}",
            workspace_source_id=f"membership-{tag}-{index}",
            catalog_status=SourceOperationStatus.SUCCEEDED,
            association_status=SourceOperationStatus.SUCCEEDED,
            readiness_status=SourceOperationStatus.FAILED,
            error_stage=SourceOperationStage.READINESS,
            error_code="readiness_refresh_failed",
            error_message="Readiness could not be refreshed.",
            revision=1,
        )
        for index in range(count)
    )


@pytest.mark.asyncio
async def test_empty_profile_mounts_zero_row_and_receipt_slots() -> None:
    """An unused Research profile pays for no slot widgets at all."""

    app = _RegionHarness()
    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        source_list = app.query_one(ResearchSourceList)
        receipts = app.query_one(ResearchSourceReceiptList)

        assert len(list(source_list.query(_ResearchSourceRowSlot))) == 0
        assert len(list(source_list.children)) == 0
        assert len(list(receipts.query(_ResearchSourceReceiptSlot))) == 0
        # The receipt frame is intact without any slots: heading then bound.
        child_ids = [child.id for child in receipts.children]
        assert child_ids == [
            "research-source-receipts-heading",
            "research-source-receipts-bound",
        ]
        assert "0 recent operation(s)." in str(
            receipts.query_one("#research-source-receipts-bound", Static).render()
        )


@pytest.mark.asyncio
async def test_slot_pool_grows_with_content_and_is_capped_at_max() -> None:
    """Slots mounted == min(rows, MAX); contents land in the grown slots."""

    app = _RegionHarness()
    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        source_list = app.query_one(ResearchSourceList)

        source_list.sync_page(_page("small", 3))
        await pilot.pause()
        assert len(list(source_list.query(_ResearchSourceRowSlot))) == 3
        assert "Source small 2" in str(
            source_list.query_one("#research-source-row-title-2", Static).render()
        )

        source_list.sync_page(_page("big", MAX_VISIBLE_SOURCE_ROWS + 5))
        await pilot.pause()
        slots = list(source_list.query(_ResearchSourceRowSlot))
        assert len(slots) == MAX_VISIBLE_SOURCE_ROWS
        assert all(slot.display for slot in slots)
        last = MAX_VISIBLE_SOURCE_ROWS - 1
        assert f"Source big {last}" in str(
            source_list.query_one(
                f"#research-source-row-title-{last}", Static
            ).render()
        )


@pytest.mark.asyncio
async def test_page_swaps_at_max_recycle_without_mount_churn() -> None:
    """At MAX rows, paging constructs and unmounts nothing (the pool's job)."""

    app = _RegionHarness()
    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        source_list = app.query_one(ResearchSourceList)
        receipts = app.query_one(ResearchSourceReceiptList)
        source_list.sync_page(_page("seed", MAX_VISIBLE_SOURCE_ROWS))
        receipts.sync_operations(
            _operations("seed", MAX_VISIBLE_SOURCE_RECEIPTS), incomplete=False
        )
        await pilot.pause()

        slots_before = list(source_list.query(_ResearchSourceRowSlot))
        receipt_slots_before = list(receipts.query(_ResearchSourceReceiptSlot))
        assert len(slots_before) == MAX_VISIBLE_SOURCE_ROWS
        assert len(receipt_slots_before) == MAX_VISIBLE_SOURCE_RECEIPTS

        constructed = {"count": 0}
        original_init = textual_widget_module.Widget.__init__

        def counting_init(self, *args, **kwargs):
            constructed["count"] += 1
            original_init(self, *args, **kwargs)

        textual_widget_module.Widget.__init__ = counting_init
        try:
            for swap in range(5):
                source_list.sync_page(
                    _page(f"swap{swap}", MAX_VISIBLE_SOURCE_ROWS)
                )
                receipts.sync_operations(
                    _operations(f"swap{swap}", MAX_VISIBLE_SOURCE_RECEIPTS),
                    incomplete=False,
                )
                await pilot.pause()
        finally:
            textual_widget_module.Widget.__init__ = original_init

        assert constructed["count"] == 0
        slots_after = list(source_list.query(_ResearchSourceRowSlot))
        receipt_slots_after = list(receipts.query(_ResearchSourceReceiptSlot))
        # Identity-stable: the same slot objects, still mounted, same order.
        assert [id(slot) for slot in slots_after] == [
            id(slot) for slot in slots_before
        ]
        assert [id(slot) for slot in receipt_slots_after] == [
            id(slot) for slot in receipt_slots_before
        ]
        # And they carry the LAST swap's data — recycled, not stale.
        assert "Source swap4 0" in str(
            source_list.query_one("#research-source-row-title-0", Static).render()
        )
        assert "operation-swap4-0" in str(
            receipts.query_one(
                "#research-source-receipt-owner-0", Static
            ).render()
        )


@pytest.mark.asyncio
async def test_shrink_recycles_slots_via_display_false_not_unmount() -> None:
    """A smaller page hides surplus slots; it never unmounts them."""

    app = _RegionHarness()
    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        source_list = app.query_one(ResearchSourceList)
        source_list.sync_page(_page("fill", MAX_VISIBLE_SOURCE_ROWS))
        await pilot.pause()

        source_list.sync_page(_page("less", 3))
        await pilot.pause()
        slots = list(source_list.query(_ResearchSourceRowSlot))
        assert len(slots) == MAX_VISIBLE_SOURCE_ROWS
        assert [slot.display for slot in slots] == [True] * 3 + [False] * (
            MAX_VISIBLE_SOURCE_ROWS - 3
        )
        assert "Source less 0" in str(
            source_list.query_one("#research-source-row-title-0", Static).render()
        )

        source_list.sync_page(None)
        await pilot.pause()
        slots = list(source_list.query(_ResearchSourceRowSlot))
        assert len(slots) == MAX_VISIBLE_SOURCE_ROWS
        assert not any(slot.display for slot in slots)


@pytest.mark.asyncio
async def test_receipt_slots_mount_before_bound_disclosure() -> None:
    """Grown receipt slots keep child order: heading, receipts…, bound."""

    app = _RegionHarness()
    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        receipts = app.query_one(ResearchSourceReceiptList)
        receipts.sync_operations(_operations("first", 2), incomplete=False)
        await pilot.pause()
        # Grow again to prove later batches also land before the bound line.
        receipts.sync_operations(
            _operations("more", MAX_VISIBLE_SOURCE_RECEIPTS + 5), incomplete=False
        )
        await pilot.pause()

        child_ids = [child.id for child in receipts.children]
        assert child_ids[0] == "research-source-receipts-heading"
        assert child_ids[-1] == "research-source-receipts-bound"
        assert child_ids[1:-1] == [
            f"research-source-receipt-{index}"
            for index in range(MAX_VISIBLE_SOURCE_RECEIPTS)
        ]
        assert "More receipts may exist" in str(
            receipts.query_one("#research-source-receipts-bound", Static).render()
        )


@pytest.mark.asyncio
async def test_dom_order_matches_slot_index_order_across_growth_steps() -> None:
    """Two growth steps still yield index order 0..N-1 in the DOM.

    `ResearchSourcesRegion.visible_owner_ids`/`selected_source_ids` pair DOM
    query order with page-row order, so out-of-order growth would silently
    mis-attribute selections.
    """

    app = _RegionHarness()
    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        source_list = app.query_one(ResearchSourceList)
        source_list.sync_page(_page("step1", 10))
        await pilot.pause()
        source_list.sync_page(_page("step2", MAX_VISIBLE_SOURCE_ROWS))
        await pilot.pause()

        indexes = [
            slot.index for slot in source_list.query(_ResearchSourceRowSlot)
        ]
        assert indexes == list(range(MAX_VISIBLE_SOURCE_ROWS))
        titles = [
            str(
                source_list.query_one(
                    f"#research-source-row-title-{index}", Static
                ).render()
            )
            for index in (0, 9, 10, MAX_VISIBLE_SOURCE_ROWS - 1)
        ]
        assert "Source step2 0" in titles[0]
        assert "Source step2 9" in titles[1]
        assert "Source step2 10" in titles[2]
        assert f"Source step2 {MAX_VISIBLE_SOURCE_ROWS - 1}" in titles[3]


@pytest.mark.asyncio
async def test_selection_reads_are_synchronous_with_first_growth() -> None:
    """The region's selection reads see fresh rows in the same message turn.

    `_render_page` calls `sync_page` and then `_sync_capabilities()` (which
    reads `selected_source_ids()`) synchronously, so slot-level state must
    be readable before the newly grown slots' subtrees finish mounting.
    """

    app = _RegionHarness()
    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        region = app.query_one(ResearchSourcesRegion)
        region.sync_workspace(
            _page("sync", 2, desired=("sync-0",)),
            readiness=(),
            capabilities={
                "set_selected_scope": AVAILABLE,
                "preview_source": AVAILABLE,
                "remove_source": AVAILABLE,
            },
            folders=(),
            operations=(),
        )
        # Deliberately NO pilot.pause() here: this is the synchronous
        # contract production relies on.
        assert region.visible_owner_ids() == ("sync-0", "sync-1")
        assert region.selected_source_ids() == ("membership-sync-0",)
        assert not region.query_one(
            "#research-source-preview-selected", Button
        ).disabled


@pytest.mark.asyncio
async def test_unmount_and_quit_mid_growth_are_safe() -> None:
    """Removing the list (or quitting) right after a growth sync cannot crash."""

    app = _RegionHarness()
    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        source_list = app.query_one(ResearchSourceList)
        source_list.sync_page(_page("grow", MAX_VISIBLE_SOURCE_ROWS))
        # Remove in the same message turn as the growth mount.
        await source_list.remove()
        await pilot.pause()
        # A post-removal sync must still be inert, not an exception.
        source_list.sync_page(_page("after", 5))
        await pilot.pause()
        assert app.query(ResearchSourceList).nodes == []

    # Quit walk: exit the app in the same turn as a fresh growth sync.
    app2 = _RegionHarness()
    async with app2.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        receipts = app2.query_one(ResearchSourceReceiptList)
        app2.query_one(ResearchSourceList).sync_page(
            _page("quit", MAX_VISIBLE_SOURCE_ROWS)
        )
        receipts.sync_operations(
            _operations("quit", MAX_VISIBLE_SOURCE_RECEIPTS), incomplete=False
        )
        # No pause: run_test teardown happens with the mounts still pending.
