from __future__ import annotations

import asyncio

import pytest

from tldw_chatbook.Research_Workspace.contracts import (
    BoundedPageResult,
    QualifiedWorkspaceRef,
    ResearchCatalogItem,
    ResearchSourcePreview,
    ResearchSourceSummary,
    ResearchWorkspaceSummary,
    SourceSelectionResult,
    SourceReadiness,
    SourceReadinessState,
    WorkspaceDataSource,
)
from tldw_chatbook.Research_Workspace.controller import ResearchWorkspaceController


class DeferredPort:
    def __init__(self) -> None:
        self.results: dict[QualifiedWorkspaceRef, asyncio.Future] = {}

    async def get_workspace(self, ref: QualifiedWorkspaceRef):
        future = asyncio.get_running_loop().create_future()
        self.results[ref] = future
        return await future


class DeferredCatalogPort:
    def __init__(self) -> None:
        self.results: list[asyncio.Future] = []

    async def list_workspaces(self, *, include_archived: bool = False):
        future = asyncio.get_running_loop().create_future()
        self.results.append(future)
        return await future


class DeferredSourcePort:
    def __init__(self) -> None:
        self.source_results: list[asyncio.Future] = []
        self.catalog_results: list[asyncio.Future] = []
        self.readiness_results: list[asyncio.Future] = []
        self.preview_results: list[asyncio.Future] = []
        self.selection_results: list[asyncio.Future] = []

    async def list_sources(self, ref, *, limit=100, offset=0):
        future = asyncio.get_running_loop().create_future()
        self.source_results.append(future)
        return await future

    async def search_catalog(
        self,
        ref,
        *,
        query="",
        source_types=(),
        sort_by="updated_desc",
        limit=25,
        offset=0,
    ):
        future = asyncio.get_running_loop().create_future()
        self.catalog_results.append(future)
        return await future

    async def get_readiness(self, ref, *, source_ids=()):
        future = asyncio.get_running_loop().create_future()
        self.readiness_results.append(future)
        return await future

    async def preview_source(
        self, ref, source_id, *, max_chars=3000, snippet_limit=3
    ):
        future = asyncio.get_running_loop().create_future()
        self.preview_results.append(future)
        return await future

    async def set_selected_scope(self, ref, source_ids):
        future = asyncio.get_running_loop().create_future()
        self.selection_results.append(future)
        return await future


def local_ref(workspace_id: str) -> QualifiedWorkspaceRef:
    return QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, workspace_id)


def local_source(
    ref: QualifiedWorkspaceRef,
    source_id: str,
    *,
    catalog_item_id: str,
    selected: bool,
) -> ResearchSourceSummary:
    return ResearchSourceSummary(
        ref=ref,
        source_id=source_id,
        catalog_item_id=catalog_item_id,
        title=f"Source {catalog_item_id}",
        source_type="media",
        selected=selected,
    )


def test_context_revision_increases_for_each_selection_and_capability_refresh() -> None:
    controller = ResearchWorkspaceController({})

    first = controller.select_workspace(local_ref("one"), capability_revision="a")
    second = controller.select_workspace(local_ref("two"), capability_revision="a")
    third = controller.set_capability_revision("b")

    assert (first, second, third) == (1, 2, 3)


def test_controller_rejects_result_for_a_different_captured_ref() -> None:
    controller = ResearchWorkspaceController({})
    ref = local_ref("one")
    controller.select_workspace(ref, capability_revision="a")
    capture = controller.capture_request()

    with pytest.raises(ValueError, match="mismatched workspace ref"):
        controller.accept_workspace_result(
            capture,
            ResearchWorkspaceSummary(ref=local_ref("two"), name="Wrong"),
        )


@pytest.mark.asyncio
async def test_stale_result_updates_owner_cache_but_not_visible_state() -> None:
    port = DeferredPort()
    controller = ResearchWorkspaceController({WorkspaceDataSource.LOCAL: port})
    old_ref = local_ref("old")
    new_ref = local_ref("new")
    controller.select_workspace(old_ref, capability_revision="old-cap")

    old_request = asyncio.create_task(controller.refresh_selected_workspace())
    await asyncio.sleep(0)
    controller.select_workspace(new_ref, capability_revision="new-cap")
    new_request = asyncio.create_task(controller.refresh_selected_workspace())
    await asyncio.sleep(0)

    port.results[new_ref].set_result(ResearchWorkspaceSummary(ref=new_ref, name="New"))
    assert await new_request is True
    port.results[old_ref].set_result(ResearchWorkspaceSummary(ref=old_ref, name="Old"))
    assert await old_request is False

    assert controller.visible_workspace == ResearchWorkspaceSummary(
        ref=new_ref, name="New"
    )
    assert controller.canonical_workspace(old_ref) == ResearchWorkspaceSummary(
        ref=old_ref, name="Old"
    )


@pytest.mark.asyncio
async def test_catalog_generation_rejects_old_local_result_after_authority_aba() -> (
    None
):
    port = DeferredCatalogPort()
    controller = ResearchWorkspaceController({WorkspaceDataSource.LOCAL: port})

    old_request = asyncio.create_task(controller.refresh_workspace_catalog())
    await asyncio.sleep(0)
    controller.select_data_source(WorkspaceDataSource.SERVER)
    controller.select_data_source(WorkspaceDataSource.LOCAL)
    new_request = asyncio.create_task(controller.refresh_workspace_catalog())
    await asyncio.sleep(0)

    new_workspace = ResearchWorkspaceSummary(ref=local_ref("new"), name="New")
    port.results[1].set_result((new_workspace,))
    await new_request
    assert controller.catalog_state is not None
    assert controller.catalog_state.workspaces == (new_workspace,)

    old_workspace = ResearchWorkspaceSummary(ref=local_ref("old"), name="Old")
    port.results[0].set_result((old_workspace,))
    await old_request

    assert controller.catalog_state is not None
    assert controller.catalog_state.workspaces == (new_workspace,)


@pytest.mark.asyncio
async def test_source_generation_rejects_old_same_ref_result_after_workspace_aba() -> (
    None
):
    port = DeferredSourcePort()
    controller = ResearchWorkspaceController({WorkspaceDataSource.LOCAL: port})
    ref = local_ref("same")
    controller.select_workspace(ref, capability_revision="cap-1")

    old_request = asyncio.create_task(controller.refresh_selected_sources())
    await asyncio.sleep(0)
    controller.select_workspace(local_ref("other"), capability_revision="cap-2")
    controller.select_workspace(ref, capability_revision="cap-1")
    new_request = asyncio.create_task(controller.refresh_selected_sources())
    await asyncio.sleep(0)

    new_source = local_source(
        ref, "membership-new", catalog_item_id="2", selected=True
    )
    port.source_results[1].set_result(
        BoundedPageResult(items=(new_source,), limit=100, total=1)
    )
    assert await new_request is True

    old_source = local_source(
        ref, "membership-old", catalog_item_id="1", selected=False
    )
    port.source_results[0].set_result(
        BoundedPageResult(items=(old_source,), limit=100, total=1)
    )
    assert await old_request is False

    assert controller.visible_source_page.items == (new_source,)
    assert controller.canonical_source(ref, "membership-old") is None


@pytest.mark.asyncio
async def test_old_readiness_and_preview_cannot_repaint_after_capability_change() -> (
    None
):
    port = DeferredSourcePort()
    controller = ResearchWorkspaceController({WorkspaceDataSource.LOCAL: port})
    ref = local_ref("one")
    controller.select_workspace(ref, capability_revision="cap-1")

    readiness_request = asyncio.create_task(controller.refresh_selected_readiness())
    preview_request = asyncio.create_task(
        controller.preview_selected_source("membership-1")
    )
    await asyncio.sleep(0)
    controller.set_capability_revision("cap-2")

    readiness = SourceReadiness(
        ref=ref,
        source_id="membership-1",
        catalog_item_id="1",
        state=SourceReadinessState.FTS_READY,
        text_ready=True,
        fts_ready=True,
    )
    preview = ResearchSourcePreview(
        ref=ref,
        source_id="membership-1",
        catalog_item_id="1",
        preview_mode="text",
        text="old",
    )
    port.readiness_results[0].set_result((readiness,))
    port.preview_results[0].set_result(preview)

    assert await readiness_request is False
    assert await preview_request is False
    assert controller.visible_readiness == ()
    assert controller.visible_preview is None


@pytest.mark.asyncio
async def test_missing_media_preview_is_cached_by_association_identity() -> None:
    class PreviewPort:
        async def preview_source(
            self, ref, source_id, *, max_chars=3000, snippet_limit=3
        ):
            return ResearchSourcePreview(
                ref=ref,
                source_id=source_id,
                catalog_item_id=None,
                preview_mode="missing_media",
            )

    ref = QualifiedWorkspaceRef(
        WorkspaceDataSource.SERVER,
        "workspace-1",
        server_profile_id="profile-1",
    )
    controller = ResearchWorkspaceController(
        {WorkspaceDataSource.SERVER: PreviewPort()}
    )
    controller.select_workspace(ref, capability_revision="cap-1")

    assert await controller.preview_selected_source("source-1") is True

    assert controller.visible_preview.catalog_item_id is None
    assert controller.canonical_source_preview(ref, "source-1") == (
        controller.visible_preview
    )


@pytest.mark.asyncio
async def test_selection_reconciliation_supersedes_older_source_refresh() -> None:
    port = DeferredSourcePort()
    controller = ResearchWorkspaceController({WorkspaceDataSource.LOCAL: port})
    ref = local_ref("one")
    controller.select_workspace(ref, capability_revision="cap-1")

    old_refresh = asyncio.create_task(controller.refresh_selected_sources())
    await asyncio.sleep(0)
    selection = asyncio.create_task(controller.set_selected_scope(("2",)))
    await asyncio.sleep(0)

    selected = local_source(
        ref, "membership-2", catalog_item_id="2", selected=True
    )
    port.selection_results[0].set_result(
        SourceSelectionResult(
            ref=ref,
            desired_source_ids=("2",),
            sources=(selected,),
        )
    )
    assert await selection is True
    assert controller.desired_source_ids == ("2",)

    stale = local_source(
        ref, "membership-2", catalog_item_id="2", selected=False
    )
    port.source_results[0].set_result(
        BoundedPageResult(items=(stale,), limit=100, total=1)
    )
    assert await old_refresh is False
    assert controller.visible_source_page is None
    assert controller.canonical_source(ref, "membership-2") == selected
    assert controller.desired_source_ids == ("2",)


@pytest.mark.asyncio
async def test_selection_of_row_101_preserves_the_current_visible_page() -> None:
    class ImmediatePort:
        async def list_sources(self, ref, *, limit=100, offset=0):
            first_page = tuple(
                local_source(
                    ref,
                    f"membership-{index}",
                    catalog_item_id=str(index),
                    selected=False,
                )
                for index in range(1, 101)
            )
            return BoundedPageResult(
                items=first_page,
                limit=100,
                total=101,
                has_more=True,
            )

        async def set_selected_scope(self, ref, source_ids):
            return SourceSelectionResult(
                ref=ref,
                desired_source_ids=("101",),
                sources=(
                    local_source(
                        ref,
                        "membership-101",
                        catalog_item_id="101",
                        selected=True,
                    ),
                ),
            )

    ref = local_ref("one")
    controller = ResearchWorkspaceController(
        {WorkspaceDataSource.LOCAL: ImmediatePort()}
    )
    controller.select_workspace(ref, capability_revision="cap-1")
    assert await controller.refresh_selected_sources() is True
    visible_before = controller.visible_source_page

    assert await controller.set_selected_scope(("101",)) is True

    assert controller.desired_source_ids == ("101",)
    assert controller.visible_source_page is visible_before


@pytest.mark.asyncio
async def test_selection_reconciliation_rejects_duplicate_requested_identity() -> None:
    class DuplicateAcceptingPort:
        async def set_selected_scope(self, ref, source_ids):
            return SourceSelectionResult(ref=ref, desired_source_ids=("1",))

    ref = local_ref("one")
    controller = ResearchWorkspaceController(
        {WorkspaceDataSource.LOCAL: DuplicateAcceptingPort()}
    )
    controller.select_workspace(ref, capability_revision="cap-1")

    with pytest.raises(ValueError, match="did not match"):
        await controller.set_selected_scope(("1", "1"))

    assert controller.desired_source_ids == ()


@pytest.mark.asyncio
async def test_late_selection_result_does_not_bleed_into_new_workspace() -> None:
    port = DeferredSourcePort()
    controller = ResearchWorkspaceController({WorkspaceDataSource.LOCAL: port})
    old_ref = local_ref("old")
    controller.select_workspace(old_ref, capability_revision="cap-1")

    request = asyncio.create_task(controller.set_selected_scope(("1",)))
    await asyncio.sleep(0)
    controller.select_workspace(local_ref("new"), capability_revision="cap-1")
    port.selection_results[0].set_result(
        SourceSelectionResult(
            ref=old_ref,
            desired_source_ids=("1",),
            sources=(
                local_source(
                    old_ref, "membership-1", catalog_item_id="1", selected=True
                ),
            ),
        )
    )

    assert await request is False
    assert controller.desired_source_ids == ()
    assert controller.visible_source_page is None


@pytest.mark.asyncio
async def test_catalog_results_are_cached_only_under_qualified_current_ref() -> None:
    port = DeferredSourcePort()
    controller = ResearchWorkspaceController({WorkspaceDataSource.LOCAL: port})
    ref = local_ref("one")
    controller.select_workspace(ref, capability_revision="cap-1")

    request = asyncio.create_task(controller.search_selected_catalog(query="paper"))
    await asyncio.sleep(0)
    item = ResearchCatalogItem(
        ref=ref,
        catalog_item_id="7",
        title="Paper",
        source_type="media",
    )
    port.catalog_results[0].set_result(
        BoundedPageResult(items=(item,), limit=25, total=1)
    )

    assert await request is True
    assert controller.visible_catalog_page.items == (item,)
    assert controller.canonical_catalog_item(ref, "7") == item
