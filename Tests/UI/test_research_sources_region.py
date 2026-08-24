"""Mounted behavior for the Research Workspace Sources workbench."""

from __future__ import annotations

from time import monotonic

import pytest
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Input, Select, Static

from tldw_chatbook.UI.Research_Workspace_Modules.sources_region import (
    ResearchSourcesRegion,
)
from tldw_chatbook.Research_Workspace import (
    QualifiedWorkspaceRef,
    ResearchSourcePage,
    ResearchSourceSummary,
    ResearchCapability,
    SourceReadiness,
    SourceReadinessState,
    WorkspaceDataSource,
)
from tldw_chatbook.Research_Workspace.overlay_store import ResearchSourceFolder
from tldw_chatbook.UI.Research_Workspace_Modules.source_list import (
    ResearchSourceList,
)
from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp


class _SourcesHarness(App[None]):
    def compose(self) -> ComposeResult:
        yield ResearchSourcesRegion(id="research-sources-pane")


class _StyledSourcesHarness(ConsolidatedCSSApp):
    CSS_PATH = str(BUNDLED_STYLESHEET)

    def compose(self) -> ComposeResult:
        with Vertical(id="research-workspace-shell"):
            yield ResearchSourcesRegion(id="research-sources-pane")


@pytest.mark.asyncio
async def test_sources_region_mounts_complete_control_inventory_once() -> None:
    app = _SourcesHarness()
    started = monotonic()
    async with app.run_test(size=(48, 36)) as pilot:
        await pilot.pause()
        mounted_seconds = monotonic() - started
        region = app.query_one(ResearchSourcesRegion)
        assert len(list(region.query("_ResearchSourceRowSlot"))) == 25
        assert len(list(region.walk_children())) < 650
        assert mounted_seconds < 1.0

        expected_ids = {
            "research-source-add",
            "research-source-quick-url",
            "research-source-search",
            "research-source-advanced",
            "research-source-sort",
            "research-source-select-all",
            "research-source-select-visible",
            "research-source-selection-clear",
            "research-source-selected-count",
            "research-source-move-copy",
            "research-source-preview-selected",
            "research-source-remove-selected",
            "research-source-folder-new",
            "research-source-folder-rename",
            "research-source-folder-focus",
            "research-source-select-folder",
            "research-source-page-prev",
            "research-source-page-next",
            "research-source-list",
            "research-source-receipts",
        }
        assert {
            widget.id for widget in region.walk_children() if widget.id
        } >= expected_ids
        assert (
            region.query_one("#research-source-quick-url", Input).placeholder
            == "Quick add URL"
        )
        assert (
            region.query_one("#research-source-search", Input).placeholder
            == "Filter current page"
        )
        assert (
            region.query_one("#research-source-preview-selected", Button).label.plain
            == "Preview visible selected"
        )
        assert (
            region.query_one("#research-source-remove-selected", Button).label.plain
            == "Remove visible selected"
        )
        assert region.query_one("#research-source-sort", Select).value == "manual"
        assert "Device-only" in str(
            region.query_one("#research-source-folders-label", Static).render()
        )
        assert "0 selected" in str(
            region.query_one("#research-source-selected-count", Static).render()
        )

        for control_id in (
            "research-source-move-copy",
            "research-source-preview-selected",
            "research-source-remove-selected",
            "research-source-folder-rename",
            "research-source-select-folder",
        ):
            assert region.query_one(f"#{control_id}", Button).disabled


@pytest.mark.asyncio
async def test_sources_region_uses_text_labels_for_honest_unavailable_actions() -> None:
    app = _SourcesHarness()
    async with app.run_test(size=(40, 24)) as pilot:
        await pilot.pause()
        region = app.query_one(ResearchSourcesRegion)

        assert "No workspace selected" in str(
            region.query_one("#research-source-recovery", Static).render()
        )
        assert "canonical owner unavailable" in str(
            region.query_one("#research-source-move-copy-reason", Static).render()
        )
        assert "association only" in str(
            region.query_one("#research-source-remove-scope", Static).render()
        )


@pytest.mark.asyncio
async def test_disabled_source_actions_use_full_opacity_with_noncolor_reason() -> None:
    app = _StyledSourcesHarness()
    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        region = app.query_one(ResearchSourcesRegion)
        disabled = region.query_one("#research-source-remove-selected", Button)

        assert disabled.disabled
        assert disabled.styles.opacity == 1.0
        assert disabled.styles.text_opacity == 1.0
        assert "canonical owner" in str(
            region.query_one("#research-source-move-copy-reason", Static).render()
        )


@pytest.mark.asyncio
async def test_nested_folders_render_ancestry_and_disable_offpage_selection() -> None:
    from tldw_chatbook.Research_Workspace.overlay_store import ResearchSourceFolder

    app = _SourcesHarness()
    ref = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "workspace-local")
    page = ResearchSourcePage(
        items=(
            ResearchSourceSummary(
                ref=ref,
                source_id="membership-visible",
                catalog_item_id="visible",
                title="Visible",
                source_type="text",
            ),
        ),
        limit=25,
        total=26,
        has_more=True,
    )
    folders = (
        ResearchSourceFolder("folder-root", "Root"),
        ResearchSourceFolder(
            "folder-child",
            "Child",
            ("membership-offpage",),
            "folder-root",
        ),
    )
    available = ResearchCapability(True, "available", "Available.", "local")

    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        region = app.query_one(ResearchSourcesRegion)
        region.sync_workspace(
            page,
            readiness=(),
            capabilities={"set_selected_scope": available},
            folders=folders,
            operations=(),
        )
        tree = region.query_one("#research-source-folder-tree", Select)
        assert any("Root" in str(prompt) for prompt, _value in tree._options)
        tree.value = "folder-child"
        await pilot.pause()

        parent = region.query_one("#research-source-folder-parent", Select)
        assert any(
            "Child" in str(prompt) and "  " in str(prompt)
            for prompt, _value in tree._options
        )
        assert parent.value == ""
        select_sources = region.query_one("#research-source-select-folder", Button)
        assert select_sources.disabled
        assert "off-page" in str(select_sources.tooltip)


@pytest.mark.asyncio
async def test_source_rows_keep_desired_intent_separate_from_readiness() -> None:
    app = _SourcesHarness()
    ref = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "workspace-local")
    source = ResearchSourceSummary(
        ref=ref,
        source_id="membership-9",
        catalog_item_id="9",
        title="Unicode evidence cafe\u0301",
        source_type="pdf",
        ready=False,
        selected=True,
        position=0,
    )
    page = ResearchSourcePage(
        items=(source,),
        limit=100,
        offset=0,
        total=1,
        has_more=False,
        desired_source_ids=("9",),
    )
    readiness = SourceReadiness(
        ref=ref,
        source_id="membership-9",
        catalog_item_id="9",
        state=SourceReadinessState.INDEXING,
        fts_ready=False,
        vector_ready=False,
        detail="Embedding index is still building.",
        next_action="Refresh status after indexing completes.",
    )

    async with app.run_test(size=(56, 38)) as pilot:
        await pilot.pause()
        source_list = app.query_one(ResearchSourceList)
        source_list.sync_page(
            page, readiness=(readiness,), folder_source_ids=frozenset()
        )
        await pilot.pause()

        assert "Unicode evidence cafe\u0301" in str(
            source_list.query_one("#research-source-row-title-0", Static).render()
        )
        assert "Selected intent: Yes" in str(
            source_list.query_one("#research-source-row-selection-0", Static).render()
        )
        assert "Readiness: Indexing" in str(
            source_list.query_one("#research-source-row-readiness-0", Static).render()
        )
        assert "Direct" in str(
            source_list.query_one("#research-source-row-badges-0", Static).render()
        )


@pytest.mark.asyncio
async def test_row_owner_actions_follow_typed_capabilities() -> None:
    app = _SourcesHarness()
    ref = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "workspace-local")
    source = ResearchSourceSummary(
        ref=ref,
        source_id="membership-9",
        catalog_item_id="9",
        title="Evidence",
        source_type="pdf",
    )
    page = ResearchSourcePage(items=(source,), limit=25, total=1)
    unavailable = ResearchCapability(
        False,
        "owner_unavailable",
        "The selected owner blocked this action.",
        "local",
    )

    async with app.run_test(size=(56, 30)) as pilot:
        await pilot.pause()
        source_list = app.query_one(ResearchSourceList)
        source_list.sync_page(
            page,
            capabilities={
                "preview_source": unavailable,
                "remove_source": unavailable,
            },
        )

        assert source_list.query_one("#research-source-row-preview-0", Button).disabled
        assert source_list.query_one("#research-source-row-remove-0", Button).disabled


@pytest.mark.asyncio
async def test_enabled_row_controls_emit_exact_owner_actions() -> None:
    messages = []
    app = _SourcesHarness()
    ref = QualifiedWorkspaceRef(
        WorkspaceDataSource.SERVER, "workspace-server", "profile", "principal"
    )
    sources = tuple(
        ResearchSourceSummary(
            ref=ref,
            source_id=f"association-{index}",
            catalog_item_id=str(index),
            title=f"Evidence {index}",
            source_type="pdf",
        )
        for index in (1, 2)
    )
    page = ResearchSourcePage(items=sources, limit=25, total=2)
    available = ResearchCapability(True, "available", "Available.", "server")

    async with app.run_test(size=(80, 30), message_hook=messages.append) as pilot:
        await pilot.pause()
        source_list = app.query_one(ResearchSourceList)
        source_list.sync_page(
            page,
            capabilities={
                "set_selected_scope": available,
                "preview_source": available,
                "remove_source": available,
                "reorder_sources": available,
            },
        )
        for widget_id in (
            "research-source-row-select-0",
            "research-source-row-details-0",
            "research-source-row-folders-0",
            "research-source-row-preview-0",
            "research-source-row-remove-0",
            "research-source-row-down-0",
        ):
            source_list.query_one(f"#{widget_id}", Button).press()
        await pilot.pause()

        selections = [
            message
            for message in messages
            if isinstance(message, ResearchSourceList.SelectionToggled)
        ]
        assert selections
        assert selections[0].source_id == "association-1"
        assert selections[0].desired_owner_id == "association-1"
        assert selections[0].selected is True
        assert {
            message.action
            for message in messages
            if isinstance(message, ResearchSourceList.ActionRequested)
        } == {"details", "folders", "preview", "remove"}
        reorders = [
            message
            for message in messages
            if isinstance(message, ResearchSourceList.ReorderRequested)
        ]
        assert reorders and reorders[0].source_id == "association-1"
        assert reorders[0].delta == 1
        copy = source_list.query_one("#research-source-row-copy-0", Button)
        assert copy.disabled
        assert "no canonical" in str(copy.tooltip)


@pytest.mark.asyncio
async def test_workspace_clear_removes_old_folders_and_disables_owner_actions() -> None:
    from tldw_chatbook.Research_Workspace.overlay_store import ResearchSourceFolder

    app = _SourcesHarness()
    ref = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "workspace-local")
    page = ResearchSourcePage(
        items=(
            ResearchSourceSummary(
                ref=ref,
                source_id="membership-9",
                catalog_item_id="9",
                title="Evidence",
                source_type="pdf",
            ),
        ),
        limit=25,
        total=1,
        desired_source_ids=("9",),
    )
    available = ResearchCapability(True, "available", "Available.", "local")

    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        region = app.query_one(ResearchSourcesRegion)
        region.sync_workspace(
            page,
            readiness=(),
            capabilities={
                "attach_existing": available,
                "preview_source": available,
                "remove_source": available,
            },
            folders=(ResearchSourceFolder("folder-old", "Old private folder"),),
            operations=(),
        )
        region.clear_workspace(authority="Server", reason="Switching owner")

        assert region.query_one("#research-source-folder-tree", Select).value == ""
        assert "Old private folder" not in str(
            region.query_one("#research-source-folder-tree", Select).render()
        )
        for widget_id in (
            "research-source-select-all",
            "research-source-select-visible",
            "research-source-selection-clear",
            "research-source-folder-new",
            "research-source-folder-rename",
            "research-source-folder-focus",
            "research-source-select-folder",
            "research-source-preview-selected",
            "research-source-remove-selected",
        ):
            assert region.query_one(f"#{widget_id}", Button).disabled


@pytest.mark.asyncio
async def test_add_and_quick_url_fail_closed_from_typed_attach_capability() -> None:
    app = _SourcesHarness()
    page = ResearchSourcePage(items=(), limit=25, total=0)
    unavailable = ResearchCapability(
        False,
        "viewer_forbidden",
        "Viewers cannot add sources.",
        "server",
        recovery_action="Ask an owner or editor for access.",
    )

    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        region = app.query_one(ResearchSourcesRegion)
        region.sync_workspace(
            page,
            readiness=(),
            capabilities={"attach_existing": unavailable},
            folders=(),
            operations=(),
        )

        add = region.query_one("#research-source-add", Button)
        quick = region.query_one("#research-source-quick-submit", Button)
        assert add.disabled and quick.disabled
        assert "Viewers cannot add sources" in str(add.tooltip)
        assert "[Unavailable]" in str(
            region.query_one("#research-source-recovery", Static).render()
        )


@pytest.mark.asyncio
async def test_enabled_source_controls_emit_owner_events_and_gates_are_honest() -> None:
    messages = []
    app = _SourcesHarness()
    ref = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "workspace-local")
    source = ResearchSourceSummary(
        ref=ref,
        source_id="membership-9",
        catalog_item_id="9",
        title="Evidence",
        source_type="pdf",
        selected=True,
        updated_at="2026-08-24T00:00:00Z",
    )
    page = ResearchSourcePage(
        items=(source,), limit=25, total=1, desired_source_ids=("9",)
    )
    available = ResearchCapability(True, "available", "Available.", "local")

    async with app.run_test(size=(80, 30), message_hook=messages.append) as pilot:
        await pilot.pause()
        region = app.query_one(ResearchSourcesRegion)
        region.sync_workspace(
            page,
            readiness=(),
            capabilities={
                "attach_existing": available,
                "preview_source": available,
                "remove_source": available,
                "reorder_sources": available,
                "set_selected_scope": available,
            },
            folders=(),
            operations=(),
        )
        region.query_one(
            "#research-source-quick-url", Input
        ).value = "https://example.invalid/paper"
        for widget_id in (
            "research-source-add",
            "research-source-refresh",
            "research-source-quick-submit",
            "research-source-select-all",
            "research-source-select-visible",
            "research-source-selection-clear",
            "research-source-preview-selected",
            "research-source-remove-selected",
        ):
            region.query_one(f"#{widget_id}", Button).press()
        region.query_one("#research-source-folder-name", Input).value = "Evidence"
        region.query_one("#research-source-folder-new", Button).press()
        await pilot.pause()

        assert any(
            isinstance(item, ResearchSourcesRegion.AddRequested) for item in messages
        )
        assert any(
            isinstance(item, ResearchSourcesRegion.RefreshRequested)
            for item in messages
        )
        assert any(
            isinstance(item, ResearchSourcesRegion.QuickUrlRequested)
            and item.url == "https://example.invalid/paper"
            for item in messages
        )
        assert {
            item.mode
            for item in messages
            if isinstance(item, ResearchSourcesRegion.SelectionScopeRequested)
        } == {"all", "visible", "clear"}
        assert {
            item.action
            for item in messages
            if isinstance(item, ResearchSourcesRegion.BatchRequested)
        } == {"preview-selected", "remove-selected"}
        assert any(
            isinstance(item, ResearchSourcesRegion.FolderRequested)
            and item.action == "new"
            and item.name == "Evidence"
            for item in messages
        )
        assert region.query_one("#research-source-move-copy", Button).disabled
        assert "not exposed" in str(
            region.query_one("#research-source-move-copy-reason", Static).render()
        )


@pytest.mark.asyncio
async def test_filters_sort_and_folder_focus_update_stable_rows_without_recompose() -> (
    None
):
    from tldw_chatbook.Research_Workspace.overlay_store import ResearchSourceFolder

    app = _SourcesHarness()
    ref = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "workspace-local")
    newer = ResearchSourceSummary(
        ref=ref,
        source_id="membership-new",
        catalog_item_id="2",
        title="Zulu",
        source_type="pdf",
        updated_at="2026-08-24T00:00:00Z",
    )
    older = ResearchSourceSummary(
        ref=ref,
        source_id="membership-old",
        catalog_item_id="1",
        title="Alpha",
        source_type="text",
        updated_at="2020-01-01T00:00:00Z",
    )
    page = ResearchSourcePage(items=(newer, older), limit=25, total=2)
    available = ResearchCapability(True, "available", "Available.", "server")

    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        region = app.query_one(ResearchSourcesRegion)
        source_list = region.query_one("#research-source-list", ResearchSourceList)
        first_slot = source_list.query_one("#research-source-row-0")
        region.sync_workspace(
            page,
            readiness=(),
            capabilities={"reorder_sources": available},
            folders=(
                ResearchSourceFolder("folder-zulu", "Zulu only", ("membership-new",)),
            ),
            operations=(),
        )
        region.query_one("#research-source-sort", Select).value = "title_asc"
        await pilot.pause()
        assert "Alpha" in str(
            source_list.query_one("#research-source-row-title-0", Static).render()
        )
        assert source_list.query_one("#research-source-row-down-0", Button).disabled

        region.query_one("#research-source-sort", Select).value = "updated_desc"
        region.query_one("#research-source-filter-date", Select).value = "week"
        await pilot.pause()
        assert "Zulu" in str(
            source_list.query_one("#research-source-row-title-0", Static).render()
        )
        assert not source_list.query_one("#research-source-row-1").display

        region.sync_workspace(
            page,
            readiness=(),
            capabilities={"reorder_sources": available},
            folders=(
                ResearchSourceFolder("folder-zulu", "Zulu only", ("membership-new",)),
            ),
            operations=(),
            focused_folder_id="folder-zulu",
        )
        assert source_list.query_one("#research-source-row-0") is first_slot
        assert "Focused folder: Zulu only" in str(
            region.query_one("#research-source-filter-summary", Static).render()
        )
        assert source_list.query_one("#research-source-row-up-0", Button).disabled
        assert source_list.query_one("#research-source-row-down-0", Button).disabled


@pytest.mark.asyncio
async def test_visible_batch_selection_excludes_selected_rows_hidden_by_each_filter() -> (
    None
):
    """A hidden selected row must never enter preview/remove batch mutation."""

    app = _SourcesHarness()
    ref = QualifiedWorkspaceRef(WorkspaceDataSource.LOCAL, "workspace-local")
    alpha = ResearchSourceSummary(
        ref=ref,
        source_id="membership-alpha",
        catalog_item_id="1",
        title="Alpha evidence",
        source_type="pdf",
        updated_at="2026-08-24T00:00:00Z",
    )
    beta = ResearchSourceSummary(
        ref=ref,
        source_id="membership-beta",
        catalog_item_id="2",
        title="Beta notes",
        source_type="text",
        updated_at="2020-01-01T00:00:00Z",
    )
    page = ResearchSourcePage(
        items=(alpha, beta),
        limit=25,
        total=2,
        desired_source_ids=("1", "2"),
    )
    available = ResearchCapability(True, "available", "Available.", "local")
    readiness = (
        SourceReadiness(
            ref=ref,
            source_id="membership-alpha",
            catalog_item_id="1",
            state=SourceReadinessState.FTS_READY,
            fts_ready=True,
            vector_ready=False,
        ),
        SourceReadiness(
            ref=ref,
            source_id="membership-beta",
            catalog_item_id="2",
            state=SourceReadinessState.ATTACHED,
            fts_ready=False,
            vector_ready=False,
        ),
    )

    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        region = app.query_one(ResearchSourcesRegion)
        region.sync_workspace(
            page,
            readiness=readiness,
            capabilities={
                "preview_source": available,
                "remove_source": available,
                "set_selected_scope": available,
            },
            folders=(
                ResearchSourceFolder(
                    "folder-alpha", "Alpha only", ("membership-alpha",)
                ),
            ),
            operations=(),
        )
        assert region.selected_source_ids() == (
            "membership-alpha",
            "membership-beta",
        )

        region.query_one("#research-source-search", Input).value = "alpha"
        await pilot.pause()
        assert region.selected_source_ids() == ("membership-alpha",)
        assert not region.query_one(
            "#research-source-preview-selected", Button
        ).disabled

        region.query_one("#research-source-search", Input).value = ""
        region.query_one("#research-source-filter-type", Select).value = "text"
        await pilot.pause()
        assert region.selected_source_ids() == ("membership-beta",)
        assert not region.query_one(
            "#research-source-preview-selected", Button
        ).disabled

        region.query_one("#research-source-filter-type", Select).value = ""
        region.query_one("#research-source-filter-status", Select).value = "ready"
        await pilot.pause()
        assert region.selected_source_ids() == ("membership-alpha",)
        assert not region.query_one(
            "#research-source-preview-selected", Button
        ).disabled

        region.query_one("#research-source-filter-status", Select).value = ""
        region.query_one("#research-source-filter-date", Select).value = "today"
        await pilot.pause()
        assert region.selected_source_ids() == ("membership-alpha",)
        assert not region.query_one(
            "#research-source-preview-selected", Button
        ).disabled

        region.query_one("#research-source-filter-date", Select).value = ""
        region._focused_folder_id = "folder-alpha"
        region._render_page()
        await pilot.pause()
        assert region.selected_source_ids() == ("membership-alpha",)
        assert not region.query_one(
            "#research-source-preview-selected", Button
        ).disabled
        assert "2 selected" in str(
            region.query_one("#research-source-selected-count", Static).render()
        )
