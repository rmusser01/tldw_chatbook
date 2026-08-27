"""Tests for the Watchlists collections screen action handlers."""

from contextlib import asynccontextmanager
import threading
from types import SimpleNamespace

import pytest
from unittest.mock import AsyncMock, Mock

from rich.text import Text
from textual.app import App, ComposeResult
from textual.geometry import Size
from textual.widgets import Button, Input, ListView, Static, TextArea

from Tests.UI.app_factory import _build_test_app
from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp
from Tests.UI.test_destination_shells import DestinationHarness, _static_text
from tldw_chatbook.Subscriptions.watchlist_item_page import WatchlistItemPage
from tldw_chatbook.UI.Screens import watchlists_collections_screen as collections_module
from tldw_chatbook.UI.Screens.watchlists_collections_screen import WatchlistsCollectionsScreen
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
from tldw_chatbook.UI.Watchlists_Modules.inspector_pane import (
    BreadcrumbScopeSelected,
    CheckNowRequested,
    InspectorPane,
    PreviewRequested,
    ViewSnapshotRequested,
)
from tldw_chatbook.UI.Watchlists_Modules.article_list import (
    ArticleListPane,
    NextItemsPageRequested,
    PreviousItemsPageRequested,
)
from tldw_chatbook.UI.Watchlists_Modules.content_pane import ContentPane
from tldw_chatbook.UI.Watchlists_Modules.items_pane import ItemSelected
from tldw_chatbook.UI.Watchlists_Modules.items_pane import ItemsFilterChanged
from tldw_chatbook.UI.Watchlists_Modules.opml_dialogs import (
    OpmlExportDialog,
    OpmlImportDialog,
    WatchlistNameDialog,
    WatchlistSourcePickerDialog,
)
from tldw_chatbook.UI.Watchlists_Modules.sources_pane import (
    ExportOpmlRequested,
    ImportOpmlRequested,
    SourceSelected,
    SourcesPane,
)
from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import (
    TreeScope,
    TreeScopeChanged,
    WatchlistTree,
)
from tldw_chatbook.Utils.input_validation import validate_url as real_validate_url


class BundledWatchlistsDestinationHarness(DestinationHarness):
    """Destination host with the same app-tier stylesheet as production."""

    CSS_PATH = str(BUNDLED_STYLESHEET)


def _controller_double() -> AsyncMock:
    """Return a controller double with the production sync/async API shape."""
    controller = AsyncMock()
    controller.create_form_source_types = Mock(
        side_effect=lambda *, runtime_backend=None: (
            ("rss", "site", "forum")
            if runtime_backend == "server"
            else ("rss", "atom", "url")
        )
    )
    return controller


def test_layout_intent_dataclasses_use_pascal_case_names() -> None:
    assert hasattr(collections_module, "ManualLayoutRollback")
    assert hasattr(collections_module, "ResponsivePriorityLease")
    assert hasattr(collections_module, "SectionViewIntent")
    assert not hasattr(collections_module, "_ManualLayoutRollback")
    assert not hasattr(collections_module, "_ResponsivePriorityLease")
    assert not hasattr(collections_module, "_SectionViewIntent")


def test_layout_width_uses_only_positive_screen_allocation() -> None:
    receiver = SimpleNamespace(size=Size(145, 50))

    assert WatchlistsCollectionsScreen._available_layout_width(receiver) == 145

    receiver.size = Size(0, 50)
    assert WatchlistsCollectionsScreen._available_layout_width(receiver) is None


@pytest.fixture
def fake_controller():
    controller = _controller_double()
    controller.preview_source = AsyncMock(
        return_value={"items": [{"title": "Post"}], "log_text": "ok"}
    )
    controller.check_now = AsyncMock(return_value={"run_id": "1"})
    controller.import_opml = AsyncMock(return_value={"created": 2})
    controller.export_opml = AsyncMock(return_value="<opml></opml>")
    controller.get_overview_data = AsyncMock(
        return_value={
            "total_sources": 0,
            "active_sources": 0,
            "sources_in_error": 0,
            "total_items": 0,
            "new_items": 0,
            "latest_run_status": "unavailable",
            "failed_runs": [],
            "active_alert_rules": 0,
        }
    )
    return controller


@asynccontextmanager
async def _open_screen(controller):
    app_instance = _build_test_app()
    host = DestinationHarness(app_instance, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        assert isinstance(screen, WatchlistsCollectionsScreen)
        screen._controller = controller
        yield screen, pilot


class _InspectorActionsApp(App[None]):
    def __init__(self, entity: dict) -> None:
        super().__init__()
        self.entity = entity
        self.snapshot_requests: list[ViewSnapshotRequested] = []

    def compose(self) -> ComposeResult:
        pane = InspectorPane(id="watchlists-entity-inspector")
        pane.set_reactive(InspectorPane.selected_entity, self.entity)
        yield pane

    def on_view_snapshot_requested(self, message: ViewSnapshotRequested) -> None:
        self.snapshot_requests.append(message)


class _PreMountServerReadHarness(ConsolidatedCSSApp):
    """Mount Watchlists after applying the Server Read deep link."""

    def __init__(self, app_instance) -> None:
        super().__init__()
        self.app_instance = app_instance

    async def on_mount(self) -> None:
        screen = WatchlistsCollectionsScreen(self.app_instance)
        screen.apply_navigation_context({"section": "items", "backend": "server"})
        await self.push_screen(screen)


@pytest.mark.parametrize("content_kind", ["article", "change"])
@pytest.mark.asyncio
async def test_item_inspector_keeps_advanced_actions(content_kind: str) -> None:
    app = _InspectorActionsApp(
        {
            "entity_kind": "watchlist_item",
            "item_id": 7,
            "title": "Selected item",
            "content_kind": content_kind,
            "queued_for_briefing": False,
        }
    )
    async with app.run_test():
        action_ids = [
            button.id
            for button in app.query_one("#inspector-actions").query(Button)
        ]

        assert "inspector-ingest-button" in action_ids
        assert "inspector-queue-briefing-button" in action_ids
        assert ("inspector-full-page-button" in action_ids) is (
            content_kind == "change"
        )
        assert ("inspector-previous-snapshot-button" in action_ids) is (
            content_kind == "change"
        )


@pytest.mark.parametrize(
    ("button_id", "which"),
    [
        ("inspector-full-page-button", "full_page"),
        ("inspector-previous-snapshot-button", "previous"),
    ],
)
@pytest.mark.asyncio
async def test_inspector_snapshot_actions_post_existing_request(
    button_id: str, which: str
) -> None:
    entity = {
        "entity_kind": "watchlist_item",
        "item_id": 7,
        "title": "Changed page",
        "content_kind": "change",
    }
    app = _InspectorActionsApp(entity)
    async with app.run_test() as pilot:
        await pilot.click(f"#{button_id}")
        await pilot.pause()

        assert len(app.snapshot_requests) == 1
        request = app.snapshot_requests[0]
        assert isinstance(request, ViewSnapshotRequested)
        assert request.item is entity
        assert request.which == which


@pytest.mark.asyncio
async def test_screen_keeps_previous_snapshot_modal_handler(monkeypatch) -> None:
    app = _build_test_app()
    app.local_watchlists_service.get_url_snapshots = AsyncMock(
        return_value=[
            {"created_at": "2026-08-23T10:00:00Z", "extracted_content": "now"},
            {
                "created_at": "2026-08-22T10:00:00Z",
                "extracted_content": "before",
            },
        ]
    )
    host = DestinationHarness(app, "watchlists_collections")
    pushed = AsyncMock(return_value=None)
    monkeypatch.setattr(host, "push_screen_wait", pushed)
    item = {"source_id": 11, "url": "https://example.com/changed"}

    async with host.run_test(size=(180, 50)) as pilot:
        screen = host.screen_stack[-1]
        screen.post_message(ViewSnapshotRequested(item, "previous"))
        assert await _wait_until(pilot, lambda: pushed.await_count == 1)

        app.local_watchlists_service.get_url_snapshots.assert_awaited_once_with(
            11, "https://example.com/changed", limit=2
        )
        modal = pushed.await_args.args[0]
        assert modal._url == "https://example.com/changed"
        assert modal._content == "before"


@pytest.mark.asyncio
async def test_preview_source_handler_calls_controller(fake_controller):
    async with _open_screen(fake_controller) as (screen, pilot):
        screen.post_message(PreviewRequested({"id": "source-1", "name": "Feed"}))
        await pilot.pause(0.2)

        fake_controller.preview_source.assert_awaited_once_with(
            runtime_backend="local", source_config={"id": "source-1", "name": "Feed"}
        )


@pytest.mark.asyncio
async def test_check_now_source_handler_calls_controller(fake_controller):
    async with _open_screen(fake_controller) as (screen, pilot):
        screen.post_message(CheckNowRequested({"id": "source-1", "name": "Feed"}))
        await pilot.pause(0.2)

        fake_controller.check_now.assert_awaited_once_with(
            runtime_backend="local", source_id="source-1"
        )


@pytest.mark.asyncio
async def test_import_opml_handler_calls_controller(fake_controller):
    async with _open_screen(fake_controller) as (screen, pilot):
        screen.post_message(ImportOpmlRequested())
        await pilot.pause(0.1)

        top_screen = screen.app.screen
        assert isinstance(top_screen, OpmlImportDialog)
        text_area = top_screen.query_one("#opml-import-text", TextArea)
        text_area.text = "<opml><outline text=\"A\" xmlUrl=\"http://a.com/feed\"/>"
        top_screen.query_one("#opml-import-confirm", Button).press()
        await pilot.pause(0.2)

        fake_controller.import_opml.assert_awaited_once_with(
            runtime_backend="local",
            xml_text="<opml><outline text=\"A\" xmlUrl=\"http://a.com/feed\"/>",
        )


@pytest.mark.asyncio
async def test_export_opml_handler_calls_controller(fake_controller):
    async with _open_screen(fake_controller) as (screen, pilot):
        screen.post_message(ExportOpmlRequested())
        await pilot.pause(0.2)

        fake_controller.export_opml.assert_awaited_once_with(runtime_backend="local")
        assert isinstance(screen.app.screen, OpmlExportDialog)


# --- task-2513: Read-first IA -- Read is tab 1, the default section, and ---
# the tab strip lives in the centre header on EVERY tab; the FEEDS region
# (and its `#watchlists-list-pane`) is gone entirely.


@pytest.mark.asyncio
async def test_the_read_tab_is_the_default_section():
    """The screen lands on Read: `active_section` defaults to "items" (the
    section id is unchanged, so deep links keep working) and the Read tab's
    own pane is what mounts first."""
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        assert screen.active_section == "items"
        assert screen.query_one("#wl-tab-items").has_class("is-active")
        assert screen.query_one("#watchlists-items-pane")


@pytest.mark.asyncio
async def test_pre_mount_server_read_is_query_free_and_enters_recovery(
    monkeypatch,
) -> None:
    """A cold Server Read deep link never starts local Reader navigation."""
    app = _build_test_app()
    scope_service = app.watchlist_scope_service
    local_async_spies = {}
    for name in ("list_watch_items", "list_items"):
        spy = AsyncMock(wraps=getattr(scope_service, name))
        monkeypatch.setattr(scope_service, name, spy)
        local_async_spies[name] = spy

    bundle = app.watchlist_bundle_service
    local_sync_spies = {}
    for name in (
        "list_watchlists",
        "list_source_rows",
        "list_all_source_rows",
        "list_unassigned_source_rows",
        "get_watchlist_item_counts",
        "get_flagged_items_count",
        "get_unread_items_count_since",
        "get_source_item_counts",
    ):
        spy = Mock(wraps=getattr(bundle, name))
        monkeypatch.setattr(bundle, name, spy)
        local_sync_spies[name] = spy

    host = _PreMountServerReadHarness(app)
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.4)
        await host.workers.wait_for_complete()
        screen = host.screen_stack[-1]

        assert screen.active_section == "items"
        assert screen.runtime_backend == "server"
        assert screen._read_recovery_active is True
        assert screen.query("#watchlists-read-local-only")
        assert screen.query("#watchlists-read-recovery-status")
        assert not screen.query("#watchlists-content-pane")
        for name, spy in local_async_spies.items():
            local_calls = [
                call
                for call in spy.await_args_list
                if call.kwargs.get("runtime_backend") == "local"
            ]
            assert not local_calls, name
        for name, spy in local_sync_spies.items():
            assert spy.call_count == 0, name


@pytest.mark.asyncio
async def test_digit_1_switches_to_read_and_7_to_overview():
    """The digit bindings follow the new tab order: Read first, Overview last."""
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]

        await pilot.press("7")
        await pilot.pause()
        assert screen.active_section == "overview"

        await pilot.press("1")
        await pilot.pause()
        assert screen.active_section == "items"

        await pilot.press("2")
        await pilot.pause()
        assert screen.active_section == "sources"


@pytest.mark.asyncio
async def test_the_tab_strip_is_mounted_on_the_read_and_sources_tabs():
    """`#wl-tabs` lives in the centre header (`_build_centre_status_header`),
    which is wired unconditionally -- the strip is on every tab, Read
    included, exactly once."""
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]

        assert screen.active_section == "items", "precondition: lands on Read"
        assert len(screen.query("#wl-tabs")) == 1
        assert screen.query_one("#wl-centre-status")

        screen.active_section = "sources"
        await pilot.pause(0.2)
        assert len(screen.query("#wl-tabs")) == 1
        assert screen.query_one("#wl-centre-status")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "size",
    [(120, 40), (180, 50), (235, 52)],
    ids=["narrow", "normal", "wide"],
)
async def test_read_snapshot_count_and_arrivals_fit_the_feed_items_pane(size):
    """Task 6 chrome stays readable through the production CSS cascade."""
    app = _build_test_app()
    host = BundledWatchlistsDestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=size) as pilot:
        await pilot.pause(0.5)
        screen = host.screen_stack[-1]
        screen._items_snapshot_count = 50
        screen._items_pending_arrivals = 3
        screen._push_items_pager_state()
        await pilot.pause(0.1)

        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        toolbar = pane.query_one("#items-toolbar")
        search_row = pane.query_one("#items-toolbar-search")
        actions_row = pane.query_one("#items-toolbar-actions")
        search = pane.query_one("#items-search-input", Input)
        pill = pane.query_one("#items-new-items-pill", Static)
        count = pane.query_one("#items-snapshot-count", Static)

        def assert_contains(parent, child) -> None:
            assert child.region.x >= parent.region.x
            assert child.region.y >= parent.region.y
            assert child.region.right <= parent.region.right
            assert child.region.bottom <= parent.region.bottom

        def composited_text(widget) -> str:
            strips = widget.screen._compositor.render_strips()
            region = widget.region
            return "\n".join(
                "".join(segment.text for segment in strips[y])[
                    region.x : region.right
                ]
                for y in range(region.y, region.bottom)
            )

        visible_children = [child for child in pane.children if child.display]
        for child in visible_children:
            assert_contains(pane, child)
        for upper, lower in zip(visible_children, visible_children[1:]):
            assert upper.region.bottom <= lower.region.y

        toolbar_rows = [search_row, actions_row]
        for row in toolbar_rows:
            assert_contains(toolbar, row)
        assert search_row.region.bottom <= actions_row.region.y
        for row in toolbar_rows:
            row_children = [child for child in row.children if child.display]
            for child in row_children:
                assert_contains(row, child)
            for left, right in zip(row_children, row_children[1:]):
                assert left.region.right <= right.region.x

        assert search.region.width >= max(8, search_row.region.width - 2)
        assert composited_text(count).strip() == "50 items in snapshot"
        assert composited_text(pill).strip() == "3 new items"


@pytest.mark.asyncio
async def test_the_list_pane_is_gone_on_every_tab():
    """The FEEDS region's `#watchlists-list-pane` died with the region -- no
    tab may mount it (the geometry tests pinned to `.watchlists-region-feeds`
    row caps died with it; the centre header now carries what it showed)."""
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]

        for section in ("items", "sources", "overview"):
            screen.active_section = section
            await pilot.pause(0.2)
            assert not screen.query("#watchlists-list-pane"), section
            assert not screen.query("#wl-region-feeds"), section
            assert not screen.query("#wl-header-feeds"), section


@pytest.mark.asyncio
async def test_server_backed_read_recovers_through_the_normal_local_load_path(
    monkeypatch,
) -> None:
    """Server-labelled Read never leaks local rows or local Reader queries."""
    import asyncio

    controller = _controller_double()
    controller.get_overview_data = AsyncMock(return_value={})
    local_rows = [
        {
            "id": "local:watchlist_item:7",
            "item_id": 7,
            "title": "Loaded after switching",
            "status": "new",
            "url": "https://example.com/7",
            "created_at": "2026-08-23T12:00:00+00:00",
        }
    ]
    local_load_entered = asyncio.Event()
    release_local_load = asyncio.Event()

    async def blocked_local_load(**_kwargs):
        local_load_entered.set()
        await release_local_load.wait()
        return WatchlistItemPage(
            items=tuple(local_rows),
            has_more=False,
            snapshot_max_item_id=7,
            snapshot_count=1,
            next_cursor=None,
        )

    controller.list_reader_items_page = AsyncMock(side_effect=blocked_local_load)
    controller.check_all = AsyncMock(return_value={"checked": 0, "failed": []})
    app = _build_test_app()
    bundle = app.watchlist_bundle_service
    count_spies = []
    for name in (
        "get_watchlist_item_counts",
        "get_flagged_items_count",
        "get_unread_items_count_since",
        "get_source_item_counts",
    ):
        spy = Mock(wraps=getattr(bundle, name))
        monkeypatch.setattr(bundle, name, spy)
        count_spies.append(spy)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.4)
        screen = host.screen_stack[-1]
        await host.workers.wait_for_complete()
        await pilot.pause()
        screen._controller = controller

        screen.active_section = "sources"
        await pilot.pause(0.3)
        selector = screen.query_one("#watchlists-backend-select")
        selector.value = "server"
        await pilot.pause(0.3)
        await host.workers.wait_for_complete()
        controller.list_reader_items_page.reset_mock()
        controller.check_all.reset_mock()
        for spy in count_spies:
            spy.reset_mock()

        screen.active_section = "items"
        await pilot.pause(0.4)
        await host.workers.wait_for_complete()
        await pilot.pause()

        assert screen.runtime_backend == "server"
        assert selector.value == "server"
        assert selector.disabled is True
        assert screen.query("#wl-region-content"), "Reader centre stays mounted"
        assert screen.query("#watchlists-read-local-only")
        switch = screen.query_one("#watchlists-switch-local", Button)
        assert switch.disabled is False
        assert "Switch to Local" in str(switch.label)
        assert "local" in _static_text(
            screen.query_one("#watchlists-read-local-only-copy", Static)
        ).lower()

        screen.post_message(ItemsFilterChanged("unread", "server search"))
        screen.action_refresh_all()
        await screen._load_tree_data().wait()
        await pilot.pause(0.5)
        await host.workers.wait_for_complete()

        controller.list_reader_items_page.assert_not_awaited()
        controller.check_all.assert_not_awaited()
        for name, spy in zip(
            (
                "get_watchlist_item_counts",
                "get_flagged_items_count",
                "get_unread_items_count_since",
                "get_source_item_counts",
            ),
            count_spies,
        ):
            assert spy.call_count == 0, name
        assert not screen.query_one(
            "#watchlists-items-pane", ArticleListPane
        ).items

        switch.press()
        assert await _wait_until(pilot, local_load_entered.is_set)

        assert screen.runtime_backend == "local"
        assert selector.value == "local"
        assert screen.query("#watchlists-read-local-only"), (
            "the recovery centre must remain until the normal load commits"
        )
        assert screen.query_one("#watchlists-switch-local", Button).disabled is False
        assert screen.query("#watchlists-read-local-only-copy")
        assert not screen.query("#watchlists-content-pane")
        assert not screen.query_one(
            "#watchlists-items-pane", ArticleListPane
        ).items

        release_local_load.set()
        await host.workers.wait_for_complete()
        await pilot.pause()

        assert screen.runtime_backend == "local"
        assert selector.value == "local"
        assert selector.disabled is True
        assert controller.list_reader_items_page.await_count == 1, (
            screen._items_page_loading,
            screen._items_inflight_page_load,
            screen._items_load_generation,
            screen._loaded_items,
        )
        assert "search" not in controller.list_reader_items_page.await_args.kwargs
        assert [
            item["title"]
            for item in screen.query_one(
                "#watchlists-items-pane", ArticleListPane
            ).items
        ] == ["Loaded after switching"]
        assert screen.query("#watchlists-content-pane")
        assert not screen.query("#watchlists-read-local-only")
        assert screen._selected_content_item is None


@pytest.mark.asyncio
async def test_failed_switch_to_local_retries_the_normal_load_path() -> None:
    controller = _controller_double()
    controller.get_overview_data = AsyncMock(return_value={})
    local_row = {
        "id": "local:watchlist_item:9",
        "item_id": 9,
        "title": "Loaded by recovery retry",
        "status": "new",
        "url": "https://example.com/9",
        "created_at": "2026-08-23T12:00:00+00:00",
    }
    controller.list_reader_items_page = AsyncMock(
        side_effect=[
            RuntimeError("local read failed"),
            WatchlistItemPage(
                items=(local_row,),
                has_more=False,
                snapshot_max_item_id=9,
                snapshot_count=1,
                next_cursor=None,
            ),
        ]
    )
    app = _build_test_app()
    bundle = app.watchlist_bundle_service
    source_id = bundle._db.add_subscription(
        name="Recovery local source",
        type="rss",
        source="https://recovery.example/feed",
    )
    watchlist = bundle.create("Recovery local watchlist")
    bundle.add_source(watchlist["id"], source_id)
    host = DestinationHarness(app, "watchlists_collections")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.4)
        screen = host.screen_stack[-1]
        await host.workers.wait_for_complete()
        screen._controller = controller
        await screen._load_tree_data().wait()
        assert screen.query(f"#wl-tree-node-watchlist-{watchlist['id']}")

        screen.active_section = "sources"
        await pilot.pause(0.3)
        selector = screen.query_one("#watchlists-backend-select")
        selector.value = "server"
        await pilot.pause(0.3)
        screen.active_section = "items"
        await pilot.pause(0.4)
        await host.workers.wait_for_complete()
        controller.list_reader_items_page.assert_not_awaited()

        screen.query_one("#watchlists-switch-local", Button).press()
        await host.workers.wait_for_complete()
        await pilot.pause()

        assert screen.runtime_backend == "local"
        assert selector.value == "local"
        controller.list_reader_items_page.assert_awaited_once()
        assert screen.query("#watchlists-read-local-only")
        assert screen.query_one("#watchlists-switch-local", Button).disabled is False
        assert screen.query("#watchlists-read-local-only-copy")
        assert not screen.query("#watchlists-content-pane")
        assert not screen.query_one(
            "#watchlists-items-pane", ArticleListPane
        ).items
        assert screen._tree_watchlists[0]["name"] == "Recovery local watchlist"
        failed_render = host.export_screenshot()
        assert "Recovery local watchlist" not in failed_render
        assert "Recovery local source" not in failed_render
        assert screen._items_retry_message == (
            "Couldn't load All Sources. Retry to load Feed Items."
        )
        app.notify.assert_not_called()

        screen.query_one("#watchlists-switch-local", Button).press()
        await host.workers.wait_for_complete()
        await pilot.pause()

        assert controller.list_reader_items_page.await_count == 2
        assert not screen.query("#watchlists-read-local-only")
        assert screen.query("#watchlists-content-pane")
        assert [
            item["title"]
            for item in screen.query_one(
                "#watchlists-items-pane", ArticleListPane
            ).items
        ] == ["Loaded by recovery retry"]
        assert screen.query(f"#wl-tree-node-watchlist-{watchlist['id']}")
        assert screen._selected_content_item is None


@pytest.mark.asyncio
async def test_same_tab_switch_to_server_replaces_local_reader_without_queries(
    monkeypatch,
) -> None:
    controller = _controller_double()
    controller.get_overview_data = AsyncMock(return_value={})
    local_row = {
        "id": "local:watchlist_item:8",
        "item_id": 8,
        "title": "Local row before server switch",
        "status": "read",
        "url": "https://example.com/8",
        "created_at": "2026-08-23T12:00:00+00:00",
    }
    controller.list_reader_items_page = AsyncMock(
        return_value=WatchlistItemPage(
            items=(local_row,),
            has_more=False,
            snapshot_max_item_id=8,
            snapshot_count=1,
            next_cursor=None,
        )
    )
    controller.list_items = AsyncMock(return_value=[local_row])
    controller.get_item_content = AsyncMock(return_value="Local reader body")
    controller.check_all = AsyncMock(return_value={"checked": 0, "failed": []})
    app = _build_test_app()
    bundle = app.watchlist_bundle_service
    source_id = bundle._db.add_subscription(
        name="Same-tab local source",
        type="rss",
        source="https://same-tab.example/feed",
    )
    watchlist = bundle.create("Same-tab local watchlist")
    bundle.add_source(watchlist["id"], source_id)
    local_spies = {}
    for name in (
        "list_watchlists",
        "list_source_rows",
        "list_all_source_rows",
        "list_unassigned_source_rows",
        "get_watchlist_item_counts",
        "get_flagged_items_count",
        "get_unread_items_count_since",
        "get_source_item_counts",
    ):
        spy = Mock(wraps=getattr(bundle, name))
        monkeypatch.setattr(bundle, name, spy)
        local_spies[name] = spy

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.4)
        screen = host.screen_stack[-1]
        await host.workers.wait_for_complete()
        screen._controller = controller

        assert screen.active_section == "items"
        assert screen.runtime_backend == "local"
        await screen._load_tree_data().wait()
        assert screen.query(f"#wl-tree-node-watchlist-{watchlist['id']}")
        assert bundle.list_source_rows(watchlist["id"])[0]["name"] == (
            "Same-tab local source"
        )
        await screen._replace_items_snapshot(reason="initial")
        screen.post_message(ItemSelected(local_row))
        assert await _wait_until(
            pilot, lambda: screen._selected_content_item is local_row
        )
        await pilot.pause()

        selector = screen.query_one("#watchlists-backend-select")
        assert [
            item["title"]
            for item in screen.query_one(
                "#watchlists-items-pane", ArticleListPane
            ).items
        ] == ["Local row before server switch"]
        assert screen.query("#watchlists-content-pane")
        assert not screen.query("#watchlists-read-local-only")

        controller.list_items.reset_mock()
        controller.get_item_content.reset_mock()
        controller.check_all.reset_mock()
        for spy in local_spies.values():
            spy.reset_mock()

        selector.value = "server"
        assert await _wait_until(
            pilot, lambda: bool(screen.query("#watchlists-read-local-only"))
        )

        assert screen.active_section == "items"
        assert screen.runtime_backend == "server"
        assert selector.value == "server"
        assert selector.disabled is True
        assert not screen.query("#watchlists-content-pane")
        assert not screen.query_one(
            "#watchlists-items-pane", ArticleListPane
        ).items
        assert screen._selected_content_item is None
        assert not screen.query(f"#wl-tree-node-watchlist-{watchlist['id']}")
        assert not screen.query(
            f"#wl-tree-node-source-{watchlist['id']}-{source_id}"
        )
        rendered = host.export_screenshot()
        assert "Same-tab local watchlist" not in rendered
        assert "Same-tab local source" not in rendered
        assert "Local Watchlists snapshot" not in rendered

        screen.post_message(ItemSelected(local_row))
        screen.post_message(ItemsFilterChanged("unread", "server query"))
        screen.post_message(PreviousItemsPageRequested())
        screen.post_message(NextItemsPageRequested())
        screen.action_refresh_all()
        await screen._load_tree_data().wait()
        await pilot.pause(0.5)
        await host.workers.wait_for_complete()

        controller.list_items.assert_not_awaited()
        controller.get_item_content.assert_not_awaited()
        controller.check_all.assert_not_awaited()
        for name, spy in local_spies.items():
            assert spy.call_count == 0, name
        assert screen._items_page_loading is False
        assert screen.query_one(
            "#watchlists-items-pane", ArticleListPane
        ).page_loading is False


@pytest.mark.asyncio
async def test_entering_server_read_hides_local_reader_navigation_without_queries(
    monkeypatch,
) -> None:
    """Sources -> Server -> Read cannot retain any local Reader state."""
    app = _build_test_app()
    service = app.watchlist_bundle_service
    db = service._db
    source_id = db.add_subscription(
        name="Local counted feed", type="rss", source="https://counted.example/feed"
    )
    watchlist = service.create("Cross-tab local watchlist")
    service.add_source(watchlist["id"], source_id)
    _seed_item(db, source_id, "Local row before cross-tab server switch")

    local_spies = {}
    for name in (
        "list_watchlists",
        "list_source_rows",
        "list_all_source_rows",
        "list_unassigned_source_rows",
        "get_watchlist_item_counts",
        "get_flagged_items_count",
        "get_unread_items_count_since",
        "get_source_item_counts",
    ):
        spy = Mock(wraps=getattr(service, name))
        monkeypatch.setattr(service, name, spy)
        local_spies[name] = spy

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.4)
        screen = host.screen_stack[-1]
        await host.workers.wait_for_complete()
        await screen._load_tree_data().wait()
        assert await _wait_until(
            pilot,
            lambda: bool(
                screen.query(f"#wl-tree-node-watchlist-{watchlist['id']}")
            ),
        )
        assert service.list_source_rows(watchlist["id"])[0]["name"] == (
            "Local counted feed"
        )
        assert await screen._replace_items_snapshot(reason="initial")
        local_row = screen._loaded_items[0]
        screen.post_message(ItemSelected(local_row))
        assert await _wait_until(
            pilot, lambda: screen._selected_content_item is local_row
        )

        assert any(bucket.get("unread", 0) for bucket in screen._tree_counts.values())
        assert screen._tree_source_counts[source_id]["unread"] == 1
        assert "1" in str(screen.query_one("#wl-tree-node-all", Button).label)
        assert screen.query("#watchlists-content-pane")
        assert screen.query_one("#watchlists-items-pane", ArticleListPane).items
        screen.query_one(f"#wl-tree-expand-{watchlist['id']}", Button).press()
        screen.post_message(
            TreeScopeChanged(
                TreeScope(kind="watchlist", watchlist_id=watchlist["id"])
            )
        )
        assert await _wait_until(
            pilot,
            lambda: bool(
                screen.query(
                    f"#wl-tree-node-source-{watchlist['id']}-{source_id}"
                )
            ),
        )
        await host.workers.wait_for_complete()
        assert screen._wc_loaded is True
        assert screen._local_watchlist_count == 1

        screen.post_message(ItemsFilterChanged("unread", "local query"))
        await pilot.pause(0.4)
        await host.workers.wait_for_complete()
        screen._items_page_index = 2
        screen._items_has_next = True
        screen._push_items_pager_state()
        parked_watchlists = list(screen._tree_watchlists)
        parked_snapshot = screen._local_watchlist_records
        parked_snapshot_count = screen._local_watchlist_count

        list_items = AsyncMock(wraps=screen._controller.list_items)
        get_item_content = AsyncMock(wraps=screen._controller.get_item_content)
        check_all = AsyncMock(wraps=screen._controller.check_all)
        screen._controller.list_items = list_items
        screen._controller.get_item_content = get_item_content
        screen._controller.check_all = check_all
        screen.active_section = "sources"
        await pilot.pause(0.3)
        selector = screen.query_one("#watchlists-backend-select")
        selector.value = "server"
        await pilot.pause(0.3)
        await host.workers.wait_for_complete()
        for spy in local_spies.values():
            spy.reset_mock()

        screen.active_section = "items"
        assert await _wait_until(
            pilot, lambda: bool(screen.query("#watchlists-read-local-only"))
        )
        await host.workers.wait_for_complete()

        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        assert screen.runtime_backend == "server"
        assert selector.value == "server"
        assert selector.disabled is True
        assert not pane.items
        assert pane.status_filter == "all"
        assert pane.search_query == ""
        assert pane.page_number == 1
        assert pane.has_previous is False
        assert pane.has_next is False
        assert pane.page_loading is False
        assert pane.selected_item is None
        assert screen._loaded_items == []
        assert screen._selected_content_item is None
        assert screen._tree_watchlists == parked_watchlists
        assert screen._tree_counts
        assert screen._tree_source_counts
        assert screen._local_watchlist_records == parked_snapshot
        assert screen._local_watchlist_count == parked_snapshot_count
        assert screen._wc_loaded is True
        assert not screen.query(f"#wl-tree-node-watchlist-{watchlist['id']}")
        assert not screen.query(
            f"#wl-tree-node-source-{watchlist['id']}-{source_id}"
        )
        rendered = host.export_screenshot()
        assert "Cross-tab local watchlist" not in rendered
        assert "Local counted feed" not in rendered
        assert "Local Watchlists snapshot" not in rendered
        assert "All sources  0" in str(
            screen.query_one("#wl-tree-node-all", Button).label
        )

        screen.post_message(ItemSelected(local_row))
        screen.post_message(ItemsFilterChanged("unread", "server query"))
        screen.post_message(NextItemsPageRequested())
        screen.action_refresh_all()
        await pilot.pause(0.5)
        await host.workers.wait_for_complete()

        list_items.assert_not_awaited()
        get_item_content.assert_not_awaited()
        check_all.assert_not_awaited()
        for name, spy in local_spies.items():
            assert spy.call_count == 0, name
        assert screen._items_page_loading is False

        screen.active_section = "sources"
        assert await _wait_until(
            pilot, lambda: bool(screen.query("#watchlists-sources-pane"))
        )
        await host.workers.wait_for_complete()
        assert await _wait_until(
            pilot,
            lambda: bool(screen.query(f"#wl-tree-node-watchlist-{watchlist['id']}"))
        )
        assert await _wait_until(
            pilot,
            lambda: bool(
                screen.query(
                    f"#wl-tree-node-source-{watchlist['id']}-{source_id}"
                )
            ),
        )
        assert screen.query("#wc-watchlists-summary")

        watchlist_node = screen.query_one(
            f"#wl-tree-node-watchlist-{watchlist['id']}", Button
        )
        source_node = screen.query_one(
            f"#wl-tree-node-source-{watchlist['id']}-{source_id}", Button
        )
        assert "Cross-tab local watchlist" in str(watchlist_node.label)
        assert "Local counted feed" in str(source_node.label)
        assert _static_text(screen.query_one("#wc-watchlists-summary", Static)) == (
            "Local Watchlists snapshot: Cross-tab local watchlist (1 source)"
        )
        assert not screen.query("#wc-loading-state")
        list_items.assert_not_awaited()
        get_item_content.assert_not_awaited()

        for spy in local_spies.values():
            spy.reset_mock()
        screen.active_section = "items"
        assert await _wait_until(
            pilot, lambda: bool(screen.query("#watchlists-read-local-only"))
        )
        await host.workers.wait_for_complete()

        recovery_render = host.export_screenshot()
        assert "Cross-tab local watchlist" not in recovery_render
        assert "Local counted feed" not in recovery_render
        assert "Local Watchlists snapshot" not in recovery_render
        list_items.assert_not_awaited()
        get_item_content.assert_not_awaited()
        for name, spy in local_spies.items():
            assert spy.call_count == 0, name


# --- Task 7: scope-driven scoped rows, with real seeded data ---------------
#
# Tests/UI/test_watchlists_destination_shell.py's own scope tests run
# against DestinationHarness's empty subscriptions DB, so the strongest
# thing they can assert is "narrowing differs, or both sides were already
# empty." These seed real rows through the same `watchlist_bundle_service`
# the screen itself reaches (`_build_test_app()` wires it to an isolated
# temp-dir SQLite file -- see that fixture's `get_subscriptions_db_path`
# patch -- never the user's real database), so the comparisons here are
# exact rather than escape-hatched.


@pytest.mark.asyncio
async def test_scoped_source_rows_narrows_by_watchlist_and_unassigned():
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        service = app.watchlist_bundle_service
        db = service._db

        morning = service.create("Morning AI Brief")
        security = service.create("Security")
        a = db.add_subscription(name="ArXiv", type="rss", source="https://a.example/f")
        b = db.add_subscription(name="Krebs", type="rss", source="https://b.example/f")
        c = db.add_subscription(
            name="Loose Feed", type="rss", source="https://c.example/f"
        )
        service.add_source(morning["id"], a)
        service.add_source(security["id"], b)
        await screen._load_tree_data().wait()

        screen.post_message(TreeScopeChanged(TreeScope(kind="all")))
        await pilot.pause()
        assert {row["id"] for row in screen.scoped_source_rows()} == {a, b, c}

        screen.post_message(
            TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=morning["id"]))
        )
        await pilot.pause()
        assert [row["id"] for row in screen.scoped_source_rows()] == [a]

        screen.post_message(TreeScopeChanged(TreeScope(kind="unassigned")))
        await pilot.pause()
        assert [row["id"] for row in screen.scoped_source_rows()] == [c]

        screen.post_message(
            TreeScopeChanged(
                TreeScope(kind="source", watchlist_id=security["id"], source_id=b)
            )
        )
        await pilot.pause()
        assert [row["id"] for row in screen.scoped_source_rows()] == [b]


@pytest.mark.asyncio
async def test_the_header_summary_names_the_scope_with_a_live_count():
    # Seeded *before* the screen mounts (unlike the narrowing test above,
    # which only needs id/type matches): the summary's watchlist-name lookup
    # resolves against `_tree_watchlists`, populated once by `_load_tree_data`
    # in `on_mount` -- the same in-memory-only, no-second-query lookup
    # `_resolve_breadcrumb_labels` already relies on. Seeding after mount
    # would leave that cache stale and fall back to "Watchlist {id}", which
    # is correct behaviour for a real race but not what this test measures.
    app = _build_test_app()
    watchlist = app.watchlist_bundle_service.create("Morning AI Brief")
    db = app.watchlist_bundle_service._db
    a = db.add_subscription(name="ArXiv", type="rss", source="https://a.example/f")
    b = db.add_subscription(name="Krebs", type="rss", source="https://b.example/f")
    app.watchlist_bundle_service.add_source(watchlist["id"], a)
    app.watchlist_bundle_service.add_source(watchlist["id"], b)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]

        # task-2513: the scope readout lives in the centre header's summary
        # line (`#wc-watchlists-summary`), mounted on every tab -- Read is
        # the default, so no section switch is needed here anymore.
        assert screen.active_section == "items"

        screen.post_message(
            TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=watchlist["id"]))
        )
        summary = ""
        for _ in range(20):
            await pilot.pause()
            node = screen.query("#wc-watchlists-summary")
            if node:
                summary = _static_text(node[0])
            if "Morning AI Brief" in summary:
                break

        assert summary == "Local Watchlists snapshot: Morning AI Brief (2 sources)"


@pytest.mark.parametrize(
    ("scope", "expected"),
    (
        (
            TreeScope(kind="source", source_id=9, parent_context="all"),
            ["All Sources", "Feed Nine"],
        ),
        (
            TreeScope(kind="source", source_id=9, parent_context="unassigned"),
            ["Unassigned", "Feed Nine"],
        ),
        (
            TreeScope(kind="source", source_id=9, parent_context="unread"),
            ["All Unread", "Feed Nine"],
        ),
        (
            TreeScope(
                kind="source",
                source_id=9,
                watchlist_id=7,
                parent_context="watchlist",
            ),
            ["Morning AI Brief", "Feed Nine"],
        ),
    ),
)
def test_contextual_source_breadcrumbs_use_snapshot_parent_and_feed_labels(
    scope: TreeScope, expected: list[str]
) -> None:
    app = Mock()
    service = Mock()
    app.watchlist_bundle_service = service
    screen = WatchlistsCollectionsScreen(app)
    screen._tree_watchlists = [{"id": 7, "name": "Morning AI Brief"}]
    screen._tree_all_source_rows = [{"id": 9, "name": "Feed Nine"}]
    screen._tree_unassigned_source_rows = [{"id": 9, "name": "Feed Nine"}]

    assert screen._resolve_breadcrumb_labels(scope) == expected
    service.list_source_rows.assert_not_called()


def test_failed_contextual_scope_names_attempted_occurrence_and_retained_scope():
    app = Mock()
    app.notify = Mock()
    screen = WatchlistsCollectionsScreen(app)
    screen._tree_all_source_rows = [{"id": 9, "name": "Feed Nine"}]
    screen.__dict__["_reactive_tree_scope"] = TreeScope(kind="all")

    screen._notify_pending_scope_failure(
        TreeScope(kind="source", source_id=9, parent_context="unread")
    )

    app.notify.assert_called_once_with(
        "Couldn't open Feed Nine under All Unread; still showing All Sources.",
        severity="error",
        markup=False,
    )


def test_unread_context_forces_effective_filter_without_overwriting_manual_choice():
    screen = WatchlistsCollectionsScreen(Mock())
    screen.__dict__["_reactive_runtime_backend"] = "local"
    screen._items_status_filter = "all"
    unread_source = TreeScope(
        kind="source", source_id=9, parent_context="unread"
    )

    assert screen._effective_items_status_filter(unread_source) == "unread"
    assert screen._items_status_filter == "all"
    query = screen._reader_item_query(scope=unread_source)
    assert query.as_kwargs()["status"] == "new"
    assert "statuses" not in query.as_kwargs()
    assert query.context_key[-2] == "unread"


@pytest.mark.parametrize(
    ("scope", "all_ids", "unassigned_ids", "watchlists", "members", "expected"),
    (
        (
            TreeScope(kind="source", source_id=9, parent_context="all"),
            set(),
            set(),
            {7},
            {9},
            TreeScope(kind="all"),
        ),
        (
            TreeScope(kind="source", source_id=9, parent_context="unassigned"),
            {9},
            set(),
            {7},
            {9},
            TreeScope(kind="unassigned"),
        ),
        (
            TreeScope(
                kind="source",
                source_id=9,
                watchlist_id=7,
                parent_context="watchlist",
            ),
            {9},
            set(),
            {7},
            set(),
            TreeScope(kind="watchlist", watchlist_id=7),
        ),
        (
            TreeScope(kind="source", source_id=9, parent_context="unread"),
            {9},
            set(),
            {7},
            set(),
            TreeScope(kind="source", source_id=9, parent_context="unread"),
        ),
    ),
)
def test_contextual_scope_reconciliation_chooses_nearest_existing_parent(
    scope: TreeScope,
    all_ids: set[int],
    unassigned_ids: set[int],
    watchlists: set[int],
    members: set[int],
    expected: TreeScope,
) -> None:
    screen = WatchlistsCollectionsScreen(Mock())
    snapshot = collections_module.TreeDataSnapshot(
        tuple({"id": watchlist_id, "name": f"List {watchlist_id}"} for watchlist_id in watchlists),
        tuple({"id": source_id, "name": f"Feed {source_id}"} for source_id in all_ids),
        tuple({"id": source_id, "name": f"Feed {source_id}"} for source_id in unassigned_ids),
        {},
        {},
        watchlist_source_ids={
            watchlist_id: frozenset(members) for watchlist_id in watchlists
        },
    )

    assert screen._reconciled_tree_scope(scope, snapshot) == expected


def test_unread_zero_count_pin_follows_pending_then_committed_authority() -> None:
    screen = WatchlistsCollectionsScreen(Mock())
    committed = TreeScope(kind="source", source_id=7, parent_context="unread")
    pending = TreeScope(kind="source", source_id=9, parent_context="unread")
    screen.__dict__["_reactive_tree_scope"] = committed

    screen._pending_tree_scope = pending
    assert screen._unread_pin_source_id() == 9

    screen._pending_tree_scope = TreeScope(kind="all")
    assert screen._unread_pin_source_id() == 7

    screen.__dict__["_reactive_tree_scope"] = TreeScope(kind="all")
    assert screen._unread_pin_source_id() is None


def test_invalid_pending_scope_is_discarded_without_committing_its_fallback() -> None:
    screen = WatchlistsCollectionsScreen(Mock())
    screen.__dict__["_reactive_tree_scope"] = TreeScope(kind="all")
    screen._pending_tree_scope = TreeScope(
        kind="source",
        source_id=9,
        parent_context="unassigned",
    )
    screen._items_snapshot_generation = 4
    screen._items_page_loading = True
    screen._apply_tree_scope = Mock()
    snapshot = collections_module.TreeDataSnapshot((), (), (), {}, {})

    screen._reconcile_tree_navigation(snapshot)

    assert screen._pending_tree_scope is None
    assert screen._items_snapshot_generation == 5
    assert screen._items_page_loading is False
    screen._apply_tree_scope.assert_not_called()


def test_membership_reconciliation_failure_preserves_contextual_scope() -> None:
    app = Mock()
    app.watchlist_bundle_service.list_source_rows.side_effect = RuntimeError(
        "membership unavailable"
    )
    screen = WatchlistsCollectionsScreen(app)
    scope = TreeScope(
        kind="source",
        source_id=9,
        watchlist_id=7,
        parent_context="watchlist",
    )
    snapshot = collections_module.TreeDataSnapshot(
        ({"id": 7, "name": "List 7"},),
        ({"id": 9, "name": "Feed 9"},),
        (),
        {},
        {},
    )

    assert screen._reconciled_tree_scope(scope, snapshot) == scope


def test_membership_reconciliation_uses_worker_snapshot_without_service_io() -> None:
    app = Mock()
    app.watchlist_bundle_service.list_source_rows.return_value = []
    screen = WatchlistsCollectionsScreen(app)
    scope = TreeScope(
        kind="source",
        source_id=9,
        watchlist_id=7,
        parent_context="watchlist",
    )
    snapshot = collections_module.TreeDataSnapshot(
        ({"id": 7, "name": "List 7"},),
        ({"id": 9, "name": "Feed 9"},),
        (),
        {},
        {},
        watchlist_source_ids={7: frozenset()},
    )

    assert screen._reconciled_tree_scope(scope, snapshot) == TreeScope(
        kind="watchlist", watchlist_id=7
    )
    app.watchlist_bundle_service.list_source_rows.assert_not_called()


def test_committed_read_scope_reconciliation_requests_atomic_fallback() -> None:
    screen = WatchlistsCollectionsScreen(Mock())
    committed = TreeScope(
        kind="source",
        source_id=9,
        parent_context="unassigned",
    )
    screen.__dict__["_reactive_tree_scope"] = committed
    screen.__dict__["_reactive_active_section"] = "items"
    screen.__dict__["_reactive_runtime_backend"] = "local"
    screen._request_tree_scope = Mock()
    screen._apply_tree_scope = Mock()
    snapshot = collections_module.TreeDataSnapshot(
        (),
        ({"id": 9, "name": "Feed 9"},),
        (),
        {},
        {},
    )

    screen._reconcile_tree_navigation(snapshot)

    fallback = TreeScope(kind="unassigned")
    assert screen.tree_scope == committed
    screen._request_tree_scope.assert_called_once_with(fallback)
    screen._apply_tree_scope.assert_not_called()


@pytest.mark.parametrize(
    ("scope", "expected"),
    (
        (
            TreeScope(kind="source", source_id=9, parent_context="all"),
            [
                TreeScope(kind="all"),
                TreeScope(kind="source", source_id=9, parent_context="all"),
            ],
        ),
        (
            TreeScope(kind="source", source_id=9, parent_context="unassigned"),
            [
                TreeScope(kind="unassigned"),
                TreeScope(
                    kind="source", source_id=9, parent_context="unassigned"
                ),
            ],
        ),
        (
            TreeScope(kind="source", source_id=9, parent_context="unread"),
            [
                TreeScope(kind="unread"),
                TreeScope(kind="source", source_id=9, parent_context="unread"),
            ],
        ),
        (
            TreeScope(
                kind="source",
                source_id=9,
                watchlist_id=7,
                parent_context="watchlist",
            ),
            [
                TreeScope(kind="watchlist", watchlist_id=7),
                TreeScope(
                    kind="source",
                    source_id=9,
                    watchlist_id=7,
                    parent_context="watchlist",
                ),
            ],
        ),
    ),
)
def test_inspector_contextual_source_breadcrumb_targets_preserve_parent(
    scope: TreeScope, expected: list[TreeScope]
) -> None:
    pane = InspectorPane()
    pane.set_reactive(InspectorPane.scope, scope)
    pane.set_reactive(InspectorPane.breadcrumb_labels, ["Parent", "Feed Nine"])

    assert [level.target_scope for level in pane._scope_levels()] == expected


def test_server_management_disables_only_individual_feed_navigation():
    screen = WatchlistsCollectionsScreen(Mock())
    screen.__dict__["_reactive_runtime_backend"] = "server"
    screen.__dict__["_reactive_active_section"] = "sources"

    assert screen._tree_selection_disabled_reason() == (
        "Individual feed selection is available in Read or the Local backend."
    )
    screen.__dict__["_reactive_runtime_backend"] = "local"
    assert screen._tree_selection_disabled_reason() is None


# --- task-2513 Task 7: the tree scope drives the items list -----------------
#
# Before this task `_load_items` fetched the newest 100 items of ANY source
# regardless of the rail selection; `_items_scope_query` is the wiring that
# makes picking "Unassigned", a watchlist, or a source in the tree show only
# that scope's items. Items are seeded straight into `subscription_items`
# (there is no service-level item insert -- items arrive via the monitoring
# engine in production), the same pattern
# `Tests/Subscriptions/test_briefing_selection.py` uses; the reads then flow
# through the screen's real controller, scope service and DB against
# `_build_test_app()`'s isolated temp-dir SQLite file.


def _seed_item(db, subscription_id: int, title: str, created_at: str | None = None) -> int:
    """Insert one `subscription_items` row for `subscription_id`.

    `created_at` (ISO text) pins the display order when a test seeds several
    items: the items query sorts newest-first, and same-second ties from a
    fast seed loop would make row order (and thus `j`/`k`/`space`
    expectations) nondeterministic.
    """
    with db.transaction() as conn:
        if created_at is None:
            cursor = conn.execute(
                "INSERT INTO subscription_items (subscription_id, url, title) "
                "VALUES (?, ?, ?)",
                (
                    subscription_id,
                    f"https://item.example/{subscription_id}/{title}",
                    title,
                ),
            )
        else:
            cursor = conn.execute(
                "INSERT INTO subscription_items (subscription_id, url, title, created_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    subscription_id,
                    f"https://item.example/{subscription_id}/{title}",
                    title,
                    created_at,
                ),
            )
        return cursor.lastrowid


async def _wait_for_items(pilot, pane, attempts: int = 60) -> None:
    """Pause until the items pane holds rows (or give up and let assert fail)."""
    for _ in range(attempts):
        await pilot.pause()
        if pane.items:
            return


@pytest.mark.asyncio
async def test_items_reload_scopes_to_watchlist():
    """A watchlist scope shows only its member sources' items."""
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        service = app.watchlist_bundle_service
        db = service._db

        watchlist = service.create("Morning AI Brief")
        member = db.add_subscription(
            name="ArXiv", type="rss", source="https://a.example/f"
        )
        outsider = db.add_subscription(
            name="Krebs", type="rss", source="https://b.example/f"
        )
        service.add_source(watchlist["id"], member)
        _seed_item(db, member, "Member item")
        _seed_item(db, outsider, "Outsider item")

        screen._apply_tree_scope(
            TreeScope(kind="watchlist", watchlist_id=watchlist["id"])
        )
        await screen._replace_items_snapshot(reason="initial")

        assert screen._loaded_items, "precondition: the watchlist's source has items"
        # `source_id` is the normalized item dict's own key for the
        # originating subscription (see `normalize_watchlist_item`).
        assert {item["source_id"] for item in screen._loaded_items} == {member}


@pytest.mark.asyncio
async def test_items_reload_scopes_to_unassigned():
    """The Unassigned scope shows only items of sources in no watchlist."""
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        service = app.watchlist_bundle_service
        db = service._db

        watchlist = service.create("Morning AI Brief")
        member = db.add_subscription(
            name="ArXiv", type="rss", source="https://a.example/f"
        )
        loose = db.add_subscription(
            name="Loose Feed", type="rss", source="https://c.example/f"
        )
        service.add_source(watchlist["id"], member)
        _seed_item(db, member, "Member item")
        _seed_item(db, loose, "Loose item")

        screen._apply_tree_scope(TreeScope(kind="unassigned"))
        await screen._replace_items_snapshot(reason="initial")

        assert screen._loaded_items, "precondition: the unassigned source has items"
        assert {item["source_id"] for item in screen._loaded_items} == {loose}


@pytest.mark.asyncio
async def test_items_reload_scopes_to_source():
    """A source scope collapses to that one source's items, even inside a
    watchlist with other members."""
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        service = app.watchlist_bundle_service
        db = service._db

        watchlist = service.create("Morning AI Brief")
        arxiv = db.add_subscription(
            name="ArXiv", type="rss", source="https://a.example/f"
        )
        krebs = db.add_subscription(
            name="Krebs", type="rss", source="https://b.example/f"
        )
        service.add_source(watchlist["id"], arxiv)
        service.add_source(watchlist["id"], krebs)
        _seed_item(db, arxiv, "ArXiv item")
        _seed_item(db, krebs, "Krebs item")

        screen._apply_tree_scope(
            TreeScope(kind="source", watchlist_id=watchlist["id"], source_id=krebs)
        )
        await screen._replace_items_snapshot(reason="initial")

        assert screen._loaded_items, "precondition: the scoped source has items"
        assert {item["source_id"] for item in screen._loaded_items} == {krebs}


@pytest.mark.asyncio
async def test_items_reload_scopes_to_starred():
    """TASK-3072 plan task 6: the Starred smart feed lists flagged items from
    every source -- membership is irrelevant, the flag is global (ADR-018
    semantics, same as the briefing queue)."""
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        service = app.watchlist_bundle_service
        db = service._db

        arxiv = db.add_subscription(
            name="ArXiv", type="rss", source="https://a.example/f"
        )
        krebs = db.add_subscription(
            name="Krebs", type="rss", source="https://b.example/f"
        )
        starred_a = _seed_item(db, arxiv, "Starred from ArXiv")
        starred_b = _seed_item(db, krebs, "Starred from Krebs")
        _seed_item(db, arxiv, "Plain ArXiv item")
        db.set_item_flagged(starred_a, True)
        db.set_item_flagged(starred_b, True)

        screen._apply_tree_scope(TreeScope(kind="starred"))
        await screen._replace_items_snapshot(reason="initial")

        assert screen._loaded_items, "precondition: two items are starred"
        assert {item["title"] for item in screen._loaded_items} == {
            "Starred from ArXiv",
            "Starred from Krebs",
        }


# --- TASK-3791 plan task 4: All Unread + Today scopes --------------------------


@pytest.mark.asyncio
async def test_items_reload_scopes_to_all_unread():
    """All Unread: every source's unread items, membership irrelevant."""
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        db = app.watchlist_bundle_service._db
        arxiv = db.add_subscription(
            name="ArXiv", type="rss", source="https://a.example/f"
        )
        krebs = db.add_subscription(
            name="Krebs", type="rss", source="https://b.example/f"
        )
        _seed_item(db, arxiv, "Unread from ArXiv")
        read_id = _seed_item(db, krebs, "Read from Krebs")
        _seed_item(db, krebs, "Unread from Krebs")
        db.mark_item_status(read_id, "reviewed")

        screen._apply_tree_scope(TreeScope(kind="unread"))
        await screen._replace_items_snapshot(reason="initial")

        assert {item["title"] for item in screen._loaded_items} == {
            "Unread from ArXiv",
            "Unread from Krebs",
        }


@pytest.mark.asyncio
async def test_all_unread_scope_wins_over_the_all_filter():
    """The node is the stronger statement: All Unread + the pane's "All"
    filter must still show unread only -- and must never trip the
    status-vs-statuses ValueError in `get_new_items`."""
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        db = app.watchlist_bundle_service._db
        source_id = db.add_subscription(
            name="ArXiv", type="rss", source="https://a.example/f"
        )
        _seed_item(db, source_id, "Still unread")
        read_id = _seed_item(db, source_id, "Already read")
        db.mark_item_status(read_id, "reviewed")

        screen._apply_tree_scope(TreeScope(kind="unread"))
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        # The pane's "All" filter would normally widen the query to the
        # reader statuses; under the All Unread scope it must not.
        pane.status_filter = "all"
        await screen._replace_items_snapshot(reason="initial")

        assert [item["title"] for item in screen._loaded_items] == ["Still unread"]


@pytest.mark.asyncio
async def test_items_reload_scopes_to_today():
    """Today: effective date at/after local midnight, across every source."""
    from datetime import datetime, timedelta, timezone

    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        db = app.watchlist_bundle_service._db
        source_id = db.add_subscription(
            name="ArXiv", type="rss", source="https://a.example/f"
        )
        now = datetime.now(timezone.utc)
        fresh = _seed_item(db, source_id, "Fresh today")
        stale = _seed_item(db, source_id, "From yesterday")
        with db.transaction() as conn:
            conn.execute(
                "UPDATE subscription_items SET published_date = ? WHERE id = ?",
                (now.isoformat(), fresh),
            )
            conn.execute(
                "UPDATE subscription_items SET published_date = ? WHERE id = ?",
                ((now - timedelta(hours=25)).isoformat(), stale),
            )

        screen._apply_tree_scope(TreeScope(kind="today"))
        await screen._replace_items_snapshot(reason="initial")

        assert [item["title"] for item in screen._loaded_items] == ["Fresh today"]


@pytest.mark.asyncio
async def test_tree_move_requests_atomic_scope_on_read_tab():
    """Read navigation uses the pending request path, not the watcher."""
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        assert screen.active_section == "items", "precondition: lands on Read"

        original_replace = screen._replace_items_snapshot
        dispatches = 0

        async def spy(**kwargs):
            nonlocal dispatches
            dispatches += 1
            return await original_replace(**kwargs)

        screen._replace_items_snapshot = spy
        try:
            screen.post_message(TreeScopeChanged(TreeScope(kind="unassigned")))
            assert await _wait_until(pilot, lambda: dispatches == 1)
            assert dispatches >= 1, (
                "a tree move on Read must dispatch a candidate snapshot"
            )
        finally:
            screen._replace_items_snapshot = original_replace


# --- task-2513 Task 10: reader verbs (m / space / a / u) --------------------
#
# The keyboard half of the reading loop: `m` toggles read on the open item,
# `space` opens the next unread, `a` catches the scope up (undo with `u`).
# Driven end to end through the real screen, the real key pipeline and
# `_build_test_app()`'s isolated SQLite file. `space` is pane-bound — the
# rail-focus and typing-guard facts have their own regression tests below.


@pytest.mark.asyncio
async def test_m_toggles_read_state_on_open_item():
    """`m` flips the open item new -> reviewed -> new (and the open itself
    already marked it read, so the observed cycle starts at reviewed)."""
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        db = app.watchlist_bundle_service._db
        source_id = db.add_subscription(
            name="ArXiv", type="rss", source="https://a.example/f"
        )
        item_id = _seed_item(db, source_id, "Toggle me")
        await screen._replace_items_snapshot(reason="initial")
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        await _wait_for_items(pilot, pane)
        assert pane.items, "precondition: the seeded item reaches the pane"

        pane.select_item_by_id(str(pane.items[0]["id"]))
        for _ in range(60):
            await pilot.pause()
            if db.get_new_items(status="reviewed", limit=10):
                break
        assert [r["id"] for r in db.get_new_items(status="reviewed", limit=10)] == [
            item_id
        ], "precondition: opening marked the item read"

        await pilot.press("m")
        for _ in range(60):
            await pilot.pause()
            if db.get_new_items(status="new", limit=10):
                break
        assert [r["id"] for r in db.get_new_items(status="new", limit=10)] == [
            item_id
        ], "`m` on a read item must restore it to unread"

        await pilot.press("m")
        for _ in range(60):
            await pilot.pause()
            if db.get_new_items(status="reviewed", limit=10):
                break
        assert [r["id"] for r in db.get_new_items(status="reviewed", limit=10)] == [
            item_id
        ], "`m` on an unread item must mark it read"


@pytest.mark.asyncio
async def test_last_unread_item_keeps_contextual_feed_and_reader_pinned():
    app = _build_test_app()
    db = app.watchlist_bundle_service._db
    source_id = db.add_subscription(
        name="Only unread feed",
        type="rss",
        source="https://only-unread.example/feed",
    )
    _seed_item(db, source_id, "Last unread item")
    host = DestinationHarness(app, "watchlists_collections")

    async with host.run_test(size=(180, 50)) as pilot:
        await host.workers.wait_for_complete()
        screen = host.screen_stack[-1]
        assert await _wait_until(
            pilot,
            lambda: bool(screen.query("#wl-tree-node-unread")),
        )
        await pilot.click("#wl-tree-expand-root-unread")
        assert await _wait_until(
            pilot,
            lambda: bool(
                screen.query(f"#wl-tree-node-source-unread-{source_id}")
            ),
        )
        await pilot.click(f"#wl-tree-node-source-unread-{source_id}")
        expected_scope = TreeScope(
            kind="source",
            source_id=source_id,
            parent_context="unread",
        )
        assert await _wait_until(
            pilot,
            lambda: screen.tree_scope == expected_scope,
        )
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        await _wait_for_items(pilot, pane)
        row_id = pane.items[0]["id"]

        pane.select_item_by_id(str(row_id))
        pane.query_one("#items-table", ListView).focus()
        assert await _wait_until(
            pilot,
            lambda: bool(db.get_new_items(status="reviewed", limit=10)),
        )
        content = screen.query_one("#watchlists-content-pane", ContentPane)
        focused_before_refresh = screen.focused
        reader_position_before_refresh = content.position
        page_before_refresh = pane.page_number
        assert await _wait_until(
            pilot,
            lambda: screen._tree_source_counts.get(source_id, {}).get("unread")
            == 0,
            ticks=140,
        )

        assert screen.query(f"#wl-tree-node-source-unread-{source_id}")
        assert "unread" in screen._tree_expanded_root_kinds
        assert screen.focused is focused_before_refresh
        assert pane.page_number == page_before_refresh
        assert content.position == reader_position_before_refresh
        assert [row["id"] for row in screen._loaded_items] == [row_id]
        assert screen._selected_content_item is not None
        assert screen._selected_content_item["id"] == row_id
        assert content.item is screen._selected_content_item

        await pilot.press("m")
        assert await _wait_until(
            pilot,
            lambda: bool(db.get_new_items(status="new", limit=10)),
        )
        await screen._load_tree_data().wait()
        assert screen._tree_source_counts[source_id]["unread"] == 1
        assert await _wait_until(
            pilot,
            lambda: bool(
                screen.query(f"#wl-tree-node-source-unread-{source_id}")
            ),
        )


@pytest.mark.asyncio
async def test_m_refuses_on_ingested_item():
    """`m` is a read/unread verb only: an ingested item is a deliberate
    record, never flipped back to `new` — the user gets a warning instead."""
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        db = app.watchlist_bundle_service._db
        source_id = db.add_subscription(
            name="ArXiv", type="rss", source="https://a.example/f"
        )
        item_id = _seed_item(db, source_id, "Ingested one")
        db.mark_item_status(item_id, "ingested")
        await screen._replace_items_snapshot(reason="initial")
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        await _wait_for_items(pilot, pane)
        assert pane.items, "precondition: the ingested item is listed (filter: all)"

        pane.select_item_by_id(str(pane.items[0]["id"]))
        await pilot.pause(0.3)
        app.notify = Mock()
        await pilot.press("m")
        for _ in range(20):
            await pilot.pause()
            if app.notify.called:
                break

        assert app.notify.called, "refusing to toggle must say so"
        _args, kwargs = app.notify.call_args
        assert kwargs.get("severity") == "warning"
        rows = db.get_new_items(status="ingested", limit=10)
        assert [r["id"] for r in rows] == [item_id], (
            "an ingested item must never be flipped by `m`"
        )


# --- TASK-3072 plan task 7: `s` and the reader's Star button -------------------


def _flagged_value(db, item_id: int) -> int:
    row = db.conn.execute(
        "SELECT is_flagged FROM subscription_items WHERE id = ?", (item_id,)
    ).fetchone()
    return int(row[0]) if row else -1


@pytest.mark.asyncio
async def test_s_toggles_star_on_the_open_item():
    """`s` stars, then unstars, the open item through the service; the row's
    star repaints in place and the Starred badge catches up through the
    debounced counts path."""
    from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import STARRED_BUCKET

    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        db = app.watchlist_bundle_service._db
        source_id = db.add_subscription(
            name="ArXiv", type="rss", source="https://a.example/f"
        )
        item_id = _seed_item(db, source_id, "Star me")
        await screen._replace_items_snapshot(reason="initial")
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        await _wait_for_items(pilot, pane)

        pane.select_item_by_id(str(pane.items[0]["id"]))
        await pilot.pause(0.3)
        assert screen._selected_content_item is not None, "precondition: item open"

        await pilot.press("s")
        for _ in range(60):
            await pilot.pause()
            if _flagged_value(db, item_id) == 1:
                break
        assert _flagged_value(db, item_id) == 1, "`s` must star the open item"

        row_widget = pane._find_row(str(pane.items[0]["id"]))
        assert pane._STAR_GLYPH in str(row_widget.render()), (
            "the row's star repaints in place -- no recompose"
        )

        from textual.widgets import Button

        assert str(screen.query_one("#content-star-button", Button).label) == (
            "★ Starred"
        ), "the reader's button flips on the success path, never optimistically"

        for _ in range(80):
            await pilot.pause(0.05)
            if screen._tree_counts.get(STARRED_BUCKET, {}).get("unread") == 1:
                break
        assert screen._tree_counts[STARRED_BUCKET]["unread"] == 1, (
            "the Starred badge must refresh through the debounced counts path"
        )

        # The open dict was patched, so the second press unstars rather than
        # re-deriving from a stale flag.
        await pilot.press("s")
        for _ in range(60):
            await pilot.pause()
            if _flagged_value(db, item_id) == 0:
                break
        assert _flagged_value(db, item_id) == 0, "a second `s` must unstar"
        assert str(screen.query_one("#content-star-button", Button).label) == "☆ Star"


@pytest.mark.asyncio
async def test_s_with_no_open_item_is_a_noop():
    """`s` with nothing open writes nothing and raises nothing."""
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        db = app.watchlist_bundle_service._db
        source_id = db.add_subscription(
            name="ArXiv", type="rss", source="https://a.example/f"
        )
        item_id = _seed_item(db, source_id, "Never opened")
        await screen._replace_items_snapshot(reason="initial")
        assert screen._selected_content_item is None, "precondition: nothing open"

        await pilot.press("s")
        await pilot.pause(0.3)
        assert _flagged_value(db, item_id) == 0


@pytest.mark.asyncio
async def test_star_toggle_requested_toggles_the_same_path():
    """The reader's Star button and the `s` key share one handler."""
    from tldw_chatbook.UI.Watchlists_Modules.content_pane import StarToggleRequested

    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        db = app.watchlist_bundle_service._db
        source_id = db.add_subscription(
            name="ArXiv", type="rss", source="https://a.example/f"
        )
        item_id = _seed_item(db, source_id, "Button-starred")
        await screen._replace_items_snapshot(reason="initial")
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        await _wait_for_items(pilot, pane)

        pane.select_item_by_id(str(pane.items[0]["id"]))
        await pilot.pause(0.3)
        assert screen._selected_content_item is not None, "precondition: item open"

        screen.post_message(
            StarToggleRequested(dict(screen._selected_content_item))
        )
        for _ in range(60):
            await pilot.pause()
            if _flagged_value(db, item_id) == 1:
                break
        assert _flagged_value(db, item_id) == 1, (
            "the button's message must reach the same write the `s` key does"
        )


# --- TASK-3072 plan task 8: `o` opens the item in the browser -----------------


async def _open_item_and_get_url(pilot, screen, db, title: str, url: str) -> int:
    """Seed one item carrying exactly `url`, open it in the reader."""
    source_id = db.add_subscription(
        name="ArXiv", type="rss", source="https://a.example/f"
    )
    item_id = _seed_item(db, source_id, title)
    with db.transaction() as conn:
        conn.execute(
            "UPDATE subscription_items SET url = ? WHERE id = ?", (url, item_id)
        )
    await screen._replace_items_snapshot(reason="initial")
    pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
    await _wait_for_items(pilot, pane)
    pane.select_item_by_id(str(pane.items[0]["id"]))
    await pilot.pause(0.3)
    assert screen._selected_content_item is not None, "precondition: item open"
    return item_id


def _successful_browser_recorder(opened: list[str]):
    """Return a browser stub that records its URL and reports success."""
    def record(url: str) -> bool:
        opened.append(url)
        return True

    return record


@pytest.mark.asyncio
async def test_o_opens_the_open_items_url(monkeypatch):
    """`o` hands the open item's http URL to the system browser."""
    opened: list[str] = []
    monkeypatch.setattr("webbrowser.open", _successful_browser_recorder(opened))

    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        db = app.watchlist_bundle_service._db
        await _open_item_and_get_url(
            pilot, screen, db, "Readable", "https://example.com/post"
        )
        app.notify = Mock()

        await pilot.press("o")
        await host.workers.wait_for_complete()
        await pilot.pause()
        assert opened == ["https://example.com/post"]
        app.notify.assert_not_called()


@pytest.mark.asyncio
async def test_o_refuses_a_non_http_url(monkeypatch):
    """A `javascript:`/`file:`/empty URL is a remote-derived string reaching
    an OS primitive: it is refused with a notification, never passed on."""
    opened: list[str] = []
    monkeypatch.setattr("webbrowser.open", opened.append)

    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        db = app.watchlist_bundle_service._db
        app.notify = Mock()
        await _open_item_and_get_url(
            pilot, screen, db, "Hostile", "javascript:alert(1)"
        )

        await pilot.press("o")
        for _ in range(20):
            await pilot.pause()
            if app.notify.called:
                break

        assert opened == [], "a non-http(s) scheme must never reach webbrowser"
        assert app.notify.called, "a refusal must say so"
        _args, kwargs = app.notify.call_args
        assert kwargs.get("severity") == "warning"


@pytest.mark.asyncio
async def test_o_strips_control_bytes_from_the_url_before_opening(monkeypatch):
    """A feed URL is remote-derived text: control bytes are stripped before
    the (already scheme- and host-validated) string reaches the OS."""
    opened: list[str] = []
    monkeypatch.setattr("webbrowser.open", _successful_browser_recorder(opened))

    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        db = app.watchlist_bundle_service._db
        await _open_item_and_get_url(
            pilot, screen, db, "Control bytes", "https://example.com/po\x07st"
        )
        app.notify = Mock()

        await pilot.press("o")
        await host.workers.wait_for_complete()
        await pilot.pause()
        assert opened == ["https://example.com/post"]
        app.notify.assert_not_called()


@pytest.mark.asyncio
async def test_o_with_no_open_item_is_a_noop(monkeypatch):
    opened: list[str] = []
    monkeypatch.setattr("webbrowser.open", opened.append)

    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        db = app.watchlist_bundle_service._db
        source_id = db.add_subscription(
            name="ArXiv", type="rss", source="https://a.example/f"
        )
        _seed_item(db, source_id, "Never opened")
        await screen._replace_items_snapshot(reason="initial")
        assert screen._selected_content_item is None, "precondition: nothing open"

        await pilot.press("o")
        await pilot.pause(0.2)
        assert opened == []


@pytest.mark.asyncio
async def test_open_in_browser_requested_takes_the_same_path(monkeypatch):
    """The reader's Open button and the `o` key share one handler."""
    from tldw_chatbook.UI.Watchlists_Modules.content_pane import (
        OpenInBrowserRequested,
    )

    opened: list[str] = []
    monkeypatch.setattr("webbrowser.open", _successful_browser_recorder(opened))

    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        db = app.watchlist_bundle_service._db
        await _open_item_and_get_url(
            pilot, screen, db, "Button-opened", "https://example.com/via-button"
        )
        app.notify = Mock()

        screen.post_message(
            OpenInBrowserRequested(dict(screen._selected_content_item))
        )
        await host.workers.wait_for_complete()
        await pilot.pause()
        assert opened == ["https://example.com/via-button"]
        app.notify.assert_not_called()


@pytest.mark.parametrize("activation", ["keyboard", "button"])
@pytest.mark.asyncio
async def test_open_validates_on_ui_thread_then_opens_in_worker(
    monkeypatch, activation: str
):
    """Both entry points converge before the UI/worker thread boundary."""
    from tldw_chatbook.UI.Watchlists_Modules.content_pane import (
        OpenInBrowserRequested,
    )

    ui_thread = threading.get_ident()
    validation_threads: list[int] = []
    browser_threads: list[int] = []
    def validate_on_recorded_thread(url: str) -> bool:
        validation_threads.append(threading.get_ident())
        return real_validate_url(url)

    def open_on_recorded_thread(url: str) -> bool:
        browser_threads.append(threading.get_ident())
        return True

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.watchlists_collections_screen.validate_url",
        validate_on_recorded_thread,
    )
    monkeypatch.setattr("webbrowser.open", open_on_recorded_thread)

    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        db = app.watchlist_bundle_service._db
        await _open_item_and_get_url(
            pilot, screen, db, "Threaded open", "https://example.com/threaded"
        )

        if activation == "keyboard":
            await pilot.press("o")
        else:
            screen.post_message(
                OpenInBrowserRequested(dict(screen._selected_content_item))
            )
        await host.workers.wait_for_complete()
        await pilot.pause()

        assert validation_threads == [ui_thread]
        assert len(browser_threads) == 1
        assert browser_threads[0] != ui_thread


# --- TASK-3072 plan task 9: the reader's position footer ----------------------


@pytest.mark.asyncio
async def test_the_reader_footer_numbers_the_open_item():
    """"N of M": M is the displayed list, N the open item's 1-based place in
    it; `j` advances the reader and the footer together."""
    from textual.widgets import Static

    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        db = app.watchlist_bundle_service._db
        source_id = db.add_subscription(
            name="ArXiv", type="rss", source="https://a.example/f"
        )
        # Newest-first display: [c (09:02), b (09:01), a (09:00)].
        _seed_item(db, source_id, "a", created_at="2026-08-06 09:00:00")
        _seed_item(db, source_id, "b", created_at="2026-08-06 09:01:00")
        _seed_item(db, source_id, "c", created_at="2026-08-06 09:02:00")
        await screen._replace_items_snapshot(reason="initial")
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        await _wait_for_items(pilot, pane)
        assert len(pane.displayed_items()) == 3, "precondition: all three listed"

        pane.select_item_by_id(str(pane.displayed_items()[1]["id"]))
        for _ in range(40):
            await pilot.pause()
            if screen._selected_content_item is not None:
                break
        assert screen._selected_content_item["title"] == "b", "precondition"
        assert str(screen.query_one("#content-position", Static).renderable) == "2 of 3"

        await pilot.press("j")
        for _ in range(40):
            await pilot.pause()
            if screen._selected_content_item.get("title") == "a":
                break
        assert screen._selected_content_item["title"] == "a", "precondition: j moved"
        assert str(screen.query_one("#content-position", Static).renderable) == "3 of 3", (
            "the footer must walk with the reader"
        )


@pytest.mark.asyncio
async def test_the_reader_footer_is_empty_with_nothing_open():
    """Nothing open: no footer at all, and definitely not "0 of 0"."""
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        assert screen._selected_content_item is None, "precondition"
        assert not screen.query("#content-position")


@pytest.mark.asyncio
async def test_the_next_unread_footer_button_opens_the_next_unread():
    """The footer's Next Unread control drives the same handler `space` does."""
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        db = app.watchlist_bundle_service._db
        source_id = db.add_subscription(
            name="ArXiv", type="rss", source="https://a.example/f"
        )
        _seed_item(db, source_id, "a", created_at="2026-08-06 09:00:00")
        b_id = _seed_item(db, source_id, "b", created_at="2026-08-06 09:01:00")
        _seed_item(db, source_id, "c", created_at="2026-08-06 09:02:00")
        db.mark_item_status(b_id, "reviewed")
        # Nothing open yet, so no footer exists -- open any item first.
        await screen._replace_items_snapshot(reason="initial")
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        await _wait_for_items(pilot, pane)
        pane.select_item_by_id(str(pane.displayed_items()[1]["id"]))
        for _ in range(40):
            await pilot.pause()
            if screen._selected_content_item is not None:
                break
        assert screen._selected_content_item["title"] == "b", "precondition"

        from textual.widgets import Button

        screen.query_one("#content-next-unread-button", Button).press()
        for _ in range(40):
            await pilot.pause()
            if screen._selected_content_item.get("title") == "a":
                break
        assert screen._selected_content_item["title"] == "a", (
            "from b, next unread walks the displayed sequence forward, "
            "past nothing, to a -- the only unread item after it"
        )


# --- TASK-3072 plan task 10: the hostile-HTML end-to-end pin ------------------


@pytest.mark.asyncio
async def test_a_hostile_item_stars_queues_and_still_renders_inert():
    """The phase-2 DoD, end to end: an item whose title and body carry
    `<script>`, `[bold red]` and control bytes renders as INERT TEXT in the
    row and the reader body while the star and queue verbs work on it --
    and both flags survive a re-persist (Task 3's pin, at the surface)."""
    from textual.widgets import Static

    from tldw_chatbook.Subscriptions.item_persist import persist_subscription_item

    hostile_title = "[bold red]x[/]<script>alert('TITLE_LITERAL')</script>\x1b[31m"
    hostile_body = "<script>alert('BODY_PAYLOAD')</script> [bold red]injected[/]\x00\x07"

    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        db = app.watchlist_bundle_service._db
        source_id = db.add_subscription(
            name="Evil Feed", type="rss", source="https://evil.example/f"
        )
        with db.transaction() as conn:
            item_id = persist_subscription_item(
                conn,
                source_id,
                {
                    "url": "https://evil.example/post",
                    "title": hostile_title,
                    "content": hostile_body,
                    "content_hash": "hash-hostile",
                },
                run_id=None,
                now="2026-08-06T09:00:00+00:00",
            )
        await screen._replace_items_snapshot(reason="initial")
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        await _wait_for_items(pilot, pane)

        pane.select_item_by_id(str(pane.items[0]["id"]))
        await pilot.pause(0.3)
        assert screen._selected_content_item is not None, "precondition: item open"

        # The verbs work on the hostile item exactly as on any other.
        await pilot.press("s")
        for _ in range(60):
            await pilot.pause()
            if _flagged_value(db, item_id) == 1:
                break
        assert _flagged_value(db, item_id) == 1, "the star write must land"
        db.set_item_briefing_queued(item_id, True)

        # The row renders the attacks as literal characters, control-stripped.
        row_widget = pane._find_row(str(pane.items[0]["id"]))
        row_text = str(row_widget.render())
        assert "[bold red]x[/]" in row_text, "markup-shaped text stays literal"
        assert "\x1b" not in row_text, "no escape sequence survives into a row"

        # The reader defends each field at its own layer. The TITLE is
        # documented plain text (render_article appends, never parses), so
        # its script-shaped characters render literally and inertly. The
        # BODY goes through `readable_body_text`, which drops script tags
        # AND their payloads. Bracket text stays literal; control bytes die.
        body = screen.query_one("#content-body", Static).renderable
        body_plain = getattr(body, "plain", str(body))
        assert "[bold red]injected[/]" in body_plain
        assert "TITLE_LITERAL" in body_plain, (
            "the title renders -- as inert literal characters"
        )
        assert "BODY_PAYLOAD" not in body_plain, (
            "the body's script payload is dropped, not shown"
        )
        assert "\x00" not in body_plain and "\x07" not in body_plain
        assert "\x1b" not in body_plain

        # Re-persist the same item (same url + content_hash, the re-fetch
        # shape): neither flag is touched by the upsert.
        with db.transaction() as conn:
            persist_subscription_item(
                conn,
                source_id,
                {
                    "url": "https://evil.example/post",
                    "title": hostile_title,
                    "content": hostile_body,
                    "content_hash": "hash-hostile",
                },
                run_id=None,
                now="2026-08-06T09:05:00+00:00",
            )
        assert _flagged_value(db, item_id) == 1, "the star survives a re-persist"
        queued = db.conn.execute(
            "SELECT queued_for_briefing FROM subscription_items WHERE id = ?",
            (item_id,),
        ).fetchone()
        assert int(queued[0]) == 1, "the queue flag survives a re-persist"


@pytest.mark.asyncio
async def test_space_opens_next_unread():
    """`space` walks to the next UNREAD item, skipping reviewed rows."""
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        db = app.watchlist_bundle_service._db
        source_id = db.add_subscription(
            name="ArXiv", type="rss", source="https://a.example/f"
        )
        # Newest-first display: [c (09:02), b (09:01), a (09:00)].
        _seed_item(db, source_id, "a", created_at="2026-08-06 09:00:00")
        b_id = _seed_item(db, source_id, "b", created_at="2026-08-06 09:01:00")
        _seed_item(db, source_id, "c", created_at="2026-08-06 09:02:00")
        db.mark_item_status(b_id, "reviewed")
        await screen._replace_items_snapshot(reason="initial")
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        await _wait_for_items(pilot, pane)
        assert len(pane.displayed_items()) == 3, "precondition: all three listed"
        displayed = pane.displayed_items()
        assert [d["title"] for d in displayed] == ["c", "b", "a"], (
            "precondition: newest-first order is what the user sees"
        )

        # Open the top row (new -> reviewed on open), then space past the
        # already-reviewed middle row to "a", the only remaining unread one.
        pane.select_item_by_id(str(displayed[0]["id"]))
        await pilot.pause(0.3)
        assert screen._selected_content_item is not None
        pane.query_one("#items-table").focus()
        await pilot.press("space")
        for _ in range(60):
            await pilot.pause()
            current = screen._selected_content_item
            if current is not None and current.get("title") == "a":
                break

        assert screen._selected_content_item.get("title") == "a", (
            "`space` must skip the reviewed row and open the next unread one"
        )


@pytest.mark.asyncio
async def test_space_at_end_notifies_all_caught_up():
    """No unread row below the current one: `space` says so, moves nothing."""
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        db = app.watchlist_bundle_service._db
        source_id = db.add_subscription(
            name="ArXiv", type="rss", source="https://a.example/f"
        )
        _seed_item(db, source_id, "only one")
        await screen._replace_items_snapshot(reason="initial")
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        await _wait_for_items(pilot, pane)
        pane.select_item_by_id(str(pane.items[0]["id"]))
        await pilot.pause(0.3)
        app.notify = Mock()
        pane.query_one("#items-table").focus()
        await pilot.press("space")
        for _ in range(20):
            await pilot.pause()
            if app.notify.called:
                break

        assert app.notify.called, "running out of unread items must say so"
        message = str(app.notify.call_args[0][0]).lower()
        assert "caught up" in message
        assert screen._selected_content_item.get("title") == "only one", (
            "the reader must stay on the current item"
        )


@pytest.mark.asyncio
async def test_space_with_rail_focused_does_not_navigate():
    """Regression: `space` is pane-bound, so the rail never triggers it.

    Assert the OUTCOME (no selection change) rather than whether a binding
    fired — the pane binding is simply unreachable from the rail.
    """
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        db = app.watchlist_bundle_service._db
        source_id = db.add_subscription(
            name="ArXiv", type="rss", source="https://a.example/f"
        )
        _seed_item(db, source_id, "unread one")
        await screen._replace_items_snapshot(reason="initial")
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        await _wait_for_items(pilot, pane)
        screen.query_one("#wl-tree-node-all", Button).focus()
        await pilot.press("space")
        await pilot.pause(0.3)

        assert screen._selected_content_item is None, (
            "space with the rail focused must not open anything"
        )


@pytest.mark.asyncio
async def test_space_in_items_search_input_still_types():
    """Typing spaces in the items search box is typing, not navigation.

    Also pins the fix for a real pre-existing bug this test exposed:
    `ItemsPane.search_query` is `reactive(..., recompose=True)`, so every
    keystroke recomposed the pane and destroyed the focused input — only the
    first character of any search ever landed. `ItemsPane.recompose()` now
    restores focus to the fresh input, so the whole query lands in the box.
    """
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        db = app.watchlist_bundle_service._db
        source_id = db.add_subscription(
            name="ArXiv", type="rss", source="https://a.example/f"
        )
        _seed_item(db, source_id, "f o matcher")
        await screen._replace_items_snapshot(reason="initial")
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        await _wait_for_items(pilot, pane)
        pane.query_one("#items-search-input", Input).focus()
        await pilot.press("f", "space", "o")
        await pilot.pause(0.2)

        # Re-query: each keystroke recomposes the pane, so the input that was
        # focused at the start was destroyed; this is the live replacement.
        search = pane.query_one("#items-search-input", Input)
        assert search.value == "f o"
        assert search.has_focus, "typing must not lose the search box"
        assert screen._selected_content_item is None


@pytest.mark.asyncio
async def test_typing_in_sources_search_survives_the_recompose():
    """Typing in the sources search box keeps focus and value, end to end.

    task-3071: the SourcesPane sibling of the items-search bug pinned above.
    `SourcesPane.search_query` was then `reactive(..., recompose=True)`, so
    every keystroke rebuilt the pane and destroyed the focused input -- and
    its `recompose()` only re-homed CREATE-FORM focus, so the box was lost
    (and with Textual's default `select_on_focus=True`, any programmatic
    refocus would have selected-all, replacing the half-typed query on the
    next keystroke).

    task-15460 removed that teardown entirely -- the filters are plain
    reactives that re-populate the table in place -- so the property this
    asserts now holds because nothing takes the focus rather than because
    `recompose()` gives it back. The assertions are unchanged on purpose:
    they are the user-facing outcome, and they must keep holding through
    whichever mechanism is underneath.
    """
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        db = app.watchlist_bundle_service._db
        db.add_subscription(name="ArXiv", type="rss", source="https://a.example/f")
        db.add_subscription(name="Krebs", type="rss", source="https://b.example/f")

        screen.active_section = "sources"
        await pilot.pause(0.3)
        pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
        pane.query_one("#sources-search-input", Input).focus()
        await pilot.press("k", "r", "e", "b", "s")
        await pilot.pause(0.2)

        # Re-query rather than reusing the reference: before task-15460
        # every keystroke recomposed the pane and destroyed the input that
        # was focused at the start, so only a fresh query saw the live one.
        # Nothing replaces it today, but reading the DOM is what makes this
        # test agnostic about that.
        search = pane.query_one("#sources-search-input", Input)
        assert search.value == "krebs"
        assert search.has_focus, "typing must not lose the sources search box"


@pytest.mark.asyncio
async def test_create_form_open_focuses_first_field_over_search():
    """Opening the create form focuses its first field even when search had focus.

    Qodo, PR #1418: task-3071 made `SourcesPane.recompose()` restore search
    focus whenever the search input was focused pre-teardown. When the SAME
    recompose is the one mounting the create form (e.g. the screen's
    deferred open timer fires while the user sits in the search box), the
    form's focus-first-field-on-open behavior must still win -- the pane
    captures whether the form was already mounted pre-teardown and only
    lets the search branch keep the caret when the form is not opening.
    """
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        screen.active_section = "sources"
        await pilot.pause(0.3)
        pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
        pane.query_one("#sources-search-input", Input).focus()
        await pilot.pause(0.1)

        pane.show_create_form = True
        await pilot.pause(0.5)

        first_field = pane.query_one("#sources-create-name", Input)
        assert first_field.has_focus, (
            "opening the create form must focus its first field even when "
            "the sources search box was focused"
        )


@pytest.mark.asyncio
async def test_mark_all_read_then_undo_roundtrip():
    """`a` catches the scope up and `u` restores it — the two-key loop."""
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        db = app.watchlist_bundle_service._db
        source_id = db.add_subscription(
            name="ArXiv", type="rss", source="https://a.example/f"
        )
        for minute in range(3):
            _seed_item(
                db, source_id, f"item {minute}",
                created_at=f"2026-08-06 09:0{minute}:00",
            )
        await screen._replace_items_snapshot(reason="initial")
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        await _wait_for_items(pilot, pane)
        assert len(pane.displayed_items()) == 3, "precondition"

        app.notify = Mock()
        await pilot.press("a")
        for _ in range(60):
            await pilot.pause()
            if len(db.get_new_items(status="reviewed", limit=10)) == 3:
                break
        assert len(db.get_new_items(status="reviewed", limit=10)) == 3, (
            "`a` must mark every unread item in scope read"
        )
        assert db.get_new_items(status="new", limit=10) == []

        await pilot.press("u")
        for _ in range(60):
            await pilot.pause()
            if len(db.get_new_items(status="new", limit=10)) == 3:
                break
        assert len(db.get_new_items(status="new", limit=10)) == 3, (
            "`u` must restore the whole mark-all-read batch to unread"
        )


@pytest.mark.asyncio
async def test_undo_failure_keeps_the_batch_for_retry(monkeypatch):
    """Qodo review (PR #1383): a failing restore must not consume the only
    undo handle -- the batch survives so a second `u` retries it."""
    from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB

    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        db = app.watchlist_bundle_service._db
        source_id = db.add_subscription(
            name="ArXiv", type="rss", source="https://a.example/f"
        )
        _seed_item(db, source_id, "item 0", created_at="2026-08-06 09:00:00")
        await screen._replace_items_snapshot(reason="initial")
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        await _wait_for_items(pilot, pane)

        await pilot.press("a")
        for _ in range(60):
            await pilot.pause()
            if screen._last_mark_all_read_batch:
                break
        batch = list(screen._last_mark_all_read_batch)
        assert batch, "precondition: `a` stored an undo batch"

        app.notify = Mock()
        original_restore = SubscriptionsDB.restore_items_new
        state = {"calls": 0}

        def fail_once(self, item_ids):
            state["calls"] += 1
            if state["calls"] == 1:
                raise RuntimeError("transient DB failure")
            return original_restore(self, item_ids)

        monkeypatch.setattr(SubscriptionsDB, "restore_items_new", fail_once)

        await pilot.press("u")
        for _ in range(60):
            await pilot.pause()
            if state["calls"] >= 1:
                break
        assert state["calls"] == 1, "precondition: the failing restore ran"
        assert screen._last_mark_all_read_batch == batch, (
            "a failed restore must leave the undo batch intact for retry"
        )
        app.notify.assert_any_call("Undo failed — press u to retry.", severity="error")

        await pilot.press("u")
        for _ in range(60):
            await pilot.pause()
            if len(db.get_new_items(status="new", limit=10)) == 1:
                break
        assert len(db.get_new_items(status="new", limit=10)) == 1, (
            "the retry must restore the batch to unread"
        )
        assert screen._last_mark_all_read_batch == [], (
            "a successful restore consumes the batch"
        )


@pytest.mark.asyncio
async def test_mark_all_read_scoped_to_watchlist():
    """`a` catches up the rail's current scope only — sources outside the
    scoped watchlist keep their unread items."""
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        service = app.watchlist_bundle_service
        db = service._db
        watchlist = service.create("Morning AI Brief")
        member = db.add_subscription(
            name="ArXiv", type="rss", source="https://a.example/f"
        )
        outsider = db.add_subscription(
            name="Krebs", type="rss", source="https://b.example/f"
        )
        service.add_source(watchlist["id"], member)
        _seed_item(db, member, "member 1", created_at="2026-08-06 09:00:00")
        _seed_item(db, member, "member 2", created_at="2026-08-06 09:01:00")
        _seed_item(db, outsider, "outsider", created_at="2026-08-06 09:02:00")

        scope = TreeScope(kind="watchlist", watchlist_id=watchlist["id"])
        assert await screen._replace_items_snapshot(
            scope=scope,
            reason="scope",
            clear_reader_on_commit=True,
        )
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        for _ in range(60):
            await pilot.pause()
            if len(pane.displayed_items()) == 2:
                break
        assert len(pane.displayed_items()) == 2, (
            "precondition: the scope shows only the member's items (task-2513 Task 7)"
        )

        await pilot.press("a")
        for _ in range(60):
            await pilot.pause()
            if len(db.get_new_items(status="reviewed", limit=10)) == 2:
                break

        reviewed = db.get_new_items(status="reviewed", limit=10)
        assert {r["subscription_id"] for r in reviewed} == {member}
        remaining = db.get_new_items(status="new", limit=10)
        assert [r["subscription_id"] for r in remaining] == [outsider], (
            "the outsider's unread item must survive a scoped catch-up"
        )


@pytest.mark.asyncio
async def test_verbs_noop_off_read_tab():
    """m/a/u are Read-tab verbs: on any other tab they change nothing."""
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        db = app.watchlist_bundle_service._db
        source_id = db.add_subscription(
            name="ArXiv", type="rss", source="https://a.example/f"
        )
        _seed_item(db, source_id, "untouched")
        await screen._replace_items_snapshot(reason="initial")
        screen.active_section = "sources"
        await pilot.pause(0.3)

        for key in ("m", "a", "u"):
            await pilot.press(key)
        await pilot.pause(0.3)

        assert len(db.get_new_items(status="new", limit=10)) == 1, (
            "no read-state writes may happen off the Read tab"
        )


# --- Fix round 1, Finding 2: a pane-row click must not discard the tree scope


@pytest.mark.asyncio
async def test_selecting_a_pane_row_keeps_the_header_summary_on_the_tree_scope():
    """Finding 2's exact reproduction: click a watchlist in the tree, then
    click a row in the Sources table to inspect it.

    Before this fix `_select_entity` reset `selected_scope` to "all", and
    since Task 7 made that same reactive drive the scoped readout, it
    silently jumped from `Morning AI Brief (1)` back to `All sources (2)` --
    an interaction in one region discarding the user's navigation in
    another, with no selection highlight in the tree to fall back on.

    task-2513: the scoped readout is the centre header's summary line now
    that the FEEDS region is gone; the behaviour being pinned is unchanged.
    """
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        service = app.watchlist_bundle_service
        db = service._db

        # Read is the default section now, and the header (with the scoped
        # summary) is mounted on every tab.
        assert screen.active_section == "items"

        morning = service.create("Morning AI Brief")
        a = db.add_subscription(name="ArXiv", type="rss", source="https://a.example/f")
        db.add_subscription(name="Loose", type="rss", source="https://c.example/f")
        service.add_source(morning["id"], a)
        screen._tree_watchlists = [{"id": morning["id"], "name": "Morning AI Brief"}]

        screen.post_message(
            TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=morning["id"]))
        )
        summary = ""
        for _ in range(20):
            await pilot.pause()
            node = screen.query("#wc-watchlists-summary")
            if node:
                summary = _static_text(node[0])
            if "Morning AI Brief" in summary:
                break
        assert [row["id"] for row in screen.scoped_source_rows()] == [a]
        assert summary == "Local Watchlists snapshot: Morning AI Brief (1 source)"

        screen.post_message(SourceSelected({"id": "source-1", "name": "Some Feed", "url": "https://x"}))
        await pilot.pause()

        assert screen.tree_scope == TreeScope(
            kind="watchlist", watchlist_id=morning["id"]
        ), "inspecting a pane row is not navigation; the tree scope must survive it"
        assert [row["id"] for row in screen.scoped_source_rows()] == [a]
        assert (
            _static_text(screen.query_one("#wc-watchlists-summary", Static))
            == "Local Watchlists snapshot: Morning AI Brief (1 source)"
        )


@pytest.mark.asyncio
async def test_pane_row_selection_still_claims_no_inspector_ancestry():
    """The half of Task 5 fix round 2 that must NOT regress while Finding 2
    is fixed.

    A pane row carries no watchlist/source ancestry, so the Inspector must
    not put a breadcrumb above it. Clearing `_breadcrumb_labels` alone is
    *not* enough: `InspectorPane._scope_levels` derives an ancestor level
    from `scope` alone and falls back to a `Watchlist {id}` label when no
    label is supplied, so the crumb would still render (just anonymously).
    The Inspector's `scope` must itself be reset -- which is why the tree's
    own navigation state now lives in a separate `tree_scope` reactive
    rather than being read back off `selected_scope`.
    """
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        screen._tree_watchlists = [{"id": 1, "name": "Morning AI Brief"}]

        screen.post_message(
            TreeScopeChanged(TreeScope(kind="source", watchlist_id=1, source_id=10))
        )
        await pilot.pause()
        screen.post_message(ItemSelected({"item_id": "item-1", "title": "RAG Eval"}))
        await pilot.pause()

        inspector = screen.query_one("#watchlists-entity-inspector", InspectorPane)
        assert screen.selected_scope == TreeScope(kind="all")
        assert inspector.scope == TreeScope(kind="all")
        assert inspector.breadcrumb_labels == []
        assert not inspector.query("#inspector-breadcrumb-0")
        assert screen.tree_scope == TreeScope(
            kind="source", watchlist_id=1, source_id=10
        ), "...while the tree itself has not moved"


# --- Fix round 1, Finding 1: staging follows the tree scope ----------------


@pytest.mark.asyncio
async def test_staged_console_payload_follows_the_tree_scope():
    """The Console handoff must send the scope the user navigated to.

    Before this fix the payload was built from `_local_watchlist_records` --
    `WatchlistScopeService.list_watch_items` over every local source,
    regardless of the tree selection -- which is also why the (now-removed)
    Feeds region printed the same sources twice. Selecting "Morning AI
    Brief" and then pressing Stage must stage Morning AI Brief.
    """
    app = _build_test_app()
    app.open_chat_with_handoff = Mock()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        service = app.watchlist_bundle_service
        db = service._db

        morning = service.create("Morning AI Brief")
        arxiv = db.add_subscription(
            name="ArXiv", type="rss", source="https://a.example/f"
        )
        db.add_subscription(name="Krebs", type="rss", source="https://b.example/f")
        service.add_source(morning["id"], arxiv)
        await screen._load_tree_data().wait()

        screen.post_message(
            TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=morning["id"]))
        )
        await pilot.pause()

        body = screen._snapshot_body()
        metadata = screen._snapshot_metadata()

    assert "Morning AI Brief" in body
    assert "ArXiv" in body
    assert "Krebs" not in body, "a source outside the scope must not be staged"
    assert metadata["scope_kind"] == "watchlist"
    assert metadata["scope_label"] == "Morning AI Brief"
    assert metadata["scope_watchlist_id"] == morning["id"]
    assert metadata["source_count"] == 1
    assert metadata["source_titles"] == ["ArXiv"]


@pytest.mark.asyncio
async def test_the_all_scope_summary_is_a_single_line_in_the_header():
    """Finding 1's headline symptom was the unscoped staging block printing
    every source a second time in the same box, in identical typography. The
    per-source rows died with the FEEDS region (task-2513); what remains on
    screen is exactly one summary line naming the scope, carried by the
    centre header on every tab.
    """
    app = _build_test_app()
    db = app.watchlist_bundle_service._db
    db.add_subscription(name="ArXiv", type="rss", source="https://a.example/f")
    db.add_subscription(name="Krebs", type="rss", source="https://b.example/f")

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        # Seeded before mount deliberately: "all" is the default scope, so
        # re-posting it would not move the reactive and nothing would
        # rebuild. This is the resting state a user actually lands on.
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        assert screen.active_section == "items", "precondition: lands on Read"
        for _ in range(20):
            await pilot.pause()
            if list(screen.query("#wc-watchlists-summary")):
                break

        summaries = list(screen.query("#wc-watchlists-summary"))
        assert len(summaries) == 1, "the summary must not be mounted twice"
        assert (
            _static_text(summaries[0])
            == "Local Watchlists snapshot: All sources (2 sources)"
        )


@pytest.mark.asyncio
async def test_the_header_summary_escapes_an_untrusted_source_name():
    """The `source` scope takes its summary label from `rows[0]["name"]` --
    a remote feed's own title. Unescaped markup reaching a rendered label
    has broken this exact screen before; the name must render as literal
    text, not parsed markup.
    """
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        # task-2513: the scoped readout lives in the centre header's summary
        # line, mounted on every tab; Read is the default now.
        assert screen.active_section == "items"
        service = app.watchlist_bundle_service
        watchlist = service.create("Morning AI Brief")
        source_id = service._db.add_subscription(
            name="[bold red]Not Actually Bold[/bold red]",
            type="rss",
            source="https://a.example/f",
        )
        service.add_source(watchlist["id"], source_id)

        screen.post_message(
            TreeScopeChanged(
                TreeScope(
                    kind="source", watchlist_id=watchlist["id"], source_id=source_id
                )
            )
        )
        summary = ""
        for _ in range(20):
            await pilot.pause()
            node = screen.query("#wc-watchlists-summary")
            if node:
                summary = _static_text(node[0])
            if "Not Actually Bold" in summary:
                break

        assert "[bold red]Not Actually Bold[/bold red]" in summary


@pytest.mark.asyncio
async def test_centre_header_summary_follows_the_tree_scope_off_the_read_tab():
    """task-1344 fix wave (Qodo correctness), kept current by task-2513:
    `#wl-centre-status` (`_build_centre_status_header`) carries the scoped
    summary on EVERY tab now -- the FEEDS region that used to have its own
    inline copy on Read is gone. `watch_tree_scope` must still rebuild that
    header in place on a scope move, or it keeps showing the PREVIOUS
    scope's summary until some unrelated recompose comes along.
    """
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        service = app.watchlist_bundle_service
        db = service._db

        morning = service.create("Morning AI Brief")
        arxiv = db.add_subscription(
            name="ArXiv", type="rss", source="https://a.example/f"
        )
        db.add_subscription(name="Krebs", type="rss", source="https://b.example/f")
        service.add_source(morning["id"], arxiv)
        await screen._load_tree_data().wait()

        screen.active_section = "sources"
        await pilot.pause(0.2)
        assert screen.query("#wl-centre-status"), (
            "precondition: the centre header is mounted off the Read tab"
        )

        summary_before = _static_text(
            screen.query_one("#wc-watchlists-summary", Static)
        )
        assert summary_before == "Local Watchlists snapshot: All sources (2 sources)"

        screen.post_message(
            TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=morning["id"]))
        )
        summary_after = summary_before
        for _ in range(20):
            await pilot.pause()
            summary_after = _static_text(
                screen.query_one("#wc-watchlists-summary", Static)
            )
            if "Morning AI Brief" in summary_after:
                break

        assert summary_after == "Local Watchlists snapshot: Morning AI Brief (1 source)", (
            f"the header must reflect the NEW scope's counts, not the old: "
            f"{summary_after!r}"
        )


@pytest.mark.asyncio
async def test_centre_header_summary_follows_the_tree_scope_on_the_read_tab_too():
    """TASK-2312: before this task, `watch_tree_scope` refreshed the centre
    header ONLY off the Read tab (`if self.active_section != "items"`) --
    the header simply did not exist on Read, which carried its own inline
    copy of the summary INSIDE FEEDS's own body instead (refreshed by
    `_refresh_feeds_region_for_scope`, unconditionally). Now that the
    header exists on every section, its refresh must be unconditional too,
    or a scope change on the Read tab would leave `#wl-centre-status`
    showing the PREVIOUS scope's summary -- silently, since nothing else
    on that tab would ever touch it.

    Kept current by task-2513: the FEEDS region itself is gone (the
    sibling test's docstring covers that removal); what this test still
    protects is the unconditional on-Read header refresh."""
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        service = app.watchlist_bundle_service
        db = service._db

        morning = service.create("Morning AI Brief")
        arxiv = db.add_subscription(
            name="ArXiv", type="rss", source="https://a.example/f"
        )
        db.add_subscription(name="Krebs", type="rss", source="https://b.example/f")
        service.add_source(morning["id"], arxiv)
        await screen._load_tree_data().wait()

        # task-2513 made Read ("items") the DEFAULT section, so the section
        # write the sibling tests use to force a rebuild would be a no-op
        # here -- the watcher only fires on a real change. Recompose
        # directly instead: exactly what the watcher's
        # `refresh(recompose=True)` does for the header.
        screen.refresh(recompose=True)
        await pilot.pause(0.2)
        assert screen.query_one("#wl-centre-status"), (
            "precondition: the header exists on the Read tab too "
            "(TASK-2312)"
        )

        summary_before = _static_text(
            screen.query_one("#wc-watchlists-summary", Static)
        )
        assert summary_before == "Local Watchlists snapshot: All sources (2 sources)"

        screen.post_message(
            TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=morning["id"]))
        )
        summary_after = summary_before
        for _ in range(20):
            await pilot.pause()
            summary_after = _static_text(
                screen.query_one("#wc-watchlists-summary", Static)
            )
            if "Morning AI Brief" in summary_after:
                break

        assert summary_after == "Local Watchlists snapshot: Morning AI Brief (1 source)", (
            f"the header must reflect the NEW scope's counts on the Read "
            f"tab too, not just every other section: {summary_after!r}"
        )


# --- task-876: the tree's own selection highlight --------------------------
#
# `WatchlistTree` never read `tree_scope`, so nothing in the rail showed
# which node the centre was scoped to. `_apply_tree_scope` is the single
# reconciliation point for BOTH a real tree click (`_on_tree_scope_changed`)
# and a breadcrumb promotion (`handle_breadcrumb_scope_selected`); these
# confirm the highlight follows either path, and survives the two rebuild
# paths (section switch, rail toggle) Phase C already had to fix once for
# `expanded`.


@pytest.mark.asyncio
async def test_breadcrumb_promotion_moves_the_tree_highlight_same_as_a_click():
    # Seeded *before* the screen mounts, like
    # `test_the_header_summary_names_the_scope_with_a_live_count` above: the
    # mounted `WatchlistTree` captures its own `_watchlists` once, from
    # whatever `_load_tree_data` populated `_tree_watchlists` with by the
    # time IT (not this test) last rebuilt the tree -- setting
    # `screen._tree_watchlists` after mount would not reach the
    # already-constructed tree instance's own copy.
    app = _build_test_app()
    morning = app.watchlist_bundle_service.create("Morning AI Brief")
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        for _ in range(20):
            await pilot.pause()
            if host.screen_stack[-1].query(f"#wl-tree-node-watchlist-{morning['id']}"):
                break
        screen = host.screen_stack[-1]

        screen.post_message(
            TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=morning["id"]))
        )
        await pilot.pause()
        assert screen.query_one(
            f"#wl-tree-node-watchlist-{morning['id']}", Button
        ).has_class("is-active")

        # Promote a breadcrumb back to "all" -- a path that never touches
        # the tree widget directly (see `handle_breadcrumb_scope_selected`)
        # -- and confirm the SAME tree instance updates exactly as a real
        # click would.
        screen.post_message(BreadcrumbScopeSelected(TreeScope(kind="all")))
        await pilot.pause()
        assert screen.query_one("#wl-tree-node-all", Button).has_class("is-active")
        assert not screen.query_one(
            f"#wl-tree-node-watchlist-{morning['id']}", Button
        ).has_class("is-active")


@pytest.mark.asyncio
async def test_tree_highlight_survives_a_section_switch_and_a_rail_toggle():
    """Both a section switch (`watch_active_section`) and a rail toggle
    (`action_toggle_left_rail`) rebuild the whole workbench, constructing a
    brand new `WatchlistTree` -- the same class of bug Phase C already fixed
    once for `expanded`/`active_tag`. Do not assume the fix generalizes;
    test it (task-876, AC #3).
    """
    app = _build_test_app()
    morning = app.watchlist_bundle_service.create("Morning AI Brief")
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        for _ in range(20):
            await pilot.pause()
            if host.screen_stack[-1].query(f"#wl-tree-node-watchlist-{morning['id']}"):
                break
        screen = host.screen_stack[-1]

        screen.post_message(
            TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=morning["id"]))
        )
        await pilot.pause()
        assert screen.query_one(
            f"#wl-tree-node-watchlist-{morning['id']}", Button
        ).has_class("is-active")

        screen.active_section = "sources"
        await pilot.pause()
        assert screen.query_one(
            f"#wl-tree-node-watchlist-{morning['id']}", Button
        ).has_class("is-active"), "the highlight must survive a section switch"

        screen.action_toggle_left_rail()
        await pilot.pause()
        assert not screen.query("#wl-tree"), "the rail should now be collapsed"

        screen.action_toggle_left_rail()
        await pilot.pause()
        assert screen.query_one(
            f"#wl-tree-node-watchlist-{morning['id']}", Button
        ).has_class("is-active"), "the highlight must survive a rail toggle"


@pytest.mark.asyncio
async def test_load_tree_data_failure_notifies_the_user():
    """A real database failure in `_load_tree_data` must not render
    identically to "you have zero watchlists" -- two empty tree roots and no
    message (task-876). Mirrors every sibling loader's own error-notify
    behaviour (`_load_sources`/`_load_runs`/`_load_notifications`, etc.).
    """
    app = _build_test_app()
    app.watchlist_bundle_service.list_watchlists = Mock(side_effect=RuntimeError("boom"))
    app.notify = Mock()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        for _ in range(20):
            await pilot.pause()
            if app.notify.called:
                break

        assert app.notify.called, "a tree-load failure must notify the user"
        _args, kwargs = app.notify.call_args
        assert kwargs.get("severity") == "error"
        screen = host.screen_stack[-1]
        assert screen.query_one("#wl-tree-node-all", Button)
        assert screen.query_one("#wl-tree-node-unassigned", Button)


@pytest.mark.asyncio
async def test_tree_snapshot_owns_complete_aggregate_rows_not_management_cache(
    monkeypatch,
) -> None:
    app = _build_test_app()
    service = app.watchlist_bundle_service
    assigned_id = service._db.add_subscription(
        name="Assigned feed", type="rss", source="https://assigned.example/feed"
    )
    unassigned_id = service._db.add_subscription(
        name="Unassigned feed",
        type="rss",
        source="https://unassigned.example/feed",
    )
    watchlist = service.create("Snapshot watchlist")
    service.add_source(watchlist["id"], assigned_id)
    spies = {}
    for name in (
        "list_watchlists",
        "list_all_source_rows",
        "list_unassigned_source_rows",
        "get_watchlist_item_counts",
        "get_source_item_counts",
    ):
        spy = Mock(wraps=getattr(service, name))
        monkeypatch.setattr(service, name, spy)
        spies[name] = spy

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await host.workers.wait_for_complete()
        screen = host.screen_stack[-1]
        screen._loaded_sources = [{"id": 999, "name": "Capped management row"}]
        for spy in spies.values():
            spy.reset_mock()

        await screen._load_tree_data().wait()
        await pilot.pause()

        assert {row["id"] for row in screen._tree_all_source_rows} == {
            assigned_id,
            unassigned_id,
        }
        assert [row["id"] for row in screen._tree_unassigned_source_rows] == [
            unassigned_id
        ]
        assert screen._loaded_sources == [
            {"id": 999, "name": "Capped management row"}
        ]
        for name, spy in spies.items():
            assert spy.call_count == 1, name


@pytest.mark.asyncio
async def test_root_and_watchlist_expansion_persist_independently_across_rebuilds() -> None:
    app = _build_test_app()
    watchlist = app.watchlist_bundle_service.create("Persistent branch")
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await host.workers.wait_for_complete()
        await pilot.pause()
        screen = host.screen_stack[-1]
        tree = screen.query_one("#wl-tree", WatchlistTree)
        tree.expanded_root_kinds = frozenset({"all", "unassigned"})
        tree.expanded = frozenset({watchlist["id"]})
        await pilot.pause()

        screen.active_section = "sources"
        await pilot.pause()
        screen.active_section = "items"
        await pilot.pause()
        tree = screen.query_one("#wl-tree", WatchlistTree)

        assert tree.expanded_root_kinds == frozenset({"all", "unassigned"})
        assert tree.expanded == frozenset({watchlist["id"]})
        assert screen._tree_expanded_root_kinds == frozenset(
            {"all", "unassigned"}
        )
        assert screen._tree_expanded_watchlist_ids == frozenset({watchlist["id"]})


@pytest.mark.asyncio
async def test_tree_snapshot_acquisition_runs_off_the_textual_event_loop(
    monkeypatch,
) -> None:
    app = _build_test_app()
    service = app.watchlist_bundle_service
    caller_thread = threading.get_ident()
    acquisition_threads: list[int] = []
    original = service.list_all_source_rows

    def record_thread() -> list[dict]:
        acquisition_threads.append(threading.get_ident())
        return original()

    monkeypatch.setattr(service, "list_all_source_rows", record_thread)
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)):
        await host.workers.wait_for_complete()

    assert acquisition_threads
    assert all(thread_id != caller_thread for thread_id in acquisition_threads)


@pytest.mark.asyncio
async def test_slow_tree_refresh_cannot_overwrite_a_newer_snapshot(monkeypatch) -> None:
    app = _build_test_app()
    service = app.watchlist_bundle_service
    slow_started = threading.Event()
    release_slow = threading.Event()
    call_lock = threading.Lock()
    call_count = 0

    def staged_all_sources() -> list[dict]:
        nonlocal call_count
        with call_lock:
            call_count += 1
            this_call = call_count
        if this_call == 1:
            slow_started.set()
            release_slow.wait(5)
            return [{"id": 1, "name": "Stale", "type": "rss"}]
        return [{"id": 2, "name": "Fresh", "type": "rss"}]

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await host.workers.wait_for_complete()
        screen = host.screen_stack[-1]
        monkeypatch.setattr(service, "list_all_source_rows", staged_all_sources)
        first = screen._load_tree_data()
        assert await _wait_until(pilot, slow_started.is_set)
        second = screen._load_tree_data()
        await second.wait()
        assert [row["name"] for row in screen._tree_all_source_rows] == ["Fresh"]

        release_slow.set()
        await first.wait()
        assert [row["name"] for row in screen._tree_all_source_rows] == ["Fresh"]


@pytest.mark.asyncio
async def test_tree_branch_failure_retains_last_snapshot_and_notifies_once_per_episode(
    monkeypatch,
) -> None:
    app = _build_test_app()
    service = app.watchlist_bundle_service
    service._db.add_subscription(
        name="Last known feed", type="rss", source="https://known.example/feed"
    )
    app.notify = Mock()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)):
        await host.workers.wait_for_complete()
        screen = host.screen_stack[-1]
        expected_all = list(screen._tree_all_source_rows)
        expected_unassigned = list(screen._tree_unassigned_source_rows)
        app.notify.reset_mock()

        failure = Mock(side_effect=RuntimeError("all-source branch failed"))
        monkeypatch.setattr(service, "list_all_source_rows", failure)
        await screen._load_tree_data().wait()
        await screen._load_tree_data().wait()

        assert screen._tree_all_source_rows == expected_all
        assert screen._tree_unassigned_source_rows == expected_unassigned
        assert screen._tree_snapshot_failures == frozenset({"all_sources"})
        app.notify.assert_called_once()

        monkeypatch.setattr(
            service, "list_all_source_rows", Mock(return_value=expected_all)
        )
        await screen._load_tree_data().wait()
        assert screen._tree_snapshot_failures == frozenset()

        monkeypatch.setattr(service, "list_all_source_rows", failure)
        await screen._load_tree_data().wait()
        assert app.notify.call_count == 2


# --- TASK-895: the tree's write verbs, end to end -------------------------
#
# Five `WatchlistBundleService` methods had no production caller: Phase C
# shipped the tree's read half, so a user could browse watchlists but never
# make one. These drive the real buttons, the real dialogs and the real
# service against `_build_test_app()`'s isolated temp-dir SQLite file (see
# that fixture's `get_subscriptions_db_path` patch -- never the user's own
# database), so they measure the wiring rather than a mock of it.


async def _wait_for_dialog(host, dialog_type, pilot, *, ticks: int = 60):
    """Return the modal `dialog_type` once the flow's worker has pushed it.

    The write flows `await push_screen_wait(...)`, so the dialog appears a
    few ticks after the button press rather than synchronously.
    """
    for _ in range(ticks):
        await pilot.pause()
        if isinstance(host.screen, dialog_type):
            return host.screen
    raise AssertionError(f"{dialog_type.__name__} never opened")


async def _wait_until(pilot, predicate, *, ticks: int = 80) -> bool:
    for _ in range(ticks):
        await pilot.pause()
        if predicate():
            return True
    return False


def _label_plain(widget) -> str:
    """The text a markup-rendering label actually paints.

    `Static`/`Label` parse markup, so a name carrying Rich syntax is only
    safe if it was escaped on the way in. Re-parsing the stored content
    here is what proves that: an unescaped `[bold]` disappears into a
    style, an escaped one survives as literal text.
    """
    renderable = widget.renderable
    raw = getattr(renderable, "plain", None)
    if raw is None:
        raw = str(renderable)
        return Text.from_markup(raw).plain
    return raw


@pytest.mark.asyncio
async def test_creating_a_watchlist_from_the_tree_shows_it_without_a_refresh():
    """AC #1. The rail must show the new watchlist on its own -- the whole
    point of wiring `create` is that the only watchlists that can exist stop
    being ones seeded outside the app.
    """
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        service = app.watchlist_bundle_service
        assert service.list_watchlists() == []

        screen.query_one("#wl-tree-new", Button).press()
        dialog = await _wait_for_dialog(host, WatchlistNameDialog, pilot)
        dialog.query_one("#watchlist-name-input", Input).value = "Morning AI Brief"
        dialog.query_one("#watchlist-name-submit", Button).press()

        assert await _wait_until(pilot, lambda: bool(service.list_watchlists()))
        rows = service.list_watchlists()
        assert [row["name"] for row in rows] == ["Morning AI Brief"]

        watchlist_id = rows[0]["id"]
        assert await _wait_until(
            pilot, lambda: bool(screen.query(f"#wl-tree-node-watchlist-{watchlist_id}"))
        ), "the new watchlist must appear in the rail with no manual refresh"


@pytest.mark.asyncio
async def test_an_empty_name_is_rejected_with_a_visible_reason():
    """AC #7. Not a silent no-op: the dialog stays open and says why."""
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]

        screen.query_one("#wl-tree-new", Button).press()
        dialog = await _wait_for_dialog(host, WatchlistNameDialog, pilot)
        dialog.query_one("#watchlist-name-input", Input).value = "   "
        dialog.query_one("#watchlist-name-submit", Button).press()
        await pilot.pause()
        await pilot.pause()

        assert isinstance(host.screen, WatchlistNameDialog), (
            "an invalid name must not dismiss the prompt"
        )
        error = _label_plain(dialog.query_one("#watchlist-name-error", Static))
        assert "cannot be empty" in error
        assert app.watchlist_bundle_service.list_watchlists() == []


@pytest.mark.asyncio
async def test_a_duplicate_name_is_rejected_and_the_reason_escapes_the_name():
    """AC #7, both halves at once.

    The duplicate is reported rather than silently suffixed to "X (2)" by
    `_unique_name` -- and because the reported name is user-authored free
    text, the reason must render it as literal characters. Unescaped remote
    and user titles have shipped as bugs on this screen before.
    """
    app = _build_test_app()
    app.watchlist_bundle_service.create("[bold red]Alpha[/bold red]")

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        assert await _wait_until(pilot, lambda: bool(screen._tree_watchlists))

        screen.query_one("#wl-tree-new", Button).press()
        dialog = await _wait_for_dialog(host, WatchlistNameDialog, pilot)
        dialog.query_one("#watchlist-name-input", Input).value = (
            "[bold red]alpha[/bold red]"
        )
        dialog.query_one("#watchlist-name-submit", Button).press()
        await pilot.pause()
        await pilot.pause()

        assert isinstance(host.screen, WatchlistNameDialog)
        error = _label_plain(dialog.query_one("#watchlist-name-error", Static))
        assert "already exists" in error
        assert "[bold red]alpha[/bold red]" in error, (
            "the rejected name must render as literal text, not as markup"
        )
        assert len(app.watchlist_bundle_service.list_watchlists()) == 1


@pytest.mark.asyncio
async def test_renaming_a_watchlist_updates_the_rail():
    """AC #2, rename half."""
    app = _build_test_app()
    service = app.watchlist_bundle_service
    watchlist = service.create("Mroning AI Brief")
    # One member source, so the header's scoped summary (the FEEDS heading's
    # successor, task-2513) renders at all -- it only exists once there is
    # anything to stage.
    source_id = service._db.add_subscription(
        name="ArXiv", type="rss", source="https://a.example/f"
    )
    service.add_source(watchlist["id"], source_id)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        # task-2513: the scope readout (asserted below) is the centre
        # header's summary line, mounted on every tab -- Read, the default,
        # included.
        assert screen.active_section == "items"
        assert await _wait_until(pilot, lambda: bool(screen._tree_watchlists))

        screen.post_message(
            TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=watchlist["id"]))
        )
        await pilot.pause()
        await pilot.pause()

        rename_button = screen.query_one("#wl-tree-rename", Button)
        assert not rename_button.disabled
        rename_button.press()

        dialog = await _wait_for_dialog(host, WatchlistNameDialog, pilot)
        assert dialog.query_one("#watchlist-name-input", Input).value == (
            "Mroning AI Brief"
        ), "the prompt should start from the current name"
        dialog.query_one("#watchlist-name-input", Input).value = "Morning AI Brief"
        dialog.query_one("#watchlist-name-submit", Button).press()

        service = app.watchlist_bundle_service
        assert await _wait_until(
            pilot,
            lambda: [row["name"] for row in service.list_watchlists()]
            == ["Morning AI Brief"],
        )
        assert await _wait_until(
            pilot,
            lambda: any(
                "Morning AI Brief" in str(button.label)
                for button in screen.query(Button)
                if (button.id or "").startswith("wl-tree-node-watchlist-")
            ),
        )
        # The rename must also reach the scope-derived copy, not just the
        # rail: `_tree_scope_label` and `_resolve_breadcrumb_labels` both
        # read `_tree_watchlists`, which a rename leaves stale until the
        # reload re-resolves it. The summary is rebuilt in place by the
        # header refresh, so poll for it.
        assert await _wait_until(
            pilot,
            lambda: bool(screen.query("#wc-watchlists-summary"))
            and _static_text(screen.query_one("#wc-watchlists-summary", Static))
            == "Local Watchlists snapshot: Morning AI Brief (1 source)",
        )
        assert screen._breadcrumb_labels == ["Morning AI Brief"]
        # ...and the MOUNTED Inspector, not only the screen's mirror of it
        # (TASK-2200). `watch_selected_scope` pushed the pre-rename label when
        # the scope moved and never fires again; nothing else rebuilds the
        # Inspector now that the tree reload patches in place instead of
        # recomposing the screen, so the reload has to push it itself.
        inspector = screen.query_one("#watchlists-entity-inspector", InspectorPane)
        assert inspector.breadcrumb_labels == ["Morning AI Brief"], (
            "the renamed watchlist must reach the Inspector's breadcrumb too"
        )


@pytest.mark.asyncio
async def test_deleting_a_watchlist_says_what_happens_to_its_sources_first():
    """AC #2, delete half: the count and the destination are stated before
    the user commits, and the name is escaped on the way into the message.
    """
    app = _build_test_app()
    service = app.watchlist_bundle_service
    watchlist = service.create("[bold]Danger[/bold]")
    db = service._db
    for index in range(2):
        service.add_source(
            watchlist["id"],
            db.add_subscription(
                name=f"Feed {index}", type="rss", source=f"https://{index}.example/f"
            ),
        )

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        assert await _wait_until(pilot, lambda: bool(screen._tree_watchlists))

        screen.post_message(
            TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=watchlist["id"]))
        )
        await pilot.pause()
        await pilot.pause()
        screen.query_one("#wl-tree-delete", Button).press()

        dialog = await _wait_for_dialog(host, ConfirmationDialog, pilot)
        message = Text.from_markup(dialog.message).plain
        assert "[bold]Danger[/bold]" in message, (
            "the watchlist name must reach the prompt as literal text"
        )
        assert "2 sources are not deleted" in message
        assert "Unassigned" in message

        dialog.query_one("#cancel-button", Button).press()
        assert await _wait_until(
            pilot, lambda: not isinstance(host.screen, ConfirmationDialog)
        )
        assert len(service.list_watchlists()) == 1, "Cancel must not delete anything"


@pytest.mark.asyncio
async def test_deleting_a_watchlist_never_orphans_its_sources_into_invisibility():
    """AC #3. Deleting cascades only the membership rows, so the sources
    survive -- but survival is worthless if nothing in the tree can reach
    them. They must land under the permanent Unassigned root, which is what
    that root exists for.
    """
    app = _build_test_app()
    service = app.watchlist_bundle_service
    watchlist = service.create("Morning AI Brief")
    db = service._db
    source_ids = [
        db.add_subscription(name="ArXiv", type="rss", source="https://a.example/f"),
        db.add_subscription(name="Krebs", type="rss", source="https://b.example/f"),
    ]
    for source_id in source_ids:
        service.add_source(watchlist["id"], source_id)
    assert service.list_unassigned_source_rows() == []

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        assert await _wait_until(pilot, lambda: bool(screen._tree_watchlists))

        screen.post_message(
            TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=watchlist["id"]))
        )
        await pilot.pause()
        await pilot.pause()
        screen.query_one("#wl-tree-delete", Button).press()

        dialog = await _wait_for_dialog(host, ConfirmationDialog, pilot)
        dialog.query_one("#confirm-button", Button).press()

        assert await _wait_until(pilot, lambda: service.list_watchlists() == [])
        assert {row["id"] for row in service.list_unassigned_source_rows()} == set(
            source_ids
        )
        # And the screen actually shows them: the scope moves to Unassigned,
        # whose rows are exactly the sources the deleted watchlist held.
        assert await _wait_until(
            pilot, lambda: screen.tree_scope == TreeScope(kind="unassigned")
        )
        assert {row["id"] for row in screen.scoped_source_rows()} == set(source_ids)
        assert screen.query("#wl-tree-node-unassigned")
        assert not screen.query(f"#wl-tree-node-watchlist-{watchlist['id']}")


@pytest.mark.asyncio
async def test_adding_a_source_to_a_watchlist_from_the_tree():
    """AC #4, add half. The picker offers only sources that are not already
    members, so adding one twice is not something the UI can even ask for.
    """
    app = _build_test_app()
    service = app.watchlist_bundle_service
    watchlist = service.create("Morning AI Brief")
    db = service._db
    member = db.add_subscription(
        name="Already In", type="rss", source="https://in.example/f"
    )
    candidate = db.add_subscription(
        name="ArXiv", type="rss", source="https://a.example/f"
    )
    service.add_source(watchlist["id"], member)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        assert await _wait_until(pilot, lambda: bool(screen._tree_watchlists))

        screen.post_message(
            TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=watchlist["id"]))
        )
        await pilot.pause()
        await pilot.pause()
        screen.query_one("#wl-tree-add-source", Button).press()

        dialog = await _wait_for_dialog(host, WatchlistSourcePickerDialog, pilot)
        assert dialog.query(f"#wl-add-source-option-{candidate}")
        assert not dialog.query(f"#wl-add-source-option-{member}"), (
            "an existing member must not be offered again"
        )
        dialog.query_one(f"#wl-add-source-option-{candidate}", Button).press()

        assert await _wait_until(
            pilot, lambda: set(service.list_sources(watchlist["id"])) == {member, candidate}
        )
        assert service.list_unassigned_source_rows() == []


@pytest.mark.asyncio
async def test_removing_a_source_from_a_watchlist_keeps_the_source():
    """AC #4, remove half -- and the other side of AC #3: a removed source
    is still reachable, it just moves to Unassigned.
    """
    app = _build_test_app()
    service = app.watchlist_bundle_service
    watchlist = service.create("Morning AI Brief")
    db = service._db
    source_id = db.add_subscription(
        name="ArXiv", type="rss", source="https://a.example/f"
    )
    service.add_source(watchlist["id"], source_id)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        assert await _wait_until(pilot, lambda: bool(screen._tree_watchlists))

        screen.post_message(
            TreeScopeChanged(
                TreeScope(
                    kind="source",
                    parent_context="watchlist",
                    watchlist_id=watchlist["id"],
                    source_id=source_id,
                )
            )
        )
        await pilot.pause()
        await pilot.pause()
        remove_button = screen.query_one("#wl-tree-remove-source", Button)
        assert not remove_button.disabled
        remove_button.press()

        assert await _wait_until(
            pilot, lambda: service.list_sources(watchlist["id"]) == []
        )
        assert [row["id"] for row in service.list_unassigned_source_rows()] == [
            source_id
        ]
        # The scope fell back to the parent watchlist rather than sitting on
        # a source node that no longer exists.
        assert await _wait_until(
            pilot,
            lambda: screen.tree_scope
            == TreeScope(kind="watchlist", watchlist_id=watchlist["id"]),
        )


@pytest.mark.asyncio
async def test_the_server_backend_disables_all_five_verbs_with_a_stated_reason():
    """AC #5. Not cosmetic hiding: `SourceUpdateRequest` carries no
    `group_ids`, neither group request carries members, and all of them are
    `extra="forbid"` -- so there is no wire path at all, and the screen says
    so rather than offering an action that cannot be sent.
    """
    app = _build_test_app()
    app.watchlist_bundle_service.create("Morning AI Brief")

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        assert await _wait_until(pilot, lambda: bool(screen._tree_watchlists))
        watchlist_id = screen._tree_watchlists[0]["id"]
        screen.post_message(
            TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=watchlist_id))
        )
        await pilot.pause()
        await pilot.pause()
        # Locally, four of the five are live on a watchlist scope.
        assert not screen.query_one("#wl-tree-rename", Button).disabled

        screen.runtime_backend = "server"
        await pilot.pause()
        await pilot.pause()

        for action_id in (
            "#wl-tree-new",
            "#wl-tree-rename",
            "#wl-tree-delete",
            "#wl-tree-add-source",
            "#wl-tree-remove-source",
        ):
            button = screen.query_one(action_id, Button)
            assert button.disabled, f"{action_id} must be disabled on the server backend"
            assert "no wire path" in str(button.tooltip)

        note = screen.query_one("#wl-tree-actions-unavailable", Static)
        assert "Switch the backend to Local" in _label_plain(note)


@pytest.mark.asyncio
async def test_the_verbs_are_disabled_when_the_bundle_service_is_missing():
    """The same degrade-don't-crash contract every other caller of
    `_watchlist_bundle_service()` follows -- and the same disabled-with-a-
    reason treatment, rather than buttons that look live over a runtime that
    cannot service them.
    """
    app = _build_test_app()
    app.watchlist_bundle_service = None

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        for action_id in ("#wl-tree-new", "#wl-tree-rename", "#wl-tree-remove-source"):
            button = screen.query_one(action_id, Button)
            assert button.disabled
            assert "unavailable" in str(button.tooltip)


def test_every_watchlist_bundle_service_method_has_a_production_caller():
    """AC #6, enforced rather than asserted once by hand.

    Five of these methods were complete, tested, and reachable from nothing
    at all before this task. A future slice that quietly drops the last
    caller of one should fail here rather than be rediscovered as dead code
    with a green suite.

    Resolved through the AST rather than by grepping for `.create(`: a plain
    text scan matches `completions.create(` in `OCR_Backends` and
    `os.rename(` in `Chat_Functions`, so it would report a caller for
    `create` and `rename` even with every real call deleted -- verified by
    mutation. This instead follows the two ways the service is actually
    reached (`self._watchlist_bundle_service()` and the
    `watchlist_bundle_service` attribute on the app) plus any local bound to
    one of them, so `self._controller.list_sources(...)` -- a different
    object with a colliding method name, in the same file -- is not counted.
    """
    import ast
    import inspect
    import warnings
    from pathlib import Path

    from tldw_chatbook.Subscriptions.watchlist_bundle_service import (
        WatchlistBundleService,
    )

    class _BundleServiceCalls(ast.NodeVisitor):
        def __init__(self) -> None:
            self.aliases: set[str] = set()
            self.called: set[str] = set()

        def _is_service(self, node: ast.AST) -> bool:
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_watchlist_bundle_service"
            ):
                return True
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "WatchlistBundleService"
            ):
                return True
            if isinstance(node, ast.Attribute) and node.attr == "watchlist_bundle_service":
                return True
            return isinstance(node, ast.Name) and node.id in self.aliases

        def visit_Assign(self, node: ast.Assign) -> None:
            if self._is_service(node.value):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.aliases.add(target.id)
            self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            for argument in (*node.args.posonlyargs, *node.args.args):
                annotation = argument.annotation
                if annotation is not None and any(
                    isinstance(part, ast.Name)
                    and part.id == "WatchlistBundleService"
                    for part in ast.walk(annotation)
                ):
                    self.aliases.add(argument.arg)
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Attribute) and self._is_service(node.func.value):
                self.called.add(node.func.attr)
            self.generic_visit(node)

    service_file = Path(inspect.getfile(WatchlistBundleService)).resolve()
    package_root = service_file.parents[1]

    called: set[str] = set()
    # `ast.parse` re-emits each file's own SyntaxWarnings (stray escape
    # sequences in unrelated modules); they are pre-existing and not this
    # test's subject, so they are silenced rather than left to bury the
    # assertion message below in noise.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        for path in package_root.rglob("*.py"):
            if path.resolve() == service_file:
                continue
            visitor = _BundleServiceCalls()
            visitor.visit(ast.parse(path.read_text(encoding="utf-8")))
            called |= visitor.called

    public_methods = {
        name
        for name, member in vars(WatchlistBundleService).items()
        if not name.startswith("_")
        and callable(getattr(member, "__func__", member))
    }
    # Guard the guard: if the reflection above ever stops seeing the class's
    # own methods, the emptiness check below would pass vacuously.
    assert {"create", "rename", "delete", "add_source", "remove_source"} <= (
        public_methods
    )

    uncalled = sorted(public_methods - called)
    assert uncalled == [], (
        f"WatchlistBundleService methods with no production caller: {uncalled}"
    )


@pytest.mark.asyncio
async def test_a_failed_tree_write_start_does_not_wedge_later_writes():
    """Qodo #3 on PR #989: `_tree_write_active` could stick True forever.

    The flag was raised before `flow_factory()` ran, and is lowered only by
    `_run_tree_write`'s `finally`. If building the flow raised synchronously
    that worker never started, so the flag stayed up and every later
    create/rename/delete returned at the guard -- silently, for the life of
    the screen.
    """
    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test() as pilot:
        await pilot.pause()
        screen = host.screen_stack[-1]

        def exploding_factory():
            raise RuntimeError("flow could not be built")

        screen._start_tree_write(exploding_factory)
        await pilot.pause()

        assert screen._tree_write_active is False, (
            "a write that never started must leave the guard down, or every "
            "later watchlist action is silently swallowed"
        )

        ran = []

        async def working_flow():
            ran.append(True)

        screen._start_tree_write(working_flow)
        for _ in range(20):
            await pilot.pause()
            if ran:
                break
        assert ran, "the next write must still be able to start"


# --- TASK-3791 plan task 3: `/` and the corpus-wide search --------------------


@pytest.mark.asyncio
async def test_slash_focuses_the_items_search_box():
    """`/` on the Read tab puts the caret in the search input."""
    from textual.widgets import Input

    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        assert screen.active_section == "items", "precondition: lands on Read"

        await pilot.press("/")
        await pilot.pause()
        focused = screen.focused
        assert isinstance(focused, Input) and focused.id == "items-search-input"


@pytest.mark.asyncio
async def test_slash_types_literally_once_the_search_box_has_focus():
    """The verb guard: once an Input has focus, `/` is text, not a verb."""
    from textual.widgets import Input

    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        await pilot.press("/")
        await pilot.pause()
        search = screen.query_one("#items-search-input", Input)

        await pilot.press("/")
        await pilot.pause()
        assert search.value == "/", "a second `/` must type into the box"


@pytest.mark.asyncio
async def test_a_search_reaches_beyond_the_first_page():
    """The corpus-wide path: an item past the newest-50 page still surfaces."""
    from textual.widgets import Input

    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        db = app.watchlist_bundle_service._db
        source_id = db.add_subscription(
            name="ArXiv", type="rss", source="https://a.example/f"
        )
        # 105 items, newest-first by created_at; the unique-token item is the
        # OLDEST, so the default page (limit 50) cannot contain it.
        for index in range(105):
            day = 1 + index // 24
            hour = index % 24
            _seed_item(
                db, source_id, f"generic {index:03d}",
                created_at=f"2026-08-0{day} {hour:02d}:00:00",
            )
        _seed_item(db, source_id, "zzqtoken oldest", created_at="2026-08-01 00:00:00")
        await screen._replace_items_snapshot(reason="initial")
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        await _wait_for_items(pilot, pane)
        assert len(pane.displayed_items()) == 50, "precondition: page is capped"
        assert all(
            "zzqtoken" not in str(item.get("title")) for item in pane.displayed_items()
        ), "precondition: the oldest item fell off the page"

        search = screen.query_one("#items-search-input", Input)
        search.value = "zzqtoken"
        found = False
        for _ in range(80):
            await pilot.pause(0.05)
            if any(
                "zzqtoken" in str(item.get("title"))
                for item in pane.displayed_items()
            ):
                found = True
                break
        assert found, "a corpus-wide search must surface items beyond the page"


@pytest.mark.asyncio
async def test_clearing_the_search_restores_the_unsearched_page():
    from textual.widgets import Input

    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        db = app.watchlist_bundle_service._db
        source_id = db.add_subscription(
            name="ArXiv", type="rss", source="https://a.example/f"
        )
        _seed_item(db, source_id, "alpha post")
        _seed_item(db, source_id, "beta post")
        await screen._replace_items_snapshot(reason="initial")
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        await _wait_for_items(pilot, pane)
        assert len(pane.displayed_items()) == 2, "precondition"

        search = screen.query_one("#items-search-input", Input)
        search.value = "no such token anywhere"
        # Track `_loaded_items` (the SERVER page), not `displayed_items()`:
        # the pane's client-side filter would answer instantly and race the
        # debounced reload this test is actually about.
        for _ in range(80):
            await pilot.pause(0.05)
            if screen._loaded_items == []:
                break
        assert screen._loaded_items == [], "the corpus query returned nothing"
        assert pane.displayed_items() == []

        # The empty-page reload recomposes the pane, rebuilding the Input --
        # and the recompose lands asynchronously, so even a FRESH query can
        # return the about-to-be-destroyed widget. Let it settle, re-query,
        # and then prove propagation through the screen's mirror before
        # waiting on the reload (a dead handle would stall every later
        # wait with no signal).
        await pilot.pause(0.5)
        screen.query_one("#items-search-input", Input).value = ""
        for _ in range(80):
            await pilot.pause(0.05)
            if screen._items_search_query == "":
                break
        assert screen._items_search_query == "", (
            "the clear must reach the screen's mirror"
        )
        for _ in range(80):
            await pilot.pause(0.05)
            if len(screen._loaded_items) == 2:
                break
        await pilot.pause(0.4)  # let any trailing debounce land
        displayed = pane.displayed_items()
        assert len(displayed) == 2, (
            "clearing the box must restore the unsearched page "
            f"(mirror={screen._items_search_query!r} pane.query={pane.search_query!r} "
            f"loaded={len(screen._loaded_items)} pane.items={len(pane.items)} "
            f"rendered={len(pane._rendered_items)})"
        )


@pytest.mark.asyncio
async def test_search_keeps_the_open_item_pinned():
    """The pin is unconditional since TASK-3072: searching away from the open
    item must not vanish the article being read."""
    from textual.widgets import Input

    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        db = app.watchlist_bundle_service._db
        source_id = db.add_subscription(
            name="ArXiv", type="rss", source="https://a.example/f"
        )
        _seed_item(db, source_id, "aaa keepme", created_at="2026-08-06 09:00:00")
        _seed_item(db, source_id, "bbb findme", created_at="2026-08-06 09:01:00")
        await screen._replace_items_snapshot(reason="initial")
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        await _wait_for_items(pilot, pane)

        pane.select_item_by_id(
            str(next(i["id"] for i in pane.items if "aaa" in str(i.get("title"))))
        )
        await pilot.pause(0.3)
        assert "aaa" in str(screen._selected_content_item.get("title")), "precondition"

        screen.query_one("#items-search-input", Input).value = "bbb"
        for _ in range(80):
            await pilot.pause(0.05)
            titles = [str(i.get("title")) for i in pane.displayed_items()]
            if any("bbb" in t for t in titles):
                break
        titles = [str(i.get("title")) for i in pane.displayed_items()]
        assert any("bbb" in t for t in titles), "the match is listed"
        assert any("aaa" in t for t in titles), "the open item stays pinned"


# --- TASK-3791 plan task 5: `r` refresh-all ------------------------------------


def _seed_checkable_sources(app):
    """Two active sources and one auto-paused one (eligible = `active`)."""
    db = app.local_watchlists_service._db()
    active_a = db.add_subscription(
        name="Active A", type="rss", source="https://a.example/f"
    )
    active_b = db.add_subscription(
        name="Active B", type="rss", source="https://b.example/f"
    )
    paused = db.add_subscription(
        name="Paused", type="rss", source="https://c.example/f"
    )
    with db.transaction() as conn:
        conn.execute(
            "UPDATE subscriptions SET is_paused = 1 WHERE id = ?", (paused,)
        )
    return active_a, active_b, paused


async def _screen_with_sources(pilot, host):
    """The mounted screen with sources and tree data both loaded."""
    await pilot.pause(0.1)
    screen = host.screen_stack[-1]
    for _ in range(40):
        await pilot.pause()
        if screen._tree_counts:
            break
    await screen._load_sources()
    await pilot.pause()
    return screen


@pytest.mark.asyncio
async def test_r_checks_every_active_source_once_and_aggregates():
    """`r` launches a check per ACTIVE source (paused are skipped), then
    ONE aggregated toast with the unread delta, and the pill shows it."""
    from unittest.mock import Mock

    from tldw_chatbook.Subscriptions.item_persist import persist_subscription_item

    app = _build_test_app()
    active_a, active_b, _paused = _seed_checkable_sources(app)
    db = app.local_watchlists_service._db()

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _screen_with_sources(pilot, host)

        calls: list[int] = []

        async def _check(*, runtime_backend=None, source_id):
            calls.append(source_id)
            with db.transaction() as conn:
                persist_subscription_item(
                    conn,
                    source_id,
                    {
                        "url": f"https://feed.test/new-{source_id}/",
                        "title": f"New from {source_id}",
                        "content_hash": f"hash-r-{source_id}",
                    },
                    run_id=None,
                    now="2026-08-08T09:00:00+00:00",
                )
            return {"status": "completed"}

        screen._controller.check_now = _check
        app.notify = Mock()

        await pilot.press("r")
        for _ in range(100):
            await pilot.pause(0.05)
            if any(
                "Checked" in str(call.args[0]) for call in app.notify.call_args_list
            ):
                break

        assert sorted(calls) == sorted([active_a, active_b]), (
            "every active source exactly once; the paused one never"
        )
        toasts = [
            str(call.args[0]) for call in app.notify.call_args_list
            if "Checked" in str(call.args[0])
        ]
        assert len(toasts) == 1, "one aggregated toast, never one per source"
        assert "2" in toasts[0] and "new items" in toasts[0]

        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        assert pane.new_items_note == "2 new items"


@pytest.mark.asyncio
async def test_refresh_all_pill_uses_arrivals_not_the_unread_delta():
    """Reading an old row during a check cannot hide a new arrival."""
    from unittest.mock import Mock

    from tldw_chatbook.Subscriptions.item_persist import persist_subscription_item

    app = _build_test_app()
    db = app.local_watchlists_service._db()
    source_id = db.add_subscription(
        name="Active", type="rss", source="https://active.example/f"
    )
    old_id = _seed_item(db, source_id, "Existing unread")

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _screen_with_sources(pilot, host)
        assert await screen._replace_items_snapshot(reason="initial") is True

        async def _check(*, runtime_backend=None, source_id):
            db.mark_item_status(old_id, "reviewed")
            with db.transaction() as conn:
                for suffix in ("a", "b"):
                    persist_subscription_item(
                        conn,
                        source_id,
                        {
                            "url": f"https://feed.test/new-{suffix}/",
                            "title": f"New {suffix}",
                            "content_hash": f"hash-arrival-{suffix}",
                        },
                        run_id=None,
                        now=f"2026-08-08T09:00:0{1 if suffix == 'a' else 2}+00:00",
                    )
            return {"status": "completed"}

        screen._controller.check_now = _check
        app.notify = Mock()
        await pilot.press("r")
        for _ in range(100):
            await pilot.pause(0.05)
            if not screen._refresh_all_in_flight:
                break

        # Live unread moved 1 -> 2, a delta of one, but two ids crossed the
        # committed creation watermark. The pill reports the latter.
        assert screen._items_pending_arrivals == 2
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        assert pane.new_items_note == "2 new items"
        assert len(screen._loaded_items) == 1
        assert screen._items_snapshot_count == 1


@pytest.mark.asyncio
async def test_arrivals_respect_scope_and_stay_outside_the_cached_snapshot():
    app = _build_test_app()
    db = app.local_watchlists_service._db()
    active = db.add_subscription(
        name="Active", type="rss", source="https://active.example/f"
    )
    outside = db.add_subscription(
        name="Outside", type="rss", source="https://outside.example/f"
    )
    committed_id = _seed_item(db, active, "Committed")

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        screen._apply_tree_scope(TreeScope(kind="source", source_id=active))
        assert await screen._replace_items_snapshot(reason="initial") is True
        snapshot = screen._items_snapshot
        rows = screen._loaded_items
        content = screen.query_one("#watchlists-content-pane", ContentPane)
        reader_item = rows[0]
        screen._selected_content_item = reader_item
        content.item = reader_item

        db.mark_item_status(committed_id, "reviewed")
        await screen._load_tree_data().wait()
        await pilot.pause(0.1)
        assert screen._items_pending_arrivals == 0
        assert screen._items_snapshot_count == 1
        assert str(screen.query_one("#wl-tree-node-all", Button).label) == (
            "All sources  0"
        )

        _seed_item(db, outside, "Out of scope")
        assert await screen._refresh_items_pending_arrivals() is True
        assert screen._items_pending_arrivals == 0

        arrival_id = _seed_item(db, active, "Matching arrival")
        count_arrivals = AsyncMock(
            wraps=screen._controller.count_reader_item_arrivals
        )
        screen._controller.count_reader_item_arrivals = count_arrivals
        await screen._load_tree_data().wait()
        await pilot.pause(0.1)

        count_arrivals.assert_awaited_once_with(
            runtime_backend="local",
            snapshot_max_item_id=snapshot.watermark,
            **snapshot.query.as_kwargs(),
        )
        assert screen._items_pending_arrivals == 1
        assert str(screen.query_one("#wl-tree-node-all", Button).label) == (
            "All sources  2"
        )
        assert screen._items_snapshot is snapshot
        assert screen._loaded_items is rows
        assert screen._items_snapshot_count == 1
        assert screen._selected_content_item is reader_item
        assert content.item is reader_item
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        assert pane.new_items_note == "1 new item"
        assert pane.snapshot_count == 1
        assert pane.items is rows

        # A later status/star patch only touches rows already admitted to the
        # committed cache; it cannot smuggle this above-watermark row in.
        screen._patch_committed_items_after_mutation(
            arrival_id, status="reviewed", is_flagged=True
        )
        assert all(
            row.get("item_id") != arrival_id
            for page in screen._items_snapshot.pages
            for row in page
        )
        assert screen._items_pending_arrivals == 1
        assert screen._items_snapshot_count == 1


@pytest.mark.asyncio
async def test_r_with_no_eligible_sources_notifies_and_dispatches_nothing():
    from unittest.mock import Mock

    app = _build_test_app()
    db = app.local_watchlists_service._db()
    paused = db.add_subscription(
        name="Paused", type="rss", source="https://c.example/f"
    )
    with db.transaction() as conn:
        conn.execute(
            "UPDATE subscriptions SET is_paused = 1 WHERE id = ?", (paused,)
        )

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _screen_with_sources(pilot, host)
        screen._controller.check_now = Mock(
            side_effect=AssertionError("must not be called")
        )
        app.notify = Mock()

        await pilot.press("r")
        for _ in range(60):
            await pilot.pause(0.05)
            if app.notify.called:
                break

        assert app.notify.called
        assert "Nothing to check" in str(app.notify.call_args.args[0])


@pytest.mark.asyncio
async def test_r_during_a_batch_is_a_noop():
    """One batch at a time: a second `r` while checks are in flight does
    not double-launch."""
    import asyncio

    app = _build_test_app()
    active_a, active_b, _paused = _seed_checkable_sources(app)

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _screen_with_sources(pilot, host)

        calls: list[int] = []
        entered = asyncio.Event()
        release = asyncio.Event()

        async def _slow_check(*, runtime_backend=None, source_id):
            calls.append(source_id)
            entered.set()
            await release.wait()
            return {"status": "completed"}

        screen._controller.check_now = _slow_check

        await pilot.press("r")
        for _ in range(60):
            await pilot.pause(0.05)
            if entered.is_set():
                break
        assert entered.is_set(), "precondition: the batch is in flight"

        await pilot.press("r")
        await pilot.pause(0.2)
        release.set()
        for _ in range(60):
            await pilot.pause(0.05)
            if len(calls) >= 2:
                break
        await pilot.pause(0.2)

        assert sorted(calls) == sorted([active_a, active_b]), (
            "the second `r` must not start a second batch"
        )


@pytest.mark.asyncio
async def test_r_names_a_failed_source_and_finishes_the_batch():
    from unittest.mock import Mock

    from tldw_chatbook.Subscriptions.item_persist import persist_subscription_item

    app = _build_test_app()
    active_a, active_b, _paused = _seed_checkable_sources(app)
    db = app.local_watchlists_service._db()

    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        screen = await _screen_with_sources(pilot, host)

        async def _check(*, runtime_backend=None, source_id):
            if source_id == active_b:
                raise RuntimeError("boom")
            with db.transaction() as conn:
                persist_subscription_item(
                    conn,
                    source_id,
                    {
                        "url": f"https://feed.test/new-{source_id}/",
                        "title": f"New from {source_id}",
                        "content_hash": f"hash-rf-{source_id}",
                    },
                    run_id=None,
                    now="2026-08-08T09:00:00+00:00",
                )
            return {"status": "completed"}

        screen._controller.check_now = _check
        app.notify = Mock()

        await pilot.press("r")
        for _ in range(100):
            await pilot.pause(0.05)
            if any(
                "Checked" in str(call.args[0]) for call in app.notify.call_args_list
            ):
                break

        toasts = [
            str(call.args[0]) for call in app.notify.call_args_list
            if "Checked" in str(call.args[0])
        ]
        assert len(toasts) == 1
        assert "1" in toasts[0] and "failed" in toasts[0], (
            "the aggregate names the failure count"
        )
        assert "1 new items" in toasts[0], (
            "the delta counts what actually arrived"
        )


# --- TASK-3791 plan task 6: the hostile-search end-to-end pin ------------------


@pytest.mark.asyncio
async def test_a_hostile_search_query_renders_inert_and_never_raises():
    """An FTS5-syntax attack typed into the reader's search box: the corpus
    query treats it as literal text (task 2's quoting), the rows render
    inert, and nothing raises into the UI."""
    from textual.widgets import Input

    app = _build_test_app()
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        db = app.watchlist_bundle_service._db
        source_id = db.add_subscription(
            name="[bold red]Evil Feed[/]", type="rss", source="https://evil.example/f"
        )
        _seed_item(db, source_id, '[script]alert("x")[/script] daily')
        await screen._replace_items_snapshot(reason="initial")
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        await _wait_for_items(pilot, pane)
        assert pane.displayed_items(), "precondition"

        for hostile in ('"unbalanced', "[bold red]", "NEAR/1 AND", '"'):
            # Each reload recomposes the pane and destroys the Input, and the
            # recompose lands asynchronously -- settle, re-query, and PROVE
            # propagation through the screen's mirror before waiting on the
            # reload, exactly the mechanic the task-3 clearing test pinned.
            await pilot.pause(0.5)
            screen.query_one("#items-search-input", Input).value = hostile
            propagated = False
            for _ in range(80):
                await pilot.pause(0.05)
                if screen._items_search_query == hostile:
                    propagated = True
                    break
            assert propagated, f"{hostile!r} must reach the screen's mirror"
            for _ in range(100):
                await pilot.pause(0.05)
                if screen._loaded_items == []:
                    break
            assert screen._loaded_items == [], (
                f"{hostile!r} matches nothing, and crucially did not raise"
            )

        # And the hostile SOURCE NAME in the surviving rows renders as
        # literal characters. Settle, re-query (each reload above recomposed
        # the pane and destroyed the previous Input), and prove the clear
        # propagated through the screen's mirror before waiting on the
        # restore -- the mechanic pinned by the clearing test in task 3.
        await pilot.pause(0.5)
        screen.query_one("#items-search-input", Input).value = ""
        for _ in range(80):
            await pilot.pause(0.05)
            if screen._items_search_query == "":
                break
        for _ in range(100):
            await pilot.pause(0.05)
            if len(screen._loaded_items) == 1:
                break
        await pilot.pause(0.5)  # let the restore recompose land

        row_widget = pane._find_row(str(pane.items[0]["id"]))
        row_text = str(row_widget.render())
        assert "[bold red]Evil Feed[/]" in row_text
        assert "[script]" in row_text


# --- TASK-3604 plan task 5: the import summary toast ----------------------------


def test_opml_import_summary_text_tells_the_whole_story():
    """The toast names new vs already-present sources, the watchlists
    created/reused, and the Unassigned remainder -- the pre-round-trip
    "Imported N source(s)" read identically for a structured import and a
    no-op re-import."""
    from tldw_chatbook.UI.Screens.watchlists_collections_screen import (
        _opml_import_summary_text,
    )

    text = _opml_import_summary_text({
        "created": 12,
        "existing": 3,
        "watchlists_created": ["AI", "News"],
        "watchlists_reused": ["Tech"],
        "assignments": 13,
    })
    assert "12 new" in text and "3 already present" in text
    assert "13 into 3 watchlists, 2 new" in text
    assert "2 unassigned" in text

    shared = _opml_import_summary_text({
        "created": 2,
        "existing": 0,
        "watchlists_created": ["News", "Tech"],
        "watchlists_reused": [],
        "assignments": 2,
        "unassigned": 1,
    })
    assert "1 unassigned" in shared, (
        "membership edges cannot be subtracted from unique source count"
    )

    assert _opml_import_summary_text({
        "created": 5, "existing": 0,
        "watchlists_created": [], "watchlists_reused": [], "assignments": 0,
    }) == "Imported 5 new source(s) from OPML.", (
        "a folderless import reads exactly like the old toast"
    )

    noop = _opml_import_summary_text({
        "created": 0, "existing": 15,
        "watchlists_created": [], "watchlists_reused": ["Tech"],
        "assignments": 15,
    })
    assert "0 new + 15 already present" in noop
    assert "unassigned" not in noop, "a full round-trip leaves no remainder"
