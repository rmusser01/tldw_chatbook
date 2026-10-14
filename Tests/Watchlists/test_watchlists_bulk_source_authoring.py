"""Production-shaped tests for Watchlists bulk source authoring."""

from __future__ import annotations

import asyncio

import pytest

# Real guarded config consumers retain their collection-time private profile.
pytestmark = pytest.mark.bootstrap_profile
from unittest.mock import AsyncMock, Mock
from textual.app import ComposeResult
from textual.widgets import Button, DataTable, Input, Select, Static, TextArea

from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_destination_shells import DestinationHarness
from tldw_chatbook.UI.Screens.watchlists_collections_screen import (
    WatchlistsCollectionsScreen,
    WatchlistsSourceSelectionCommandProvider,
)
from tldw_chatbook.UI.Watchlists_Modules.bulk_sources_modal import (
    BulkSourceRequestRow,
    BulkSourcesContinueRequested,
    BulkSourcesCreateRequested,
    BulkSourcesModal,
)
from tldw_chatbook.UI.Watchlists_Modules.sources_pane import (
    CreateWatchlistFromSelectedRequested,
    SourcesPane,
)
from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import TreeScope


class BulkSourcesHarness(ConsolidatedCSSApp):
    CSS_PATH = [str(sheet) for sheet in APP_STYLESHEETS]

    def __init__(self) -> None:
        super().__init__()
        self.create_requests: list[BulkSourcesCreateRequested] = []
        self.continue_requests: list[BulkSourcesContinueRequested] = []

    def compose(self) -> ComposeResult:
        yield Static("Watchlists")

    def on_mount(self) -> None:
        self.push_screen(BulkSourcesModal())

    def on_bulk_sources_create_requested(
        self, event: BulkSourcesCreateRequested
    ) -> None:
        self.create_requests.append(event)

    def on_bulk_sources_continue_requested(
        self, event: BulkSourcesContinueRequested
    ) -> None:
        self.continue_requests.append(event)


class BulkSourcesScreenHarness(DestinationHarness):
    CSS_PATH = [str(sheet) for sheet in APP_STYLESHEETS]


class ConfiguredWatchlistsHarness(ConsolidatedCSSApp):
    CSS_PATH = [str(sheet) for sheet in APP_STYLESHEETS]

    def __init__(self, screen: WatchlistsCollectionsScreen) -> None:
        super().__init__()
        self.configured_screen = screen

    async def on_mount(self) -> None:
        await self.push_screen(self.configured_screen)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(160, 42), (220, 52)])
async def test_source_authoring_controls_fit_production_layout(size):
    """Primary and disclosed controls remain visible at supported widths."""
    app_instance = _build_test_app()
    host = BulkSourcesScreenHarness(app_instance, "watchlists_collections")

    async with host.run_test(size=size) as pilot:
        screen = host.screen_stack[-1]
        screen.active_section = "sources"
        await pilot.pause()
        pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
        for selector in (
            "#sources-search-input",
            "#sources-new-button",
            "#sources-add-several-button",
            "#sources-filter-toggle",
            "#sources-selection-status",
            "#sources-create-watchlist-selected",
            "#sources-table",
        ):
            widget = pane.query_one(selector)
            assert widget.region.width > 0 and widget.region.height > 0
            assert widget.region.x >= pane.region.x
            assert widget.region.right <= pane.region.right
            assert widget.region.bottom <= size[1]
        assert not pane.query("#sources-type-select")

        pane.query_one("#sources-filter-toggle", Button).press()
        await pilot.pause()
        editor = pane.query_one("#sources-filter-editor")
        assert editor.region.right <= pane.region.right
        for selector in (
            "#sources-type-select",
            "#sources-status-filter",
            "#sources-active-filter",
            "#sources-tags-filter",
        ):
            control = editor.query_one(selector)
            assert control.region.width > 0 and control.region.height > 0
            assert control.region.right <= pane.region.right
        assert [
            str(label.render()) for label in editor.query(".sources-filter-label")
        ] == [
            "Type",
            "Status",
            "Active",
            "Tags",
        ]


@pytest.mark.asyncio
async def test_bulk_modal_escape_restores_focus_to_its_launch_action():
    """Catches dismiss returning keyboard users to an unrelated screen control."""
    app_instance = _build_test_app()
    host = BulkSourcesScreenHarness(app_instance, "watchlists_collections")

    async with host.run_test(size=(160, 42)) as pilot:
        screen = host.screen_stack[-1]
        screen.active_section = "sources"
        await pilot.pause()
        pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
        launch = pane.query_one("#sources-add-several-button", Button)
        launch.focus()
        launch.press()
        await pilot.pause()

        assert isinstance(host.screen, BulkSourcesModal)
        await pilot.press("escape")
        await pilot.pause()

        restored = screen.query_one("#sources-add-several-button", Button)
        assert screen.focused is restored


@pytest.mark.asyncio
async def test_bulk_modal_cannot_dismiss_an_admitted_write():
    """An admitted exact batch must report its outcome before the modal closes."""
    app = BulkSourcesHarness()
    async with app.run_test(size=(160, 42)) as pilot:
        modal = app.screen
        assert isinstance(modal, BulkSourcesModal)
        modal.query_one(
            "#bulk-sources-draft", TextArea
        ).text = "https://admitted.example/feed"
        modal.query_one("#bulk-sources-create", Button).press()
        await pilot.pause()

        assert modal.query_one("#bulk-sources-create", Button).disabled is True
        assert modal.query_one("#bulk-sources-cancel", Button).disabled is True
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is modal

        modal.show_write_failure("Sources could not be saved. Return to the draft.")
        await pilot.pause()
        assert modal.query_one("#bulk-sources-cancel", Button).disabled is False
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is not modal


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(160, 42), (220, 52)])
async def test_partial_result_escape_returns_to_the_preserved_draft(size):
    """Escape is Return to draft, not a hidden third partial-result choice."""
    app = BulkSourcesHarness()
    async with app.run_test(size=size) as pilot:
        modal = app.screen
        assert isinstance(modal, BulkSourcesModal)
        draft = modal.query_one("#bulk-sources-draft", TextArea)
        original = "https://ready.example/feed\nnot a URL"
        draft.text = original
        modal.query_one("#bulk-sources-create", Button).press()
        await pilot.pause()
        modal.apply_results(
            [
                {
                    "input_index": 0,
                    "outcome": "created",
                    "source": {"id": "local:subscription:11"},
                }
            ]
        )
        await pilot.pause()
        assert modal.query_one("#bulk-sources-actions").display is False
        assert modal.query_one("#bulk-sources-decisions").display is True

        await pilot.press("escape")
        await pilot.pause()

        assert app.screen is modal
        assert draft.text == original
        assert modal.query_one("#bulk-sources-actions").display is True
        assert modal.query_one("#bulk-sources-decisions").display is False
        assert modal.query_one("#bulk-sources-create", Button).disabled is False
        assert modal.query_one("#bulk-sources-cancel", Button).disabled is False
        assert modal.focused is draft


@pytest.mark.asyncio
async def test_help_advertises_source_selection_keys_only_for_the_focused_table():
    """Focus-scoped keys must be discoverable without being promised globally."""
    app_instance = _build_test_app()
    app_instance.notify = Mock()
    host = BulkSourcesScreenHarness(app_instance, "watchlists_collections")

    async with host.run_test(size=(160, 42)) as pilot:
        screen = host.screen_stack[-1]
        screen.active_section = "sources"
        await pilot.pause()
        pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
        pane.query_one("#sources-table", DataTable).focus()
        await pilot.pause()
        screen.action_show_help()
        focused_copy = app_instance.notify.call_args.args[0]
        assert "Space=toggle source" in focused_copy
        assert "space=next-unread" not in focused_copy.casefold()
        assert "Shift+Up/Down=range" in focused_copy
        assert "v=visible" in focused_copy
        assert "x=clear selected" in focused_copy

        app_instance.notify.reset_mock()
        pane.query_one("#sources-search-input").focus()
        await pilot.pause()
        screen.action_show_help()
        typing_copy = app_instance.notify.call_args.args[0]
        assert "Space=toggle source" not in typing_copy
        assert "space=next-unread" in typing_copy.casefold()
        assert "v=visible" not in typing_copy


@pytest.mark.asyncio
async def test_command_palette_exposes_only_focus_valid_source_selection_actions():
    """Palette discoverability follows the same focus contract as key handling."""
    app_instance = _build_test_app()
    host = BulkSourcesScreenHarness(app_instance, "watchlists_collections")

    async with host.run_test(size=(160, 42)) as pilot:
        screen = host.screen_stack[-1]
        screen.active_section = "sources"
        await pilot.pause()
        pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
        pane.query_one("#sources-table", DataTable).focus()
        await pilot.pause()
        provider = WatchlistsSourceSelectionCommandProvider(screen)

        assert [label for label, _callback, _help in provider.commands()] == [
            "Sources: Toggle highlighted source",
            "Sources: Extend selection up",
            "Sources: Extend selection down",
            "Sources: Toggle visible sources",
            "Sources: Clear selected sources",
        ]
        assert all(
            "next unread" not in f"{label} {help_text}".casefold()
            for label, _callback, help_text in provider.commands()
        )

        pane.query_one("#sources-search-input").focus()
        await pilot.pause()
        assert provider.commands() == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(160, 42), (220, 52)])
async def test_bulk_modal_preserves_order_and_pauses_on_partial_results(size):
    """Catches invalid rows being dropped or partial success auto-continuing."""
    app = BulkSourcesHarness()
    async with app.run_test(size=size) as pilot:
        modal = app.screen
        assert isinstance(modal, BulkSourcesModal)
        assert modal.query_one("#bulk-sources-continue", Button).disabled is True
        assert modal.query_one("#bulk-sources-destination").value == "all_sources"
        assert [
            str(label.render())
            for label in modal.query(".bulk-sources-field-label").results(Static)
        ] == ["Type", "Tags", "Next"]
        draft = modal.query_one("#bulk-sources-draft", TextArea)
        draft.text = "https://one.example/feed\nnot a URL\nhttps://two.example/feed"
        modal.query_one("#bulk-sources-create", Button).press()
        modal.query_one("#bulk-sources-create", Button).press()
        await pilot.pause()

        assert len(app.create_requests) == 1
        request = app.create_requests[0]
        assert [row.input_index for row in request.rows] == [0, 2]
        assert [row.payload["url"] for row in request.rows] == [
            "https://one.example/feed",
            "https://two.example/feed",
        ]
        assert draft.text.endswith("https://two.example/feed")

        modal.apply_results(
            [
                {
                    "input_index": 0,
                    "outcome": "created",
                    "source": {"id": "local:subscription:11"},
                },
                {
                    "input_index": 1,
                    "outcome": "existing",
                    "source": {"id": "local:subscription:22"},
                },
            ]
        )
        await pilot.pause()

        table = modal.query_one("#bulk-sources-results", DataTable)
        assert [str(table.get_row_at(i)[1]) for i in range(table.row_count)] == [
            "Created",
            "Invalid",
            "Existing",
        ]
        decisions = [
            str(button.label)
            for button in modal.query("#bulk-sources-decisions Button")
        ]
        assert decisions == [
            "Continue with successful sources",
            "Return to draft",
        ]
        for selector in (
            "#bulk-sources-dialog",
            "#bulk-sources-draft-label",
            "#bulk-sources-draft",
            "#bulk-sources-options",
            "#bulk-sources-results",
            "#bulk-sources-create",
            "#bulk-sources-cancel",
            "#bulk-sources-continue",
            "#bulk-sources-return",
        ):
            widget = modal.query_one(selector)
            assert widget.region.right <= size[0], f"{selector} clips horizontally"
            assert widget.region.bottom <= size[1], f"{selector} clips vertically"
        assert app.continue_requests == []
        assert modal.query_one("#bulk-sources-continue", Button).disabled is False

        modal.query_one("#bulk-sources-destination").value = "create_watchlist"
        modal.query_one("#bulk-sources-continue", Button).press()
        modal.query_one("#bulk-sources-continue", Button).press()
        await pilot.pause()
        assert len(app.continue_requests) == 1
        assert app.continue_requests[0].destination == "create_watchlist"
        assert app.continue_requests[0].source_ids == (
            "local:subscription:11",
            "local:subscription:22",
        )


@pytest.mark.asyncio
async def test_bulk_modal_fails_closed_on_unknown_or_idless_results():
    """Only recognized outcomes with canonical IDs can enable Continue."""
    app = BulkSourcesHarness()
    async with app.run_test(size=(160, 42)) as pilot:
        modal = app.screen
        assert isinstance(modal, BulkSourcesModal)
        modal.query_one("#bulk-sources-draft", TextArea).text = (
            "https://unknown.example/feed\n"
            "https://idless.example/feed\n"
            "https://missing.example/feed"
        )
        modal.query_one("#bulk-sources-create", Button).press()
        await pilot.pause()

        modal.apply_results(
            [
                {
                    "input_index": 0,
                    "outcome": "surprising",
                    "source": {"id": "local:subscription:11"},
                },
                {"input_index": 1, "outcome": "created", "source": {}},
            ]
        )
        await pilot.pause()

        table = modal.query_one("#bulk-sources-results", DataTable)
        assert [str(table.get_row_at(i)[1]) for i in range(3)] == [
            "Invalid",
            "Invalid",
            "Invalid",
        ]
        assert modal.query_one("#bulk-sources-decisions").display is True
        assert modal.query_one("#bulk-sources-return", Button).disabled is False
        assert modal.query_one("#bulk-sources-continue", Button).disabled is True

        modal.apply_results(
            [
                {
                    "input_index": index,
                    "outcome": "created" if index == 0 else "existing",
                    "source": {"id": "local:subscription:11"},
                }
                for index in range(3)
            ]
        )
        modal.query_one("#bulk-sources-continue", Button).press()
        await pilot.pause()
        assert app.continue_requests[-1].source_ids == ("local:subscription:11",)


@pytest.mark.asyncio
async def test_dismissed_bulk_modal_ignores_late_batch_callback():
    """A batch may finish after Escape without touching the detached modal."""
    app_instance = _build_test_app()
    host = BulkSourcesScreenHarness(app_instance, "watchlists_collections")

    async with host.run_test(size=(160, 42)) as pilot:
        screen = host.screen_stack[-1]
        screen.active_section = "sources"
        modal = BulkSourcesModal(message_target=screen)
        host.push_screen(modal)
        await pilot.pause()

        entered = asyncio.Event()
        release = asyncio.Event()

        async def delayed_batch(_payloads):
            entered.set()
            await release.wait()
            return [
                {
                    "input_index": 0,
                    "outcome": "created",
                    "source": {"id": "local:subscription:11"},
                }
            ]

        service = Mock()
        service.create_sources_exact_batch = AsyncMock(side_effect=delayed_batch)
        screen._local_watchlists_service = Mock(return_value=service)
        modal.apply_results = Mock()
        modal.query_one(
            "#bulk-sources-draft", TextArea
        ).text = "https://late.example/feed"
        modal.query_one("#bulk-sources-create", Button).press()
        await entered.wait()

        modal.dismiss(None)
        await pilot.pause()
        release.set()
        for _ in range(10):
            await pilot.pause()

        modal.apply_results.assert_not_called()


@pytest.mark.asyncio
async def test_server_mode_rejects_local_bulk_and_selected_bundle_mutations():
    """Changing backend before execution must close both local write seams."""
    app_instance = _build_test_app()
    host = BulkSourcesScreenHarness(app_instance, "watchlists_collections")

    async with host.run_test(size=(160, 42)) as pilot:
        screen = host.screen_stack[-1]
        screen.active_section = "sources"
        modal = BulkSourcesModal(message_target=screen)
        host.push_screen(modal)
        await pilot.pause()
        screen.runtime_backend = "server"

        service = Mock()
        service.create_sources_exact_batch = AsyncMock(return_value=[])
        screen._local_watchlists_service = Mock(return_value=service)
        event = BulkSourcesCreateRequested(
            modal,
            (
                BulkSourceRequestRow(
                    0,
                    {
                        "name": "One",
                        "url": "https://one.example/feed",
                        "source_type": "rss",
                    },
                ),
            ),
        )
        await screen._create_bulk_sources(event)
        service.create_sources_exact_batch.assert_not_awaited()

        screen._start_tree_write = Mock()
        screen.post_message(
            CreateWatchlistFromSelectedRequested(("local:subscription:11",))
        )
        await pilot.pause()
        screen._start_tree_write.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(160, 42), (220, 52)])
async def test_server_mode_never_enables_local_only_source_actions(size):
    """Backend switches preserve the live table while clearing its entity."""
    app_instance = _build_test_app()
    screen = WatchlistsCollectionsScreen(app_instance)
    screen.set_reactive(WatchlistsCollectionsScreen.active_section, "sources")
    screen.set_reactive(WatchlistsCollectionsScreen.runtime_backend, "server")
    host = ConfiguredWatchlistsHarness(screen)

    async with host.run_test(size=size) as pilot:
        await pilot.pause()
        pane = screen.query_one("#watchlists-sources-pane", SourcesPane)

        def assert_server_actions_disabled() -> None:
            add_several = pane.query_one("#sources-add-several-button", Button)
            create_watchlist = pane.query_one(
                "#sources-create-watchlist-selected", Button
            )
            assert add_several.disabled is True
            assert create_watchlist.disabled is True
            assert "Local only" in str(add_several.label)
            assert "Local only" in str(create_watchlist.label)
            assert add_several.region.right <= pane.region.right
            assert create_watchlist.region.right <= pane.region.right

        assert_server_actions_disabled()

        screen.runtime_backend = "local"
        await pilot.pause()
        first = await screen._local_watchlists_service().create_source(
            {
                "name": "Alpha feed",
                "url": "https://alpha.example/feed",
                "source_type": "rss",
                "active": True,
            }
        )
        second = await screen._local_watchlists_service().create_source(
            {
                "name": "Beta feed",
                "url": "https://beta.example/feed",
                "source_type": "rss",
                "active": True,
            }
        )
        await screen._load_sources()
        await pilot.pause()
        table = pane.query_one("#sources-table", DataTable)
        table.focus()
        table.move_cursor(row=1, animate=False)
        pane.set_selected_source_ids((first["id"],))
        await pilot.pause()
        assert screen.focused is table
        assert table.cursor_row == 1
        assert screen.selected_source is not None
        assert screen.selected_source["id"] == second["id"]
        assert pane.query_one("#sources-add-several-button", Button).disabled is False
        assert (
            pane.query_one("#sources-create-watchlist-selected", Button).disabled
            is False
        )

        screen.runtime_backend = "server"
        await pilot.pause()
        assert screen.query_one("#watchlists-sources-pane", SourcesPane) is pane
        assert pane.query_one("#sources-table", DataTable) is table
        assert screen.focused is table
        assert table.cursor_row == 1
        assert screen.selected_source is None
        assert screen.selected_entity is None
        assert pane.selected_source is None
        assert pane.selected_source_ids == frozenset({first["id"]})
        assert_server_actions_disabled()
        screen._start_tree_write = Mock()
        pane.query_one("#sources-add-several-button", Button).press()
        pane.query_one("#sources-create-watchlist-selected", Button).press()
        await pilot.pause()
        assert host.screen is screen
        screen._start_tree_write.assert_not_called()

        screen.runtime_backend = "local"
        await pilot.pause()
        assert screen.query_one("#watchlists-sources-pane", SourcesPane) is pane
        assert pane.query_one("#sources-table", DataTable) is table
        assert screen.focused is table
        assert table.cursor_row == 1
        assert screen.selected_source is None
        assert screen.selected_entity is None
        assert pane.selected_source is None
        assert pane.selected_source_ids == frozenset({first["id"]})
        assert pane.query_one("#sources-add-several-button", Button).disabled is False
        assert (
            pane.query_one("#sources-create-watchlist-selected", Button).disabled
            is False
        )


def _hold_loader(monkeypatch, loader):
    """Hold one screen loader until the returned ``land`` lets it finish.

    Args:
        monkeypatch: The test's monkeypatch.
        loader: ``"tree"`` (the rail snapshot), ``"overview"`` (its counts) or
            ``"sources"`` (the Sources rows).

    Returns:
        ``land(host, screen)``: release the loader and await its workers.
    """
    import threading

    from tldw_chatbook.UI.Watchlists_Modules.watchlists_backend_controller import (
        WatchlistsBackendController,
    )

    release = threading.Event()
    if loader == "tree":
        group, read = "wc_tree", WatchlistsCollectionsScreen._read_tree_data_snapshot

        def held(*args):
            assert release.wait(30), "the tree load was never released"
            return read(*args)

        monkeypatch.setattr(
            WatchlistsCollectionsScreen, "_read_tree_data_snapshot", staticmethod(held)
        )
    else:
        group, name = {
            "overview": ("wc_overview", "get_overview_data"),
            "sources": ("wc_sources", "list_sources"),
        }[loader]
        read = getattr(WatchlistsBackendController, name)

        async def held(self, **kwargs):
            await asyncio.to_thread(release.wait, 30)
            return await read(self, **kwargs)

        monkeypatch.setattr(WatchlistsBackendController, name, held)

    async def land(host, screen):
        release.set()
        await asyncio.gather(
            *(
                worker.wait()
                for worker in host.workers
                if worker.node is screen and worker.group == group
            ),
            return_exceptions=True,  # an exclusive refresh cancels its forerunner
        )

    return land


async def _until(pilot, done):
    for _ in range(100):
        if done():
            return
        await pilot.pause(0.05)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("target", "loader"),
    [("sources", "tree"), ("overview", "tree"), ("overview", "overview")],
)
async def test_a_load_landing_while_a_section_pane_mounts_reaches_that_pane(
    monkeypatch, target, loader
):
    """TASK-34951: the section swap builds the new pane, then awaits its
    remove/mount; a loader landing in that gap found no pane to push into,
    and the post-mount reseed restored the rows only. The Sources create form
    then offered no watchlist, and choosing one raised InvalidSelectValueError
    (the backend-switch test's CI failure); the Overview kept counting no
    watchlists, or kept its loading layout (review of #3107, item V2)."""
    from tldw_chatbook.UI.Watchlists_Modules.overview_pane import OverviewPane
    from tldw_chatbook.UI.Watchlists_Modules.watchlists_workbench import (
        WatchlistsWorkbench,
    )

    app_instance = _build_test_app()
    watchlist = app_instance.watchlist_bundle_service.create("Research")
    host = BulkSourcesScreenHarness(app_instance, "watchlists_collections")
    land = _hold_loader(monkeypatch, loader)
    replace = WatchlistsWorkbench._replace_region_content
    swaps, landed = [], []

    async def load_lands_first(workbench, container, replacement):
        screen = workbench.screen
        swaps.append(screen.active_section)
        if screen.active_section == target and swaps.count(target) == 1:
            # The target pane is built but not mounted: let the load land.
            await land(host, screen)
            landed.append(
                watchlist["id"] in {r["id"] for r in screen._tree_watchlists}
                if loader == "tree"
                else bool(screen.overview_data)
            )
        await replace(workbench, container, replacement)

    monkeypatch.setattr(
        WatchlistsWorkbench, "_replace_region_content", load_lands_first
    )

    async with host.run_test(size=(220, 52)) as pilot:
        screen = host.screen_stack[-1]
        for section in ("sources",) if target == "sources" else ("sources", target):
            screen.active_section = section
            await _until(pilot, lambda section=section: section in swaps)
        await _until(pilot, lambda: bool(landed))
        await pilot.pause()
        if target == "sources":
            pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
            shown = {int(row["id"]) for row in pane.watchlist_choices}
            expected = {watchlist["id"]}
        elif loader == "tree":
            pane = screen.query_one("#watchlists-overview-pane", OverviewPane)
            shown, expected = pane.watchlist_count, 1
        else:
            pane = screen.query_one("#watchlists-overview-pane", OverviewPane)
            shown, expected = pane.data, screen.overview_data

    assert landed == [True]
    assert shown == expected


@pytest.mark.asyncio
async def test_a_load_landing_while_the_items_region_is_recreated_reaches_its_pane(
    monkeypatch,
):
    """Review of #3107, item V1: re-expanding a collapsed Items region on Read
    rebuilds its pane through the same factory-then-mount path, and a Read
    reload committing in between (no pane to present into) was dropped: the
    re-created pane showed the rows from before the reload."""
    from datetime import UTC, datetime

    from textual.widget import Widget

    from Tests.Watchlists.test_watchlists_scoped_rebuilds import _open, _seed, _settle
    from tldw_chatbook.Subscriptions.item_persist import persist_subscription_item
    from tldw_chatbook.UI.Watchlists_Modules.article_list import ArticleListPane
    from tldw_chatbook.UI.Watchlists_Modules.region_layout import Region

    app = _build_test_app()
    watchlist_id = _seed(app, items=2)
    async with _open(app, watchlist_id) as (screen, pilot, host):
        await _settle(pilot, host)
        assert len(screen._loaded_items) == 2
        screen._toggle_preferred_region(Region.ITEMS)
        await _until(pilot, lambda: not screen.query("#watchlists-items-pane"))
        assert not screen.query("#watchlists-items-pane"), "Items did not collapse"

        db = app.watchlist_bundle_service.db
        source_id = db.get_all_subscriptions()[0]["id"]
        with db.transaction() as conn:
            persist_subscription_item(
                conn,
                source_id,
                {
                    "url": "https://ai-news.example/late",
                    "title": "Late story",
                    "content": "body of the late story",
                    "content_hash": "hash-late",
                    "content_kind": "article",
                    "content_format": "text",
                },
                run_id=None,
                now=datetime.now(UTC).isoformat(),
            )
        release = asyncio.Event()
        list_page = screen._controller.list_reader_items_page

        async def held_page(**kwargs):
            await release.wait()
            return await list_page(**kwargs)

        screen._controller.list_reader_items_page = held_page
        screen._load_active_section_data()  # the reload is held until the rebuild
        mount = Widget.mount
        landed = []

        def mount_after_the_reload(self, *widgets, **kwargs):
            if landed or not any(w.id == "wl-region-items" for w in widgets):
                return mount(self, *widgets, **kwargs)

            async def later():
                # The re-created Items region is built but not mounted.
                release.set()
                await asyncio.gather(
                    *(
                        worker.wait()
                        for worker in host.workers
                        if worker.node is screen and worker.group == "wc_items"
                    ),
                    return_exceptions=True,
                )
                landed.append(len(screen._loaded_items))
                return await mount(self, *widgets, **kwargs)

            return later()

        monkeypatch.setattr(Widget, "mount", mount_after_the_reload)
        screen._toggle_preferred_region(Region.ITEMS)
        await _until(pilot, lambda: bool(landed))
        await pilot.pause()
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        shown = len(pane.items)

    assert landed == [3]
    assert shown == 3


@pytest.mark.asyncio
async def test_an_open_create_form_offers_a_watchlist_the_tree_load_adds():
    """Review of #3107, item V3: a tree load pushing new watchlist choices into
    a pane whose create form is already open also refreshes the form's
    Select, and keeps the destination the user chose."""
    app_instance = _build_test_app()
    chosen = app_instance.watchlist_bundle_service.create("Research")
    host = BulkSourcesScreenHarness(app_instance, "watchlists_collections")

    async def tree_loaded(screen):
        await asyncio.gather(
            *(
                worker.wait()
                for worker in host.workers
                if worker.node is screen and worker.group == "wc_tree"
            )
        )

    async with host.run_test(size=(220, 52)) as pilot:
        screen = host.screen_stack[-1]
        screen.active_section = "sources"
        await pilot.pause()
        await tree_loaded(screen)
        pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
        pane.query_one("#sources-new-button", Button).press()
        await pilot.pause()
        destination = pane.query_one("#sources-create-watchlist", Select)
        destination.value = chosen["id"]
        await pilot.pause()
        later = app_instance.watchlist_bundle_service.create("Later")
        screen._load_tree_data()
        await pilot.pause()
        await tree_loaded(screen)
        await pilot.pause()
        offered = {value for _label, value in destination._options}

    assert later["id"] in offered
    assert destination.value == chosen["id"]
    assert pane.create_draft_destination == chosen["id"]


@pytest.mark.asyncio
async def test_a_sources_load_landing_while_the_pane_mounts_prunes_its_selection(
    monkeypatch,
):
    """Review of #3107, item V4: the swap gap also dropped the Sources loader's
    authoritative ids and pruned selection, so the mounted pane kept a
    selected source the reload had already found deleted."""
    from tldw_chatbook.UI.Watchlists_Modules.watchlists_workbench import (
        WatchlistsWorkbench,
    )

    app_instance = _build_test_app()
    db = app_instance.watchlist_bundle_service.db
    kept_id = db.add_subscription(name="Kept", type="rss", source="https://k.example/f")
    gone_id = db.add_subscription(name="Gone", type="rss", source="https://g.example/f")
    kept, gone = (f"local:subscription:{n}" for n in (kept_id, gone_id))
    host = BulkSourcesScreenHarness(app_instance, "watchlists_collections")
    replace = WatchlistsWorkbench._replace_region_content
    landed = []

    async with host.run_test(size=(220, 52)) as pilot:
        screen = host.screen_stack[-1]
        screen.active_section = "sources"
        await _until(pilot, lambda: len(screen._loaded_sources) == 2)
        pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
        pane.set_selected_source_ids((kept, gone))
        await _until(pilot, lambda: len(screen._selected_source_ids) == 2)
        assert set(screen._selected_source_ids) == {kept, gone}
        screen.active_section = "rules"
        await _until(pilot, lambda: not screen.query("#watchlists-sources-pane"))
        db.delete_subscription(gone_id)
        land = _hold_loader(monkeypatch, "sources")

        async def sources_load_lands_first(workbench, container, replacement):
            if workbench.screen.active_section == "sources" and not landed:
                # The Sources pane is built but not mounted: let the load land.
                await land(host, workbench.screen)
                landed.append(workbench.screen._selected_source_ids)
            await replace(workbench, container, replacement)

        monkeypatch.setattr(
            WatchlistsWorkbench, "_replace_region_content", sources_load_lands_first
        )
        screen.active_section = "sources"
        await _until(pilot, lambda: bool(landed))
        await pilot.pause()
        pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
        selected = pane.selected_source_ids

    assert landed == [(kept,)]
    assert selected == {kept}


@pytest.mark.asyncio
@pytest.mark.parametrize("rail", ["right_rail", "left_rail"])
async def test_a_load_landing_while_a_rail_is_recreated_reaches_it(monkeypatch, rail):
    """Review of #3107, round 2, item Y1: expanding a collapsed rail builds
    its content before mounting it, so an overview refresh (the Inspector's
    profile state) or a tree load (the rail's watchlists) landing in between
    reached nothing and the rail showed the state from before the load."""
    from textual.widget import Widget

    from tldw_chatbook.UI.Watchlists_Modules.inspector_pane import InspectorPane
    from tldw_chatbook.UI.Watchlists_Modules.overview_pane import OverviewPane
    from tldw_chatbook.UI.Watchlists_Modules.region_layout import Region
    from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import WatchlistTree

    app_instance = _build_test_app()
    watchlist = app_instance.watchlist_bundle_service.create("Research")
    host = BulkSourcesScreenHarness(app_instance, "watchlists_collections")
    land = _hold_loader(monkeypatch, "overview" if rail == "right_rail" else "tree")
    region, node_id = Region(rail), f"wl-region-{rail}"
    mount = Widget.mount
    landed = []

    def mount_after_the_load(self, *widgets, **kwargs):
        if landed or not any(w.id == node_id for w in widgets):
            return mount(self, *widgets, **kwargs)

        async def later():
            # The re-created rail is built but not mounted.
            await land(host, self.screen)
            landed.append(True)
            return await mount(self, *widgets, **kwargs)

        return later()

    async with host.run_test(size=(220, 52)) as pilot:
        screen = host.screen_stack[-1]
        screen.active_section = "sources"
        await pilot.pause()
        if screen.query(f"#{node_id}"):
            screen._toggle_preferred_region(region)
            await _until(pilot, lambda: not screen.query(f"#{node_id}"))
        assert not screen.query(f"#{node_id}"), f"{rail} did not collapse"
        monkeypatch.setattr(Widget, "mount", mount_after_the_load)
        screen._toggle_preferred_region(region)
        await _until(pilot, lambda: bool(landed))
        if rail == "right_rail":

            def shown():
                return screen.query_one(InspectorPane).profile_state

            expected = screen._watchlists_profile_state()
            assert expected != OverviewPane.LOADING
        else:

            def shown():
                tree = screen.query_one("#wl-tree", WatchlistTree)
                return [int(row["id"]) for row in tree._watchlists]

            expected = [watchlist["id"]]
        await _until(pilot, lambda: shown() == expected)
        result = shown()

    assert landed == [True]
    assert result == expected


@pytest.mark.asyncio
async def test_an_open_destination_list_keeps_its_highlight_when_choices_change():
    """Review of #3107, round 2, item Y2: re-offering the choices under an
    expanded destination list kept it open but dropped its highlight; the
    source-type list was re-set too. Both now keep their state."""
    from textual.widgets._select import SelectOverlay

    app_instance = _build_test_app()
    service = app_instance.watchlist_bundle_service
    service.create("Alpha")
    second = service.create("Beta")
    host = BulkSourcesScreenHarness(app_instance, "watchlists_collections")

    async def tree_loaded(screen):
        await asyncio.gather(
            *(
                worker.wait()
                for worker in host.workers
                if worker.node is screen and worker.group == "wc_tree"
            )
        )

    async with host.run_test(size=(220, 52)) as pilot:
        screen = host.screen_stack[-1]
        screen.active_section = "sources"
        await pilot.pause()
        await tree_loaded(screen)
        pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
        pane.query_one("#sources-new-button", Button).press()
        await pilot.pause()
        destination = pane.query_one("#sources-create-watchlist", Select)
        kind = pane.query_one("#sources-create-type", Select)
        kind.expanded = True
        await pilot.pause()
        kind_overlay = kind.query_one(SelectOverlay)
        kind_overlay.highlighted = 1
        kind.expanded = False
        destination.expanded = True
        await pilot.pause()
        overlay = destination.query_one(SelectOverlay)
        overlay.highlighted = 2  # Unassigned, Alpha, Beta: two rows down
        assert destination._options[2][1] == second["id"]
        service.create("Gamma")
        screen._load_tree_data()
        await pilot.pause()
        await tree_loaded(screen)
        await pilot.pause()
        expanded, highlighted = destination.expanded, overlay.highlighted
        offered = len(destination._options)
        kind_highlighted = kind_overlay.highlighted

    assert offered == 4
    assert expanded
    assert highlighted == 2
    assert kind_highlighted == 1


@pytest.mark.asyncio
async def test_the_reseed_keeps_a_selection_the_screen_has_not_heard_of_yet():
    """Review of #3107, round 2, item Y3: a reseed handled one message after
    the pane was shown (the Items re-creation path) replaced the pane's
    selection with the screen's copy, which lags the pane by that message,
    so a toggle made in the fresh pane in that instant was lost."""
    app_instance = _build_test_app()
    db = app_instance.watchlist_bundle_service.db
    source_id = db.add_subscription(
        name="One", type="rss", source="https://o.example/f"
    )
    selected = f"local:subscription:{source_id}"
    host = BulkSourcesScreenHarness(app_instance, "watchlists_collections")

    async with host.run_test(size=(220, 52)) as pilot:
        screen = host.screen_stack[-1]
        screen.active_section = "sources"
        await _until(pilot, lambda: len(screen._loaded_sources) == 1)
        pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
        assert screen._selected_source_ids == ()
        pane.set_selected_source_ids((selected,))  # its message is still queued
        screen._reseed_active_section_pane()
        kept = pane.selected_source_ids
        await pilot.pause()
        mirrored = screen._selected_source_ids

    assert kept == {selected}
    assert mirrored == (selected,)


def _land_before_mounting(monkeypatch, node_id, landing):
    """Run ``landing(screen)`` once, after the region ``node_id`` is built and
    before it mounts; returns the list ``landing`` results are appended to."""
    from textual.widget import Widget

    mount = Widget.mount
    landed = []

    def mount_after_landing(self, *widgets, **kwargs):
        if landed or not any(w.id == node_id for w in widgets):
            return mount(self, *widgets, **kwargs)

        async def later():
            landed.append(await landing(self.screen))
            return await mount(self, *widgets, **kwargs)

        return later()

    monkeypatch.setattr(Widget, "mount", mount_after_landing)
    return landed


async def _reopen(pilot, screen, region, node_id):
    """Collapse ``region`` if it is open, and return once it is gone."""
    if screen.query(f"#{node_id}"):
        screen._toggle_preferred_region(region)
        await _until(pilot, lambda: not screen.query(f"#{node_id}"))
    assert not screen.query(f"#{node_id}"), f"{node_id} did not collapse"


@pytest.mark.asyncio
async def test_a_check_starting_while_the_right_rail_is_recreated_reaches_the_inspector(
    monkeypatch,
):
    """Review of #3107, round 3, item Q1: the Inspector's factory seeds the
    busy source ids, and _set_check_now_busy pushes them into a mounted
    Inspector only, so a check starting while the right rail re-expanded
    left its Check-now state stale."""
    from tldw_chatbook.UI.Watchlists_Modules.inspector_pane import InspectorPane
    from tldw_chatbook.UI.Watchlists_Modules.region_layout import Region

    host = BulkSourcesScreenHarness(_build_test_app(), "watchlists_collections")

    async def a_check_starts(screen):
        screen._checks_in_flight.add("local:subscription:1")
        screen._set_check_now_busy()
        return True

    async with host.run_test(size=(220, 52)) as pilot:
        screen = host.screen_stack[-1]
        screen.active_section = "sources"
        await pilot.pause()
        await _reopen(pilot, screen, Region.RIGHT_RAIL, "wl-region-right_rail")
        landed = _land_before_mounting(
            monkeypatch, "wl-region-right_rail", a_check_starts
        )
        screen._toggle_preferred_region(Region.RIGHT_RAIL)
        await _until(pilot, lambda: bool(landed))
        await pilot.pause()
        busy = screen.query_one(InspectorPane).busy_source_ids

    assert landed == [True]
    assert busy == {"local:subscription:1"}


@pytest.mark.asyncio
async def test_a_rail_reopened_after_a_discarded_build_shows_the_landed_tree(
    monkeypatch,
):
    """Review of #3107, round 3, item Q2: recording the tree snapshot when a
    rail was BUILT let a build that was then discarded claim the reopened
    rail was current, so the rebuild the landed load needed was skipped."""
    from tldw_chatbook.UI.Watchlists_Modules.region_layout import Region
    from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import WatchlistTree

    app_instance = _build_test_app()
    watchlist = app_instance.watchlist_bundle_service.create("Research")
    host = BulkSourcesScreenHarness(app_instance, "watchlists_collections")
    land = _hold_loader(monkeypatch, "tree")

    async def the_load_lands_then_a_build_is_discarded(screen):
        await land(host, screen)
        screen._build_tree_pane()  # built for nothing; never mounted
        return True

    async with host.run_test(size=(220, 52)) as pilot:
        screen = host.screen_stack[-1]
        screen.active_section = "sources"
        await pilot.pause()
        await _reopen(pilot, screen, Region.LEFT_RAIL, "wl-region-left_rail")
        landed = _land_before_mounting(
            monkeypatch, "wl-region-left_rail", the_load_lands_then_a_build_is_discarded
        )
        screen._toggle_preferred_region(Region.LEFT_RAIL)
        await _until(pilot, lambda: bool(landed))

        def shown():
            tree = screen.query_one("#wl-tree", WatchlistTree)
            return [int(row["id"]) for row in tree._watchlists]

        await _until(pilot, lambda: shown() == [watchlist["id"]])
        result = shown()

    assert landed == [True]
    assert result == [watchlist["id"]]


@pytest.mark.asyncio
async def test_an_ordinary_rail_reopen_builds_the_tree_once(monkeypatch):
    """Review of #3107, round 3, item Q2: with nothing landing during the
    rebuild, a reopened rail is not rebuilt a second time (a rebuild costs
    200-500 ms on a 30-watchlist rail)."""
    from tldw_chatbook.UI.Watchlists_Modules.region_layout import Region
    from tldw_chatbook.UI.Watchlists_Modules.watchlist_tree import WatchlistTree

    app_instance = _build_test_app()
    app_instance.watchlist_bundle_service.create("Research")
    host = BulkSourcesScreenHarness(app_instance, "watchlists_collections")

    async with host.run_test(size=(220, 52)) as pilot:
        screen = host.screen_stack[-1]
        screen.active_section = "sources"
        await pilot.pause()
        await host.workers.wait_for_complete()
        await _reopen(pilot, screen, Region.LEFT_RAIL, "wl-region-left_rail")
        init, builds = WatchlistTree.__init__, []

        def counted(self, *args, **kwargs):
            builds.append(True)
            init(self, *args, **kwargs)

        monkeypatch.setattr(WatchlistTree, "__init__", counted)
        screen._toggle_preferred_region(Region.LEFT_RAIL)
        await _until(pilot, lambda: bool(screen.query("#wl-tree")))
        await pilot.pause()
        await pilot.pause(0.3)

    assert len(builds) == 1


@pytest.mark.asyncio
async def test_the_reseed_keeps_the_selection_range_anchor():
    """Review of #3107, round 3, item Q4: re-applying the pane's selection
    through replace() also reset its shift-range anchor, so a shift-click
    range started in that moment lost its start."""
    app_instance = _build_test_app()
    db = app_instance.watchlist_bundle_service.db
    source_id = db.add_subscription(
        name="One", type="rss", source="https://o.example/f"
    )
    picked = f"local:subscription:{source_id}"
    host = BulkSourcesScreenHarness(app_instance, "watchlists_collections")

    async with host.run_test(size=(220, 52)) as pilot:
        screen = host.screen_stack[-1]
        screen.active_section = "sources"
        await _until(pilot, lambda: len(screen._loaded_sources) == 1)
        pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
        pane._multi_selection.toggle(picked)  # a click: the pick is the anchor
        screen._reseed_active_section_pane()
        anchor = pane._multi_selection.anchor_id
        selected = pane.selected_source_ids

    assert selected == {picked}
    assert anchor == picked


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(160, 42), (220, 52)])
async def test_backend_switch_preserves_open_source_draft_widget_and_focus(size):
    """Backend-only repaints must not replace a focused create-form field."""
    app_instance = _build_test_app()
    watchlist = app_instance.watchlist_bundle_service.create("Research")
    host = BulkSourcesScreenHarness(app_instance, "watchlists_collections")

    async with host.run_test(size=size) as pilot:
        screen = host.screen_stack[-1]
        screen.active_section = "sources"
        await pilot.pause()
        pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
        await asyncio.gather(
            *(
                worker.wait()
                for worker in host.workers
                if worker.node is screen and worker.group == "wc_tree"
            )
        )
        assert not screen._tree_snapshot_failures
        assert watchlist["id"] in {row["id"] for row in screen._tree_watchlists}
        pane.query_one("#sources-new-button", Button).press()
        await pilot.pause()
        table = pane.query_one("#sources-table", DataTable)
        name = pane.query_one("#sources-create-name", Input)
        url = pane.query_one("#sources-create-url", Input)
        tags = pane.query_one("#sources-create-tags", Input)
        destination = pane.query_one("#sources-create-watchlist", Select)
        name.value = "Daily intelligence"
        url.value = "https://intel.example/feed"
        tags.value = "threat-intel, daily"
        destination.value = watchlist["id"]
        url.focus()
        await pilot.pause()
        assert pane.create_draft_destination == watchlist["id"]

        screen.runtime_backend = "server"
        await pilot.pause()

        assert screen.query_one("#watchlists-sources-pane", SourcesPane) is pane
        assert pane.query_one("#sources-table", DataTable) is table
        assert pane.query_one("#sources-create-name", Input) is name
        assert pane.query_one("#sources-create-url", Input) is url
        assert pane.query_one("#sources-create-tags", Input) is tags
        assert pane.query_one("#sources-create-watchlist", Select) is destination
        assert screen.focused is url
        assert (name.value, url.value, tags.value) == (
            "Daily intelligence",
            "https://intel.example/feed",
            "threat-intel, daily",
        )
        assert pane.query_one("#sources-add-several-button", Button).disabled is True
        assert destination.disabled is True
        assert destination.value == SourcesPane.UNASSIGNED_DESTINATION
        assert pane.create_draft_destination == watchlist["id"]
        destination_label = destination.parent.query_one(Static)
        assert "Local only" in str(destination_label.render())
        assert destination_label.region.width >= len("Watchlist (Local only)")

        screen.runtime_backend = "local"
        await pilot.pause()

        assert pane.query_one("#sources-table", DataTable) is table
        assert pane.query_one("#sources-create-name", Input) is name
        assert pane.query_one("#sources-create-url", Input) is url
        assert pane.query_one("#sources-create-tags", Input) is tags
        assert pane.query_one("#sources-create-watchlist", Select) is destination
        assert screen.focused is url
        assert (name.value, url.value, tags.value) == (
            "Daily intelligence",
            "https://intel.example/feed",
            "threat-intel, daily",
        )
        assert pane.query_one("#sources-add-several-button", Button).disabled is False
        assert destination.disabled is False
        assert destination.value == watchlist["id"]
        assert str(destination.parent.query_one(Static).render()) == "Watchlist"

        screen.runtime_backend = "server"
        await pilot.pause()
        screen._create_source = AsyncMock()
        pane.query_one("#sources-create-submit", Button).press()
        for _ in range(10):
            await pilot.pause()
            if screen._create_source.await_count:
                break

        screen._create_source.assert_awaited_once()
        payload = screen._create_source.await_args.args[0]
        assert payload["watchlist_id"] is None


@pytest.mark.asyncio
async def test_bulk_modal_keeps_draft_after_validation_and_write_failure():
    """Catches either validation or persistence failure erasing the draft."""
    app = BulkSourcesHarness()
    async with app.run_test(size=(160, 42)) as pilot:
        modal = app.screen
        assert isinstance(modal, BulkSourcesModal)
        draft = modal.query_one("#bulk-sources-draft", TextArea)
        draft.text = "\n".join(
            f"https://source-{index}.example/feed" for index in range(51)
        )
        modal.query_one("#bulk-sources-create", Button).press()
        await pilot.pause()

        assert app.create_requests == []
        assert len(draft.text.splitlines()) == 51
        assert "50" in str(modal.query_one("#bulk-sources-status", Static).render())

        draft.text = "https://recover.example/feed"
        modal.query_one("#bulk-sources-create", Button).press()
        await pilot.pause()
        assert len(app.create_requests) == 1
        modal.show_write_failure(
            "Sources could not be saved. Return to the draft and retry."
        )
        await pilot.pause()

        assert draft.text == "https://recover.example/feed"
        assert "could not be saved" in str(
            modal.query_one("#bulk-sources-status", Static).render()
        )


@pytest.mark.asyncio
async def test_watchlists_screen_uses_exact_batch_and_continuation_changes_no_membership():
    """Catches bulk UI bypassing the owner seam or silently filing memberships."""
    app_instance = _build_test_app()
    watchlist = app_instance.watchlist_bundle_service.create("Threat brief")
    host = BulkSourcesScreenHarness(app_instance, "watchlists_collections")

    async with host.run_test(size=(160, 42)) as pilot:
        screen = host.screen_stack[-1]
        modal = BulkSourcesModal(
            message_target=screen,
        )
        host.push_screen(modal)
        await pilot.pause()
        modal.query_one(
            "#bulk-sources-draft", TextArea
        ).text = "https://one.example/feed\nhttps://two.example/feed"
        modal.query_one("#bulk-sources-create", Button).press()

        for _ in range(100):
            await pilot.pause()
            if modal.query_one("#bulk-sources-results", DataTable).row_count == 2:
                break
        assert modal.query_one("#bulk-sources-results", DataTable).row_count == 2
        stored = app_instance.local_watchlists_service._db().get_all_subscriptions(
            include_inactive=True
        )
        assert len(stored) == 2
        assert app_instance.watchlist_bundle_service.list_sources(watchlist["id"]) == []

        modal.query_one("#bulk-sources-continue", Button).press()
        await pilot.pause()
        assert app_instance.watchlist_bundle_service.list_sources(watchlist["id"]) == []


@pytest.mark.asyncio
async def test_bulk_continue_preserves_created_ids_from_a_watchlist_scope():
    """Unassigned results stay selected by moving to a scope that contains them."""
    app_instance = _build_test_app()
    watchlist = app_instance.watchlist_bundle_service.create("Existing scope")
    host = BulkSourcesScreenHarness(app_instance, "watchlists_collections")

    async with host.run_test(size=(160, 42)) as pilot:
        screen = host.screen_stack[-1]
        screen.active_section = "sources"
        screen.tree_scope = TreeScope(
            kind="watchlist",
            watchlist_id=watchlist["id"],
        )
        await pilot.pause()
        service = screen._local_watchlists_service()
        created = await service.create_source(
            {
                "name": "New unassigned source",
                "url": "https://unassigned.example/feed",
                "source_type": "rss",
                "active": True,
            }
        )
        await screen._load_sources()
        modal = BulkSourcesModal(message_target=screen)
        host.push_screen(modal)
        await pilot.pause()

        screen.post_message(
            BulkSourcesContinueRequested(modal, (created["id"],), "all_sources")
        )
        await pilot.pause()

        assert screen.tree_scope == TreeScope(kind="all")
        pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
        assert pane.selected_source_ids == frozenset({created["id"]})
        assert screen._selected_source_ids == (created["id"],)


@pytest.mark.asyncio
async def test_scoped_rows_never_prune_selection_until_authoritative_reload_deletes_id():
    """Scope/filter/reorder are visibility; only a full reload may prune IDs."""
    app_instance = _build_test_app()
    bundles = app_instance.watchlist_bundle_service
    db = bundles._db
    first_watchlist = bundles.create("First collection")
    second_watchlist = bundles.create("Second collection")
    first_id = db.add_subscription(
        name="Alpha feed", type="rss", source="https://alpha.example/feed"
    )
    second_id = db.add_subscription(
        name="Beta feed", type="rss", source="https://beta.example/feed"
    )
    bundles.add_source(first_watchlist["id"], first_id)
    bundles.add_source(second_watchlist["id"], second_id)
    canonical_ids = (
        f"local:subscription:{first_id}",
        f"local:subscription:{second_id}",
    )
    host = BulkSourcesScreenHarness(app_instance, "watchlists_collections")

    async with host.run_test(size=(160, 42)) as pilot:
        screen = host.screen_stack[-1]
        screen.active_section = "sources"
        await screen._load_sources()
        await pilot.pause()
        pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
        pane.set_selected_source_ids(canonical_ids)
        await pilot.pause()

        for watchlist_id in (first_watchlist["id"], second_watchlist["id"]):
            screen.tree_scope = TreeScope(
                kind="watchlist",
                watchlist_id=watchlist_id,
            )
            await pilot.pause()
            assert pane.selected_source_ids == frozenset(canonical_ids)
            assert screen._selected_source_ids == canonical_ids

        pane.search_query = "Beta"
        await pilot.pause()
        assert pane.selected_source_ids == frozenset(canonical_ids)
        assert screen._selected_source_ids == canonical_ids

        screen.tree_scope = TreeScope(kind="all")
        pane.search_query = ""
        await pilot.pause()
        pane.sources = list(reversed(pane.sources))
        await pilot.pause()
        assert pane.selected_source_ids == frozenset(canonical_ids)
        assert screen._selected_source_ids == canonical_ids

        await app_instance.local_watchlists_service.delete_source(second_id)
        await screen._load_sources()
        await pilot.pause()

        assert pane.selected_source_ids == frozenset({canonical_ids[0]})
        assert screen._selected_source_ids == (canonical_ids[0],)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(160, 42), (220, 52)])
@pytest.mark.parametrize(
    ("destination", "target_selector", "expected_notification"),
    [
        (
            "all_sources",
            "#sources-table",
            "1 source(s) selected in All Sources. Choose any available source action to continue.",
        ),
        (
            "create_watchlist",
            "#sources-create-watchlist-selected",
            "1 source(s) are ready. Focus moved to Create Watchlist from selected.",
        ),
    ],
)
async def test_bulk_continue_focuses_the_chosen_non_mutating_next_step(
    size,
    destination,
    target_selector,
    expected_notification,
):
    app_instance = _build_test_app()
    host = BulkSourcesScreenHarness(app_instance, "watchlists_collections")

    async with host.run_test(size=size) as pilot:
        screen = host.screen_stack[-1]
        screen.active_section = "sources"
        await pilot.pause()
        created = await screen._local_watchlists_service().create_source(
            {
                "name": "Next-step source",
                "url": "https://next-step.example/feed",
                "source_type": "rss",
                "active": True,
            }
        )
        await screen._load_sources()
        modal = BulkSourcesModal(message_target=screen)
        host.push_screen(modal)
        await pilot.pause()
        screen._notify_watchlists = Mock()

        screen.post_message(
            BulkSourcesContinueRequested(modal, (created["id"],), destination)
        )
        for _ in range(5):
            await pilot.pause()

        pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
        target = pane.query_one(target_selector)
        assert screen.tree_scope == TreeScope(kind="all")
        assert pane.selected_source_ids == frozenset({created["id"]})
        assert screen.focused is target
        assert app_instance.watchlist_bundle_service.list_watchlists() == []
        screen._notify_watchlists.assert_called_once_with(expected_notification)


@pytest.mark.asyncio
async def test_selected_sources_create_one_atomic_watchlist_bundle():
    """Catches selected creation looping through membership dialogs or writes."""
    app_instance = _build_test_app()
    outcomes = app_instance.local_watchlists_service.create_sources_exact_batch_sync(
        [
            {"name": "One", "source_type": "rss", "url": "https://one.example/feed"},
            {"name": "Two", "source_type": "rss", "url": "https://two.example/feed"},
        ]
    )
    canonical_ids = tuple(str(row["source"]["id"]) for row in outcomes)
    host = BulkSourcesScreenHarness(app_instance, "watchlists_collections")

    async with host.run_test(size=(160, 42)) as pilot:
        screen = host.screen_stack[-1]
        screen._prompt_watchlist_name = AsyncMock(return_value="Selected feeds")
        screen.post_message(CreateWatchlistFromSelectedRequested(canonical_ids))
        for _ in range(100):
            await pilot.pause()
            if app_instance.watchlist_bundle_service.list_watchlists():
                break

        watchlists = app_instance.watchlist_bundle_service.list_watchlists()
        assert [row["name"] for row in watchlists] == ["Selected feeds"]
        assert app_instance.watchlist_bundle_service.list_sources(
            watchlists[0]["id"]
        ) == [
            int(canonical_ids[0].rsplit(":", 1)[-1]),
            int(canonical_ids[1].rsplit(":", 1)[-1]),
        ]
