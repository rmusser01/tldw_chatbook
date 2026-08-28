"""Production-shaped tests for Watchlists bulk source authoring."""

from __future__ import annotations

import asyncio

import pytest
from unittest.mock import AsyncMock, Mock
from textual.app import ComposeResult
from textual.widgets import Button, DataTable, Input, Select, Static, TextArea

from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp
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
    CSS_PATH = [str(BUNDLED_STYLESHEET)]

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
    CSS_PATH = str(BUNDLED_STYLESHEET)


class ConfiguredWatchlistsHarness(ConsolidatedCSSApp):
    CSS_PATH = [str(BUNDLED_STYLESHEET)]

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
        assert [str(label.render()) for label in editor.query(".sources-filter-label")] == [
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
        modal.query_one("#bulk-sources-draft", TextArea).text = (
            "https://admitted.example/feed"
        )
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
        draft.text = (
            "https://one.example/feed\n"
            "not a URL\n"
            "https://two.example/feed"
        )
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
        modal.query_one("#bulk-sources-draft", TextArea).text = (
            "https://late.example/feed"
        )
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
        assert pane.query_one(
            "#sources-create-watchlist-selected", Button
        ).disabled is False

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
        assert pane.query_one(
            "#sources-create-watchlist-selected", Button
        ).disabled is False


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
        modal.show_write_failure("Sources could not be saved. Return to the draft and retry.")
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
        modal.query_one("#bulk-sources-draft", TextArea).text = (
            "https://one.example/feed\nhttps://two.example/feed"
        )
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
