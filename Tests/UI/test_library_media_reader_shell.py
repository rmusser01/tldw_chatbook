"""Production-shaped geometry contracts for the permanent Media reader shell."""

from __future__ import annotations

import pytest
from textual.containers import Horizontal
from textual.widgets import Button

from Tests.UI.app_factory import _build_test_app
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _active_library_screen,
    _seed_conversations,
    _two_conversations,
    _two_media_items,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.Library.library_media_reader_state import (
    PANE_GRIP_WIDTH,
    MediaReaderLayoutPreferences,
    resolve_media_reader_layout,
)
from tldw_chatbook.Library.library_media_viewer_state import (
    build_library_media_viewer_state,
)
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.Library import (
    LibraryMediaCanvas,
    LibraryMediaReaderShell,
    LibraryMediaViewer,
    LibraryNavigationRailHandle,
)
from tldw_chatbook.app import TldwCli


def _painted_text_in_region(app, region) -> str:
    strips = list(app.screen._compositor.render_strips())
    return "\n".join(
        strips[y].crop(region.x, region.right).text.rstrip()
        for y in range(region.y, region.bottom)
    )


def _build_media_test_app():
    app = _build_test_app()
    app.library_new_profile_admission = False
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    return app


async def _open_media_shell(host, pilot) -> tuple[LibraryScreen, LibraryMediaReaderShell]:
    screen = _active_library_screen(host)
    await _wait_for_library_shell(screen, pilot)
    screen.query_one("#library-row-browse-media", Button).press()
    await _wait_for_selector(screen, pilot, "#library-media-reader-shell")
    await _wait_for_selector(screen, pilot, "#library-media-row-0")
    await pilot.pause()
    return screen, screen.query_one(
        "#library-media-reader-shell", LibraryMediaReaderShell
    )


@pytest.mark.asyncio
async def test_media_shell_mounts_library_items_reader_and_two_five_column_grips():
    host = LibraryProductionCSSHarness(_build_media_test_app())

    async with host.run_test(size=(170, 48)) as pilot:
        screen, shell = await _open_media_shell(host, pilot)
        library = shell.query_one("#library-rail")
        items = shell.query_one("#library-media-canvas", LibraryMediaCanvas)
        reader = shell.query_one("#library-media-viewer", LibraryMediaViewer)
        grips = list(shell.query(".library-media-pane-grip"))

        assert library.display and items.display and reader.display, (
            shell.region,
            shell.effective_layout,
        )
        assert [grip.region.width for grip in grips] == [
            PANE_GRIP_WIDTH,
            PANE_GRIP_WIDTH,
        ]
        assert shell.content_region.contains_region(reader.region)
        assert "Select a media item to read it here." in _painted_text_in_region(
            pilot.app, reader.region
        )
        assert screen.query_one("#library-rail-collapse", Button).display is False


@pytest.mark.asyncio
async def test_expanded_and_collapsed_grip_copy_names_its_action():
    host = LibraryProductionCSSHarness(_build_media_test_app())

    async with host.run_test(size=(170, 48)) as pilot:
        _, shell = await _open_media_shell(host, pilot)
        for pane in ("library", "items"):
            grip = shell.query_one(f"#library-media-{pane}-grip", Button)
            assert str(grip.label) == "<---"
            assert str(grip.tooltip) == f"Collapse {pane.title()} pane"
            assert grip.name == f"Collapse {pane.title()} pane"
            assert "<---" in _painted_text_in_region(pilot.app, grip.region)

            grip.press()
            await pilot.pause()
            assert str(grip.label) == "--->"
            assert str(grip.tooltip) == f"Expand {pane.title()} pane"
            assert grip.name == f"Expand {pane.title()} pane"
            assert "--->" in _painted_text_in_region(pilot.app, grip.region)


@pytest.mark.asyncio
async def test_grips_are_focusable_clickable_and_geometry_stable():
    host = LibraryProductionCSSHarness(_build_media_test_app())

    async with host.run_test(size=(170, 48)) as pilot:
        _, shell = await _open_media_shell(host, pilot)
        library_grip = shell.query_one("#library-media-library-grip", Button)
        items_grip = shell.query_one("#library-media-items-grip", Button)

        for grip in (library_grip, items_grip):
            before = grip.region
            grip.focus()
            await pilot.pause()
            assert grip.has_focus and grip.region == before
            await pilot.press("enter")
            await pilot.pause(0.4)
            assert str(grip.label) == "--->"
            assert grip.region.width == PANE_GRIP_WIDTH
            await pilot.press("space")
            await pilot.pause(0.4)
            assert str(grip.label) == "<---"
            assert grip.region.width == PANE_GRIP_WIDTH


@pytest.mark.asyncio
async def test_reader_is_never_a_collapse_target():
    host = LibraryProductionCSSHarness(_build_media_test_app())

    async with host.run_test(size=(170, 48)) as pilot:
        _, shell = await _open_media_shell(host, pilot)
        reader = shell.query_one("#library-media-viewer", LibraryMediaViewer)

        shell.query_one("#library-media-library-grip", Button).press()
        shell.query_one("#library-media-items-grip", Button).press()
        await pilot.pause()

        assert reader.display
        assert reader.region.width > 0
        assert shell.content_region.contains_region(reader.region)
        assert not shell.query("#library-media-reader-grip")
        assert {grip.pane for grip in shell.query(".library-media-pane-grip")} == {
            "library",
            "items",
        }


@pytest.mark.asyncio
async def test_row_activation_keeps_items_mounted_and_loads_permanent_reader():
    app = _build_media_test_app()
    service = app.media_reading_scope_service
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=(170, 48)) as pilot:
        screen, shell = await _open_media_shell(host, pilot)
        items = shell.query_one("#library-media-canvas", LibraryMediaCanvas)

        items.query_one("#library-media-row-0", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-viewer-title")
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_reader_session.loaded_id is not None,
            message="Activated media detail never settled into Reader",
        )

        session = screen._library_media_reader_session
        assert shell.query_one("#library-media-canvas") is items
        assert service.detail_calls
        assert session.selected_id == session.loaded_id == "local:media:2"
        assert str(
            shell.query_one("#library-media-viewer-title").renderable
        ) == "Product Demo Video"


@pytest.mark.asyncio
async def test_non_media_library_routes_keep_the_existing_shell():
    host = LibraryProductionCSSHarness(_build_media_test_app())

    async with host.run_test(size=(170, 48)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        assert screen.query_one(
            "#library-rail-handle", LibraryNavigationRailHandle
        )
        assert screen.query_one("#library-canvas")
        assert not screen.query("#library-media-reader-shell")
        assert not screen.query(".library-media-pane-grip")


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(160, 50), (120, 35), (100, 30), (80, 24)])
async def test_media_shell_resize_uses_resolver_without_reads_or_recompose(size):
    app = _build_media_test_app()
    service = app.media_reading_scope_service
    service.progress_calls = []
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=(160, 50)) as pilot:
        screen, shell = await _open_media_shell(host, pilot)
        items = shell.query_one("#library-media-canvas", LibraryMediaCanvas)
        controller = screen._library_media_browse_controller
        calls = (
            len(service.search_calls),
            len(service.type_calls),
            len(service.detail_calls),
            len(service.progress_calls),
            len(service.update_calls),
            len(service.delete_calls),
        )
        scope = controller.applied_scope
        selected = screen._selected_media_id

        await pilot.resize_terminal(*size)
        await _wait_for_condition(
            pilot,
            lambda: shell.effective_layout == resolve_media_reader_layout(
                shell.region.width,
                screen._library_media_reader_preferences,
                previous=shell.effective_layout,
            ),
            message=f"Media shell did not settle at {size}",
        )
        await pilot.pause()

        assert shell.query_one("#library-media-canvas") is items
        assert controller.applied_scope == scope
        assert screen._selected_media_id == selected
        assert (
            len(service.search_calls),
            len(service.type_calls),
            len(service.detail_calls),
            len(service.progress_calls),
            len(service.update_calls),
            len(service.delete_calls),
        ) == calls
        expected = resolve_media_reader_layout(
            shell.region.width,
            MediaReaderLayoutPreferences(),
            previous=shell.effective_layout,
        )
        assert (
            shell.effective_layout.library_open,
            shell.effective_layout.items_open,
        ) == (
            expected.library_open,
            expected.items_open,
        )
        for grip in shell.query(".library-media-pane-grip"):
            assert grip.region.width == PANE_GRIP_WIDTH
            assert shell.content_region.contains_region(grip.region)
        reader = shell.query_one("#library-media-viewer")
        assert shell.content_region.contains_region(reader.region)
        assert reader.region.right <= shell.content_region.right


class _SixtyColumnMediaShellApp(ConsolidatedCSSApp):
    CSS_PATH = TldwCli.CSS_PATH

    def compose(self):
        # This direct host pins the Media shell's own allocation to the design
        # floor independently of application chrome.
        layout = resolve_media_reader_layout(60, MediaReaderLayoutPreferences())
        shell = LibraryMediaReaderShell(
            Horizontal(id="library-rail"),
            Horizontal(id="library-media-canvas"),
            LibraryMediaViewer(
                build_library_media_viewer_state(None),
                id="library-media-viewer",
            ),
            layout,
            id="library-media-reader-shell",
        )
        shell.styles.width = 60
        yield shell


@pytest.mark.asyncio
async def test_two_grips_leave_fifty_columns_for_reader_at_sixty_shell_columns():
    app = _SixtyColumnMediaShellApp()

    async with app.run_test(size=(60, 24)) as pilot:
        await pilot.pause()
        shell = app.query_one("#library-media-reader-shell", LibraryMediaReaderShell)
        reader = shell.query_one("#library-media-viewer", LibraryMediaViewer)

        assert shell.region.width == 60
        assert reader.region.width == 50
        assert reader.region.right <= shell.region.right
        assert sum(
            grip.region.width for grip in shell.query(".library-media-pane-grip")
        ) == 10
