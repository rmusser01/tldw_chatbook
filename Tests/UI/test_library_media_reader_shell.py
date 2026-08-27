"""Production-shaped geometry contracts for the permanent Media reader shell."""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import Mock

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
from tldw_chatbook.UI.Screens import library_screen as library_screen_module
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.Library import (
    AdaptiveReaderShellResized,
    LibraryAdaptiveReaderShell,
    LibraryMediaCanvas,
    LibraryMediaPaneGrip,
    LibraryMediaReaderShell,
    LibraryMediaViewer,
    LibraryNavigationRailHandle,
    MediaShellResized,
    PaneToggleRequested,
)
from tldw_chatbook.Widgets.Library.library_adaptive_reader_shell import (
    PaneToggleRequested as SharedPaneToggleRequested,
)
from tldw_chatbook.app import TldwCli


def test_media_grip_preserves_legacy_constructor_signature():
    grip = LibraryMediaPaneGrip("library", open=True, id="legacy-media-grip")

    assert grip.id == "legacy-media-grip"
    assert grip.has_class("library-media-pane-grip")
    assert grip.name == "Collapse Library pane"
    assert str(grip.tooltip) == "Collapse Library pane"


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


async def _open_media_shell(
    host, pilot
) -> tuple[LibraryScreen, LibraryMediaReaderShell]:
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
async def test_media_wrapper_preserves_shell_ids_classes_messages_and_aliases():
    host = LibraryProductionCSSHarness(_build_media_test_app())

    async with host.run_test(size=(170, 48)) as pilot:
        _, shell = await _open_media_shell(host, pilot)

        assert isinstance(shell, LibraryAdaptiveReaderShell)
        assert shell.id == "library-media-reader-shell"
        assert shell.reader is shell.work
        assert shell.library.id == "library-rail"
        assert shell.items.id == "library-canvas"
        assert shell.reader.id == "library-media-viewer"
        assert shell.library_grip.id == "library-media-library-grip"
        assert shell.items_grip.id == "library-media-items-grip"
        assert all(
            grip.has_class("library-media-pane-grip")
            for grip in (shell.library_grip, shell.items_grip)
        )
        assert PaneToggleRequested is SharedPaneToggleRequested
        assert MediaShellResized is AdaptiveReaderShellResized


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
async def test_compact_reader_keeps_every_toolbar_action_inside_reader():
    host = LibraryProductionCSSHarness(_build_media_test_app())

    async with host.run_test(size=(100, 30)) as pilot:
        screen, shell = await _open_media_shell(host, pilot)
        shell.items.query_one("#library-media-row-0", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._library_media_reader_session.pending_request is None
                and screen._library_media_reader_session.loaded_id is not None
            ),
            message="Compact Reader detail never settled.",
        )
        await pilot.pause()

        reader = shell.reader
        loaded_row = shell.items.query_one("#library-media-row-0", Button)
        assert "Loaded in Reader" in str(loaded_row.label)
        assert "loading preview" not in str(loaded_row.label)
        for selector in (
            "#library-media-reader-find",
            "#library-media-read-later",
            "#library-media-use-in-chat",
            "#library-media-reader-more",
            "#library-media-reader-select-read",
            "#library-media-reader-select-analysis",
            "#library-media-reader-select-highlights",
            "#library-media-reader-select-info",
        ):
            action = reader.query_one(selector, Button)
            assert action.region.width > 0
            assert reader.content_region.contains_region(action.region), (
                selector,
                action.region,
                reader.content_region,
            )


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
        assert (
            str(shell.query_one("#library-media-viewer-title").renderable)
            == "Product Demo Video"
        )


@pytest.mark.asyncio
async def test_non_media_library_routes_keep_the_existing_shell():
    host = LibraryProductionCSSHarness(_build_media_test_app())

    async with host.run_test(size=(170, 48)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        assert screen.query_one("#library-rail-handle", LibraryNavigationRailHandle)
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
            lambda: (
                shell.effective_layout
                == resolve_media_reader_layout(
                    shell.region.width,
                    screen._library_media_reader_preferences,
                    previous=shell.effective_layout,
                )
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
        assert (
            sum(grip.region.width for grip in shell.query(".library-media-pane-grip"))
            == 10
        )


@pytest.mark.asyncio
async def test_manual_grip_persists_preference_but_responsive_collapse_does_not(
    monkeypatch,
):
    app = _build_media_test_app()
    app.app_config["library"] = {
        "reader": {
            "library_open": False,
            "custom_widths_enabled": False,
            "library_width": 28,
            "future_shared": "keep",
        },
        "media_reader": {
            "library_open": "legacy-keep",
            "items_open": True,
            "items_width": 40,
            "future_media": "keep",
        },
    }
    writes = []

    def save_setting(section, key, value):
        writes.append((section, key, value))
        return True

    monkeypatch.setattr(
        library_screen_module, "save_setting_to_cli_config", save_setting
    )
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=(170, 48)) as pilot:
        screen, shell = await _open_media_shell(host, pilot)
        assert shell.effective_layout.library_open is False

        shell.library_grip.press()
        await _wait_for_condition(
            pilot,
            lambda: bool(writes),
            message="Manual pane preference was not persisted.",
        )
        assert writes == [("library.reader", "library_open", True)]
        assert app.app_config["library"]["reader"] == {
            "library_open": True,
            "custom_widths_enabled": False,
            "library_width": 28,
            "future_shared": "keep",
        }
        assert app.app_config["library"]["media_reader"]["library_open"] == (
            "legacy-keep"
        )

        shell.items_grip.press()
        await _wait_for_condition(
            pilot,
            lambda: len(writes) == 2,
            message="Media Items preference was not persisted.",
        )
        assert writes[-1] == ("library.media_reader", "items_open", False)
        assert app.app_config["library"]["media_reader"]["items_open"] is False

        await pilot.resize_terminal(80, 24)
        await pilot.pause()
        assert len(writes) == 2

    next_screen = LibraryScreen(app)
    assert next_screen._library_media_reader_preferences.library_open is True


@pytest.mark.asyncio
async def test_shared_library_pane_choice_round_trips_between_media_and_conversations(
    monkeypatch,
):
    app = _build_media_test_app()
    writes = []
    monkeypatch.setattr(
        library_screen_module,
        "save_setting_to_cli_config",
        lambda *args: writes.append(args) or True,
    )
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=(170, 48)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-conversations", Button).press()
        conversation_shell = await _wait_for_selector(
            screen, pilot, "#library-conversations-reader-shell"
        )

        conversation_shell.library_grip.press()
        await _wait_for_condition(
            pilot,
            lambda: len(writes) == 1,
            message="Conversations Library-pane choice was not persisted.",
        )
        assert not screen._library_conversation_reader_preferences.library_open
        assert not screen._library_media_reader_preferences.library_open

        screen.query_one("#library-row-browse-media", Button).press()
        media_shell = await _wait_for_selector(
            screen, pilot, "#library-media-reader-shell"
        )
        assert not media_shell.effective_layout.library_open

        media_shell.library_grip.press()
        await _wait_for_condition(
            pilot,
            lambda: len(writes) == 2,
            message="Media Library-pane choice was not persisted.",
        )
        assert screen._library_media_reader_preferences.library_open
        assert screen._library_conversation_reader_preferences.library_open

        screen.query_one("#library-row-browse-conversations", Button).press()
        conversation_shell = await _wait_for_selector(
            screen, pilot, "#library-conversations-reader-shell"
        )
        assert conversation_shell.effective_layout.library_open
        assert writes == [
            ("library.reader", "library_open", False),
            ("library.reader", "library_open", True),
        ]


@pytest.mark.asyncio
async def test_failed_conversations_library_pane_write_restores_shared_choice_and_warns(
    monkeypatch,
):
    app = _build_media_test_app()
    write_started = threading.Event()
    notices = []

    def fail_save(*_args):
        write_started.set()
        return False

    monkeypatch.setattr(library_screen_module, "save_setting_to_cli_config", fail_save)
    monkeypatch.setattr(
        app,
        "notify",
        lambda message, **kwargs: notices.append((message, kwargs)),
    )
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=(170, 48)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-conversations", Button).press()
        conversation_shell = await _wait_for_selector(
            screen, pilot, "#library-conversations-reader-shell"
        )

        conversation_shell.library_grip.press()
        await asyncio.to_thread(write_started.wait, 10)
        await _wait_for_condition(
            pilot,
            lambda: screen._library_conversation_reader_preferences.library_open,
            message="Failed Conversations pane write did not roll back.",
        )

        assert screen._library_media_reader_preferences.library_open
        assert notices[-1][1]["severity"] == "warning"
        assert "could not be saved" in notices[-1][0]


@pytest.mark.asyncio
async def test_failed_manual_grip_persistence_restores_previous_preference(
    monkeypatch,
):
    app = _build_media_test_app()
    app.app_config["library"] = {
        "reader": {"library_open": False},
        "media_reader": {"library_open": "legacy-keep", "items_open": True},
    }
    notices = []
    monkeypatch.setattr(
        library_screen_module,
        "save_setting_to_cli_config",
        lambda *_args: False,
    )
    monkeypatch.setattr(
        app,
        "notify",
        lambda message, **kwargs: notices.append((message, kwargs)),
    )
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=(170, 48)) as pilot:
        screen, shell = await _open_media_shell(host, pilot)
        shell.library_grip.press()
        await _wait_for_condition(
            pilot,
            lambda: bool(notices),
            message="Failed pane persistence did not report or restore.",
        )

        assert screen._library_media_reader_preferences.library_open is False
        assert app.app_config["library"]["reader"]["library_open"] is False
        assert (
            app.app_config["library"]["media_reader"]["library_open"] == "legacy-keep"
        )
        assert shell.effective_layout.library_open is False
        assert notices[-1][1]["severity"] == "warning"


@pytest.mark.asyncio
async def test_rapid_manual_grip_changes_persist_in_order(monkeypatch):
    app = _build_media_test_app()
    app.app_config["library"] = {
        "reader": {"library_open": False},
        "media_reader": {"library_open": "legacy-keep", "items_open": True},
    }
    writes = []
    first_started = threading.Event()
    release_first = threading.Event()

    def save_setting(section, key, value):
        writes.append((section, key, value))
        if len(writes) == 1:
            first_started.set()
            release_first.wait(timeout=3)
        return True

    monkeypatch.setattr(
        library_screen_module, "save_setting_to_cli_config", save_setting
    )
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=(170, 48)) as pilot:
        _, shell = await _open_media_shell(host, pilot)
        shell.library_grip.press()
        await _wait_for_condition(
            pilot,
            first_started.is_set,
            message="First pane preference write never started.",
        )
        shell.library_grip.press()
        release_first.set()
        await _wait_for_condition(
            pilot,
            lambda: len(writes) == 2,
            message="Newer pane preference was not serialized after the first.",
        )

        assert writes == [
            ("library.reader", "library_open", True),
            ("library.reader", "library_open", False),
        ]
        assert app.app_config["library"]["reader"]["library_open"] is False
        assert (
            app.app_config["library"]["media_reader"]["library_open"] == "legacy-keep"
        )


@pytest.mark.asyncio
async def test_shared_library_pane_writes_settle_latest_across_destinations(
    monkeypatch,
):
    app = _build_media_test_app()
    disk = {"library_open": True}
    writes = []
    older_started = threading.Event()
    release_older = threading.Event()
    newer_started = threading.Event()

    def save_setting(section, key, value):
        if value is False:
            older_started.set()
            release_older.wait(timeout=10)
        else:
            newer_started.set()
        disk[key] = value
        writes.append((section, key, value))
        return True

    monkeypatch.setattr(
        library_screen_module, "save_setting_to_cli_config", save_setting
    )
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=(170, 48)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-conversations", Button).press()
        conversations = await _wait_for_selector(
            screen, pilot, "#library-conversations-reader-shell"
        )
        conversations.library_grip.press()
        await asyncio.to_thread(older_started.wait, 10)

        screen.query_one("#library-row-browse-media", Button).press()
        media = await _wait_for_selector(screen, pilot, "#library-media-reader-shell")
        assert not media.effective_layout.library_open
        media.library_grip.press()
        try:
            await _wait_for_condition(
                pilot,
                lambda: screen._library_reader_persistence_generations["library"] == 2,
                message="Newer shared Library-pane intent was not claimed.",
            )
            await pilot.pause()
            assert not newer_started.is_set()
        finally:
            release_older.set()
        await screen.workers.wait_for_complete()

        assert newer_started.is_set()
        assert disk["library_open"] is True, writes
        assert set(writes) == {
            ("library.reader", "library_open", False),
            ("library.reader", "library_open", True),
        }


@pytest.mark.asyncio
async def test_shared_library_pane_double_failure_restores_durable_choice(
    monkeypatch,
):
    app = _build_media_test_app()
    writes = []
    first_started = threading.Event()
    release_first = threading.Event()

    def fail_save(section, key, value):
        writes.append((section, key, value))
        if len(writes) == 1:
            first_started.set()
            release_first.wait(timeout=10)
        return False

    monkeypatch.setattr(library_screen_module, "save_setting_to_cli_config", fail_save)
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=(170, 48)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-conversations", Button).press()
        conversations = await _wait_for_selector(
            screen, pilot, "#library-conversations-reader-shell"
        )
        conversations.library_grip.press()
        await asyncio.to_thread(first_started.wait, 10)
        try:
            screen.query_one("#library-row-browse-media", Button).press()
            media = await _wait_for_selector(
                screen, pilot, "#library-media-reader-shell"
            )
            assert not media.effective_layout.library_open
            media.library_grip.press()
            await _wait_for_condition(
                pilot,
                lambda: screen._library_reader_persistence_generations["library"] == 2,
                message="Newer shared Library-pane intent was not claimed.",
            )
        finally:
            release_first.set()

        await screen.workers.wait_for_complete()
        await pilot.pause()
        assert writes == [
            ("library.reader", "library_open", False),
            ("library.reader", "library_open", True),
        ]
        assert screen._library_media_reader_preferences.library_open
        assert screen._library_conversation_reader_preferences.library_open
        assert app.app_config["library"]["reader"]["library_open"] is True
        assert media.effective_layout.library_open


@pytest.mark.parametrize(
    ("destination", "pane", "config_section", "preference_key", "authority"),
    (
        ("media", "library", "reader", "library_open", "library"),
        ("conversations", "library", "reader", "library_open", "library"),
        ("media", "items", "media_reader", "items_open", "media_items"),
        (
            "conversations",
            "items",
            "conversations_reader",
            "items_open",
            "conversations_items",
        ),
    ),
)
@pytest.mark.asyncio
async def test_settings_refresh_repairs_started_stale_pane_write(
    monkeypatch,
    destination,
    pane,
    config_section,
    preference_key,
    authority,
):
    app = _build_media_test_app()
    disk = {preference_key: True}
    writes = []
    stale_started = threading.Event()
    release_stale = threading.Event()
    expected_section = f"library.{config_section}"

    def save_setting(section, key, value):
        if (section, key) != (expected_section, preference_key):
            return True
        writes.append((section, key, value))
        if len(writes) == 1:
            stale_started.set()
            release_stale.wait(timeout=10)
        disk[key] = value
        return True

    monkeypatch.setattr(
        library_screen_module, "save_setting_to_cli_config", save_setting
    )
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=(170, 48)) as pilot:
        if destination == "media":
            screen, shell = await _open_media_shell(host, pilot)
        else:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            screen.query_one("#library-row-browse-conversations", Button).press()
            shell = await _wait_for_selector(
                screen, pilot, "#library-conversations-reader-shell"
            )
        getattr(shell, f"{pane}_grip").press()
        await asyncio.to_thread(stale_started.wait, 10)
        intent_generation = screen._library_reader_persistence_generations[authority]
        try:
            app.app_config["library"].setdefault(config_section, {})[preference_key] = (
                True
            )
            screen.request_library_reader_layout_refresh(
                screen._library_reader_layout_refresh_generation + 1
            )
            await pilot.pause()
            assert getattr(shell.effective_layout, preference_key)
        finally:
            release_stale.set()

        await screen.workers.wait_for_complete()
        await pilot.pause()
        assert writes == [
            (expected_section, preference_key, False),
            (expected_section, preference_key, True),
        ]
        assert disk[preference_key] is True
        assert screen._library_reader_persistence_generations[authority] > (
            intent_generation
        )
        preferences = (
            screen._library_conversation_reader_preferences
            if destination == "conversations"
            else screen._library_media_reader_preferences
        )
        assert getattr(preferences, preference_key)
        assert app.app_config["library"][config_section][preference_key] is True
        assert getattr(shell.effective_layout, preference_key)


@pytest.mark.asyncio
async def test_failed_settings_repair_rolls_back_to_physical_durable_value(
    monkeypatch,
):
    app = _build_media_test_app()
    disk = {"library_open": True}
    writes = []
    stale_started = threading.Event()
    release_stale = threading.Event()
    notices = []

    def save_setting(section, key, value):
        if key != "library_open":
            return True
        writes.append((section, key, value))
        if len(writes) == 1:
            stale_started.set()
            release_stale.wait(timeout=10)
            disk[key] = value
            return True
        return False

    monkeypatch.setattr(
        library_screen_module, "save_setting_to_cli_config", save_setting
    )
    monkeypatch.setattr(
        library_screen_module,
        "read_cli_config_serialized",
        lambda: (
            "[library.media_reader]\n"
            f"library_open = {str(disk['library_open']).lower()}\n"
        ),
    )
    monkeypatch.setattr(
        app,
        "notify",
        lambda message, **kwargs: notices.append((message, kwargs)),
    )
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=(170, 48)) as pilot:
        screen, shell = await _open_media_shell(host, pilot)
        shell.library_grip.press()
        await asyncio.to_thread(stale_started.wait, 10)
        try:
            app.app_config["library"]["reader"]["library_open"] = True
            screen.request_library_reader_layout_refresh(
                screen._library_reader_layout_refresh_generation + 1
            )
        finally:
            release_stale.set()

        await screen.workers.wait_for_complete()
        await pilot.pause()
        assert writes == [
            ("library.reader", "library_open", False),
            ("library.reader", "library_open", True),
        ]
        assert disk["library_open"] is False
        assert not screen._library_media_reader_preferences.library_open
        assert not screen._library_conversation_reader_preferences.library_open
        assert app.app_config["library"]["reader"]["library_open"] is False
        assert not shell.effective_layout.library_open
        assert notices and notices[-1][1]["severity"] == "warning"


@pytest.mark.asyncio
async def test_settings_repair_coalesces_newer_grip_intent(monkeypatch):
    app = _build_media_test_app()
    disk = {"library_open": True}
    writes = []
    stale_started = threading.Event()
    release_stale = threading.Event()
    repair_started = threading.Event()
    release_repair = threading.Event()

    def save_setting(section, key, value):
        if key != "library_open":
            return True
        writes.append((section, key, value))
        if len(writes) == 1:
            stale_started.set()
            release_stale.wait(timeout=10)
        elif len(writes) == 2:
            repair_started.set()
            release_repair.wait(timeout=10)
        disk[key] = value
        return True

    monkeypatch.setattr(
        library_screen_module, "save_setting_to_cli_config", save_setting
    )
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=(170, 48)) as pilot:
        screen, shell = await _open_media_shell(host, pilot)
        shell.library_grip.press()
        await asyncio.to_thread(stale_started.wait, 10)
        app.app_config["library"]["reader"]["library_open"] = True
        screen.request_library_reader_layout_refresh(
            screen._library_reader_layout_refresh_generation + 1
        )
        release_stale.set()
        await asyncio.to_thread(repair_started.wait, 10)
        try:
            shell.library_grip.press()
            await _wait_for_condition(
                pilot,
                lambda: screen._library_reader_persistence_generations["library"] == 3,
                message="Newer grip intent did not supersede the repair.",
            )
        finally:
            release_repair.set()

        await screen.workers.wait_for_complete()
        await pilot.pause()
        assert writes == [
            ("library.reader", "library_open", False),
            ("library.reader", "library_open", True),
            ("library.reader", "library_open", False),
        ]
        assert disk["library_open"] is False
        assert not screen._library_media_reader_preferences.library_open
        assert not screen._library_conversation_reader_preferences.library_open
        assert app.app_config["library"]["reader"]["library_open"] is False
        assert not shell.effective_layout.library_open


@pytest.mark.asyncio
async def test_delayed_settings_refresh_repairs_exited_stale_grip_write(monkeypatch):
    app = _build_media_test_app()
    disk = {"library_open": True}
    writes = []
    grip_started = threading.Event()
    release_grip = threading.Event()

    def save_setting(section, key, value):
        if key != "library_open":
            return True
        writes.append((section, key, value))
        if len(writes) == 1:
            grip_started.set()
            release_grip.wait(timeout=10)
        disk[key] = value
        return True

    monkeypatch.setattr(
        library_screen_module, "save_setting_to_cli_config", save_setting
    )
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=(170, 48)) as pilot:
        screen, shell = await _open_media_shell(host, pilot)
        shell.library_grip.press()
        await asyncio.to_thread(grip_started.wait, 10)

        disk["library_open"] = True
        app.app_config["library"]["reader"]["library_open"] = True
        release_grip.set()
        await screen.workers.wait_for_complete()
        assert disk["library_open"] is False
        assert "library" in screen._library_reader_dirty_persistence_authorities

        screen.request_library_reader_layout_refresh(
            screen._library_reader_layout_refresh_generation + 1
        )
        await screen.workers.wait_for_complete()
        await pilot.pause()

        assert writes == [
            ("library.reader", "library_open", False),
            ("library.reader", "library_open", True),
        ]
        assert disk["library_open"] is True
        assert screen._library_reader_durable_preferences["library"] is True
        assert screen._library_media_reader_preferences.library_open
        assert screen._library_conversation_reader_preferences.library_open
        assert app.app_config["library"]["reader"]["library_open"] is True
        assert shell.effective_layout.library_open


@pytest.mark.asyncio
async def test_clean_first_mounted_settings_refresh_starts_no_repair(monkeypatch):
    app = _build_media_test_app()
    disk = {"library_open": True}
    save_attempts = []
    snapshot_reads = 0

    def save_setting(section, key, value):
        save_attempts.append((section, key, value))
        return False

    def read_snapshot():
        nonlocal snapshot_reads
        snapshot_reads += 1
        return f"[library.reader]\nlibrary_open = {str(disk['library_open']).lower()}\n"

    monkeypatch.setattr(
        library_screen_module, "save_setting_to_cli_config", save_setting
    )
    monkeypatch.setattr(
        library_screen_module,
        "read_cli_config_serialized",
        read_snapshot,
    )
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=(170, 48)) as pilot:
        screen, shell = await _open_media_shell(host, pilot)
        workers = Mock(wraps=screen.run_worker)
        monkeypatch.setattr(screen, "run_worker", workers)
        app.app_config["library"]["reader"]["library_open"] = True
        screen.request_library_reader_layout_refresh(
            screen._library_reader_layout_refresh_generation + 1
        )
        await screen.workers.wait_for_complete()
        await pilot.pause()

        assert save_attempts == []
        assert workers.call_count == 0
        assert snapshot_reads == 0
        assert disk["library_open"] is True
        assert screen._library_reader_durable_preferences["library"] is True
        assert screen._library_media_reader_preferences.library_open
        assert screen._library_conversation_reader_preferences.library_open
        assert app.app_config["library"]["reader"]["library_open"] is True
        assert shell.effective_layout.library_open


@pytest.mark.asyncio
async def test_completed_manual_write_yields_to_later_settings_without_repair(
    monkeypatch,
):
    app = _build_media_test_app()
    writes = []

    def save_setting(section, key, value):
        writes.append((section, key, value))
        return True

    monkeypatch.setattr(
        library_screen_module, "save_setting_to_cli_config", save_setting
    )
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=(170, 48)) as pilot:
        screen, shell = await _open_media_shell(host, pilot)
        shell.library_grip.press()
        await screen.workers.wait_for_complete()
        await pilot.pause()
        assert writes == [("library.reader", "library_open", False)]

        workers = Mock(wraps=screen.run_worker)
        monkeypatch.setattr(screen, "run_worker", workers)
        app.app_config["library"]["reader"]["library_open"] = True
        screen.request_library_reader_layout_refresh(
            screen._library_reader_layout_refresh_generation + 1
        )
        await screen.workers.wait_for_complete()
        await pilot.pause()

        assert writes == [("library.reader", "library_open", False)]
        assert workers.call_count == 0
        assert screen._library_media_reader_preferences.library_open
        assert shell.effective_layout.library_open


@pytest.mark.asyncio
async def test_failed_manual_write_gets_one_bounded_settings_repair(monkeypatch):
    app = _build_media_test_app()
    writes = []

    def save_setting(section, key, value):
        writes.append((section, key, value))
        return len(writes) > 1

    monkeypatch.setattr(
        library_screen_module, "save_setting_to_cli_config", save_setting
    )
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=(170, 48)) as pilot:
        screen, shell = await _open_media_shell(host, pilot)
        shell.library_grip.press()
        await screen.workers.wait_for_complete()
        await pilot.pause()
        assert writes == [("library.reader", "library_open", False)]
        assert screen._library_media_reader_preferences.library_open

        workers = Mock(wraps=screen.run_worker)
        monkeypatch.setattr(screen, "run_worker", workers)
        screen.request_library_reader_layout_refresh(
            screen._library_reader_layout_refresh_generation + 1
        )
        await screen.workers.wait_for_complete()
        await pilot.pause()

        assert writes == [
            ("library.reader", "library_open", False),
            ("library.reader", "library_open", True),
        ]
        assert workers.call_count == 1
        assert shell.effective_layout.library_open

        workers.reset_mock()
        screen.request_library_reader_layout_refresh(
            screen._library_reader_layout_refresh_generation + 1
        )
        await screen.workers.wait_for_complete()
        assert len(writes) == 2
        assert workers.call_count == 0


@pytest.mark.asyncio
async def test_failed_settings_reconciliation_does_not_project_cached_guess(
    monkeypatch,
):
    app = _build_media_test_app()
    cache_reads = 0
    save_attempts = []

    def save_setting(section, key, value):
        save_attempts.append((section, key, value))
        return False

    def fail_snapshot():
        raise ValueError("physical config unavailable")

    def read_cached_setting(*_args):
        nonlocal cache_reads
        cache_reads += 1
        return False

    monkeypatch.setattr(
        library_screen_module, "save_setting_to_cli_config", save_setting
    )
    monkeypatch.setattr(
        library_screen_module, "read_cli_config_serialized", fail_snapshot
    )
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=(170, 48)) as pilot:
        screen, shell = await _open_media_shell(host, pilot)
        monkeypatch.setattr(
            library_screen_module, "get_cli_setting", read_cached_setting
        )
        shell.library_grip.press()
        await screen.workers.wait_for_complete()
        await pilot.pause()
        assert save_attempts == [("library.reader", "library_open", False)]

        screen.request_library_reader_layout_refresh(
            screen._library_reader_layout_refresh_generation + 1
        )
        await screen.workers.wait_for_complete()
        await pilot.pause()

        assert cache_reads == 0
        assert save_attempts == [
            ("library.reader", "library_open", False),
            ("library.reader", "library_open", True),
        ]
        assert "library" in screen._library_reader_dirty_persistence_authorities
        assert screen._library_reader_durable_preferences["library"] is True
        assert screen._library_media_reader_preferences.library_open
        assert screen._library_conversation_reader_preferences.library_open
        assert app.app_config["library"]["reader"]["library_open"] is True
        assert shell.effective_layout.library_open


@pytest.mark.asyncio
async def test_persisted_shared_library_read_honors_real_legacy_config(
    monkeypatch, tmp_path
):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[library.media_reader]\nlibrary_open = false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    physical_value = await LibraryScreen._read_library_reader_persisted_preference(
        object(), "library.reader", "library_open"
    )

    assert physical_value is False


@pytest.mark.asyncio
async def test_persisted_items_read_uses_default_when_key_is_absent(
    monkeypatch, tmp_path
):
    config_path = tmp_path / "config.toml"
    config_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    physical_value = await LibraryScreen._read_library_reader_persisted_preference(
        object(), "library.media_reader", "items_open"
    )

    assert physical_value is True


@pytest.mark.asyncio
async def test_failed_delayed_settings_repair_projects_stale_disk_truth(monkeypatch):
    app = _build_media_test_app()
    disk = {"library_open": True}
    library_writes = []
    grip_started = threading.Event()
    release_grip = threading.Event()
    notices = []

    def save_setting(section, key, value):
        if key != "library_open":
            return True
        library_writes.append((section, key, value))
        if len(library_writes) == 1:
            grip_started.set()
            release_grip.wait(timeout=10)
            disk[key] = value
            return True
        return False

    monkeypatch.setattr(
        library_screen_module, "save_setting_to_cli_config", save_setting
    )
    monkeypatch.setattr(
        library_screen_module,
        "read_cli_config_serialized",
        lambda: (
            f"[library.reader]\nlibrary_open = {str(disk['library_open']).lower()}\n"
        ),
    )
    monkeypatch.setattr(
        app,
        "notify",
        lambda message, **kwargs: notices.append((message, kwargs)),
    )
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=(170, 48)) as pilot:
        screen, shell = await _open_media_shell(host, pilot)
        shell.library_grip.press()
        await asyncio.to_thread(grip_started.wait, 10)
        disk["library_open"] = True
        app.app_config["library"]["reader"]["library_open"] = True
        release_grip.set()
        await screen.workers.wait_for_complete()

        screen.request_library_reader_layout_refresh(
            screen._library_reader_layout_refresh_generation + 1
        )
        await screen.workers.wait_for_complete()
        await pilot.pause()

        assert library_writes == [
            ("library.reader", "library_open", False),
            ("library.reader", "library_open", True),
        ]
        assert disk["library_open"] is False
        assert screen._library_reader_durable_preferences["library"] is False
        assert not screen._library_media_reader_preferences.library_open
        assert not screen._library_conversation_reader_preferences.library_open
        assert app.app_config["library"]["reader"]["library_open"] is False
        assert not shell.effective_layout.library_open
        assert notices and notices[-1][1]["severity"] == "warning"


@pytest.mark.asyncio
async def test_failed_library_pane_write_resyncs_mounted_peer_shell(monkeypatch):
    app = _build_media_test_app()
    write_started = threading.Event()
    release_write = threading.Event()
    notices = []

    def fail_save(*_args):
        write_started.set()
        release_write.wait(timeout=10)
        return False

    monkeypatch.setattr(library_screen_module, "save_setting_to_cli_config", fail_save)
    monkeypatch.setattr(
        app,
        "notify",
        lambda message, **kwargs: notices.append((message, kwargs)),
    )
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=(170, 48)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-conversations", Button).press()
        conversations = await _wait_for_selector(
            screen, pilot, "#library-conversations-reader-shell"
        )
        conversations.library_grip.press()
        await asyncio.to_thread(write_started.wait, 10)
        try:
            screen.query_one("#library-row-browse-media", Button).press()
            media = await _wait_for_selector(
                screen, pilot, "#library-media-reader-shell"
            )
            assert not media.effective_layout.library_open
        finally:
            release_write.set()

        await _wait_for_condition(
            pilot,
            lambda: bool(notices),
            message="Failed pane write did not report or restore.",
        )
        await pilot.pause()
        assert screen._library_media_reader_preferences.library_open
        assert media.effective_layout.library_open


@pytest.mark.asyncio
async def test_settings_refresh_reconciles_panes_without_media_reads(
    monkeypatch,
):
    app = _build_media_test_app()
    host = LibraryProductionCSSHarness(app)
    writes = []
    monkeypatch.setattr(
        library_screen_module,
        "save_setting_to_cli_config",
        lambda *args: writes.append(args) or True,
    )

    async with host.run_test(size=(170, 48)) as pilot:
        screen, shell = await _open_media_shell(host, pilot)
        service = app.media_reading_scope_service
        reads = (len(service.search_calls), len(service.detail_calls))
        app.app_config["library"] = {
            "reader": {
                "library_open": False,
                "custom_widths_enabled": True,
                "library_width": 36,
            },
            "media_reader": {
                "items_open": False,
                "items_width": 56,
            },
        }

        screen.request_library_reader_layout_refresh(1)
        await screen.workers.wait_for_complete()
        await pilot.pause()

        assert screen.query_one("#library-media-reader-shell") is shell
        assert screen._library_media_reader_preferences == MediaReaderLayoutPreferences(
            library_open=False,
            items_open=False,
            custom_widths_enabled=True,
            library_width=36,
            items_width=56,
        )
        assert not shell.effective_layout.library_open
        assert not shell.effective_layout.items_open
        assert (len(service.search_calls), len(service.detail_calls)) == reads
        assert writes == []

        app.app_config["library"]["media_reader"]["items_open"] = True
        screen.request_library_media_layout_refresh(2)
        await screen.workers.wait_for_complete()
        await pilot.pause()
        assert shell.effective_layout.items_open
        assert (len(service.search_calls), len(service.detail_calls)) == reads
        assert writes == []
