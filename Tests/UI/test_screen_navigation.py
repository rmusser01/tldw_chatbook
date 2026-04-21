"""Focused screen wiring tests for screen-navigation mode."""

from types import SimpleNamespace

import pytest
from textual import on
from textual.app import App
from textual.widgets import Button
from unittest.mock import MagicMock, patch

from tldw_chatbook.app import TldwCli
from tldw_chatbook.Media import (
    LocalMediaReadingService,
    MediaReadingScopeService,
    ServerMediaReadingService,
)
from tldw_chatbook.Constants import ALL_TABS
from tldw_chatbook.UI.Navigation.base_app_screen import BaseAppScreen
from tldw_chatbook.UI.Navigation.main_navigation import MainNavigationBar
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Screens.media_ingest_screen import MediaIngestScreen
from tldw_chatbook.UI.Screens.media_screen import MediaScreen


def _build_test_app() -> TldwCli:
    with patch("tldw_chatbook.app.load_settings", return_value={"tldw_api": {"base_url": "http://localhost:8000"}}):
        with patch("tldw_chatbook.app.get_cli_setting", side_effect=lambda _section, _key, default=None: default):
            with patch("tldw_chatbook.app.get_chachanotes_db_lazy", return_value=None):
                with patch("tldw_chatbook.app.ServerNotesWorkspaceService.from_config", return_value=MagicMock()):
                    with patch("tldw_chatbook.app.ServerCharacterPersonaService.from_config", return_value=MagicMock()):
                        with patch.object(TldwCli, "_init_notes_service", lambda self, _user: setattr(self, "notes_service", None)):
                            with patch.object(TldwCli, "_init_prompts_service", lambda self: setattr(self, "prompts_service_initialized", False)):
                                with patch.object(TldwCli, "_init_providers_models", lambda self: setattr(self, "providers_models", {})):
                                    with patch.object(TldwCli, "_init_media_db", lambda self: (setattr(self, "media_db", None), setattr(self, "_media_types_for_ui", ["All Media"]))):
                                        return TldwCli()


def test_app_uses_screen_navigation_and_wires_media_services():
    app = _build_test_app()

    assert app._use_screen_navigation is True
    assert isinstance(app.local_media_reading_service, LocalMediaReadingService)
    assert isinstance(app.server_media_reading_service, ServerMediaReadingService)
    assert isinstance(app.media_reading_scope_service, MediaReadingScopeService)
    assert app.media_runtime_state.runtime_backend == "local"


def test_media_screen_uses_shared_runtime_state():
    app = _build_test_app()
    screen = MediaScreen(app)

    widgets = list(screen.compose_content())

    assert len(widgets) == 1
    assert screen.media_runtime_state is app.media_runtime_state
    assert screen.media_window is widgets[0]
    assert screen.media_window.runtime_state is app.media_runtime_state


def test_media_ingest_screen_uses_shared_runtime_state():
    app = _build_test_app()
    screen = MediaIngestScreen(app)

    widgets = list(screen.compose_content())

    assert len(widgets) == 1
    assert screen.media_runtime_state is app.media_runtime_state
    assert screen.media_ingest_window is widgets[0]
    assert screen.media_ingest_window.runtime_state is app.media_runtime_state


@pytest.mark.asyncio
async def test_tab_links_emit_navigation_messages():
    from tldw_chatbook.UI.Tab_Links import TabLinks

    messages_received = []

    class TestApp(App):
        def compose(self):
            yield TabLinks(tab_ids=ALL_TABS, initial_active_tab="chat")

        @on(NavigateToScreen)
        def capture_navigation(self, message: NavigateToScreen) -> None:
            messages_received.append(message)

    app = TestApp()

    async with app.run_test() as pilot:
        tab_links = pilot.app.query_one(TabLinks)
        notes_link = tab_links.query_one("#tab-link-notes")

        original_get_widget_at = tab_links.app.get_widget_at
        tab_links.app.get_widget_at = lambda _x, _y: (notes_link, None)
        try:
            await tab_links.on_click(SimpleNamespace(screen_x=0, screen_y=0))
            await pilot.pause(0.05)
        finally:
            tab_links.app.get_widget_at = original_get_widget_at

    assert len(messages_received) == 1
    assert messages_received[0].screen_name == "notes"


def test_screen_state_preservation():
    class TestScreen(BaseAppScreen):
        def __init__(self, app_instance):
            super().__init__(app_instance, "test")
            self.state_data = {"value": "saved"}

    app = _build_test_app()
    original = TestScreen(app)
    state = original.save_state()

    restored = TestScreen(app)
    restored.restore_state(state)

    assert restored.state_data == {"value": "saved"}


def test_screen_lifecycle_methods():
    class TestScreen(BaseAppScreen):
        def __init__(self, app_instance):
            super().__init__(app_instance, "test")
            self.mount_called = False

        def on_mount(self) -> None:
            self.mount_called = True
            super().on_mount()

    screen = TestScreen(_build_test_app())
    screen.on_mount()

    assert screen.mount_called is True


@pytest.mark.asyncio
async def test_main_navigation_copy_and_order():
    expected_button_order = [
        ("nav-chat", "Chat"),
        ("nav-chatbooks", "Chatbooks"),
        ("nav-notes", "Notes"),
        ("nav-media", "Media"),
        ("nav-ingest", "Ingest"),
        ("nav-search", "Search"),
        ("nav-subscriptions", "Subscriptions"),
        ("nav-ccp", "Library"),
        ("nav-study", "Study"),
        ("nav-llm", "LLM"),
        ("nav-stts", "S/TT/S"),
        ("nav-evals", "Evals"),
        ("nav-tools_settings", "Settings"),
        ("nav-customize", "Customize"),
        ("nav-logs", "Logs"),
        ("nav-stats", "Stats"),
        ("nav-coding", "Coding"),
    ]

    class TestApp(App):
        def compose(self):
            yield MainNavigationBar(active="chat")

    app = TestApp()

    async with app.run_test(size=(160, 20)) as pilot:
        await pilot.pause(0.1)

        nav_buttons = list(app.query(".nav-button"))
        actual_button_order = [(button.id, str(button.label).strip()) for button in nav_buttons]

        assert actual_button_order == expected_button_order
        assert str(app.query_one("#nav-ccp", Button).label).strip() == "Library"
        assert nav_buttons[0].id == "nav-chat"
        assert nav_buttons[1].id == "nav-chatbooks"
        assert nav_buttons[-1].id == "nav-coding"


@pytest.mark.asyncio
async def test_main_navigation_route_ids_remain_intact():
    class TestApp(App):
        def compose(self):
            yield MainNavigationBar(active="chat")

    app = TestApp()

    async with app.run_test(size=(160, 20)) as pilot:
        await pilot.pause(0.1)

        expected_route_ids = {
            "nav-chat",
            "nav-chatbooks",
            "nav-notes",
            "nav-media",
            "nav-ingest",
            "nav-search",
            "nav-subscriptions",
            "nav-ccp",
            "nav-study",
            "nav-llm",
            "nav-stts",
            "nav-evals",
            "nav-tools_settings",
            "nav-customize",
            "nav-logs",
            "nav-stats",
            "nav-coding",
        }

        actual_route_ids = [button.id for button in app.query(".nav-button")]

        assert actual_route_ids == [
            "nav-chat",
            "nav-chatbooks",
            "nav-notes",
            "nav-media",
            "nav-ingest",
            "nav-search",
            "nav-subscriptions",
            "nav-ccp",
            "nav-study",
            "nav-llm",
            "nav-stts",
            "nav-evals",
            "nav-tools_settings",
            "nav-customize",
            "nav-logs",
            "nav-stats",
            "nav-coding",
        ]
        assert set(actual_route_ids) == expected_route_ids


@pytest.mark.asyncio
async def test_screen_navigation_routes_reach_real_app_handler():
    app = _build_test_app()
    captured_destinations = []

    async def fake_switch_screen(screen):
        captured_destinations.append(type(screen).__name__)

    app.switch_screen = fake_switch_screen

    cases = [
        ("chatbooks", "ChatbooksScreen"),
        ("subscriptions", "SubscriptionScreen"),
        ("study", "StudyScreen"),
        ("stts", "STTSScreen"),
    ]

    async with app.run_test(size=(160, 40)) as pilot:
        await pilot.pause(0.1)

        for route, expected_screen_class in cases:
            captured_destinations.clear()

            await app.handle_screen_navigation(NavigateToScreen(route))
            await pilot.pause(0.05)

            assert app.current_tab == route
            assert captured_destinations == [expected_screen_class]
