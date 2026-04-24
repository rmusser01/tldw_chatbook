"""Focused screen wiring tests for screen-navigation mode."""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from textual import on
from textual.app import App
from unittest.mock import MagicMock, patch

from tldw_chatbook.app import TldwCli
from tldw_chatbook.Chat.chat_conversation_scope_service import ChatConversationScopeService
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.chat_loop_scope_service import ServerChatLoopScopeService
from tldw_chatbook.Chat.server_chat_conversation_service import ServerChatConversationService
from tldw_chatbook.Chat.server_chat_loop_service import ServerChatLoopService
from tldw_chatbook.Media import (
    LocalMediaReadingService,
    MediaReadingScopeService,
    ServerMediaReadingService,
)
from tldw_chatbook.Notifications.client_notifications_db import ClientNotificationsDB
from tldw_chatbook.Outputs import ServerOutputsScopeService, ServerOutputsService
from tldw_chatbook.Sharing import ServerSharingScopeService, ServerSharingService
from tldw_chatbook.Subscriptions.local_watchlists_service import LocalWatchlistsService
from tldw_chatbook.Subscriptions.server_watchlists_service import ServerWatchlistsService
from tldw_chatbook.Subscriptions.watchlist_scope_service import WatchlistScopeService
from tldw_chatbook.Constants import ALL_TABS, TAB_WRITING
from tldw_chatbook.Constants import TAB_RESEARCH
from tldw_chatbook.DB.Research_DB import ResearchDatabase
from tldw_chatbook.DB.Writing_DB import WritingDatabase
from tldw_chatbook.Research_Interop import (
    LocalResearchService,
    ResearchScopeService,
    ServerResearchService,
)
from tldw_chatbook.UI.Navigation.base_app_screen import BaseAppScreen
from tldw_chatbook.UI.Navigation.main_navigation import MainNavigationBar, NavigateToScreen
from tldw_chatbook.UI.Screens.media_ingest_screen import MediaIngestScreen
from tldw_chatbook.UI.Screens.media_screen import MediaScreen
from tldw_chatbook.UI.Screens.research_screen import ResearchScreen
from tldw_chatbook.UI.Screens.writing_screen import WritingScreen
from tldw_chatbook.Writing_Interop import (
    LocalWritingService,
    ServerWritingService,
    WritingScopeService,
)


def _build_test_app() -> TldwCli:
    temp_dir = Path(tempfile.mkdtemp())
    notifications_db_path = temp_dir / "notifications.sqlite3"
    subscriptions_db_path = temp_dir / "subscriptions.sqlite3"
    writing_db_path = temp_dir / "writing.sqlite3"
    research_db_path = temp_dir / "research.sqlite3"

    with patch("tldw_chatbook.app.load_settings", return_value={"tldw_api": {"base_url": "http://localhost:8000"}}):
        with patch("tldw_chatbook.app.get_cli_setting", side_effect=lambda _section, _key, default=None: default):
            with patch("tldw_chatbook.app.load_runtime_policy_for_app", return_value=SimpleNamespace(state=None)):
                with patch("tldw_chatbook.app.get_notifications_db_path", return_value=notifications_db_path, create=True):
                        with patch("tldw_chatbook.app.get_subscriptions_db_path", return_value=subscriptions_db_path, create=True):
                            with patch("tldw_chatbook.app.get_writing_db_path", return_value=writing_db_path, create=True):
                                with patch("tldw_chatbook.app.get_research_db_path", return_value=research_db_path, create=True):
                                    with patch("tldw_chatbook.app.get_chachanotes_db_lazy", return_value=None):
                                        with patch("tldw_chatbook.app.ServerNotesWorkspaceService.from_config", return_value=MagicMock()):
                                            with patch("tldw_chatbook.app.ServerCharacterPersonaService.from_config", return_value=MagicMock()):
                                                with patch("tldw_chatbook.app.NotesScopeService", return_value=MagicMock()):
                                                    with patch("tldw_chatbook.app.RAGAdminScopeService", return_value=MagicMock()):
                                                        with patch.object(TldwCli, "_wire_evaluation_services", lambda self: None):
                                                            with patch.object(TldwCli, "_wire_study_services", lambda self: None):
                                                                with patch.object(TldwCli, "_wire_character_persona_services", lambda self: None):
                                                                    with patch.object(TldwCli, "_init_notes_service", lambda self, _user: setattr(self, "notes_service", None)):
                                                                        with patch.object(TldwCli, "_init_prompts_service", lambda self: setattr(self, "prompts_service_initialized", False)):
                                                                            with patch.object(TldwCli, "_init_providers_models", lambda self: setattr(self, "providers_models", {})):
                                                                                with patch.object(TldwCli, "_init_media_db", lambda self: (setattr(self, "media_db", None), setattr(self, "_media_types_for_ui", ["All Media"]))):
                                                                                    return TldwCli()


@pytest.fixture
def app() -> TldwCli:
    return _build_test_app()


def test_app_uses_screen_navigation_and_wires_media_services():
    app = _build_test_app()

    assert app._use_screen_navigation is True
    assert isinstance(app.local_media_reading_service, LocalMediaReadingService)
    assert isinstance(app.server_media_reading_service, ServerMediaReadingService)
    assert isinstance(app.media_reading_scope_service, MediaReadingScopeService)
    assert app.media_runtime_state.runtime_backend == "local"


def test_app_initializes_watchlists_and_notifications_services(app):
    assert isinstance(app.local_watchlists_service, LocalWatchlistsService)
    assert isinstance(app.server_watchlists_service, ServerWatchlistsService)
    assert isinstance(app.watchlist_scope_service, WatchlistScopeService)
    assert app.watchlist_scope_service.local_service is app.local_watchlists_service
    assert app.watchlist_scope_service.server_service is app.server_watchlists_service
    assert app.watchlist_scope_service.policy_enforcer is app.service_policy_enforcer
    assert isinstance(app.client_notifications_db, ClientNotificationsDB)
    assert app.notification_dispatch_service.store is app.client_notifications_db
    assert app.local_media_reading_service.notification_dispatch_service is app.notification_dispatch_service
    assert app.local_media_reading_service.notification_app is app


def test_app_initializes_server_sharing_services(app):
    assert isinstance(app.server_sharing_service, ServerSharingService)
    assert isinstance(app.server_sharing_scope_service, ServerSharingScopeService)
    assert app.server_sharing_scope_service.server_service is app.server_sharing_service
    assert app.server_sharing_scope_service.policy_enforcer is app.service_policy_enforcer


def test_app_initializes_server_outputs_services(app):
    assert isinstance(app.server_outputs_service, ServerOutputsService)
    assert isinstance(app.server_outputs_scope_service, ServerOutputsScopeService)
    assert app.server_outputs_scope_service.server_service is app.server_outputs_service
    assert app.server_outputs_scope_service.policy_enforcer is app.service_policy_enforcer


def test_app_initializes_chat_conversation_scope_services(app):
    assert isinstance(app.local_chat_conversation_service, ChatConversationService)
    assert isinstance(app.server_chat_conversation_service, ServerChatConversationService)
    assert isinstance(app.chat_conversation_scope_service, ChatConversationScopeService)
    assert app.local_chat_conversation_service.db is app.chachanotes_db
    assert app.chat_conversation_scope_service.local_service is app.local_chat_conversation_service
    assert app.chat_conversation_scope_service.server_service is app.server_chat_conversation_service
    assert app.chat_conversation_scope_service.policy_enforcer is app.service_policy_enforcer


def test_app_initializes_server_chat_loop_services(app):
    assert isinstance(app.server_chat_loop_service, ServerChatLoopService)
    assert isinstance(app.server_chat_loop_scope_service, ServerChatLoopScopeService)
    assert app.server_chat_loop_scope_service.server_service is app.server_chat_loop_service
    assert app.server_chat_loop_scope_service.policy_enforcer is app.service_policy_enforcer


def test_writing_tab_and_screen_are_registered(app):
    screen_name, current_tab, screen_class = app._resolve_screen_navigation_target(TAB_WRITING)

    assert TAB_WRITING in ALL_TABS
    assert screen_name == TAB_WRITING
    assert current_tab == TAB_WRITING
    assert screen_class is WritingScreen


def test_research_tab_and_screen_are_registered(app):
    screen_name, current_tab, screen_class = app._resolve_screen_navigation_target(TAB_RESEARCH)

    assert TAB_RESEARCH in ALL_TABS
    assert screen_name == TAB_RESEARCH
    assert current_tab == TAB_RESEARCH
    assert screen_class is ResearchScreen


def test_main_navigation_bar_includes_writing_item():
    navigation = MainNavigationBar(active=TAB_WRITING)

    assert (TAB_WRITING, "Writing") in navigation.nav_items


def test_main_navigation_bar_includes_research_item():
    navigation = MainNavigationBar(active=TAB_RESEARCH)

    assert (TAB_RESEARCH, "Research") in navigation.nav_items


def test_app_initializes_writing_services(app):
    assert isinstance(app.writing_db, WritingDatabase)
    assert isinstance(app.local_writing_service, LocalWritingService)
    assert isinstance(app.server_writing_service, ServerWritingService)
    assert isinstance(app.writing_scope_service, WritingScopeService)
    assert app.local_writing_service.db is app.writing_db
    assert app.writing_scope_service.local_service is app.local_writing_service
    assert app.writing_scope_service.server_service is app.server_writing_service
    assert app.writing_scope_service.policy_enforcer is app.service_policy_enforcer


def test_app_initializes_research_services(app):
    assert isinstance(app.research_db, ResearchDatabase)
    assert isinstance(app.local_research_service, LocalResearchService)
    assert isinstance(app.server_research_service, ServerResearchService)
    assert isinstance(app.research_scope_service, ResearchScopeService)
    assert app.local_research_service.db is app.research_db
    assert app.research_scope_service.local_service is app.local_research_service
    assert app.research_scope_service.server_service is app.server_research_service
    assert app.research_scope_service.policy_enforcer is app.service_policy_enforcer


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
