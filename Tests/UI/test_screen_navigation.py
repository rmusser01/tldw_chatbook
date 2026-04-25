"""Focused screen wiring tests for screen-navigation mode."""

from types import SimpleNamespace

import pytest
from textual import on
from textual.app import App
from unittest.mock import MagicMock, patch

from tldw_chatbook.app import TldwCli
from tldw_chatbook.Chat import (
    ChatConversationScopeService,
    ServerChatConversationService,
)
from tldw_chatbook.Character_Chat.chat_dictionary_scope_service import ChatDictionaryScopeService
from tldw_chatbook.Character_Chat.local_chat_dictionary_service import LocalChatDictionaryService
from tldw_chatbook.Character_Chat.server_chat_dictionary_service import ServerChatDictionaryService
from tldw_chatbook.Media import (
    LocalMediaReadingService,
    MediaReadingScopeService,
    ServerMediaReadingService,
)
from tldw_chatbook.Notifications import (
    ClientNotificationsDB,
    ClientNotificationsService,
    NotificationDispatchService,
    NotificationsScopeService,
    ServerNotificationsService,
)
from tldw_chatbook.Outputs_Interop import OutputsScopeService, ServerOutputsService
from tldw_chatbook.Prompt_Management import (
    LocalPromptService,
    PromptChatbookScopeService,
    ServerPromptService,
)
from tldw_chatbook.Research_Interop import (
    LocalResearchSearchService,
    LocalResearchService,
    ResearchSearchScopeService,
    ResearchScopeService,
    ServerResearchSearchService,
    ServerResearchService,
)
from tldw_chatbook.Chatbooks import LocalChatbookService, ServerChatbookService
from tldw_chatbook.Chat_Grammars_Interop import ChatGrammarsScopeService, ServerChatGrammarsService
from tldw_chatbook.Collections_Interop import CollectionsFeedsScopeService, ServerCollectionsFeedsService
from tldw_chatbook.External_Connectors_Interop import ConnectorsScopeService, ServerConnectorsService
from tldw_chatbook.Feedback_Interop import FeedbackScopeService, ServerFeedbackService
from tldw_chatbook.Sharing_Interop import ServerSharingService, SharingScopeService
from tldw_chatbook.Skills_Interop import ServerSkillsService, SkillsScopeService
from tldw_chatbook.Web_Clipper_Interop import ServerWebClipperService, WebClipperScopeService
from tldw_chatbook.Writing_Interop import LocalWritingService, ServerWritingService, WritingScopeService
from tldw_chatbook.Subscriptions import (
    LocalWatchlistsService,
    ServerWatchlistsService,
    WatchlistScopeService,
)
from tldw_chatbook.Constants import ALL_TABS
from tldw_chatbook.UI.Navigation.base_app_screen import BaseAppScreen
from tldw_chatbook.UI.Navigation.main_navigation import MainNavigationBar
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Screens.media_ingest_screen import MediaIngestScreen
from tldw_chatbook.UI.Screens.media_screen import MediaScreen
from tldw_chatbook.runtime_policy.types import RuntimeSourceState


def _build_test_app() -> TldwCli:
    def fake_runtime_policy(app):
        context = SimpleNamespace(
            state=RuntimeSourceState(active_source="local", server_configured=True),
            persist=lambda: None,
        )
        app.runtime_policy = context
        app.current_runtime_source = "local"
        app.current_runtime_backend = "local"
        return context

    with patch("tldw_chatbook.app.load_settings", return_value={"tldw_api": {"base_url": "http://localhost:8000"}}):
        with patch("tldw_chatbook.app.get_cli_setting", side_effect=lambda _section, _key, default=None: default):
            with patch("tldw_chatbook.app.get_chachanotes_db_lazy", return_value=None):
                with patch("tldw_chatbook.app.ServerNotesWorkspaceService.from_config", return_value=MagicMock()):
                    with patch("tldw_chatbook.app.ServerCharacterPersonaService.from_config", return_value=MagicMock()):
                        with patch.object(TldwCli, "_init_notes_service", lambda self, _user: setattr(self, "notes_service", None)):
                            with patch.object(TldwCli, "_init_prompts_service", lambda self: setattr(self, "prompts_service_initialized", False)):
                                with patch.object(TldwCli, "_init_providers_models", lambda self: setattr(self, "providers_models", {})):
                                    with patch.object(TldwCli, "_init_media_db", lambda self: (setattr(self, "media_db", None), setattr(self, "_media_types_for_ui", ["All Media"]))):
                                        with patch("tldw_chatbook.app.load_runtime_policy_for_app", side_effect=fake_runtime_policy):
                                            with patch("tldw_chatbook.app.get_notifications_db_path", return_value=":memory:"):
                                                with patch("tldw_chatbook.app.get_subscriptions_db_path", return_value=":memory:"):
                                                    with patch("tldw_chatbook.app.get_research_db_path", return_value=":memory:"):
                                                        with patch("tldw_chatbook.app.get_writing_db_path", return_value=":memory:"):
                                                            return TldwCli()


def test_app_uses_screen_navigation_and_wires_media_services():
    app = _build_test_app()

    assert app._use_screen_navigation is True
    assert isinstance(app.local_media_reading_service, LocalMediaReadingService)
    assert isinstance(app.server_media_reading_service, ServerMediaReadingService)
    assert isinstance(app.media_reading_scope_service, MediaReadingScopeService)
    assert app.media_runtime_state.runtime_backend == "local"


def test_app_initializes_watchlists_and_notifications_services():
    app = _build_test_app()

    assert isinstance(app.local_watchlists_service, LocalWatchlistsService)
    assert isinstance(app.server_watchlists_service, ServerWatchlistsService)
    assert isinstance(app.watchlist_scope_service, WatchlistScopeService)
    assert isinstance(app.client_notifications_db, ClientNotificationsDB)
    assert isinstance(app.client_notifications_service, ClientNotificationsService)
    assert isinstance(app.notification_dispatch_service, NotificationDispatchService)
    assert isinstance(app.server_notifications_service, ServerNotificationsService)
    assert isinstance(app.notifications_scope_service, NotificationsScopeService)
    assert isinstance(app.server_outputs_service, ServerOutputsService)
    assert isinstance(app.outputs_scope_service, OutputsScopeService)
    assert isinstance(app.local_research_service, LocalResearchService)
    assert isinstance(app.server_research_service, ServerResearchService)
    assert isinstance(app.research_scope_service, ResearchScopeService)
    assert isinstance(app.local_research_search_service, LocalResearchSearchService)
    assert isinstance(app.server_research_search_service, ServerResearchSearchService)
    assert isinstance(app.research_search_scope_service, ResearchSearchScopeService)
    assert isinstance(app.server_chat_grammars_service, ServerChatGrammarsService)
    assert isinstance(app.chat_grammars_scope_service, ChatGrammarsScopeService)
    assert isinstance(app.server_feedback_service, ServerFeedbackService)
    assert isinstance(app.feedback_scope_service, FeedbackScopeService)
    assert isinstance(app.server_collections_feeds_service, ServerCollectionsFeedsService)
    assert isinstance(app.collections_feeds_scope_service, CollectionsFeedsScopeService)
    assert isinstance(app.server_connectors_service, ServerConnectorsService)
    assert isinstance(app.connectors_scope_service, ConnectorsScopeService)
    assert isinstance(app.server_skills_service, ServerSkillsService)
    assert isinstance(app.skills_scope_service, SkillsScopeService)
    assert isinstance(app.server_sharing_service, ServerSharingService)
    assert isinstance(app.sharing_scope_service, SharingScopeService)
    assert isinstance(app.server_web_clipper_service, ServerWebClipperService)
    assert isinstance(app.web_clipper_scope_service, WebClipperScopeService)
    assert isinstance(app.local_writing_service, LocalWritingService)
    assert isinstance(app.server_writing_service, ServerWritingService)
    assert isinstance(app.writing_scope_service, WritingScopeService)
    assert isinstance(app.server_chat_conversation_service, ServerChatConversationService)
    assert isinstance(app.chat_conversation_scope_service, ChatConversationScopeService)
    assert isinstance(app.server_chat_dictionary_service, ServerChatDictionaryService)
    assert isinstance(app.local_chat_dictionary_service, LocalChatDictionaryService)
    assert isinstance(app.chat_dictionary_scope_service, ChatDictionaryScopeService)
    assert isinstance(app.local_prompt_service, LocalPromptService)
    assert isinstance(app.server_prompt_service, ServerPromptService)
    assert isinstance(app.local_chatbook_service, LocalChatbookService)
    assert isinstance(app.server_chatbook_service, ServerChatbookService)
    assert isinstance(app.prompt_chatbook_scope_service, PromptChatbookScopeService)


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


@pytest.mark.asyncio
async def test_main_navigation_exposes_all_routed_primary_screens():
    class TestApp(App):
        def compose(self):
            yield MainNavigationBar(active="chat")

    app = TestApp()

    async with app.run_test() as pilot:
        nav = pilot.app.query_one(MainNavigationBar)
        for screen_id in ("study", "stts", "chatbooks", "subscriptions"):
            assert nav.query_one(f"#nav-{screen_id}") is not None


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
