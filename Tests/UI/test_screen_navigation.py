"""Focused screen wiring tests for screen-navigation mode."""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from textual import on
from textual.app import App
from textual.widgets import Button
from unittest.mock import MagicMock, patch

from tldw_chatbook.app import TldwCli
from tldw_chatbook.Chat import (
    ChatConversationScopeService,
    ServerChatConversationService,
)
from tldw_chatbook.Auth_Account_Interop import AuthAccountScopeService, ServerAuthAccountService
from tldw_chatbook.Audio_Services_Interop import (
    AudioServicesScopeService,
    LocalAudioServicesService,
    ServerAudioServicesService,
)
from tldw_chatbook.Character_Chat.chat_dictionary_scope_service import ChatDictionaryScopeService
from tldw_chatbook.Character_Chat.character_persona_scope_service import CharacterPersonaScopeService
from tldw_chatbook.Character_Chat.local_chat_dictionary_service import LocalChatDictionaryService
from tldw_chatbook.Character_Chat.local_character_persona_service import LocalCharacterPersonaService
from tldw_chatbook.Character_Chat.server_chat_dictionary_service import ServerChatDictionaryService
from tldw_chatbook.Character_Chat.server_character_persona_service import ServerCharacterPersonaService
from tldw_chatbook.Media import (
    LocalMediaReadingService,
    MediaReadingScopeService,
    ServerMediaReadingService,
)
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService
from tldw_chatbook.Notes.server_notes_workspace_service import ServerNotesWorkspaceService
from tldw_chatbook.Meetings_Interop import MeetingsScopeService, ServerMeetingsService
from tldw_chatbook.MCP.local_control_service import LocalMCPControlService
from tldw_chatbook.MCP.local_store import LocalMCPStore
from tldw_chatbook.MCP.server_target_store import ConfiguredServerTargetStore
from tldw_chatbook.MCP.server_unified_service import ServerUnifiedMCPService
from tldw_chatbook.MCP.unified_context_store import UnifiedMCPContextStore
from tldw_chatbook.MCP.unified_control_plane_service import UnifiedMCPControlPlaneService
from tldw_chatbook.Notifications import (
    ClientNotificationsDB,
    ClientNotificationsService,
    EventStateRepository,
    NotificationDispatchService,
    NotificationsScopeService,
    ServerNotificationsService,
)
from tldw_chatbook.Outputs_Interop import OutputsScopeService, ServerOutputsService
from tldw_chatbook.Personalization_Interop import (
    PersonalizationScopeService,
    ServerPersonalizationService,
)
from tldw_chatbook.Prompt_Management import (
    LocalPromptService,
    PromptChatbookScopeService,
    ServerPromptService,
)
from tldw_chatbook.Prompt_Studio_Interop import PromptStudioScopeService, ServerPromptStudioService
from tldw_chatbook.Research_Interop import (
    LocalResearchSearchService,
    LocalResearchService,
    ResearchSearchScopeService,
    ResearchScopeService,
    ServerResearchSearchService,
    ServerResearchService,
)
from tldw_chatbook.Chatbooks import LocalChatbookService, ServerChatbookService
from tldw_chatbook.Chat_Grammars_Interop import (
    ChatGrammarsScopeService,
    LocalChatGrammarsService,
    ServerChatGrammarsService,
)
from tldw_chatbook.Claims_Interop import ClaimsScopeService, ServerClaimsService
from tldw_chatbook.Companion_Interop import CompanionScopeService, ServerCompanionService
from tldw_chatbook.Collections_Interop import CollectionsFeedsScopeService, ServerCollectionsFeedsService
from tldw_chatbook.External_Connectors_Interop import ConnectorsScopeService, ServerConnectorsService
from tldw_chatbook.Feedback_Interop import FeedbackScopeService, LocalFeedbackService, ServerFeedbackService
from tldw_chatbook.Home.active_work_adapter import LocalNotificationHomeActiveWorkAdapter
from tldw_chatbook.Kanban_Interop import KanbanScopeService, LocalKanbanService, ServerKanbanService
from tldw_chatbook.LLM_Provider_Catalog import (
    LLMProviderCatalogScopeService,
    LocalLLMProviderCatalogService,
    ServerLLMProviderCatalogService,
)
from tldw_chatbook.Server_Runtime_Interop import ServerRuntimeScopeService, ServerRuntimeService
from tldw_chatbook.Sharing_Interop import ServerSharingService, SharingScopeService
from tldw_chatbook.Skills_Interop import LocalSkillsService, ServerSkillsService, SkillsScopeService
from tldw_chatbook.Sync_Interop import ServerSyncService, SyncScopeService, SyncStateRepository
from tldw_chatbook.Text2SQL_Interop import ServerText2SQLService, Text2SQLScopeService
from tldw_chatbook.Tools_Interop import ServerToolsService, ToolsScopeService
from tldw_chatbook.MCP_Governance_Interop import MCPGovernanceScopeService, ServerMCPGovernanceService
from tldw_chatbook.User_Governance_Interop import ServerUserGovernanceService, UserGovernanceScopeService
from tldw_chatbook.Web_Clipper_Interop import ServerWebClipperService, WebClipperScopeService
from tldw_chatbook.Web_Scraping_Interop import ServerWebScrapingService, WebScrapingScopeService
from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService
from tldw_chatbook.Writing_Interop import LocalWritingService, ServerWritingService, WritingScopeService
from tldw_chatbook.Subscriptions import (
    LocalWatchlistsService,
    ServerWatchlistsService,
    WatchlistScopeService,
)
from tldw_chatbook.Translation_Interop import ServerTranslationService, TranslationScopeService
from tldw_chatbook.Voice_Assistant_Interop import ServerVoiceAssistantService, VoiceAssistantScopeService
from tldw_chatbook.Constants import ALL_TABS
from tldw_chatbook.UI.Navigation.base_app_screen import BaseAppScreen
from tldw_chatbook.UI.Navigation.main_navigation import MainNavigationBar
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Screens.media_ingest_screen import MediaIngestScreen
from tldw_chatbook.UI.Screens.media_screen import MediaScreen
from tldw_chatbook.runtime_policy.types import RuntimeSourceState
from tldw_chatbook.runtime_policy.server_capabilities import ActiveServerCapabilityService
from tldw_chatbook.runtime_policy import KeyringServerCredentialStore, RuntimeServerContextProvider
from tldw_chatbook.runtime_policy.server_credentials import UnavailableServerCredentialStore
from tldw_chatbook.runtime_policy.server_parity_state import ServerParityStateRepositories


PRIMARY_ROUTE_IDS = [
    "chat",
    "notes",
    "media",
    "ingest",
    "search",
    "study",
    "ccp",
    "chatbooks",
]


def test_master_shell_route_inventory_has_known_legacy_routes():
    expected_legacy_routes = {
        "chat",
        "notes",
        "media",
        "ingest",
        "search",
        "study",
        "ccp",
        "conversation",
        "chatbooks",
        "subscriptions",
        "tools_settings",
        "llm",
        "stts",
        "evals",
        "logs",
        "stats",
        "coding",
        "customize",
    }

    app = _build_test_app()
    unresolved = []
    for route in expected_legacy_routes:
        _screen_name, _tab_id, screen_class = app._resolve_screen_navigation_target(route)
        if screen_class is None:
            unresolved.append(route)

    assert unresolved == []


def test_home_route_resolves_to_home_screen():
    app = _build_test_app()

    screen_name, current_tab, screen_class = app._resolve_screen_navigation_target("home")

    assert screen_name == "home"
    assert current_tab == "home"
    assert screen_class.__name__ == "HomeScreen"


def test_first_run_initial_route_defaults_to_home():
    app = _build_test_app()
    app.app_config["_first_run"] = True
    app._initial_tab_value = "chat"

    assert app._resolve_initial_shell_route() == "home"


@pytest.mark.asyncio
async def test_deferred_initial_tab_uses_first_run_home_route():
    app = _build_test_app()
    app.app_config["_first_run"] = True
    app._initial_tab_value = "chat"

    await app._set_initial_tab()

    assert app.current_tab == "home"


@pytest.mark.parametrize("configured_route", ["home", "library", "settings", "notes"])
def test_returning_user_initial_route_preserves_configured_default(configured_route):
    app = _build_test_app()
    app.app_config["_first_run"] = False
    app._initial_tab_value = configured_route

    assert app._resolve_initial_shell_route() == configured_route


def test_startup_route_validation_accepts_shell_and_legacy_defaults():
    app = _build_test_app()

    for route in ["home", "library", "settings", "notes"]:
        assert app._normalize_initial_tab_from_config(route) == route


def test_startup_route_validation_rejects_unknown_default():
    app = _build_test_app()

    assert app._normalize_initial_tab_from_config("definitely-not-a-route") == "chat"


def test_ccp_default_tab_initializes_before_reactive_watcher_runs():
    app = _build_test_app(configured_default="conversations_characters_prompts")

    assert app._initial_tab_value == "conversations_characters_prompts"
    assert app._ui_ready is False


def test_all_master_shell_primary_routes_resolve_before_nav_exposure():
    app = _build_test_app()
    expected_routes = {
        "home",
        "chat",
        "library",
        "conversation",
        "artifacts",
        "personas",
        "watchlists_collections",
        "schedules",
        "workflows",
        "mcp",
        "acp",
        "skills",
        "settings",
    }

    unresolved = []
    for route in expected_routes:
        _screen_name, _tab_id, screen_class = app._resolve_screen_navigation_target(route)
        if screen_class is None:
            unresolved.append(route)

    assert unresolved == []


def test_conversation_route_uses_library_conversation_context():
    app = _build_test_app()

    screen_name, current_tab, screen_class = app._resolve_screen_navigation_target("conversation")

    assert screen_name == "conversation"
    assert current_tab == "conversation"
    assert screen_class.__name__ == "LibraryConversationsScreen"


def test_legacy_tools_settings_route_uses_mcp_context():
    app = _build_test_app()

    screen_name, current_tab, screen_class = app._resolve_screen_navigation_target("tools_settings")

    assert screen_name == "tools_settings"
    assert current_tab == "mcp"
    assert screen_class.__name__ == "MCPScreen"


def _build_test_app(configured_default: str | None = None) -> TldwCli:
    user_data_dir = Path(tempfile.mkdtemp(prefix="tldw-chatbook-test-"))

    def fake_runtime_policy(app):
        context = SimpleNamespace(
            state=RuntimeSourceState(active_source="local", server_configured=True),
            persist=lambda: None,
        )
        app.runtime_policy = context
        app.current_runtime_source = "local"
        app.current_runtime_backend = "local"
        return context

    def fake_cli_setting(_section, _key=None, default=None):
        if _section == "general" and _key == "default_tab" and configured_default is not None:
            return configured_default
        return default

    with patch("tldw_chatbook.app.load_settings", return_value={"tldw_api": {"base_url": "http://localhost:8000"}}):
        with patch("tldw_chatbook.app.get_cli_setting", side_effect=fake_cli_setting):
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
                                                            with patch("tldw_chatbook.app.get_user_data_dir", return_value=user_data_dir):
                                                                with patch("tldw_chatbook.app.get_workspaces_db_path", return_value=user_data_dir / "workspaces.sqlite"):
                                                                    return TldwCli()


def test_app_uses_screen_navigation_and_wires_media_services():
    app = _build_test_app()

    assert app._use_screen_navigation is True
    assert isinstance(app.workspace_registry_service, LocalWorkspaceRegistryService)
    assert isinstance(app.local_media_reading_service, LocalMediaReadingService)
    assert isinstance(app.server_media_reading_service, ServerMediaReadingService)
    assert isinstance(app.media_reading_scope_service, MediaReadingScopeService)
    assert app.media_runtime_state.runtime_backend == "local"
    assert app.auth_account_scope_service.server_context_provider is app.server_context_provider
    assert app.server_media_reading_service.client_provider is app.server_context_provider
    assert app.server_chat_conversation_service.client_provider is app.server_context_provider
    assert app.server_notes_workspace_service.client_provider is app.server_context_provider
    assert app.server_character_persona_service.client_provider is app.server_context_provider
    assert app.server_chat_dictionary_service.client_provider is app.server_context_provider
    assert app.server_prompt_service.client_provider is app.server_context_provider
    assert app.server_chatbook_service.client_provider is app.server_context_provider
    assert app.server_prompt_studio_service.client_provider is app.server_context_provider
    assert app.server_runtime_service.client_provider is app.server_context_provider
    assert app.server_auth_account_service.client_provider is app.server_context_provider


@pytest.mark.asyncio
async def test_app_shutdown_helper_closes_server_context_provider_cached_client():
    class FakeServerContextProvider:
        def __init__(self) -> None:
            self.close_calls = 0

        async def close_cached_client(self) -> None:
            self.close_calls += 1

    provider = FakeServerContextProvider()
    app_like = SimpleNamespace(server_context_provider=provider)

    await TldwCli._close_server_context_provider_cached_client(app_like)

    assert provider.close_calls == 1


def test_app_initializes_watchlists_and_notifications_services():
    app = _build_test_app()

    assert isinstance(app.local_watchlists_service, LocalWatchlistsService)
    assert isinstance(app.server_watchlists_service, ServerWatchlistsService)
    assert isinstance(app.watchlist_scope_service, WatchlistScopeService)
    assert isinstance(app.client_notifications_db, ClientNotificationsDB)
    assert isinstance(app.server_parity_state, ServerParityStateRepositories)
    assert isinstance(app.event_state_repository, EventStateRepository)
    assert isinstance(app.sync_state_repository, SyncStateRepository)
    assert app.server_parity_state.local_notifications_db is app.client_notifications_db
    assert app.server_parity_state.event_state_repository is app.event_state_repository
    assert app.server_parity_state.sync_state_repository is app.sync_state_repository
    assert isinstance(app.client_notifications_service, ClientNotificationsService)
    assert isinstance(app.notification_dispatch_service, NotificationDispatchService)
    assert isinstance(app.server_notifications_service, ServerNotificationsService)
    assert isinstance(app.notifications_scope_service, NotificationsScopeService)
    assert app.notifications_scope_service.local_service is app.client_notifications_service
    assert isinstance(app.home_active_work_adapter, LocalNotificationHomeActiveWorkAdapter)
    assert app.home_active_work_adapter.notification_service is app.client_notifications_service
    assert app.home_active_work_adapter.watchlist_service is app.local_watchlists_service
    assert app.home_active_work_adapter.chatbook_service is app.local_chatbook_service
    assert isinstance(app.server_outputs_service, ServerOutputsService)
    assert isinstance(app.outputs_scope_service, OutputsScopeService)
    assert isinstance(app.local_research_service, LocalResearchService)
    assert app.local_research_service.notification_dispatcher is app.notification_dispatch_service
    assert app.local_research_service.notification_app is app
    assert app.local_media_reading_service.notification_dispatcher is app.notification_dispatch_service
    assert app.local_media_reading_service.notification_app is app
    assert isinstance(app.server_research_service, ServerResearchService)
    assert isinstance(app.research_scope_service, ResearchScopeService)
    assert isinstance(app.local_research_search_service, LocalResearchSearchService)
    assert isinstance(app.server_research_search_service, ServerResearchSearchService)
    assert isinstance(app.research_search_scope_service, ResearchSearchScopeService)
    assert isinstance(app.local_chat_grammars_service, LocalChatGrammarsService)
    assert isinstance(app.server_chat_grammars_service, ServerChatGrammarsService)
    assert isinstance(app.chat_grammars_scope_service, ChatGrammarsScopeService)
    assert isinstance(app.local_feedback_service, LocalFeedbackService)
    assert isinstance(app.server_feedback_service, ServerFeedbackService)
    assert isinstance(app.feedback_scope_service, FeedbackScopeService)
    assert isinstance(app.server_claims_service, ServerClaimsService)
    assert isinstance(app.claims_scope_service, ClaimsScopeService)
    assert isinstance(app.server_meetings_service, ServerMeetingsService)
    assert isinstance(app.meetings_scope_service, MeetingsScopeService)
    assert isinstance(app.server_prompt_studio_service, ServerPromptStudioService)
    assert isinstance(app.prompt_studio_scope_service, PromptStudioScopeService)
    assert isinstance(app.server_kanban_service, ServerKanbanService)
    assert isinstance(app.local_kanban_service, LocalKanbanService)
    assert isinstance(app.kanban_scope_service, KanbanScopeService)
    assert app.kanban_scope_service.local_service is app.local_kanban_service
    assert app.kanban_scope_service.server_service is app.server_kanban_service
    assert isinstance(app.server_translation_service, ServerTranslationService)
    assert isinstance(app.translation_scope_service, TranslationScopeService)
    assert isinstance(app.server_voice_assistant_service, ServerVoiceAssistantService)
    assert isinstance(app.voice_assistant_scope_service, VoiceAssistantScopeService)
    assert isinstance(app.server_companion_service, ServerCompanionService)
    assert isinstance(app.companion_scope_service, CompanionScopeService)
    assert isinstance(app.server_personalization_service, ServerPersonalizationService)
    assert isinstance(app.personalization_scope_service, PersonalizationScopeService)
    assert isinstance(app.server_collections_feeds_service, ServerCollectionsFeedsService)
    assert isinstance(app.collections_feeds_scope_service, CollectionsFeedsScopeService)
    assert app.collections_feeds_scope_service.local_service is app.local_watchlists_service
    assert isinstance(app.server_connectors_service, ServerConnectorsService)
    assert isinstance(app.connectors_scope_service, ConnectorsScopeService)
    assert isinstance(app.local_skills_service, LocalSkillsService)
    assert isinstance(app.server_skills_service, ServerSkillsService)
    assert isinstance(app.skills_scope_service, SkillsScopeService)
    assert app.skills_scope_service.local_service is app.local_skills_service
    assert app.skills_scope_service.server_service is app.server_skills_service
    assert isinstance(app.server_tools_service, ServerToolsService)
    assert isinstance(app.tools_scope_service, ToolsScopeService)
    assert isinstance(app.server_mcp_governance_service, ServerMCPGovernanceService)
    assert isinstance(app.mcp_governance_scope_service, MCPGovernanceScopeService)
    assert isinstance(app.local_mcp_store, LocalMCPStore)
    assert isinstance(app.local_mcp_control_service, LocalMCPControlService)
    assert isinstance(app.unified_mcp_target_store, ConfiguredServerTargetStore)
    assert isinstance(app.unified_mcp_context_store, UnifiedMCPContextStore)
    assert isinstance(app.server_unified_mcp_service, ServerUnifiedMCPService)
    assert isinstance(app.unified_mcp_service, UnifiedMCPControlPlaneService)
    assert app.unified_mcp_service.local_service is app.local_mcp_control_service
    assert app.unified_mcp_service.server_service is app.server_unified_mcp_service
    target = app.unified_mcp_target_store.resolve_active_target(None)
    assert target is not None
    assert target.auth_reference == "legacy:tldw_api"
    unified_client = app.server_unified_mcp_service.client_factory(target)
    assert unified_client.root_client.base_url == "http://localhost:8000"
    assert unified_client.root_client.token != "legacy:tldw_api"
    assert isinstance(app.server_text2sql_service, ServerText2SQLService)
    assert isinstance(app.text2sql_scope_service, Text2SQLScopeService)
    assert isinstance(app.server_sync_service, ServerSyncService)
    assert isinstance(app.sync_scope_service, SyncScopeService)
    assert app.media_reading_scope_service.sync_scope_service is app.sync_scope_service
    assert app.notes_scope_service.sync_scope_service is app.sync_scope_service
    assert app.research_scope_service.sync_scope_service is app.sync_scope_service
    assert isinstance(app.server_runtime_service, ServerRuntimeService)
    assert isinstance(app.server_runtime_scope_service, ServerRuntimeScopeService)
    assert isinstance(app.active_server_capability_service, ActiveServerCapabilityService)
    assert isinstance(
        app.server_credential_store,
        (KeyringServerCredentialStore, UnavailableServerCredentialStore),
    )
    assert isinstance(app.server_context_provider, RuntimeServerContextProvider)
    assert app.server_context_provider.runtime_context is app.runtime_policy
    assert app.server_context_provider.target_store is app.unified_mcp_target_store
    assert app.server_context_provider.credential_store is app.server_credential_store
    assert isinstance(app.local_llm_provider_catalog_service, LocalLLMProviderCatalogService)
    assert isinstance(app.server_llm_provider_catalog_service, ServerLLMProviderCatalogService)
    assert isinstance(app.llm_provider_catalog_scope_service, LLMProviderCatalogScopeService)
    assert isinstance(app.server_auth_account_service, ServerAuthAccountService)
    assert isinstance(app.auth_account_scope_service, AuthAccountScopeService)
    assert isinstance(app.local_audio_services_service, LocalAudioServicesService)
    assert isinstance(app.server_audio_services_service, ServerAudioServicesService)
    assert isinstance(app.audio_services_scope_service, AudioServicesScopeService)
    assert isinstance(app.server_user_governance_service, ServerUserGovernanceService)
    assert isinstance(app.user_governance_scope_service, UserGovernanceScopeService)
    assert isinstance(app.server_sharing_service, ServerSharingService)
    assert isinstance(app.sharing_scope_service, SharingScopeService)
    assert isinstance(app.server_web_clipper_service, ServerWebClipperService)
    assert isinstance(app.web_clipper_scope_service, WebClipperScopeService)
    assert isinstance(app.server_web_scraping_service, ServerWebScrapingService)
    assert isinstance(app.web_scraping_scope_service, WebScrapingScopeService)
    assert isinstance(app.local_writing_service, LocalWritingService)
    assert isinstance(app.server_writing_service, ServerWritingService)
    assert isinstance(app.writing_scope_service, WritingScopeService)
    assert isinstance(app.server_chat_conversation_service, ServerChatConversationService)
    assert isinstance(app.chat_conversation_scope_service, ChatConversationScopeService)
    assert isinstance(app.server_notes_workspace_service, ServerNotesWorkspaceService)
    assert isinstance(app.notes_scope_service, NotesScopeService)
    assert isinstance(app.server_character_persona_service, ServerCharacterPersonaService)
    assert isinstance(app.local_character_persona_service, LocalCharacterPersonaService)
    assert isinstance(app.character_persona_scope_service, CharacterPersonaScopeService)
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
    from tldw_chatbook.UI.Navigation.shell_destinations import SHELL_DESTINATION_ORDER

    class TestApp(App):
        def compose(self):
            yield MainNavigationBar(active="chat")

    app = TestApp()

    async with app.run_test() as pilot:
        nav = pilot.app.query_one(MainNavigationBar)
        for destination in SHELL_DESTINATION_ORDER:
            assert nav.query_one(f"#nav-{destination.destination_id}") is not None


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
        ("nav-home", "Home"),
        ("nav-console", "Console"),
        ("nav-library", "Library"),
        ("nav-artifacts", "Artifacts"),
        ("nav-personas", "Personas"),
        ("nav-watchlists_collections", "Watchlists"),
        ("nav-schedules", "Schedules"),
        ("nav-workflows", "Workflows"),
        ("nav-mcp", "MCP"),
        ("nav-acp", "ACP"),
        ("nav-skills", "Skills"),
        ("nav-settings", "Settings"),
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
        assert str(app.query_one("#nav-console", Button).label).strip() == "Console"
        assert nav_buttons[0].id == "nav-home"
        assert nav_buttons[1].id == "nav-console"
        assert nav_buttons[-1].id == "nav-settings"
        assert str(app.query_one("#nav-overflow-hint").renderable) == "More: Ctrl+P"


@pytest.mark.asyncio
async def test_main_navigation_buttons_explain_compact_labels():
    from tldw_chatbook.UI.Navigation.shell_destinations import SHELL_DESTINATION_ORDER

    expected_tooltips = {
        f"nav-{destination.destination_id}": destination.tooltip
        for destination in SHELL_DESTINATION_ORDER
    }

    class TestApp(App):
        def compose(self):
            yield MainNavigationBar(active="chat")

    app = TestApp()

    async with app.run_test(size=(160, 20)) as pilot:
        await pilot.pause(0.1)

        actual_tooltips = {
            button.id: str(button.tooltip)
            for button in app.query(".nav-button")
        }

        assert actual_tooltips == expected_tooltips


@pytest.mark.asyncio
async def test_main_navigation_route_ids_match_shell_destinations():
    from tldw_chatbook.UI.Navigation.shell_destinations import SHELL_DESTINATION_ORDER

    class TestApp(App):
        def compose(self):
            yield MainNavigationBar(active="chat")

    app = TestApp()

    async with app.run_test(size=(160, 20)) as pilot:
        await pilot.pause(0.1)

        actual_route_ids = [button.id for button in app.query(".nav-button")]
        expected_route_ids = [
            f"nav-{destination.destination_id}"
            for destination in SHELL_DESTINATION_ORDER
        ]

        assert actual_route_ids == expected_route_ids


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


def test_primary_routed_screens_use_base_app_screen():
    app = _build_test_app()

    offenders = []
    for route_id in PRIMARY_ROUTE_IDS:
        _screen_name, _tab_id, screen_class = app._resolve_screen_navigation_target(route_id)
        if screen_class is None or not issubclass(screen_class, BaseAppScreen):
            offenders.append((route_id, screen_class))

    assert offenders == []
