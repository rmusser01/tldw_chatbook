"""Focused screen wiring tests for screen-navigation mode."""

import asyncio
import shutil
import subprocess
from dataclasses import replace
from types import SimpleNamespace

import pytest
from textual.app import ComposeResult

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.widgets import Button, Input
from unittest.mock import AsyncMock, patch

from tldw_chatbook.app import TldwCli
from tldw_chatbook.Chat import (
    ChatConversationScopeService,
    ServerChatConversationService,
)
from tldw_chatbook.Auth_Account_Interop import (
    AuthAccountScopeService,
    ServerAuthAccountService,
)
from tldw_chatbook.Audio_Services_Interop import (
    AudioServicesScopeService,
    LocalAudioServicesService,
    ServerAudioServicesService,
)
from tldw_chatbook.Character_Chat.chat_dictionary_scope_service import (
    ChatDictionaryScopeService,
)
from tldw_chatbook.Character_Chat.character_persona_scope_service import (
    CharacterPersonaScopeService,
)
from tldw_chatbook.Character_Chat.local_chat_dictionary_service import (
    LocalChatDictionaryService,
)
from tldw_chatbook.Character_Chat.local_character_persona_service import (
    LocalCharacterPersonaService,
)
from tldw_chatbook.Character_Chat.server_chat_dictionary_service import (
    ServerChatDictionaryService,
)
from tldw_chatbook.Character_Chat.server_character_persona_service import (
    ServerCharacterPersonaService,
)
from tldw_chatbook.Media import (
    LocalMediaReadingService,
    MediaReadingScopeService,
    ServerMediaReadingService,
)
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService
from tldw_chatbook.Notes.file_notes_session_owner import SessionChange
from tldw_chatbook.Notes.server_notes_workspace_service import (
    ServerNotesWorkspaceService,
)
from tldw_chatbook.Meetings_Interop import MeetingsScopeService, ServerMeetingsService
from tldw_chatbook.MCP.local_control_service import LocalMCPControlService
from tldw_chatbook.MCP.local_store import LocalMCPStore
from tldw_chatbook.MCP.server_target_store import ConfiguredServerTargetStore
from tldw_chatbook.MCP.server_unified_service import ServerUnifiedMCPService
from tldw_chatbook.MCP.unified_context_store import UnifiedMCPContextStore
from tldw_chatbook.MCP.unified_control_plane_service import (
    UnifiedMCPControlPlaneService,
)
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
from tldw_chatbook.Prompt_Studio_Interop import (
    PromptStudioScopeService,
    ServerPromptStudioService,
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
from tldw_chatbook.Chat_Grammars_Interop import (
    ChatGrammarsScopeService,
    LocalChatGrammarsService,
    ServerChatGrammarsService,
)
from tldw_chatbook.Claims_Interop import ClaimsScopeService, ServerClaimsService
from tldw_chatbook.Companion_Interop import (
    CompanionScopeService,
    ServerCompanionService,
)
from tldw_chatbook.Collections_Interop import (
    CollectionsFeedsScopeService,
    ServerCollectionsFeedsService,
)
from tldw_chatbook.External_Connectors_Interop import (
    ConnectorsScopeService,
    ServerConnectorsService,
)
from tldw_chatbook.Feedback_Interop import (
    FeedbackScopeService,
    LocalFeedbackService,
    ServerFeedbackService,
)
from tldw_chatbook.Home.active_work_adapter import (
    LocalNotificationHomeActiveWorkAdapter,
)
from tldw_chatbook.Kanban_Interop import (
    KanbanScopeService,
    LocalKanbanService,
    ServerKanbanService,
)
from tldw_chatbook.LLM_Provider_Catalog import (
    LLMProviderCatalogScopeService,
    LocalLLMProviderCatalogService,
    ServerLLMProviderCatalogService,
)
from tldw_chatbook.Server_Runtime_Interop import (
    ServerRuntimeScopeService,
    ServerRuntimeService,
)
from tldw_chatbook.Sharing_Interop import ServerSharingService, SharingScopeService
from tldw_chatbook.Skills_Interop import (
    LocalSkillsService,
    ServerSkillsService,
    SkillTrustService,
    SkillsScopeService,
)
from tldw_chatbook.Sync_Interop import (
    LocalFirstSyncService,
    ManualSyncControlService,
    ServerSyncService,
    SyncScopeService,
    SyncStateRepository,
)
from tldw_chatbook.Text2SQL_Interop import ServerText2SQLService, Text2SQLScopeService
from tldw_chatbook.Tools_Interop import ServerToolsService, ToolsScopeService
from tldw_chatbook.MCP_Governance_Interop import (
    MCPGovernanceScopeService,
    ServerMCPGovernanceService,
)
from tldw_chatbook.User_Governance_Interop import (
    ServerUserGovernanceService,
    UserGovernanceScopeService,
)
from tldw_chatbook.Web_Clipper_Interop import (
    ServerWebClipperService,
    WebClipperScopeService,
)
from tldw_chatbook.Web_Scraping_Interop import (
    ServerWebScrapingService,
    WebScrapingScopeService,
)
from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService
from tldw_chatbook.Writing_Interop import (
    LocalWritingService,
    ServerWritingService,
    WritingScopeService,
)
from tldw_chatbook.Subscriptions import (
    LocalWatchlistsService,
    ServerWatchlistsService,
    WatchlistScopeService,
)
from tldw_chatbook.Translation_Interop import (
    ServerTranslationService,
    TranslationScopeService,
)
from tldw_chatbook.Voice_Assistant_Interop import (
    ServerVoiceAssistantService,
    VoiceAssistantScopeService,
)
from tldw_chatbook.Constants import ALL_TABS
from tldw_chatbook.UI.Navigation.base_app_screen import BaseAppScreen
from tldw_chatbook.UI.Navigation.main_navigation import MainNavigationBar
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Screens.media_screen import MediaScreen
from tldw_chatbook.runtime_policy.server_capabilities import (
    ActiveServerCapabilityService,
)
from tldw_chatbook.runtime_policy import (
    KeyringServerCredentialStore,
    RuntimeServerContextProvider,
)
from tldw_chatbook.runtime_policy.server_credentials import (
    UnavailableServerCredentialStore,
)
from tldw_chatbook.runtime_policy.server_parity_state import (
    ServerParityStateRepositories,
)


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


def test_master_shell_route_inventory_has_known_legacy_routes(monkeypatch):
    monkeypatch.setattr(
        "tldw_chatbook.Utils.optional_deps.check_subscriptions_deps",
        lambda: True,
    )
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
        _screen_name, _tab_id, screen_class = app._resolve_screen_navigation_target(
            route
        )
        if screen_class is None:
            unresolved.append(route)

    assert unresolved == []


def test_home_route_resolves_to_home_screen():
    app = _build_test_app()

    screen_name, current_tab, screen_class = app._resolve_screen_navigation_target(
        "home"
    )

    assert screen_name == "home"
    assert current_tab == "home"
    assert screen_class.__name__ == "HomeScreen"


def test_customize_route_resolves_to_settings_screen():
    app = _build_test_app()

    screen_name, current_tab, screen_class = app._resolve_screen_navigation_target(
        "customize"
    )

    assert screen_name == "settings"
    assert current_tab == "settings"
    assert screen_class.__name__ == "SettingsScreen"


def test_first_run_initial_route_defaults_to_home():
    app = _build_test_app()
    app.app_config["_first_run"] = True
    app._initial_tab_value = "chat"

    assert app._resolve_initial_shell_route() == "home"


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


def test_notes_is_not_a_navigable_tab():
    """The standalone Notes tab is retired: Notes now lives entirely inside
    Library, so "notes" must not appear as a top-level tab id, and the
    underlying ``NotesScreen`` must no longer be reachable. The legacy
    ``"notes"`` route id stays valid for backward compatibility (e.g. an
    existing user's saved startup config) but now resolves to
    ``LibraryScreen`` via a compatibility alias instead of ``NotesScreen``.
    """
    from tldw_chatbook.UI.Navigation.screen_registry import resolve_screen_target
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    assert "notes" not in ALL_TABS
    assert not hasattr(
        __import__("tldw_chatbook.Constants", fromlist=["TAB_NOTES"]), "TAB_NOTES"
    )

    _screen_name, _canonical_tab, screen_class = resolve_screen_target("notes")
    assert screen_class is LibraryScreen


def test_open_notes_workspace_routes_to_library_notes_list():
    """``open_notes_workspace`` (Study's "return to workspace" action) used
    to route to the standalone Notes tab; it must now re-point into Library
    with a ``mode=notes`` navigation context that lands on the Notes list,
    since Library has no equivalent to the retired per-workspace scope.
    """
    from tldw_chatbook.Constants import TAB_LIBRARY

    app = _build_test_app()
    posted = []
    app.post_message = posted.append

    app.open_notes_workspace("ws-1", subview="details")

    assert len(posted) == 1
    message = posted[0]
    assert message.screen_name == TAB_LIBRARY
    assert message.screen_context == {"mode": "notes"}


def test_prompts_route_resolves_to_library_screen():
    """The Personas "prompts" mode chip is retired (Task 7): prompt
    management now lives entirely inside Library. The legacy "prompts"
    route id must resolve to ``LibraryScreen`` instead of ``PersonasScreen``,
    mirroring the "notes" compatibility alias above.
    """
    from tldw_chatbook.UI.Navigation.screen_registry import resolve_screen_target
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    _screen_name, _canonical_tab, screen_class = resolve_screen_target("prompts")
    assert screen_class is LibraryScreen


def test_skills_route_resolves_to_library_screen():
    """The standalone Skills tab is retired (Skills sub-project Task 5):
    skill management now lives entirely inside Library (its own Skills
    rail row, built in Tasks 1-4). The legacy "skills" route id must
    resolve to ``LibraryScreen`` instead of ``SkillsScreen``, mirroring the
    "notes"/"prompts" compatibility aliases above. ``SkillsScreen`` itself
    is not deleted -- its passphrase modal is reused by the Library skill
    editor's trust panel, and it stays directly reachable by its own
    destination-shell test suite (``Tests/UI/test_destination_shells.py``).
    """
    from tldw_chatbook.UI.Navigation.screen_registry import resolve_screen_target
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    _screen_name, _canonical_tab, screen_class = resolve_screen_target("skills")
    assert screen_class is LibraryScreen


def test_research_route_resolves_to_research_screen():
    """task-16322 (ADR-068) reverses task-255's library alias: the research
    route is a real screen again.

    The local research execution engine now drives launched local runs
    (planning -> collecting -> synthesizing -> packaging), so
    ``ResearchWindow`` -- the only run/event observation surface -- is
    reachable from navigation under the legacy "research" route id
    (still a command-palette direct command via ``TAB_RESEARCH`` and valid
    in saved startup configs). The Workbench migration owner stays
    "library" (route_inventory).
    """
    from tldw_chatbook.UI.Navigation.screen_registry import resolve_screen_target
    from tldw_chatbook.UI.Screens.research_screen import ResearchScreen

    screen_name, canonical_tab, screen_class = resolve_screen_target("research")
    assert screen_class is ResearchScreen
    assert screen_name == "research"
    assert canonical_tab == "research"


def test_media_route_resolves_to_library_screen():
    """task-2851: the legacy standalone Media Library screen is retired.

    Library already reimplements full media browsing/management as its own
    canvas (rail row "media", ``LIBRARY_ROW_BROWSE_MEDIA``) -- the standalone
    ``MediaScreen`` (nav: Media Types / All Media / Analysis Review /
    Collections-Tags / Multi-Item Review) is a dead-end duplicate that used
    to render UNDER the active Library tab highlight (the "media" legacy
    route folds into the "library" shell destination for nav-bar purposes,
    but its screen route pointed at a completely different screen). The
    legacy "media" route id must resolve to ``LibraryScreen`` instead of
    ``MediaScreen``, mirroring the "notes"/"prompts"/"skills"/"research"
    compatibility aliases above. ``MediaScreen`` itself is not deleted --
    its save_state/restore_state contracts stay directly exercised by their
    own unit tests below, mirroring the "skills" precedent.
    """
    from tldw_chatbook.UI.Navigation.screen_registry import resolve_screen_target
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    _screen_name, _canonical_tab, screen_class = resolve_screen_target("media")
    assert screen_class is LibraryScreen


def test_no_route_reaches_the_retired_media_screen():
    """task-2851 AC#1: nothing may still resolve to the legacy MediaScreen."""
    from tldw_chatbook.UI.Navigation import screen_registry

    for route_id in screen_registry.registered_screen_route_ids():
        _screen_name, _canonical_tab, screen_class = (
            screen_registry.resolve_screen_target(route_id)
        )
        assert screen_class is None or screen_class.__name__ != "MediaScreen", route_id
    for alias_id in screen_registry.registered_screen_aliases():
        _screen_name, _canonical_tab, screen_class = (
            screen_registry.resolve_screen_target(alias_id)
        )
        assert screen_class is None or screen_class.__name__ != "MediaScreen", alias_id


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
        "settings",
    }

    unresolved = []
    for route in expected_routes:
        _screen_name, _tab_id, screen_class = app._resolve_screen_navigation_target(
            route
        )
        if screen_class is None:
            unresolved.append(route)

    assert unresolved == []


def test_lazy_screen_registry_resolves_visible_shell_destinations():
    from tldw_chatbook.UI.Navigation.screen_registry import resolve_screen_target
    from tldw_chatbook.UI.Navigation.shell_destinations import SHELL_DESTINATION_ORDER

    expected_class_names = {
        "home": "HomeScreen",
        "chat": "ChatScreen",
        "library": "LibraryScreen",
        "artifacts": "ArtifactsScreen",
        "personas": "PersonasScreen",
        "watchlists_collections": "WatchlistsCollectionsScreen",
        "schedules": "SchedulesWorkbench",
        "workflows": "WorkflowsScreen",
        "mcp": "MCPScreen",
        "acp": "ACPScreen",
        "llm": "LLMScreen",
        "logs": "LogsScreen",
        "settings": "SettingsScreen",
    }

    resolved = {}
    for destination in SHELL_DESTINATION_ORDER:
        _screen_name, _tab_id, screen_class = resolve_screen_target(
            destination.primary_route
        )
        resolved[destination.primary_route] = (
            screen_class.__name__ if screen_class else None
        )

    assert resolved == expected_class_names


def test_subscriptions_route_resolves_to_watchlists_collections_via_alias():
    from tldw_chatbook.UI.Navigation import screen_registry

    screen_name, canonical_tab, screen_class = screen_registry.resolve_screen_target(
        "subscriptions"
    )

    assert screen_name == "watchlists_collections"
    assert canonical_tab == "watchlists_collections"
    assert screen_class.__name__ == "WatchlistsCollectionsScreen"


def test_subscription_route_resolves_to_watchlists_collections_via_alias():
    from tldw_chatbook.UI.Navigation import screen_registry

    screen_name, canonical_tab, screen_class = screen_registry.resolve_screen_target(
        "subscription"
    )

    assert screen_name == "watchlists_collections"
    assert canonical_tab == "watchlists_collections"
    assert screen_class.__name__ == "WatchlistsCollectionsScreen"


def test_conversation_route_uses_library_conversation_context():
    app = _build_test_app()

    screen_name, current_tab, screen_class = app._resolve_screen_navigation_target(
        "conversation"
    )

    assert screen_name == "conversation"
    assert current_tab == "conversation"
    assert screen_class.__name__ == "LibraryConversationsScreen"


def test_legacy_tools_settings_route_uses_mcp_context():
    app = _build_test_app()

    screen_name, current_tab, screen_class = app._resolve_screen_navigation_target(
        "tools_settings"
    )

    assert screen_name == "tools_settings"
    assert current_tab == "mcp"
    assert screen_class.__name__ == "MCPScreen"


@pytest.mark.asyncio
async def test_screen_navigation_always_constructs_fresh_instances(monkeypatch):
    """Regression lock for the rapid-tab-switch freeze (2026-07-11).

    Navigation used to cache Screen INSTANCES for allowlisted routes and
    re-mount them after ``switch_screen`` had already unmounted them. Under
    rapid switching the re-mount interleaved with the still-in-flight
    unmount, leaving zombie widgets (``mounted=True`` with stopped message
    pumps), a compositor stuck on a stale frame, and an app that silently
    swallowed every subsequent click -- a permanent, exception-free freeze.
    Every navigation must therefore construct a FRESH screen instance; this
    test fails if instance reuse ever returns.
    """
    app = _build_test_app()
    constructed = {"chat": 0, "library": 0}

    class FakeChatScreen:
        screen_name = "chat"

        def __init__(self, app_instance):
            self.app_instance = app_instance
            constructed["chat"] += 1

    class FakeLibraryScreen:
        screen_name = "library"

        def __init__(self, app_instance):
            self.app_instance = app_instance
            constructed["library"] += 1

    def fake_resolve(target):
        if target == "chat":
            return "chat", "chat", FakeChatScreen
        if target == "library":
            return "library", "library", FakeLibraryScreen
        return target, target, None

    switched_screens = []

    async def fake_switch_screen(screen):
        switched_screens.append(screen)

    monkeypatch.setattr(app, "_resolve_screen_navigation_target", fake_resolve)
    monkeypatch.setattr(app, "switch_screen", fake_switch_screen)

    async with app.run_test(size=(160, 40)) as pilot:
        await pilot.pause(0.1)
        # These direct handler calls simulate post-startup navigation; mark
        # startup complete so the pre-initial-screen guard lets them through.
        app._initial_screen_pushed = True

        await app.handle_screen_navigation(NavigateToScreen("chat"))
        await app.handle_screen_navigation(NavigateToScreen("library"))
        await app.handle_screen_navigation(NavigateToScreen("chat"))

    assert constructed == {"chat": 2, "library": 1}
    assert switched_screens[0] is not switched_screens[2]


@pytest.mark.asyncio
async def test_navigation_flushes_outgoing_screen_and_honors_veto(monkeypatch):
    """Navigating away must flush the outgoing screen's pending work first.

    Screens are no longer cached, so anything not persisted before the
    switch is destroyed with the old instance (e.g. a Library note edit
    whose debounced autosave has not fired yet). The app awaits the
    outgoing screen's ``flush_pending_work()`` and aborts the switch when
    it returns False (unresolved save conflict needs the user).
    """
    app = _build_test_app()

    class FakeTargetScreen:
        screen_name = "chat"

        def __init__(self, app_instance):
            self.app_instance = app_instance

    def fake_resolve(target):
        return "chat", "chat", FakeTargetScreen

    switched_screens = []

    async def fake_switch_screen(screen):
        switched_screens.append(screen)

    monkeypatch.setattr(app, "_resolve_screen_navigation_target", fake_resolve)
    monkeypatch.setattr(app, "switch_screen", fake_switch_screen)

    flush_results = {"value": False}
    flush_calls = []

    class FakeOutgoingScreen:
        screen_name = "library"

        async def flush_pending_work(self):
            flush_calls.append(True)
            return flush_results["value"]

    outgoing = FakeOutgoingScreen()
    # The handler only touches self.screen for the outgoing flush/save-state
    # steps, so it is callable without a running app once switch_screen is
    # stubbed -- patching the screen property this way would break the live
    # compositor under run_test.
    monkeypatch.setattr(type(app), "screen", property(lambda self: outgoing))
    # Simulate post-startup state so the pre-initial-screen guard lets the
    # direct handler call through.
    app._initial_screen_pushed = True

    await app.handle_screen_navigation(NavigateToScreen("chat"))
    assert flush_calls, "outgoing screen's flush_pending_work was never awaited"
    assert switched_screens == [], "veto (False) must abort the switch"

    flush_results["value"] = True
    await app.handle_screen_navigation(NavigateToScreen("chat"))
    assert len(switched_screens) == 1, "flush returning True must allow the switch"


@pytest.mark.asyncio
async def test_navigation_flush_exception_warns_and_aborts_switch(monkeypatch):
    """A broken outgoing flush must fail closed while pending edits may exist."""
    app = _build_test_app()
    created_screens = []
    switched_screens = []
    saved_states = []
    notifications = []

    class FakeTargetScreen:
        screen_name = "chat"

        def __init__(self, app_instance):
            created_screens.append(app_instance)

    class FakeOutgoingScreen:
        screen_name = "library"

        async def flush_pending_work(self):
            raise RuntimeError("simulated flush failure")

        def save_state(self):
            saved_states.append(True)
            return {}

    async def fake_switch_screen(screen):
        switched_screens.append(screen)

    monkeypatch.setattr(
        app,
        "_resolve_screen_navigation_target",
        lambda target: ("chat", "chat", FakeTargetScreen),
    )
    monkeypatch.setattr(app, "switch_screen", fake_switch_screen)
    monkeypatch.setattr(
        app,
        "notify",
        lambda message, **kwargs: notifications.append((message, kwargs)),
    )
    outgoing = FakeOutgoingScreen()
    monkeypatch.setattr(type(app), "screen", property(lambda self: outgoing))
    # Simulate post-startup state so the pre-initial-screen guard lets the
    # direct handler call through.
    app._initial_screen_pushed = True

    await app.handle_screen_navigation(NavigateToScreen("chat"))

    assert switched_screens == []
    assert created_screens == []
    assert saved_states == []
    assert notifications == [
        (
            "Couldn't save pending changes before switching screens.",
            {"severity": "warning"},
        )
    ]


@pytest.mark.asyncio
async def test_navigation_confirms_with_outgoing_screen_and_honors_veto(monkeypatch):
    """TASK-1143 (F5): navigating away must consult the outgoing screen's
    ``confirm_navigation()`` the same way it already consults
    ``flush_pending_work()``. Console (``ChatScreen``) implements this to
    warn when the agent fleet is busy -- unmounting cancels every
    in-flight run and denies every pending/parked approval round. False
    vetoes the switch, leaving the screen (and its live fleet) mounted
    exactly like a flush veto; True (idle fleet, or the user chose
    "Leave") lets it proceed.
    """
    app = _build_test_app()
    app._initial_screen_pushed = True

    class FakeTargetScreen:
        screen_name = "chat"

        def __init__(self, app_instance):
            self.app_instance = app_instance

    def fake_resolve(target):
        return "chat", "chat", FakeTargetScreen

    switched_screens = []

    async def fake_switch_screen(screen):
        switched_screens.append(screen)

    monkeypatch.setattr(app, "_resolve_screen_navigation_target", fake_resolve)
    monkeypatch.setattr(app, "switch_screen", fake_switch_screen)

    confirm_results = {"value": False}
    confirm_calls = []

    class FakeOutgoingScreen:
        screen_name = "library"

        async def confirm_navigation(self):
            confirm_calls.append(True)
            return confirm_results["value"]

    outgoing = FakeOutgoingScreen()
    monkeypatch.setattr(type(app), "screen", property(lambda self: outgoing))

    await app.handle_screen_navigation(NavigateToScreen("chat"))
    assert confirm_calls, "outgoing screen's confirm_navigation was never awaited"
    assert switched_screens == [], "veto (False) must abort the switch"

    confirm_results["value"] = True
    await app.handle_screen_navigation(NavigateToScreen("chat"))
    assert len(switched_screens) == 1, "confirm returning True must allow the switch"


@pytest.mark.asyncio
async def test_navigation_confirm_exception_warns_and_aborts_switch(monkeypatch):
    """A broken outgoing confirm_navigation must fail closed, not silently
    let navigation proceed and tear down live work nobody was asked about.
    """
    app = _build_test_app()
    app._initial_screen_pushed = True
    created_screens = []
    switched_screens = []
    notifications = []

    class FakeTargetScreen:
        screen_name = "chat"

        def __init__(self, app_instance):
            created_screens.append(app_instance)

    class FakeOutgoingScreen:
        screen_name = "library"

        async def confirm_navigation(self):
            raise RuntimeError("simulated confirm failure")

    async def fake_switch_screen(screen):
        switched_screens.append(screen)

    monkeypatch.setattr(
        app,
        "_resolve_screen_navigation_target",
        lambda target: ("chat", "chat", FakeTargetScreen),
    )
    monkeypatch.setattr(app, "switch_screen", fake_switch_screen)
    monkeypatch.setattr(
        app,
        "notify",
        lambda message, **kwargs: notifications.append((message, kwargs)),
    )
    outgoing = FakeOutgoingScreen()
    monkeypatch.setattr(type(app), "screen", property(lambda self: outgoing))

    await app.handle_screen_navigation(NavigateToScreen("chat"))

    assert switched_screens == []
    assert created_screens == []
    assert any(
        "Couldn't confirm leaving this screen" in message
        for message, _kwargs in notifications
    )


@pytest.mark.asyncio
async def test_rapid_tab_switch_storm_leaves_no_zombie_widgets():
    """Live-repro regression lock for the rapid-tab-switch freeze.

    Storm real navigation across real screens with no settling pauses, then
    assert the app is still responsive and the active screen's widget tree
    contains no zombie widgets (attached but with a stopped message pump) --
    the wedged state the instance cache used to produce, where the compositor
    froze on a stale frame and dead pumps swallowed every click.
    """
    app = _build_test_app()

    async with app.run_test(size=(160, 40)) as pilot:
        # Wait for the app's own initial navigation screen before storming --
        # in production the nav bar only exists once that screen is mounted,
        # so a pre-boot NavigateToScreen is unreachable by real input.
        for _ in range(150):
            await pilot.pause(0.02)
            if type(app.screen).__name__ != "Screen":
                break
        assert type(app.screen).__name__ != "Screen", (
            "app never mounted its initial screen"
        )
        routes = ("home", "library", "workflows", "schedules")
        for _round in range(3):
            for route in routes:
                app.post_message(NavigateToScreen(route))
                await pilot.pause(0)
        # Let the queued switches drain, then prove the app still navigates.
        app.post_message(NavigateToScreen("library"))
        zombies: list = []
        for _ in range(200):
            await pilot.pause(0.02)
            if type(app.screen).__name__ == "LibraryScreen" and app.screen.is_running:
                # TASK-1230: `handle_screen_navigation` now runs each
                # attempt as its own worker (`_dispatch_screen_navigation`)
                # instead of inline on the App's own message-processing
                # task, specifically so a busy-fleet confirm dialog can
                # never starve that task's own input routing (see that
                # method's docstring). Twelve back-to-back navigations
                # posted with zero pacing (`pilot.pause(0)` above) now
                # queue behind `_screen_navigation_lock` with real worker-
                # scheduling overhead each, so the LAST one's own children
                # can still be finishing their own mount for a brief beat
                # after `app.screen` first reports the target screen and
                # `is_running` -- keep polling for the zombie check itself
                # to clear rather than asserting on the very first tick;
                # if the app ever regresses to genuinely stuck/dead
                # widgets (the historical instance-cache bug this test
                # guards against), `zombies` never clears and the
                # assertion below still fails.
                zombies = [
                    widget
                    for widget in app.screen.walk_children()
                    if not widget.is_running
                ]
                if not zombies:
                    break
        assert type(app.screen).__name__ == "LibraryScreen"
        assert app.screen.is_running
        assert not zombies, f"zombie widgets on active screen: {zombies[:5]}"
        # One more hop for responsiveness.
        app.post_message(NavigateToScreen("home"))
        for _ in range(150):
            await pilot.pause(0.02)
            if type(app.screen).__name__ == "HomeScreen":
                break
        assert type(app.screen).__name__ == "HomeScreen"
        assert app.screen.is_running


@pytest.mark.asyncio
async def test_overlapping_navigate_requests_complete_in_fifo_order() -> None:
    """TASK-1230 review follow-up: `TldwCli._dispatch_screen_navigation`
    runs each `NavigateToScreen` attempt as its own worker instead of
    awaiting it inline on the App's own message-processing task (see that
    method's docstring for why -- a busy-fleet confirm dialog must never
    starve that task's own input routing). Workers are otherwise
    independent tasks, so nothing but `_screen_navigation_lock` stops
    three overlapping attempts from racing on shared state
    (``self.current_tab``, ``ScreenStateStore`` save/restore, and
    ``switch_screen``'s screen stack -- all inside the region the lock
    guards, since they run from within ``_handle_screen_navigation_locked``
    while the lock is held).

    This asserts the STRONGER property than "the last target wins" (which
    the storm test above already covers): three back-to-back
    ``NavigateToScreen`` messages -- posted with NO awaited gap between
    them, an idle fleet so no confirm dialog ever gates any of them --
    must still MOUNT in EXACTLY the order they were posted. Recorded via
    the real ``BaseAppScreen.on_mount`` seam every screen (Home/Library/
    Workflows alike) calls once actually mounted -- the point AFTER
    ``switch_screen``'s own async unmount/mount work has run, which is
    where two attempts racing without the lock could genuinely finish out
    of order. (Recording at ``TldwCli._create_navigation_screen`` instead
    -- called synchronously, early in each attempt, before any of that
    async work -- was tried and rejected: it recorded FIFO order even with
    the lock temporarily replaced by a fresh, unshared ``asyncio.Lock()``
    per call [i.e. no real serialization at all], because nothing before
    that point yields the event loop for an idle-fleet attempt -- not a
    discriminating check.)

    Incidental asyncio scheduling alone turned out to be an unreliable way
    to PROVE the lock matters (verified directly: with the lock replaced
    by a fresh, unshared lock per call, reordering was observed on some
    runs but not others -- real, but not deterministic, since nothing
    forces the three attempts' async work to overlap in a particular way).
    So this test manufactures a deterministic race instead: it wraps
    `_complete_screen_navigation` (called from inside the guarded region)
    with a per-target `asyncio.sleep` -- LONGEST for "home" (posted
    FIRST), zero for "workflows" (posted LAST) -- so that without
    serialization the last-posted, zero-delay attempt would provably
    finish first. Confirmed this setup, with the real lock temporarily
    replaced by a fresh lock per call, reliably reorders `mounted_order`
    (5/5 runs; "workflows" -- zero delay -- mounts first every time, then
    either `['workflows', 'library', 'home']` [the exact reverse, 4/5
    runs] or `['workflows', 'home', 'library']` [1/5 runs] depending on
    exactly how "library"'s short delay lands relative to "home"'s longer
    one); restoring the real lock forces `['home', 'library', 'workflows']`
    every time despite the same delays, because the lock keeps "library"
    from even starting its own (short) delay until "home" -- delay
    included -- fully finishes, and likewise for "workflows" after
    "library". Not polling ``app.screen`` after the fact: polling can only
    ever observe whichever attempt happens to be current when it looks,
    never prove the two that came before it also landed in order.
    """
    from tldw_chatbook.UI.Navigation.base_app_screen import BaseAppScreen

    app = _build_test_app()
    mounted_order: list[str] = []
    original_on_mount = BaseAppScreen.on_mount

    def _recording_on_mount(self) -> None:
        mounted_order.append(self.screen_name)
        return original_on_mount(self)

    # Deterministic race pressure: "home" (posted first) is slowest,
    # "workflows" (posted last) is instant. Without the lock this
    # guarantees "workflows" mounts before "home" even finishes; with the
    # lock, "workflows" cannot start until "library" (and, transitively,
    # "home") has fully completed, delay included.
    delays = {"home": 0.2, "library": 0.05, "workflows": 0.0}
    original_complete = type(app)._complete_screen_navigation

    async def _delayed_complete(self, **kwargs):
        delay = delays.get(kwargs.get("screen_name"), 0.0)
        if delay:
            await asyncio.sleep(delay)
        return await original_complete(self, **kwargs)

    with (
        patch.object(BaseAppScreen, "on_mount", _recording_on_mount),
        patch.object(type(app), "_complete_screen_navigation", _delayed_complete),
    ):
        async with app.run_test(size=(160, 40)) as pilot:
            for _ in range(150):
                await pilot.pause(0.02)
                if type(app.screen).__name__ != "Screen":
                    break
            assert type(app.screen).__name__ != "Screen", (
                "app never mounted its initial screen"
            )
            # Deliberately NOT clearing `mounted_order` here: the app's own
            # delayed initial-tab switch (a known cold-start gotcha -- it
            # can re-navigate to "chat" shortly after the placeholder
            # screen clears) settles on an unpredictable timeline under
            # load. Filtering the recorded names down to this test's own
            # three targets below is immune to that race regardless of how
            # long the boot noise takes to settle.

            # Three DIFFERENT targets, posted back-to-back with NO await
            # between them -- this is exactly what lets their
            # `_dispatch_screen_navigation` workers get CREATED in a tight
            # burst; only `_screen_navigation_lock` then decides which one
            # actually gets to run its body first. A `pilot.pause(0)`
            # between posts (as the storm test above uses) would let each
            # attempt fully settle before the next is even posted, which
            # would never exercise the lock's ordering guarantee at all.
            app.post_message(NavigateToScreen("home"))
            app.post_message(NavigateToScreen("library"))
            app.post_message(NavigateToScreen("workflows"))

            # Wait for all THREE targets to have mounted at least once,
            # rather than asserting `app.screen`'s final type: without the
            # lock, completion order reverses (the induced delays mean
            # "home" -- posted first, slowest -- finishes LAST), so
            # whichever screen ends up current when this loop times out
            # differs run to run and isn't itself the property under
            # test. The FIFO check below, on `mounted_order`, is.
            this_tests_targets = {"home", "library", "workflows"}
            for _ in range(150):
                await pilot.pause(0.02)
                seen = {name for name in mounted_order if name in this_tests_targets}
                if len(seen) >= 3:
                    break

    # `mounted_order` can also carry the app's own cold-start noise (its
    # delayed initial-tab switch to "chat"; see above) and
    # `BaseAppScreen.on_mount` fires twice per real mount (a pre-existing,
    # harmless duplication -- also visible as a doubled "Screen X mounted"
    # log line, unrelated to this fix). Filter down to this test's own
    # three targets and dedupe consecutive repeats before asserting order,
    # so the check is immune to both and verifies FIFO ordering only.
    this_tests_targets = {"home", "library", "workflows"}
    filtered = [name for name in mounted_order if name in this_tests_targets]
    deduped = [
        name for i, name in enumerate(filtered) if i == 0 or name != filtered[i - 1]
    ]
    assert deduped == ["home", "library", "workflows"], (
        f"navigation attempts mounted out of FIFO order: {mounted_order}"
    )



@pytest.mark.asyncio
async def test_navigation_keypress_during_splash_is_safely_ignored():
    """Regression lock for the F9-during-splash crash (task-1339).

    Pressing a shell-destination key (F7/F8/F9 or Ctrl+digit) while the
    splash screen is still up posted ``NavigateToScreen`` before the initial
    screen existed; ``switch_screen`` then hit Textual's empty
    result-callback stack and raised ``IndexError: pop from empty list``.
    Navigation requests must be ignored until the initial screen has been
    pushed: pressing F9 mid-splash must raise nothing and must not navigate,
    leaving the app to finish startup on its configured initial screen.
    """
    app = _build_test_app()  # splash enabled by default (skip_on_keypress=True)

    def force_splash_config(section, key=None, default=None):
        # ``_build_test_app`` only patches config lookups during __init__, so
        # compose() at run_test time would read the real user config; force a
        # deterministic splash here so "splash still active at press time"
        # does not depend on the developer's machine. 1.5s matches the real
        # default; pressing at 0.3s leaves ~1.2s of splash window.
        if section == "splash_screen":
            splash_defaults = {
                "enabled": True,
                "duration": 1.5,
                "skip_on_keypress": True,
                "show_progress": True,
                "card_selection": "random",
            }
            return splash_defaults.get(key, default)
        return default

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=force_splash_config):
        async with app.run_test(size=(120, 35)) as pilot:
            await pilot.pause(0.3)
            assert app.splash_screen_active, "splash must still be active at press time"
            assert not getattr(app, "_initial_screen_pushed", False)

            await pilot.press("f9")  # F9 = Settings destination; must not crash
            await pilot.pause(0.2)

            # Wait for startup to finish: splash dismissed, initial screen pushed.
            for _ in range(60):
                await pilot.pause(0.05)
                if not app.splash_screen_active and getattr(
                    app, "_initial_screen_pushed", False
                ):
                    break
            assert not app.splash_screen_active, "splash never dismissed"
            assert app._initial_screen_pushed, "initial screen never mounted"

            # The swallowed navigation must not have landed on Settings; the
            # app settles on its configured initial screen instead.
            assert type(app.screen).__name__ != "SettingsScreen"
            assert app.screen.is_running

# The shared app factory moved to Tests/UI/app_factory.py (task-1458) so a
# test module no longer hosts suite-wide infrastructure and its temp dirs
# get drained after every test. Re-exported here for in-flight branches
# that still import it from this module.
from Tests.UI.app_factory import _build_test_app  # noqa: F401,E402


# task-3316 introduced `_wait_for_background_signal` / `_await_background_task`
# here; task-14912 moved them to Tests/UI/background_signals.py so every UI test
# gets the bound by default rather than only this file. Read that module's
# header for the incident and the rules. Re-exported under the original private
# names so in-flight branches that import them from this module keep working.
from Tests.UI.background_signals import (  # noqa: E402
    BACKGROUND_SIGNAL_TIMEOUT_SECONDS as _BACKGROUND_SIGNAL_TIMEOUT_SECONDS,  # noqa: F401
    await_background_task as _await_background_task,  # noqa: F401
    wait_for_background_signal as _wait_for_background_signal,  # noqa: F401
)


def test_local_watchlists_service_db_factory_resolves_the_same_path_as_the_eager_subscriptions_db():
    """task-1631: the watchlists service and the eager `subscriptions_db`
    must be one database -- and, since task-15463, one *instance*.

    `watchlist_bundle_service.db` is the `subscriptions_db` built during
    `__init__` (`self.watchlist_bundle_service = WatchlistBundleService(
    subscriptions_db)`); `local_watchlists_service.db_factory()` is what
    every watchlists read and write goes through. task-1631 pinned that they
    resolve the same on-disk file, because `db_factory` used to be a lambda
    re-resolving `get_subscriptions_db_path()` per call and
    `_build_test_app`'s patch could fall out of scope before it ran.

    task-15463 made the factory return the ONE instance wired at startup
    instead of constructing a fresh `SubscriptionsDB` per call (~52-statement
    schema script each time), so the assertion is now identity -- strictly
    stronger than the path comparison it replaces, and the production
    property AC#1 asks for: one instance per app session. The path assertion
    is kept underneath it so a failure still says which files diverged.
    """
    app = _build_test_app()

    eager_db = app.watchlist_bundle_service.db
    lazy_db = app.local_watchlists_service.db_factory()

    assert lazy_db.db_path == eager_db.db_path, (
        "local_watchlists_service.db_factory() resolved a DIFFERENT on-disk "
        f"file ({lazy_db.db_path}) than the eagerly-built subscriptions_db "
        f"({eager_db.db_path}) -- the get_subscriptions_db_path patch fell "
        "out of scope before this call, splitting the app across two databases"
    )
    assert lazy_db is eager_db, (
        "the watchlists service must hand out the SAME SubscriptionsDB the "
        "rest of the app was wired with (task-15463), not a second instance "
        "on the same file"
    )


def test_file_notes_owner_is_injected_into_fresh_library_workspaces(
    monkeypatch,
    tmp_path,
):
    """Fresh production Library screens share only the app-scoped owner."""
    from tldw_chatbook.Notes.file_notes_session_owner import (
        FileSystemIdentity,
        HeadIdentity,
        IndexBaseline,
        IndexEntry,
        RepositoryIdentity,
        SessionChangeGroup,
        SessionGitRow,
        SessionGitStatus,
        StagingOwnership,
    )
    from tldw_chatbook.UI.Screens import library_screen as library_module
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    constructed = []

    class WorkspaceProbe:
        def __init__(self, *, session_owner):
            self.session_owner = session_owner
            self.editor = object()
            self.replica = object()
            self.service = object()
            constructed.append(self)

    monkeypatch.setattr(
        library_module,
        "LibraryFileNotesWorkspace",
        WorkspaceProbe,
    )
    app = _build_test_app()
    first_screen = app._create_navigation_screen("library", LibraryScreen)
    first = first_screen._library_file_notes_workspace_factory()

    binding = app.file_notes_session_owner.select_root(tmp_path / "notes")
    assert app.file_notes_session_owner.record_change(
        binding,
        SessionChange("modified", "note.md"),
    )
    filesystem_identity = FileSystemIdentity(device=1, inode=2)
    repository = RepositoryIdentity(
        worktree_root="/repo",
        git_dir="/repo/.git",
        git_common_dir="/repo/.git",
        worktree_identity=filesystem_identity,
        git_dir_identity=filesystem_identity,
        git_common_dir_identity=filesystem_identity,
    )
    group = SessionChangeGroup(
        group_id=1,
        endpoints=("note.md",),
        source_path="note.md",
        destination_path=None,
        current_path="note.md",
        latest_action="modified",
        latest_sequence=1,
    )
    staged_entry = IndexEntry("note.md", "100644", "a" * 40)
    ownership = StagingOwnership(
        repository=repository,
        head=HeadIdentity.attached("refs/heads/main", "b" * 40),
        approved_endpoint_topology=("note.md",),
        approved_move_edges=(),
        approved_current_path="note.md",
        original_baselines={"note.md": IndexBaseline(None)},
        post_stage_entries={"note.md": staged_entry},
    )
    status_generation = app.file_notes_session_owner.next_status_generation(binding)
    assert status_generation is not None
    status = SessionGitStatus(
        binding_generation=binding.generation,
        status_generation=status_generation,
        state="ready",
        rows=(SessionGitRow(group, "owned", unstage_eligible=True),),
        repository=repository,
        head=ownership.head,
    )
    assert app.file_notes_session_owner.publish_trust(binding, repository)
    assert app.file_notes_session_owner.publish_status(binding, status)
    assert app.file_notes_session_owner.publish_ownership(binding, {1: ownership})

    second_screen = app._create_navigation_screen("library", LibraryScreen)
    second = second_screen._library_file_notes_workspace_factory()

    assert constructed == [first, second]
    assert first.session_owner is app.file_notes_session_owner
    assert second.session_owner is app.file_notes_session_owner
    assert first is not second
    assert first.editor is not second.editor
    assert first.replica is not second.replica
    assert first.service is not second.service
    assert [
        change.change.relative_path
        for change in second.session_owner.snapshot(binding).changes
    ] == ["note.md"]
    retained = second.session_owner.snapshot(binding)
    assert retained.trusted_repository == repository
    assert retained.git_status == status
    assert retained.staging_ownership == {1: ownership}
    assert retained.git_status.rows[0].unstage_eligible

    replacement = app.file_notes_session_owner.select_root(tmp_path / "replacement")
    cleared = app.file_notes_session_owner.snapshot(replacement)
    assert cleared.changes == ()
    assert cleared.trusted_repository is None
    assert cleared.git_status is None
    assert not cleared.staging_ownership
    replacement_app = _build_test_app()
    assert replacement_app.file_notes_session_owner is not app.file_notes_session_owner
    assert replacement_app.file_notes_session_owner.current_binding() is None


@pytest.mark.asyncio
async def test_file_notes_new_app_owner_classifies_prior_stage_as_external_without_unstage(
    tmp_path,
):
    """A replacement app observes prior-process staging without inheriting authority."""
    git = shutil.which("git")
    if git is None:
        pytest.skip("Git is not installed")
    repository = tmp_path / "repository"
    repository.mkdir()

    def run_git(*arguments: str) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            [git, *arguments],
            cwd=repository,
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    run_git("init", "--initial-branch=main")
    run_git("config", "--local", "user.name", "Chatbook Test")
    run_git("config", "--local", "user.email", "chatbook@example.invalid")
    note = repository / "note.md"
    note.write_text("initial\n", encoding="utf-8")
    run_git("add", "--", "note.md")
    run_git("commit", "-m", "initial")
    note.write_text("edited\n", encoding="utf-8")

    prior_app = _build_test_app()
    replacement_app = _build_test_app()
    prior_owner = prior_app.file_notes_session_owner
    replacement_owner = replacement_app.file_notes_session_owner
    prior_service = prior_owner.attached_git_service()
    replacement_service = replacement_owner.attached_git_service()
    assert prior_service is not None
    assert replacement_service is not None
    try:
        prior_binding = prior_owner.select_root(repository)
        assert prior_owner.record_change(
            prior_binding,
            SessionChange("modified", "note.md"),
        )
        discovery = await prior_service.discover(prior_binding)
        assert discovery.repository is not None
        assert prior_owner.publish_trust(prior_binding, discovery.repository)
        result = await prior_service.start_stage(prior_binding, (1,))
        assert result.state == "success"
        assert prior_owner.snapshot(prior_binding).staging_ownership
        assert run_git("diff", "--cached", "--name-only").stdout == b"note.md\n"

        replacement_binding = replacement_owner.select_root(repository)
        assert replacement_owner.record_change(
            replacement_binding,
            SessionChange("modified", "note.md"),
        )
        replacement_discovery = await replacement_service.discover(
            replacement_binding
        )
        assert replacement_discovery.repository is not None
        assert replacement_owner.publish_trust(
            replacement_binding,
            replacement_discovery.repository,
        )
        status = await replacement_service.start_status(
            replacement_binding,
            replacement_owner.snapshot(replacement_binding).changes,
        )

        assert status.state == "ready"
        assert len(status.rows) == 1
        assert status.rows[0].state == "external_staged"
        assert not status.rows[0].unstage_eligible
        assert not replacement_owner.snapshot(
            replacement_binding
        ).staging_ownership
    finally:
        await prior_owner.shutdown_async()
        await replacement_owner.shutdown_async()


@pytest.mark.asyncio
async def test_file_notes_navigation_transition_blocks_mutation_until_switch_finishes(
    monkeypatch,
    tmp_path,
):
    """Transition admission immediately after flush closes the Stage race."""
    app = _build_test_app()
    app._initial_screen_pushed = True
    owner = app.file_notes_session_owner
    binding = owner.select_root(tmp_path / "notes")
    switched = []

    class FakeTargetScreen:
        screen_name = "chat"

        def __init__(self, app_instance):
            self.app_instance = app_instance

    class FakeOutgoingScreen:
        screen_name = "library"

        async def flush_pending_work(self):
            return True

        def acquire_navigation_transition(self):
            lease = owner.try_acquire_transition(binding, "screen")
            return False if lease is None else lease.release

    async def fake_switch_screen(screen):
        admission = owner.admit_mutation(binding)
        assert admission.lease is None
        assert admission.reason == "transition_active"
        switched.append(screen)

    outgoing = FakeOutgoingScreen()
    monkeypatch.setattr(type(app), "screen", property(lambda self: outgoing))
    monkeypatch.setattr(
        app,
        "_resolve_screen_navigation_target",
        lambda _target: ("chat", "chat", FakeTargetScreen),
    )
    monkeypatch.setattr(app, "switch_screen", fake_switch_screen)

    await app.handle_screen_navigation(NavigateToScreen("chat"))

    assert len(switched) == 1
    after_switch = owner.try_acquire_mutation(binding)
    assert after_switch is not None
    after_switch.release()


@pytest.mark.asyncio
async def test_file_notes_mutation_admitted_during_flush_vetoes_navigation(
    monkeypatch,
    tmp_path,
):
    """A Stage lease won during flush leaves the current screen mounted."""
    app = _build_test_app()
    app._initial_screen_pushed = True
    owner = app.file_notes_session_owner
    binding = owner.select_root(tmp_path / "notes")
    flush_started = asyncio.Event()
    finish_flush = asyncio.Event()
    switched = []

    class FakeTargetScreen:
        screen_name = "chat"

        def __init__(self, app_instance):
            self.app_instance = app_instance

    class FakeOutgoingScreen:
        screen_name = "library"

        async def flush_pending_work(self):
            flush_started.set()
            await finish_flush.wait()
            return not owner.mutation_active(binding)

        def acquire_navigation_transition(self):
            raise AssertionError("a vetoed flush must not attempt transition admission")

    outgoing = FakeOutgoingScreen()
    monkeypatch.setattr(type(app), "screen", property(lambda self: outgoing))
    monkeypatch.setattr(
        app,
        "_resolve_screen_navigation_target",
        lambda _target: ("chat", "chat", FakeTargetScreen),
    )
    monkeypatch.setattr(
        app,
        "switch_screen",
        lambda screen: switched.append(screen),
    )

    navigation = asyncio.create_task(
        app.handle_screen_navigation(NavigateToScreen("chat"))
    )
    await _wait_for_background_signal(
        flush_started, navigation, what="the outgoing screen's flush"
    )
    mutation = owner.try_acquire_mutation(binding)
    assert mutation is not None
    finish_flush.set()
    await _await_background_task(navigation, what="the vetoed navigation")

    assert switched == []
    assert app.screen is outgoing
    mutation.release()


@pytest.mark.asyncio
async def test_file_notes_source_transition_blocks_mutation_through_recompose(
    monkeypatch,
    tmp_path,
):
    """The Files-to-Database switch holds exact source admission to completion."""
    from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_NOTES
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    app = _build_test_app()
    owner = app.file_notes_session_owner
    binding = owner.select_root(tmp_path / "notes")

    class WorkspaceProbe:
        async def flush_pending_work(self):
            return not owner.mutation_active(binding)

        def acquire_transition(self, kind):
            lease = owner.try_acquire_transition(binding, kind)
            return False if lease is None else lease.release

    class EventProbe:
        def stop(self):
            return None

    workspace = WorkspaceProbe()
    screen = LibraryScreen(app, file_notes_workspace_factory=lambda: workspace)
    screen._library_file_notes_workspace = workspace
    screen._library_notes_source = "files"
    screen._library_selected_row_id = LIBRARY_ROW_BROWSE_NOTES
    recompose_calls = []

    async def recompose():
        admission = owner.admit_mutation(binding)
        assert admission.lease is None
        assert admission.reason == "transition_active"
        recompose_calls.append(True)

    monkeypatch.setattr(screen, "recompose", recompose)

    await screen._show_library_database_notes(EventProbe())

    assert screen._library_notes_source == "database"
    assert recompose_calls == [True]
    after_recompose = owner.try_acquire_mutation(binding)
    assert after_recompose is not None
    after_recompose.release()


@pytest.mark.asyncio
async def test_file_notes_create_route_returns_to_database_notes(monkeypatch):
    """Create Note leaves Files only after its source transition is admitted."""
    from tldw_chatbook.Library.library_shell_state import (
        LIBRARY_ROW_BROWSE_NOTES,
        LIBRARY_ROW_CREATE_NOTE,
    )
    from tldw_chatbook.UI.Screens.library_screen import (
        LIBRARY_NOTES_SOURCE_DATABASE,
        LIBRARY_NOTES_SOURCE_FILES,
        LibraryScreen,
    )

    app = _build_test_app()
    transition_events = []

    class WorkspaceProbe:
        async def flush_pending_work(self):
            transition_events.append("flushed")
            return True

        def acquire_transition(self, kind):
            transition_events.append(f"admitted:{kind}")
            return lambda: transition_events.append("released")

    workspace = WorkspaceProbe()
    screen = LibraryScreen(app, file_notes_workspace_factory=lambda: workspace)
    screen._library_file_notes_workspace = workspace
    screen._library_notes_source = LIBRARY_NOTES_SOURCE_FILES
    screen._library_selected_row_id = LIBRARY_ROW_BROWSE_NOTES
    recompose = AsyncMock()
    monkeypatch.setattr(screen, "recompose", recompose)

    await screen._select_library_rail_row(LIBRARY_ROW_CREATE_NOTE)

    assert transition_events == ["flushed", "admitted:source", "released"]
    assert screen._library_selected_row_id == LIBRARY_ROW_CREATE_NOTE
    assert screen._library_notes_source == LIBRARY_NOTES_SOURCE_DATABASE
    assert screen.check_action("library_notes_escape", ()) is True
    recompose.assert_awaited_once()


@pytest.mark.asyncio
async def test_file_notes_collections_source_transition_blocks_mutation_through_targeted_reconcile(
    monkeypatch,
    tmp_path,
):
    """Files-to-Collections keeps source admission through targeted reconcile."""
    from tldw_chatbook.Library.library_shell_state import (
        LIBRARY_ROW_BROWSE_COLLECTIONS,
        LIBRARY_ROW_BROWSE_NOTES,
    )
    from tldw_chatbook.Library.library_notes_session import (
        NoteFlushOutcome,
        NoteFlushOutcomeKind,
    )
    from tldw_chatbook.UI.Screens.library_screen import (
        LibraryEntryReconcileResult,
        LibraryScreen,
    )

    app = _build_test_app()
    owner = app.file_notes_session_owner
    binding = owner.select_root(tmp_path / "notes")
    sync_returned = asyncio.Event()
    reconcile_started = asyncio.Event()
    finish_reconcile = asyncio.Event()

    class WorkspaceProbe:
        async def flush_pending_work(self):
            return not owner.mutation_active(binding)

        def acquire_transition(self, kind):
            lease = owner.try_acquire_transition(binding, kind)
            return False if lease is None else lease.release

    workspace = WorkspaceProbe()
    screen = LibraryScreen(app, file_notes_workspace_factory=lambda: workspace)
    screen._library_file_notes_workspace = workspace
    screen._library_notes_source = "files"
    screen._library_selected_row_id = LIBRARY_ROW_BROWSE_NOTES
    screen._library_collections_loaded = False

    async def sync_collections_panel(*, refresh_snapshot, wait_for_recompose):
        assert refresh_snapshot is True
        assert wait_for_recompose is True
        screen._library_collections_loaded = True
        sync_returned.set()
        reconcile_started.set()
        admission = owner.admit_mutation(binding)
        assert admission.lease is None
        assert admission.reason == "transition_active"
        await finish_reconcile.wait()
        return LibraryEntryReconcileResult.APPLIED

    async def flush_note():
        # task-3316: this stub MUST honour ``_flush_library_note_save``'s
        # contract. It was authored against the old ``-> None`` signature and
        # kept returning None after PR #1439 retyped the seam to
        # NoteFlushOutcome, so the awaited path died on
        # ``NoneType.kind`` inside the fire-and-forget task below.
        return NoteFlushOutcome(NoteFlushOutcomeKind.PERMITTED)

    async def flush_editor():
        return True

    monkeypatch.setattr(
        screen,
        "_sync_collections_panel",
        sync_collections_panel,
    )
    monkeypatch.setattr(screen, "refresh", lambda *, recompose: None)
    monkeypatch.setattr(screen, "_flush_library_note_save", flush_note)
    monkeypatch.setattr(screen, "_flush_library_prompt_save", flush_editor)
    monkeypatch.setattr(screen, "_flush_library_skill_save", flush_editor)

    source_switch = asyncio.create_task(
        screen._select_library_rail_row(LIBRARY_ROW_BROWSE_COLLECTIONS)
    )
    await _wait_for_background_signal(
        sync_returned, source_switch, what="the Collections snapshot refresh"
    )
    await _wait_for_background_signal(
        reconcile_started, source_switch, what="the Collections targeted reconcile"
    )

    assert reconcile_started.is_set()
    assert not source_switch.done()
    admission = owner.admit_mutation(binding)
    assert admission.lease is None
    assert admission.reason == "transition_active"

    finish_reconcile.set()
    await _await_background_task(source_switch, what="the Collections source switch")
    after_recompose = owner.try_acquire_mutation(binding)
    assert after_recompose is not None
    after_recompose.release()


@pytest.mark.asyncio
async def test_file_notes_mutation_admitted_during_source_flush_vetoes_switch(
    monkeypatch,
    tmp_path,
):
    """Stage winning during source flush preserves the mounted Files source."""
    from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_NOTES
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    app = _build_test_app()
    owner = app.file_notes_session_owner
    binding = owner.select_root(tmp_path / "notes")
    flush_started = asyncio.Event()
    finish_flush = asyncio.Event()

    class WorkspaceProbe:
        async def flush_pending_work(self):
            flush_started.set()
            await finish_flush.wait()
            return not owner.mutation_active(binding)

        def acquire_transition(self, kind):
            raise AssertionError("a vetoed flush must not admit a source transition")

    class EventProbe:
        def stop(self):
            return None

    workspace = WorkspaceProbe()
    screen = LibraryScreen(app, file_notes_workspace_factory=lambda: workspace)
    screen._library_file_notes_workspace = workspace
    screen._library_notes_source = "files"
    screen._library_selected_row_id = LIBRARY_ROW_BROWSE_NOTES
    recompose = AsyncMock()
    monkeypatch.setattr(screen, "recompose", recompose)

    source_switch = asyncio.create_task(
        screen._show_library_database_notes(EventProbe())
    )
    await _wait_for_background_signal(
        flush_started, source_switch, what="the File Notes workspace flush"
    )
    mutation = owner.try_acquire_mutation(binding)
    assert mutation is not None
    finish_flush.set()
    await _await_background_task(source_switch, what="the vetoed source switch")

    assert screen._library_notes_source == "files"
    recompose.assert_not_awaited()
    mutation.release()


def test_check_action_gates_notes_files_back_to_active_files_mode():
    """task-2850 AC3: the Files-mode Escape binding only fires while Files
    mode genuinely owns the Notes canvas -- everywhere else it must behave
    as if unbound (``check_action`` returning ``False``), exactly like the
    sibling ``library_skill_back`` gate it shares the "escape" key with.
    """
    from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_NOTES
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    app = _build_test_app()
    workspace = object()
    screen = LibraryScreen(app, file_notes_workspace_factory=lambda: workspace)

    # Default state: Database Notes, no workspace mounted -- inactive.
    assert screen.check_action("library_notes_files_back", ()) is False

    # Files mode selected but the row isn't Notes (stale source flag from a
    # prior visit) -- still inactive, mirroring ``_file_notes_active()``.
    screen._library_notes_source = "files"
    screen._library_file_notes_workspace = workspace
    assert screen.check_action("library_notes_files_back", ()) is False

    # Files mode genuinely owns the Notes canvas -- active.
    screen._library_selected_row_id = LIBRARY_ROW_BROWSE_NOTES
    assert screen.check_action("library_notes_files_back", ()) is True
    assert screen.check_action("library_notes_escape", ()) is False

    # Back to Database Notes -- inactive again.
    screen._library_notes_source = "database"
    assert screen.check_action("library_notes_files_back", ()) is False

    # Unrelated actions are untouched by the new gate. "library_rag_use_
    # in_console" is no longer a valid stand-in for "unrelated" -- task-2858
    # AC#2 gates it too (see test_check_action_gates_rag_use_in_console_to_
    # search_row below) -- so this uses a genuinely nonexistent action,
    # which check_action's final `return True` fallback still covers.
    assert screen.check_action("some_nonexistent_action", ()) is True


@pytest.mark.asyncio
async def test_action_library_notes_files_back_returns_to_database(
    monkeypatch,
    tmp_path,
):
    """task-2850 AC3: Escape's action reuses the SAME guarded return path as
    the "Database" strip button (``_return_to_library_database_notes``) --
    one seam, not a parallel key-driven shortcut that could drift from the
    button's flush/leave-guard sequence.
    """
    from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_NOTES
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    app = _build_test_app()
    owner = app.file_notes_session_owner
    binding = owner.select_root(tmp_path / "notes")

    class WorkspaceProbe:
        async def flush_pending_work(self):
            return not owner.mutation_active(binding)

        def acquire_transition(self, kind):
            lease = owner.try_acquire_transition(binding, kind)
            return False if lease is None else lease.release

        def cancel_reload_confirmation(self):
            # Faithful to the real workspace for this test's state: with no
            # reload confirmation pending, ``LibraryFileNotesWorkspace.
            # cancel_reload_confirmation`` returns False
            # (``_dismiss_reload_confirmation``'s None guard; task-15767).
            return False

    workspace = WorkspaceProbe()
    screen = LibraryScreen(app, file_notes_workspace_factory=lambda: workspace)
    screen._library_file_notes_workspace = workspace
    screen._library_notes_source = "files"
    screen._library_selected_row_id = LIBRARY_ROW_BROWSE_NOTES
    recompose_calls = []

    async def recompose():
        recompose_calls.append(True)

    monkeypatch.setattr(screen, "recompose", recompose)
    footer_calls = []
    monkeypatch.setattr(
        screen,
        "_register_footer_shortcuts",
        lambda: footer_calls.append(screen._library_notes_source),
    )

    await screen.action_library_notes_files_back()

    assert screen._library_notes_source == "database"
    assert recompose_calls == [True]
    # The footer's "esc" hint must drop the moment the source flips back,
    # not on some later, separate recompose (task-2850).
    assert footer_calls == ["database"]
    after_recompose = owner.try_acquire_mutation(binding)
    assert after_recompose is not None
    after_recompose.release()


@pytest.mark.asyncio
async def test_action_library_notes_files_back_cancels_open_reload_confirmation_first(
    monkeypatch,
    tmp_path,
):
    """task-15767 AC3 (task-15503's Escape contract at the screen seam):
    pressing back while the destructive-reload confirmation is open must
    CANCEL the confirmation and stay on Files -- it must never run the
    leave flush/transition or navigate away while the inline decision is
    still pending. Only the SECOND back (confirmation gone) takes the
    normal guarded return to Database Notes."""
    from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_NOTES
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    app = _build_test_app()
    owner = app.file_notes_session_owner
    binding = owner.select_root(tmp_path / "notes")
    confirmation_open = True
    cancel_returns = []

    class WorkspaceProbe:
        async def flush_pending_work(self):
            assert not confirmation_open, (
                "back-mid-confirmation must cancel the pending reload "
                "decision, not run the leave flush"
            )
            return not owner.mutation_active(binding)

        def acquire_transition(self, kind):
            assert not confirmation_open, (
                "back-mid-confirmation must cancel the pending reload "
                "decision, not admit a source transition"
            )
            lease = owner.try_acquire_transition(binding, kind)
            return False if lease is None else lease.release

        def cancel_reload_confirmation(self):
            # Faithful to the real workspace: True exactly when a pending
            # confirmation was dismissed, False when none was open
            # (``_dismiss_reload_confirmation``'s None guard).
            nonlocal confirmation_open
            was_open = confirmation_open
            confirmation_open = False
            cancel_returns.append(was_open)
            return was_open

    workspace = WorkspaceProbe()
    screen = LibraryScreen(app, file_notes_workspace_factory=lambda: workspace)
    screen._library_file_notes_workspace = workspace
    screen._library_notes_source = "files"
    screen._library_selected_row_id = LIBRARY_ROW_BROWSE_NOTES
    recompose_calls = []

    async def recompose():
        recompose_calls.append(True)

    monkeypatch.setattr(screen, "recompose", recompose)
    footer_calls = []
    monkeypatch.setattr(
        screen,
        "_register_footer_shortcuts",
        lambda: footer_calls.append(screen._library_notes_source),
    )

    # First back: cancels the open confirmation and STAYS on Files.
    await screen.action_library_notes_files_back()
    assert cancel_returns == [True]
    assert screen._library_notes_source == "files"
    assert recompose_calls == []
    # The footer must drop its "esc cancel reload" hint immediately
    # (task-15503 registered that hint while the decision is pending).
    assert footer_calls == ["files"]

    # Second back: no confirmation pending -- the normal guarded return
    # runs and lands on Database Notes.
    await screen.action_library_notes_files_back()
    assert cancel_returns == [True, False]
    assert screen._library_notes_source == "database"
    assert recompose_calls == [True]
    assert footer_calls == ["files", "database"]
    after_recompose = owner.try_acquire_mutation(binding)
    assert after_recompose is not None
    after_recompose.release()


def test_files_back_navigation_workspace_contract_matches_real_workspace():
    """task-15767 AC2: the regression behind this task was production
    widening the File Notes workspace contract (task-15503's
    ``cancel_reload_confirmation`` call in ``action_library_notes_files_
    back``) while this file's ``WorkspaceProbe`` doubles stayed on the old
    shape -- surfacing as an opaque ``AttributeError`` mid-path. Pin the
    contract structurally: enumerate every attribute the Files-mode leave
    seams actually access on the workspace, and require (a) that set to be
    exactly the pinned contract the probes implement, and (b) every pinned
    name to exist on the REAL ``LibraryFileNotesWorkspace`` with the
    async-ness the screen assumes. Future widening then fails HERE, naming
    the probes to update, instead of deep inside an unrelated test."""
    import ast
    import inspect
    import textwrap

    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
    from tldw_chatbook.Widgets.Library.library_file_notes_workspace import (
        LibraryFileNotesWorkspace,
    )

    def workspace_attribute_accesses(func) -> set[str]:
        """Attributes accessed on the workspace object inside ``func``."""
        tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
        found: set[str] = set()

        class Visitor(ast.NodeVisitor):
            def visit_Attribute(self, node: ast.Attribute) -> None:
                value = node.value
                if isinstance(value, ast.Name) and value.id == "workspace":
                    found.add(node.attr)
                elif (
                    isinstance(value, ast.Attribute)
                    and value.attr == "_library_file_notes_workspace"
                ):
                    found.add(node.attr)
                self.generic_visit(node)

        Visitor().visit(tree)
        return found

    # The seams the Files-mode back navigation runs through (Escape action,
    # the shared strip-button return path, and its two workspace helpers).
    back_seams = (
        LibraryScreen.action_library_notes_files_back,
        LibraryScreen._return_to_library_database_notes,
        LibraryScreen._flush_active_file_notes,
        LibraryScreen._acquire_file_notes_transition,
    )
    called = set()
    for seam in back_seams:
        called |= workspace_attribute_accesses(seam)
    probe_contract = {
        "flush_pending_work",
        "acquire_transition",
        "cancel_reload_confirmation",
    }
    assert called == probe_contract, (
        "the Files-mode back-navigation seams now touch a different "
        f"workspace contract ({sorted(called)}) than the pinned probe "
        f"contract ({sorted(probe_contract)}) -- update every "
        "WorkspaceProbe in this file that those seams can reach, then "
        "re-pin here (task-15767)"
    )

    # The cancel branch also re-registers footer shortcuts, whose chooser
    # reads ``reload_confirmation_active``; the nav tests patch
    # ``_register_footer_shortcuts`` out, so probes never see it -- but the
    # real widget must still satisfy it.
    footer_contract = workspace_attribute_accesses(
        LibraryScreen._library_footer_shortcuts_for_current_state
    )
    assert "reload_confirmation_active" in footer_contract

    # Every pinned name must exist on the REAL workspace with the
    # async-ness the screen assumes (flush is awaited; the rest are called
    # synchronously) -- so this pin cannot itself drift from the widget.
    for name in probe_contract | footer_contract:
        assert inspect.getattr_static(LibraryFileNotesWorkspace, name) is not None
    assert inspect.iscoroutinefunction(LibraryFileNotesWorkspace.flush_pending_work)
    assert not inspect.iscoroutinefunction(LibraryFileNotesWorkspace.acquire_transition)
    assert not inspect.iscoroutinefunction(
        LibraryFileNotesWorkspace.cancel_reload_confirmation
    )
    assert isinstance(
        inspect.getattr_static(LibraryFileNotesWorkspace, "reload_confirmation_active"),
        property,
    )


def test_check_action_gates_media_viewer_back_to_active_viewer():
    """task-2856 AC2: the media viewer's Escape binding only fires while
    the media canvas genuinely shows its viewer sub-view -- everywhere
    else it must behave as if unbound, the same posture every other
    context-gated "escape" binding on this screen uses."""
    from tldw_chatbook.Library.library_shell_state import (
        LIBRARY_ROW_BROWSE_MEDIA,
        LIBRARY_ROW_BROWSE_NOTES,
    )
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    app = _build_test_app()
    screen = LibraryScreen(app)

    # Default state: landing, no row selected -- inactive.
    assert screen.check_action("library_media_viewer_back", ()) is False

    # Media selected but showing the LIST, not the viewer -- inactive.
    screen._library_selected_row_id = LIBRARY_ROW_BROWSE_MEDIA
    screen._library_media_view = "list"
    assert screen.check_action("library_media_viewer_back", ()) is False

    # Media viewer genuinely open -- active.
    screen._library_media_view = "viewer"
    assert screen.check_action("library_media_viewer_back", ()) is True

    # A different row selected (stale view flag) -- inactive.
    screen._library_selected_row_id = LIBRARY_ROW_BROWSE_NOTES
    assert screen.check_action("library_media_viewer_back", ()) is False

    # Unrelated actions are untouched by the new gate. "library_rag_use_
    # in_console" is no longer a valid stand-in for "unrelated" -- task-2858
    # AC#2 gates it too -- so this uses a genuinely nonexistent action.
    assert screen.check_action("some_nonexistent_action", ()) is True


def test_register_footer_shortcuts_distinguishes_plain_viewer_from_a_media_sub_state():
    """task-2856 review round 3: ``LIBRARY_DETAIL_BACK_SHORTCUTS`` used to be
    selected whenever the media canvas showed its viewer, unconditionally
    advertising "esc back to list" -- but ``action_library_media_viewer_
    back`` (review round 2) only steps back ONE level while an edit/
    delete-confirm/analysis-edit sub-state is active (viewer -> plain
    viewer, NOT viewer -> list), so a SECOND Escape is actually needed to
    reach the list in that state. The footer must advertise a different,
    accurate hint whenever any of the three sub-state flags is set, not
    repeat the plain viewer's "back to list" claim."""
    from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_MEDIA
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    app = _build_test_app()
    screen = LibraryScreen(app)
    screen._library_selected_row_id = LIBRARY_ROW_BROWSE_MEDIA
    screen._library_media_view = "viewer"

    # Plain read-only viewer -- no sub-state active -- Escape genuinely
    # goes straight to the list, so "back to list" is true here.
    screen._library_media_editing = False
    screen._library_media_confirming_delete = False
    screen._library_media_editing_analysis = False
    screen._register_footer_shortcuts()
    _source, plain_shortcuts = screen._footer_shortcut_registration
    assert dict(plain_shortcuts)["esc"] == "back to list"

    # Mid-edit sub-state active -- Escape only steps back to the plain
    # viewer (see action_library_media_viewer_back's staged exit), so the
    # footer must NOT repeat "back to list" here.
    screen._library_media_editing = True
    screen._register_footer_shortcuts()
    _source, edit_shortcuts = screen._footer_shortcut_registration
    assert dict(edit_shortcuts)["esc"] != "back to list"
    assert edit_shortcuts != plain_shortcuts

    # Same for the delete-confirm and analysis-edit sub-states.
    screen._library_media_editing = False
    screen._library_media_confirming_delete = True
    screen._register_footer_shortcuts()
    _source, delete_shortcuts = screen._footer_shortcut_registration
    assert dict(delete_shortcuts)["esc"] != "back to list"

    screen._library_media_confirming_delete = False
    screen._library_media_editing_analysis = True
    screen._register_footer_shortcuts()
    _source, analysis_shortcuts = screen._footer_shortcut_registration
    assert dict(analysis_shortcuts)["esc"] != "back to list"


def test_register_footer_shortcuts_advertises_skill_editor_working_keys():
    """task-3020 AC4: the skill editor's footer must advertise its own
    working ``ctrl+s``/``esc`` -- before this task,
    ``_library_footer_shortcuts_for_current_state`` had no skill-editor
    branch, so it fell through to ``_library_list_canvas_showing_list()``
    (False here -- the skills view is "editor", not "list") and landed on
    the bare ``LIBRARY_GENERAL_SHORTCUTS`` (just "/" and "F6"), an
    asymmetry beside the note/prompt editors, which already advertise
    their own "esc" hint."""
    from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_SKILLS
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    app = _build_test_app()
    screen = LibraryScreen(app)
    screen._library_selected_row_id = LIBRARY_ROW_BROWSE_SKILLS
    screen._library_skills_view = "editor"

    screen._register_footer_shortcuts()
    _source, shortcuts = screen._footer_shortcut_registration
    shortcut_dict = dict(shortcuts)
    assert shortcut_dict["esc"] == "back to skills list"
    assert shortcut_dict["ctrl+s"] == "save skill"

    # The plain skills LIST is unaffected -- it still advertises "focus
    # rail", never the editor's keys.
    screen._library_skills_view = "list"
    screen._register_footer_shortcuts()
    _source, list_shortcuts = screen._footer_shortcut_registration
    assert "ctrl+s" not in dict(list_shortcuts)
    assert dict(list_shortcuts)["esc"] == "focus rail"


def test_action_show_workbench_help_includes_skill_editor_keys(monkeypatch):
    """task-3020 AC4: F1 inherits the fix automatically via the shared
    ``_library_footer_shortcuts_for_current_state`` helper both the
    footer and F1 read -- pinned directly rather than just trusted."""
    from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_SKILLS
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
    from tldw_chatbook.UI.Workbench.help import WorkbenchHelpPanel

    app = _build_test_app()
    screen = LibraryScreen(app)
    screen._library_selected_row_id = LIBRARY_ROW_BROWSE_SKILLS
    screen._library_skills_view = "editor"

    pushed = []

    class _FakeApp:
        def push_screen(self, panel):
            pushed.append(panel)

    # An un-mounted Screen's ``.app`` raises NoActiveAppError -- mirrors
    # ``test_action_show_workbench_help_filters_bindings_by_check_action``'s
    # own fake-app override.
    monkeypatch.setattr(
        LibraryScreen, "app", property(lambda self: _FakeApp()), raising=False
    )

    screen.action_show_workbench_help()

    assert len(pushed) == 1
    panel = pushed[0]
    assert isinstance(panel, WorkbenchHelpPanel)
    shortcut_keys = {key for key, _description in panel.state.shortcuts}
    assert "ctrl+s" in shortcut_keys
    assert "esc" in shortcut_keys


def test_check_action_gates_note_editor_back_to_active_editor():
    """task-2856 AC2: the note editor's Escape binding only fires for the
    DATABASE note editor -- Files mode (its own dedicated Escape gate) and
    the plain list must both leave it inactive."""
    from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_NOTES
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    app = _build_test_app()
    screen = LibraryScreen(app)

    assert screen.check_action("library_note_editor_back", ()) is False

    screen._library_selected_row_id = LIBRARY_ROW_BROWSE_NOTES
    screen._library_notes_view = "list"
    assert screen.check_action("library_note_editor_back", ()) is False

    screen._library_notes_view = "editor"
    assert screen.check_action("library_note_editor_back", ()) is True

    # Files mode never activates this gate, even mid-editor-looking state --
    # it owns a dedicated Escape binding instead (``library_notes_files_back``).
    screen._library_notes_source = "files"
    assert screen.check_action("library_note_editor_back", ()) is False


def test_check_action_gates_prompt_editor_back_to_active_editor():
    """task-2856 AC2: the prompt editor's Escape binding fires for BOTH the
    Browse > Prompts editor and the Create > New prompt editor -- mirroring
    ``_library_skill_editor_active``'s dual-row-id gate, since
    ``_enter_library_prompt_create_editor`` never reassigns
    ``_library_selected_row_id`` away from ``LIBRARY_ROW_CREATE_PROMPT``."""
    from tldw_chatbook.Library.library_shell_state import (
        LIBRARY_ROW_BROWSE_PROMPTS,
        LIBRARY_ROW_CREATE_PROMPT,
    )
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    app = _build_test_app()
    screen = LibraryScreen(app)

    assert screen.check_action("library_prompt_editor_back", ()) is False

    screen._library_selected_row_id = LIBRARY_ROW_BROWSE_PROMPTS
    screen._library_prompts_view = "list"
    assert screen.check_action("library_prompt_editor_back", ()) is False

    screen._library_prompts_view = "editor"
    assert screen.check_action("library_prompt_editor_back", ()) is True

    screen._library_selected_row_id = LIBRARY_ROW_CREATE_PROMPT
    assert screen.check_action("library_prompt_editor_back", ()) is True


def test_check_action_gates_list_focus_rail_to_showing_list():
    """task-2856 AC2: the "focus rail" Escape binding only fires while one
    of the four list canvases shows its plain LIST sub-view -- viewer/
    editor/create/sync sub-views, Files mode, the landing, and other
    canvases (e.g. Search/RAG) must all leave it inactive."""
    from tldw_chatbook.Library.library_shell_state import (
        LIBRARY_ROW_BROWSE_MEDIA,
        LIBRARY_ROW_BROWSE_NOTES,
        LIBRARY_ROW_BROWSE_PROMPTS,
        LIBRARY_ROW_BROWSE_SEARCH,
        LIBRARY_ROW_BROWSE_SKILLS,
    )
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    app = _build_test_app()
    screen = LibraryScreen(app)

    # Landing -- inactive.
    assert screen.check_action("library_list_focus_rail", ()) is False

    screen._library_selected_row_id = LIBRARY_ROW_BROWSE_MEDIA
    screen._library_media_view = "list"
    assert screen.check_action("library_list_focus_rail", ()) is True
    screen._library_media_view = "viewer"
    assert screen.check_action("library_list_focus_rail", ()) is False

    screen._library_selected_row_id = LIBRARY_ROW_BROWSE_NOTES
    screen._library_notes_view = "list"
    screen._library_notes_source = "database"
    assert screen.check_action("library_list_focus_rail", ()) is True
    screen._library_notes_view = "editor"
    assert screen.check_action("library_list_focus_rail", ()) is False
    screen._library_notes_view = "list"
    screen._library_notes_source = "files"
    assert screen.check_action("library_list_focus_rail", ()) is False

    screen._library_selected_row_id = LIBRARY_ROW_BROWSE_PROMPTS
    screen._library_prompts_view = "list"
    assert screen.check_action("library_list_focus_rail", ()) is True
    screen._library_prompts_view = "editor"
    assert screen.check_action("library_list_focus_rail", ()) is False

    screen._library_selected_row_id = LIBRARY_ROW_BROWSE_SKILLS
    screen._library_skills_view = "list"
    assert screen.check_action("library_list_focus_rail", ()) is True
    screen._library_skills_view = "editor"
    assert screen.check_action("library_list_focus_rail", ()) is False

    # A canvas outside the four list canvases (Search/RAG) -- inactive.
    screen._library_selected_row_id = LIBRARY_ROW_BROWSE_SEARCH
    assert screen.check_action("library_list_focus_rail", ()) is False


def test_check_action_gates_media_bulk_delete_cancel_to_armed_confirm():
    """task-3020 AC2: Escape cancels an ARMED bulk-delete confirmation --
    parity with the single-item viewer confirm's own Escape-cancels
    behavior -- gated so it never fires outside that exact state."""
    from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_MEDIA
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    app = _build_test_app()
    screen = LibraryScreen(app)

    # Landing -- inactive.
    assert screen.check_action("library_media_bulk_delete_cancel", ()) is False

    # Media list, Select mode active, but no confirmation armed -- inactive.
    screen._library_selected_row_id = LIBRARY_ROW_BROWSE_MEDIA
    screen._library_media_view = "list"
    screen._library_media_select_mode = True
    assert screen.check_action("library_media_bulk_delete_cancel", ()) is False

    # The confirmation is armed -- active. Note ``library_list_focus_rail``
    # is ALSO True on this exact state (the list is still genuinely
    # showing) -- see ``test_library_media_bulk_delete_cancel_binding_
    # precedes_focus_rail`` for the ordering guarantee that keeps only
    # ONE of the two from ever actually firing.
    screen._library_media_confirming_bulk_delete = True
    assert screen.check_action("library_media_bulk_delete_cancel", ()) is True
    assert screen.check_action("library_list_focus_rail", ()) is True

    # Cancelling (or completing) the confirmation drops it again.
    screen._library_media_confirming_bulk_delete = False
    assert screen.check_action("library_media_bulk_delete_cancel", ()) is False

    # Unrelated actions are untouched by the new gate.
    assert screen.check_action("some_nonexistent_action", ()) is True


def test_library_media_bulk_delete_cancel_binding_precedes_focus_rail():
    """task-3020 AC2: both ``library_media_bulk_delete_cancel`` and
    ``library_list_focus_rail`` can be simultaneously check_action-True
    while a bulk-delete confirmation is armed on the plain Media list
    (``_library_list_canvas_showing_list()`` stays True -- the toolbar is
    swapped in place, not a distinct sub-view). Textual resolves same-key
    ``Binding``s in DECLARATION ORDER, trying each until one's
    ``check_action`` passes, so the more specific cancel action MUST be
    declared before the broader focus-rail action, or Escape would still
    reproduce the original defect (moving focus to the rail and stranding
    the armed confirm row behind it)."""
    from textual.binding import Binding

    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    escape_actions = []
    for entry in LibraryScreen.BINDINGS:
        if isinstance(entry, Binding):
            key, action = entry.key, entry.action
        elif isinstance(entry, (tuple, list)) and entry:
            key = str(entry[0])
            action = str(entry[1]) if len(entry) > 1 else ""
        else:
            continue
        if key == "escape":
            escape_actions.append(action)

    assert "library_media_bulk_delete_cancel" in escape_actions
    assert "library_list_focus_rail" in escape_actions
    assert escape_actions.index(
        "library_media_bulk_delete_cancel"
    ) < escape_actions.index("library_list_focus_rail")


def test_action_library_media_bulk_delete_cancel_dismisses_confirmation():
    """task-3020 AC2: the Escape action reuses the exact same cancel path
    as the confirm row's own "Cancel" button -- one seam, not a
    duplicated one that could drift."""
    from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_MEDIA
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    app = _build_test_app()
    screen = LibraryScreen(app)
    screen._library_selected_row_id = LIBRARY_ROW_BROWSE_MEDIA
    screen._library_media_view = "list"
    screen._library_media_select_mode = True
    screen._library_media_confirming_bulk_delete = True

    refresh_calls = []
    screen.refresh = lambda recompose=False: refresh_calls.append(recompose)

    screen.action_library_media_bulk_delete_cancel()

    assert screen._library_media_confirming_bulk_delete is False
    # ``_sync_library_canvas`` fails closed to a full recompose here (no
    # ``#library-media-canvas`` mounted on this bare screen) -- the same
    # fallback the button handler's own test relies on.
    assert refresh_calls == [True]


def test_register_footer_shortcuts_advertises_cancel_while_bulk_delete_confirm_armed():
    """task-3020 AC2 footer honesty: while a bulk-delete confirmation is
    armed, the footer must advertise "esc cancel delete", not the plain
    list's "esc focus rail" -- the list canvas is still genuinely showing
    (``_library_list_canvas_showing_list()`` is True), so without a
    dedicated branch the footer would keep the OLD, now-inaccurate hint."""
    from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_MEDIA
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    app = _build_test_app()
    screen = LibraryScreen(app)
    screen._library_selected_row_id = LIBRARY_ROW_BROWSE_MEDIA
    screen._library_media_view = "list"
    screen._library_media_select_mode = True
    screen._library_media_confirming_bulk_delete = True

    screen._register_footer_shortcuts()
    _source, shortcuts = screen._footer_shortcut_registration
    assert dict(shortcuts)["esc"] == "cancel delete"


def test_check_action_gates_rag_use_in_console_to_search_row():
    """task-2858 AC#2 (LIB-09): the "u" binding
    (``library_rag_use_in_console``) had no ``check_action`` gate before
    this task -- its OWN action body already no-ops off the Search/RAG
    row (``if self._library_selected_row_id != LIBRARY_ROW_BROWSE_SEARCH:
    return``), but nothing told ``check_action`` that, so F1's help panel
    kept advertising "u" on every other Library surface (the media
    viewer, browsing skills, etc.) -- reproducing the original finding.
    This pins the gate to the EXACT same predicate the action itself
    uses.
    """
    from tldw_chatbook.Library.library_shell_state import (
        LIBRARY_ROW_BROWSE_MEDIA,
        LIBRARY_ROW_BROWSE_SEARCH,
    )
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    app = _build_test_app()
    screen = LibraryScreen(app)

    # Landing -- inactive.
    assert screen.check_action("library_rag_use_in_console", ()) is False

    # A different canvas (media) -- inactive, exactly LIB-09's finding.
    screen._library_selected_row_id = LIBRARY_ROW_BROWSE_MEDIA
    assert screen.check_action("library_rag_use_in_console", ()) is False

    # Search/RAG row -- active.
    screen._library_selected_row_id = LIBRARY_ROW_BROWSE_SEARCH
    assert screen.check_action("library_rag_use_in_console", ()) is True


def test_check_action_gates_rag_result_card_actions_to_focused_card():
    """task-2858 AC#2 (LIB-09): the Enter/``o`` evidence-card actions
    (``library_rag_result_card_select``/``_open``) had no ``check_action``
    gate either -- both already no-op unless a
    ``.library-rag-result-card`` widget holds focus (see
    ``_focused_library_rag_result_card_index``), so this pins the SAME
    predicate at the ``check_action`` layer (monkeypatched directly here
    rather than building a real focus chain, mirroring how other
    predicate-only gates in this suite are exercised)."""
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    app = _build_test_app()
    screen = LibraryScreen(app)

    screen._focused_library_rag_result_card_index = lambda: None
    assert screen.check_action("library_rag_result_card_select", ()) is False
    assert screen.check_action("library_rag_result_card_open", ()) is False

    screen._focused_library_rag_result_card_index = lambda: 0
    assert screen.check_action("library_rag_result_card_select", ()) is True
    assert screen.check_action("library_rag_result_card_open", ()) is True


def test_library_screen_bindings_are_all_gated_or_universal():
    """task-2858 AC#2 (LIB-09): audit the FULL static ``BINDINGS`` list.

    F1's help panel (``LibraryScreen.action_show_workbench_help``) now
    filters ``BINDINGS`` through ``check_action`` -- so an action added to
    ``BINDINGS`` in the future WITHOUT a matching ``check_action`` branch
    would silently fall through to the default ``return True`` and leak
    into F1 on every Library surface again, reproducing this exact
    finding. This audits every action currently on the class: on a bare
    (landing) screen instance -- where none of today's context-specific
    actions legitimately apply -- each one must either be gated (``check_
    action`` returns ``False``) or be explicitly declared universal below
    (works identically on every surface, so ``True`` would be correct
    even on the landing). Nothing is declared universal today; the
    allowlist exists so a genuinely screen-wide binding could be added
    later without failing this test for the right reason.
    """
    from textual.binding import Binding

    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    # No LibraryScreen binding is meant to work identically on every
    # surface today -- "/" (focus search) and F6 (next pane) are real
    # screen-wide keys, but they are NOT Bindings (see
    # ``LibraryScreen.on_key``/``action_focus_next_workbench_pane``'s own
    # wiring), so they never appear in BINDINGS and are out of scope here.
    universal_actions: frozenset[str] = frozenset()

    app = _build_test_app()
    screen = LibraryScreen(app)

    actions: set[str] = set()
    for entry in LibraryScreen.BINDINGS:
        if isinstance(entry, Binding):
            actions.add(entry.action)
        elif isinstance(entry, (tuple, list)) and len(entry) > 1:
            actions.add(str(entry[1]))

    assert actions, "BINDINGS must not be empty for this audit to mean anything"

    for action in sorted(actions):
        result = screen.check_action(action, ())
        if action in universal_actions:
            assert result is True, (
                f"{action!r} is declared universal but check_action "
                f"returned {result!r} on the landing."
            )
        else:
            assert result is False, (
                f"{action!r} has no check_action gate (or its gate passes "
                "on the bare landing) -- it will leak into F1's help "
                "panel on every Library surface (LIB-09 contamination)."
            )


def test_action_show_workbench_help_filters_bindings_by_check_action(monkeypatch):
    """task-2858 AC#2 (LIB-09): F1 on a non-skills, non-Search canvas must
    NOT advertise the skill editor's or Search/RAG's dead keys -- the
    original finding, reproduced live at ``6ffa56516``: F1 on the Media
    canvas was titled "LibraryScreen Shortcuts" and listed "ctrl+s: Save
    skill"/"escape: Back to skills list". Sets the screen to the Media
    LIST canvas (nothing skill/Search/viewer related active) and asserts
    the resulting help state's shortcuts include none of those dead keys.
    """
    from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_MEDIA
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
    from tldw_chatbook.UI.Workbench.help import WorkbenchHelpPanel

    app = _build_test_app()
    screen = LibraryScreen(app)
    screen._library_selected_row_id = LIBRARY_ROW_BROWSE_MEDIA
    screen._library_media_view = "list"

    pushed = []

    class _FakeApp:
        def push_screen(self, panel):
            pushed.append(panel)

    # A class-level property override (auto-reverted by monkeypatch) --
    # mirrors test_settings_rag_profile_region.py's ``fake_app`` fixture:
    # an un-mounted Screen's ``.app`` raises NoActiveAppError otherwise.
    monkeypatch.setattr(
        LibraryScreen, "app", property(lambda self: _FakeApp()), raising=False
    )

    screen.action_show_workbench_help()

    assert len(pushed) == 1
    panel = pushed[0]
    assert isinstance(panel, WorkbenchHelpPanel)
    keys = {key for key, _description in panel.state.shortcuts}
    descriptions = {description for _key, description in panel.state.shortcuts}
    # Dead on the Media list canvas -- must not appear.
    assert "ctrl+s" not in keys
    assert "Save skill" not in descriptions
    assert "Back to skills list" not in descriptions
    assert "u" not in keys
    assert "enter" not in keys
    assert "o" not in keys
    # Genuinely active here: Escape moves focus to the rail (list canvas).
    # task-3312 (#1): advertised exactly ONCE, as the footer set's "esc"
    # row -- the raw-key dedupe used to keep the BINDINGS spelling too and
    # F1 listed the same exit twice ("esc: focus rail" + "escape: Focus
    # rail"; live in Ingest, same mechanism here).
    assert "esc" in keys
    assert "escape" not in keys


def test_action_show_workbench_help_includes_landing_footer_keys(monkeypatch):
    """task-2858 review (Important #1): F1 on the Library LANDING (and every
    other surface whose real keyboard story is on_key/footer-set wiring,
    never a ``Binding``) must not render an EMPTY panel.

    Before this fix ``action_show_workbench_help`` only listed check_action-
    filtered ``BINDINGS`` entries -- and ``test_library_screen_bindings_are_
    all_gated_or_universal`` above pins that EVERY ``BINDINGS`` action gates
    ``False`` on the bare landing, so the panel rendered a title and a Close
    button and nothing else. Yet the landing footer teaches four real keys
    (``/``, ``i``, ``n``, F6 -- ``LIBRARY_LANDING_SHORTCUTS``,
    ``_register_footer_shortcuts``) that never reach a ``Binding`` (``/``
    and the hub accelerators are ``on_key`` wiring; F6 is the app-global
    pane-cycle key). F1 must now include that same per-mode footer set, so
    the landing's F1 is never empty and matches what the footer already
    teaches -- including F6, which task-2860's reserved-global filtering
    drops from the FOOTER's compact rendering but must still reach F1 (F1
    is not subject to that filter, mirroring how
    ``SettingsScreen.action_show_workbench_help`` reads its per-category
    shortcuts directly rather than through the footer widget).
    """
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
    from tldw_chatbook.UI.Workbench.help import WorkbenchHelpPanel

    app = _build_test_app()
    screen = LibraryScreen(app)
    # Landing: no row selected (the default -- see LibraryScreen.__init__).
    assert screen._library_selected_row_id == ""

    pushed = []

    class _FakeApp:
        def push_screen(self, panel):
            pushed.append(panel)

    monkeypatch.setattr(
        LibraryScreen, "app", property(lambda self: _FakeApp()), raising=False
    )

    screen.action_show_workbench_help()

    assert len(pushed) == 1
    panel = pushed[0]
    assert isinstance(panel, WorkbenchHelpPanel)
    assert panel.state.shortcuts, "F1 must not be empty on the Library landing"
    keys = {key for key, _description in panel.state.shortcuts}
    # LIBRARY_LANDING_SHORTCUTS -- the exact keys the landing footer teaches.
    assert "/" in keys
    assert "i" in keys
    assert "n" in keys
    assert "F6" in keys
    # Minor #2 (same review): the title must not leak the raw class name.
    # task-4023 (dev) then added a mode suffix for F1 honesty, so the title
    # is "Library Shortcuts — Landing" here. The claim under test is the
    # original one -- no raw class name -- so assert THAT rather than an
    # exact string that a later honest change breaks again.
    assert panel.state.title.startswith("Library Shortcuts")
    assert "LibraryScreen" not in panel.state.title


def test_action_library_media_viewer_back_returns_to_list_and_refocuses_it():
    """task-2856 AC1/AC2: Escape from the PLAIN read-only media viewer (no
    edit/delete-confirm/analysis sub-state active) reuses the exact same
    reset sequence as the "‹ Back to list" button
    (``_exit_library_media_viewer``) and then re-focuses the list's first
    row, one seam for both exits."""
    from tldw_chatbook.Library.library_media_state import (
        LIBRARY_MEDIA_BROWSE_PAGE_SIZE,
        MediaBrowseResult,
        MediaBrowseScope,
    )
    from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_MEDIA
    from tldw_chatbook.UI.Screens.library_screen import (
        LIBRARY_LIST_ENTRY_FOCUS_ARMED_SECONDS,
        LibraryScreen,
    )

    app = _build_test_app()
    screen = LibraryScreen(app)
    screen._library_selected_row_id = LIBRARY_ROW_BROWSE_MEDIA
    screen._library_media_view = "viewer"
    screen._library_media_editing = False
    screen._library_media_confirming_delete = False
    screen._library_media_editing_analysis = False
    # The Escape-from-viewer flow this test pins starts from a BROWSED
    # list: the user loaded a Media page, opened an item, and Escape
    # returns to that same applied page. Seed the browse controller with
    # that applied page (task-19193). Without it the controller reports
    # the DEEP-LINK entry state (no page ever applied), and since
    # f86c636af ("fix(library): close media lifecycle authority gaps")
    # ``_exit_library_media_viewer`` closes that gap by requesting a real
    # page+facet load (``_load_library_media_list_if_needed`` ->
    # ``run_worker``) -- app-context work this synchronous harness cannot
    # host (``self.app`` raised NoActiveAppError). The deep-link exit flow
    # is covered live in ``Tests/UI/test_library_shell.py::
    # test_library_media_deep_link_back_loads_exact_page_and_facets``.
    applied_scope = MediaBrowseScope()
    screen._library_media_browse_controller.applied_result = MediaBrowseResult(
        scope=applied_scope,
        items=(
            {
                "id": "local:media:1",
                "backing_media_id": 1,
                "title": "Clip",
                "media_type": "video",
                "updated_at": "2026-08-20T00:00:00Z",
            },
        ),
        total=1,
        limit=LIBRARY_MEDIA_BROWSE_PAGE_SIZE,
        offset=0,
    )

    refresh_calls = []
    focus_calls = []
    timer_calls = []
    worker_requests = []
    screen.refresh = lambda recompose=False: refresh_calls.append(recompose)
    screen.call_after_refresh = lambda callback: focus_calls.append(callback)
    # ``_arm_library_list_entry_focus`` also arms a settle-window timer
    # (task-2856) -- stub it out, a real ``set_timer`` needs a running
    # event loop this synchronous test has none of.
    screen.set_timer = lambda delay, callback: timer_calls.append((delay, callback))

    def _capture_worker(work, **kwargs):
        # Close a captured coroutine so it never warns as never-awaited.
        if hasattr(work, "close"):
            work.close()
        worker_requests.append(kwargs)

    screen.run_worker = _capture_worker

    screen.action_library_media_viewer_back()

    assert screen._library_media_view == "list"
    assert refresh_calls == [True]
    assert timer_calls == [
        (LIBRARY_LIST_ENTRY_FOCUS_ARMED_SECONDS, screen._disarm_library_list_entry_focus)
    ]
    assert focus_calls == [screen._focus_library_list_entry]
    # With a page already applied, the exit must NOT re-request it:
    # ``_load_library_media_list_if_needed`` is for deep-link entries that
    # never applied a page (f86c636af). A worker request here would mean
    # the exit path started reloading the list the user is returning to.
    assert worker_requests == []


@pytest.mark.parametrize(
    "sub_state_flag",
    [
        "_library_media_editing",
        "_library_media_confirming_delete",
        "_library_media_editing_analysis",
    ],
)
def test_action_library_media_viewer_back_steps_out_of_a_sub_state_first(
    sub_state_flag: str,
):
    """task-2856 AC2 review round 2: the media edit/delete-confirm/
    analysis-edit sub-states have no dirty-tracking field to veto on (
    unlike the note/prompt editors), so Escape instead mirrors each
    sub-state's OWN existing Cancel button -- one step back to the plain
    viewer, NOT straight through to the list. This is strictly less
    aggressive than jumping to the list mid-edit would be, and requires no
    new dirty-state invention."""
    from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_MEDIA
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    app = _build_test_app()
    screen = LibraryScreen(app)
    screen._library_selected_row_id = LIBRARY_ROW_BROWSE_MEDIA
    screen._library_media_view = "viewer"
    screen._library_media_editing = False
    screen._library_media_confirming_delete = False
    screen._library_media_editing_analysis = False
    setattr(screen, sub_state_flag, True)

    refresh_calls = []
    focus_calls = []
    screen.refresh = lambda recompose=False: refresh_calls.append(recompose)
    screen.call_after_refresh = lambda callback: focus_calls.append(callback)
    screen.set_timer = lambda delay, callback: None

    screen.action_library_media_viewer_back()

    # Still on the viewer -- Escape did NOT jump to the list.
    assert screen._library_media_view == "viewer"
    assert getattr(screen, sub_state_flag) is False
    assert refresh_calls == [True]
    # No entry-focus request armed -- the list was never re-entered.
    assert focus_calls == []


@pytest.mark.asyncio
async def test_action_library_note_editor_back_honors_dirty_guard():
    """task-2856 AC2: Escape from the note editor reuses the SAME guarded
    exit as the "‹ Back to list" button -- a dirty edit that survives the
    flush vetoes the exit exactly like it vetoes the button."""
    from tldw_chatbook.Library.library_notes_session import (
        NoteFlushOutcome,
        NoteFlushOutcomeKind,
    )
    from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_NOTES
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    app = _build_test_app()
    screen = LibraryScreen(app)
    screen._library_selected_row_id = LIBRARY_ROW_BROWSE_NOTES
    screen._library_notes_source = "database"
    screen._library_notes_view = "editor"

    # task-3316: "still dirty after the flush" is expressed by the flush
    # OUTCOME now -- ``_library_note_dirty`` became a read-only property over
    # the session coordinator's snapshot, so the exit gate reads
    # ``NoteFlushOutcome.kind`` rather than a screen flag.
    async def flush_still_dirty():
        return NoteFlushOutcome(NoteFlushOutcomeKind.BLOCKED)

    screen._flush_library_note_save = flush_still_dirty
    refresh_calls = []
    screen.refresh = lambda recompose=False: refresh_calls.append(recompose)

    await screen.action_library_note_editor_back()

    assert screen._library_notes_view == "editor", "dirty veto must not exit"
    assert refresh_calls == []

    async def flush_clean():
        return NoteFlushOutcome(NoteFlushOutcomeKind.PERMITTED)

    screen._flush_library_note_save = flush_clean
    screen._refresh_local_source_snapshot = lambda: None
    focus_calls = []
    screen.call_after_refresh = lambda callback, *args: focus_calls.append(callback)

    await screen.action_library_note_editor_back()

    assert screen._library_notes_view == "list"
    assert refresh_calls == [True]
    assert focus_calls == [screen._restore_library_notes_focus_identity]


@pytest.mark.asyncio
async def test_action_library_prompt_editor_back_honors_dirty_guard():
    """task-2856 AC2: Escape from the prompt editor reuses the SAME
    guarded exit as the "‹ Back to list" button."""
    from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_PROMPTS
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    app = _build_test_app()
    screen = LibraryScreen(app)
    screen._library_selected_row_id = LIBRARY_ROW_BROWSE_PROMPTS
    screen._library_prompts_view = "editor"

    async def flush_fails():
        return False

    screen._flush_library_prompt_save = flush_fails
    refresh_calls = []
    screen.refresh = lambda recompose=False: refresh_calls.append(recompose)

    await screen.action_library_prompt_editor_back()

    assert screen._library_prompts_view == "editor", "a failed flush must veto"
    assert refresh_calls == []

    async def flush_ok():
        return True

    screen._flush_library_prompt_save = flush_ok
    screen._reset_library_prompt_editor_state = (
        lambda: setattr(screen, "_library_prompts_view", "list")
    )
    screen._refresh_local_source_snapshot = lambda: None
    # task-3316: the guarded exit now re-requests the Prompts page through
    # the browse controller, which needs a running App. This screen is
    # deliberately unmounted, so stand in for that request -- the exit's
    # veto/reset/focus contract is what this test asserts.
    browse_requests = []
    screen._request_library_prompts_browse = (
        lambda scope, **kwargs: browse_requests.append(scope)
    )
    focus_calls = []
    screen.call_after_refresh = lambda callback, *args: focus_calls.append(callback)

    await screen.action_library_prompt_editor_back()

    assert screen._library_prompts_view == "list"
    # The exit's redraw is now carried by the prompts-page refetch it
    # requests (whose reply recomposes), not by a direct ``refresh`` call.
    assert len(browse_requests) == 1, "the exit must refetch the prompts page"
    assert refresh_calls == []
    assert focus_calls == [screen._focus_library_list_entry]


def test_action_library_list_focus_rail_focuses_search_input(monkeypatch):
    """task-2856 AC2/AC3: Escape from a list canvas focuses the SAME rail
    search box `/` and F6 already target -- one converged "the rail"
    destination across all three routes."""
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    app = _build_test_app()
    screen = LibraryScreen(app)

    focused_widgets = []

    class _FakeInput:
        def focus(self):
            focused_widgets.append(self)

    fake_input = _FakeInput()
    monkeypatch.setattr(screen, "query_one", lambda *a, **k: fake_input)

    screen.action_library_list_focus_rail()

    assert focused_widgets == [fake_input]


def test_compose_content_reapplies_pending_list_entry_focus_on_every_recompose():
    """task-2856: this screen has SEVERAL independent background workers
    that each end in their own ``self.refresh(recompose=True)`` (the
    snapshot refresh several "back to list" exits kick off, the skills
    trust-posture load IT chains into, per-item detail fetches, ...) -- any
    one of them landing after an earlier recompose rebuilds the list's row
    Buttons as fresh instances and silently drops the focus already set.
    Reproduced live: the skill editor's Escape correctly returned to the
    list and briefly focused its first row, but the trust-posture worker's
    LATER recompose (a SECOND recompose after the first) still dropped it
    -- clearing the flag on the first consumption (an earlier version of
    this fix) lost that same race one level up. ``compose_content`` is the
    one choke point every recompose passes through regardless of which
    worker triggered it, so it re-fires the focus request on EVERY run
    while the flag stays armed (cleared only by the settle-window timer or
    an explicit Up/Down -- see ``_arm_library_list_entry_focus``), proven
    here by draining ``compose_content`` twice in a row."""
    from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_SKILLS
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    app = _build_test_app()
    screen = LibraryScreen(app)
    screen._library_selected_row_id = LIBRARY_ROW_BROWSE_SKILLS
    screen._library_pending_list_entry_focus = True

    focus_calls = []
    screen.call_after_refresh = lambda callback: focus_calls.append(callback)
    screen._register_footer_shortcuts = lambda: None
    screen._library_rail_preferences = lambda: None

    for _ in range(2):
        try:
            # compose_content is a generator; draining it (list(...)) runs
            # the body up to its first yield, which is all the
            # flag-consuming code (before any widget construction) needs.
            list(screen.compose_content())
        except Exception:
            # A real recompose needs a live rail/canvas build this bare,
            # unmounted screen cannot supply -- irrelevant to what this
            # test checks (the flag-consume runs before any of that).
            pass

    assert focus_calls == [screen._focus_library_list_entry] * 2
    assert screen._library_pending_list_entry_focus is True


def test_compose_content_leaves_focus_alone_without_a_pending_request():
    """The common case -- an ordinary recompose while no "enter/return to
    list" seam has armed the flag -- must never touch focus. Without this
    guard, EVERY recompose would yank a user who manually pressed Down
    back to the list's first row."""
    from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_SKILLS
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    app = _build_test_app()
    screen = LibraryScreen(app)
    screen._library_selected_row_id = LIBRARY_ROW_BROWSE_SKILLS
    screen._library_pending_list_entry_focus = False

    focus_calls = []
    screen.call_after_refresh = lambda callback: focus_calls.append(callback)
    screen._register_footer_shortcuts = lambda: None
    screen._library_rail_preferences = lambda: None

    try:
        list(screen.compose_content())
    except Exception:
        pass

    assert focus_calls == []


def test_arm_library_list_entry_focus_schedules_immediate_attempt_and_settle_timer():
    """``_arm_library_list_entry_focus`` sets the pending flag, schedules
    the immediate ``call_after_refresh`` attempt, AND arms a bounded
    settle-window timer (``LIBRARY_LIST_ENTRY_FOCUS_ARMED_SECONDS``) that
    disarms the flag once no more of the chained background workers this
    task's "back to list" exits kick off are expected to still be running
    (task-2856)."""
    from tldw_chatbook.UI.Screens.library_screen import (
        LIBRARY_LIST_ENTRY_FOCUS_ARMED_SECONDS,
        LibraryScreen,
    )

    app = _build_test_app()
    screen = LibraryScreen(app)
    assert screen._library_pending_list_entry_focus is False

    focus_calls = []
    timer_calls = []
    screen.call_after_refresh = lambda callback: focus_calls.append(callback)
    screen.set_timer = lambda delay, callback: timer_calls.append((delay, callback))

    screen._arm_library_list_entry_focus()

    assert screen._library_pending_list_entry_focus is True
    assert focus_calls == [screen._focus_library_list_entry]
    assert timer_calls == [
        (LIBRARY_LIST_ENTRY_FOCUS_ARMED_SECONDS, screen._disarm_library_list_entry_focus)
    ]

    screen._disarm_library_list_entry_focus()
    assert screen._library_pending_list_entry_focus is False


def test_arm_library_list_entry_focus_twice_cancels_the_stale_timer_handle():
    """Qodo review (PR #1410): ``_arm_library_list_entry_focus`` called
    ``self.set_timer(...)`` every time it ran but never kept the returned
    handle. Calling it twice within ``LIBRARY_LIST_ENTRY_FOCUS_ARMED_SECONDS``
    (e.g. a rail-row press immediately followed by a chained background
    recompose re-requesting focus) left the FIRST timer still scheduled
    underneath the second -- if it fired later it called
    ``_disarm_library_list_entry_focus`` and cleared the flag even though
    the SECOND arm was still meant to be active, making the armed window
    non-deterministic. The fix stores the handle
    (``_library_list_entry_focus_timer``) and stops any existing one before
    scheduling a new one, so this test pins that the stale handle is
    actually cancelled -- not merely that the flag happens to still read
    ``True`` a moment later (a real Textual timer must never be allowed to
    fire once superseded, so the assertion is on ``.stop()`` having been
    called on the STALE handle, not on sleeping past its deadline)."""
    from tldw_chatbook.UI.Screens.library_screen import (
        LIBRARY_LIST_ENTRY_FOCUS_ARMED_SECONDS,
        LibraryScreen,
    )

    class _FakeTimer:
        def __init__(self, delay, callback):
            self.delay = delay
            self.callback = callback
            self.stop_calls = 0

        def stop(self):
            self.stop_calls += 1

    app = _build_test_app()
    screen = LibraryScreen(app)
    screen.call_after_refresh = lambda callback: None

    created_timers: list[_FakeTimer] = []

    def fake_set_timer(delay, callback):
        timer = _FakeTimer(delay, callback)
        created_timers.append(timer)
        return timer

    screen.set_timer = fake_set_timer

    screen._arm_library_list_entry_focus()
    assert len(created_timers) == 1
    first_timer = created_timers[0]
    assert first_timer.delay == LIBRARY_LIST_ENTRY_FOCUS_ARMED_SECONDS
    assert screen._library_list_entry_focus_timer is first_timer
    assert first_timer.stop_calls == 0

    # Re-arm before the first timer's own deadline -- e.g. a chained
    # background recompose re-requesting focus while the first window is
    # still open. The stale first timer must be cancelled, not left
    # dangling alongside the new one.
    screen._arm_library_list_entry_focus()
    assert len(created_timers) == 2
    second_timer = created_timers[1]

    assert first_timer.stop_calls == 1, (
        "the stale first timer must be stopped once a second arm supersedes it"
    )
    assert screen._library_list_entry_focus_timer is second_timer
    assert second_timer.stop_calls == 0
    assert screen._library_pending_list_entry_focus is True

    # Disarming (whether fired by the second timer's own deadline, an
    # interaction hook, or an explicit consumer) must stop and clear the
    # CURRENT handle too, so nothing is left running after the flag drops.
    screen._disarm_library_list_entry_focus()
    assert screen._library_pending_list_entry_focus is False
    assert second_timer.stop_calls == 1
    assert screen._library_list_entry_focus_timer is None


class _FakeMediaRowQuery(list):
    """Stands in for the ``DOMQuery`` ``self.query(...)`` returns: a
    ``NoMatches``-raising ``.first()`` plus plain iteration, both of which
    ``_focus_library_list_entry`` relies on."""

    def first(self):
        from textual.css.query import NoMatches

        if not self:
            raise NoMatches("no rows")
        return self[0]


class _FakeMediaRowButton:
    def __init__(self, media_id: str):
        self.media_id = media_id
        self.focused = False

    def focus(self) -> None:
        self.focused = True


def test_focus_library_list_entry_prefers_still_checked_row_in_select_mode():
    """task-3020 AC3: after a partial (or total) bulk-delete failure,
    keyboard focus must land on a STILL-CHECKED row (the failed id(s),
    retained for retry) rather than the literal first row in the list --
    landing on a row the user never selected would be a worse regression
    than the original "nothing focused" bug this method exists to fix."""
    from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_MEDIA
    from tldw_chatbook.Library.row_selection import RowSelection
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    row_a = _FakeMediaRowButton("1")
    row_b = _FakeMediaRowButton("2")
    row_c = _FakeMediaRowButton("3")
    selection = RowSelection("media")
    selection.select_all(["2"])  # only the second row is still checked

    app = _build_test_app()
    screen = LibraryScreen(app)
    screen._library_selected_row_id = LIBRARY_ROW_BROWSE_MEDIA
    screen._library_media_select_mode = True
    screen._library_media_row_selection = selection
    screen.query = lambda selector: _FakeMediaRowQuery([row_a, row_b, row_c])

    screen._focus_library_list_entry()

    assert row_a.focused is False
    assert row_b.focused is True
    assert row_c.focused is False


def test_focus_library_list_entry_falls_back_to_first_row_outside_active_selection():
    """Regression guard: the AC3 preference must not change the existing
    "focus the first row" behavior when Select mode isn't active, or
    nothing is checked (e.g. the bulk delete's own full-success path,
    which clears the selection before arming entry focus)."""
    from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_MEDIA
    from tldw_chatbook.Library.row_selection import RowSelection
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    app = _build_test_app()
    screen = LibraryScreen(app)
    screen._library_selected_row_id = LIBRARY_ROW_BROWSE_MEDIA

    # Outside Select mode entirely.
    row_a = _FakeMediaRowButton("1")
    row_b = _FakeMediaRowButton("2")
    screen._library_media_select_mode = False
    screen._library_media_row_selection = RowSelection("media")
    screen.query = lambda selector: _FakeMediaRowQuery([row_a, row_b])
    screen._focus_library_list_entry()
    assert row_a.focused is True
    assert row_b.focused is False

    # Select mode active but nothing checked (e.g. a full-success bulk
    # delete already cleared the selection before arming this).
    row_c = _FakeMediaRowButton("3")
    row_d = _FakeMediaRowButton("4")
    screen._library_media_select_mode = True
    screen._library_media_row_selection = RowSelection("media")
    screen.query = lambda selector: _FakeMediaRowQuery([row_c, row_d])
    screen._focus_library_list_entry()
    assert row_c.focused is True
    assert row_d.focused is False


def test_focus_library_list_entry_checked_row_preference_is_media_only():
    """The AC3 preference must not leak into the other three list
    canvases (Notes/Prompts/Skills) -- none of them have a bulk Select-
    mode row_selection concept the way Media does, so the guard requires
    the Media row class specifically."""
    from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_NOTES
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    app = _build_test_app()
    screen = LibraryScreen(app)
    screen._library_selected_row_id = LIBRARY_ROW_BROWSE_NOTES
    screen._library_notes_source = "database"
    screen._library_notes_view = "list"
    # Deliberately True/non-empty -- MUST be ignored for a non-Media list.
    screen._library_media_select_mode = True

    row_a = _FakeMediaRowButton("n1")
    row_b = _FakeMediaRowButton("n2")
    screen.query = lambda selector: _FakeMediaRowQuery([row_a, row_b])

    screen._focus_library_list_entry()

    assert row_a.focused is True
    assert row_b.focused is False


def test_on_key_disarms_a_pending_list_entry_focus_on_any_key():
    """task-2856 review round 2: review found the settle-window timer was
    the ONLY disarm path, so a background recompose landing after the
    user had already Tabbed/clicked away could silently steal focus back
    to row 0. ``on_key`` now disarms unconditionally the instant ANY key
    is pressed while a request is armed -- disarming an idle flag is a
    harmless no-op, so this needs no gate on which key."""
    from types import SimpleNamespace

    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    app = _build_test_app()
    screen = LibraryScreen(app)
    screen._library_pending_list_entry_focus = True
    screen.focused = None

    # An arbitrary key with no other special handling in on_key.
    screen.on_key(SimpleNamespace(key="x", character="x"))

    assert screen._library_pending_list_entry_focus is False


def test_on_key_disarm_is_a_harmless_noop_when_nothing_is_armed():
    """The unconditional on_key disarm must never raise or misbehave when
    there is nothing pending -- the overwhelmingly common case."""
    from types import SimpleNamespace

    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    app = _build_test_app()
    screen = LibraryScreen(app)
    assert screen._library_pending_list_entry_focus is False
    screen.focused = None

    screen.on_key(SimpleNamespace(key="x", character="x"))

    assert screen._library_pending_list_entry_focus is False


def test_on_descendant_focus_disarms_when_focus_leaves_the_armed_list():
    """task-2856 review round 2: ``on_descendant_focus`` is what catches
    focus changes ``on_key`` cannot -- a mouse click never reaches
    ``on_key`` at all. Focus landing on any widget that is NOT a row of
    the currently-armed list (Tab, Shift+Tab, a click elsewhere in the
    canvas) disarms the pending request."""
    from types import SimpleNamespace

    from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_MEDIA
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    app = _build_test_app()
    screen = LibraryScreen(app)
    screen._library_selected_row_id = LIBRARY_ROW_BROWSE_MEDIA
    screen._library_pending_list_entry_focus = True

    foreign_widget = SimpleNamespace(has_class=lambda name: False)
    screen.on_descendant_focus(SimpleNamespace(widget=foreign_widget))

    assert screen._library_pending_list_entry_focus is False


def test_on_descendant_focus_does_not_disarm_for_the_systems_own_row_refocus():
    """The system's own ``_focus_library_list_entry()`` call ALSO posts a
    ``DescendantFocus`` (Textual posts it for every focus change,
    including programmatic ``.focus()`` calls) -- that case must NOT
    immediately disarm what it just armed, or the settle window would be
    self-defeating."""
    from types import SimpleNamespace

    from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_MEDIA
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    app = _build_test_app()
    screen = LibraryScreen(app)
    screen._library_selected_row_id = LIBRARY_ROW_BROWSE_MEDIA
    screen._library_pending_list_entry_focus = True

    own_row = SimpleNamespace(has_class=lambda name: name == "library-media-row")
    screen.on_descendant_focus(SimpleNamespace(widget=own_row))

    assert screen._library_pending_list_entry_focus is True


def test_on_descendant_focus_is_a_noop_when_nothing_is_armed():
    """The hook must never touch focus/state when no request is pending --
    the overwhelmingly common case (every OTHER focus change on the whole
    screen also flows through here)."""
    from types import SimpleNamespace

    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    app = _build_test_app()
    screen = LibraryScreen(app)
    assert screen._library_pending_list_entry_focus is False

    foreign_widget = SimpleNamespace(has_class=lambda name: False)
    screen.on_descendant_focus(SimpleNamespace(widget=foreign_widget))

    assert screen._library_pending_list_entry_focus is False


def test_app_uses_screen_navigation_and_wires_media_services():
    app = _build_test_app()

    assert app._use_screen_navigation is True
    assert isinstance(app.workspace_registry_service, LocalWorkspaceRegistryService)
    assert isinstance(app.local_media_reading_service, LocalMediaReadingService)
    assert isinstance(app.server_media_reading_service, ServerMediaReadingService)
    assert isinstance(app.media_reading_scope_service, MediaReadingScopeService)
    assert not hasattr(app, "media_runtime_state")
    assert (
        app.auth_account_scope_service.server_context_provider
        is app.server_context_provider
    )
    assert (
        app.server_media_reading_service.client_provider is app.server_context_provider
    )
    assert (
        app.server_chat_conversation_service.client_provider
        is app.server_context_provider
    )
    assert (
        app.server_notes_workspace_service.client_provider
        is app.server_context_provider
    )
    assert (
        app.server_character_persona_service.client_provider
        is app.server_context_provider
    )
    assert (
        app.server_chat_dictionary_service.client_provider
        is app.server_context_provider
    )
    assert app.server_prompt_service.client_provider is app.server_context_provider
    assert app.server_chatbook_service.client_provider is app.server_context_provider
    assert (
        app.server_prompt_studio_service.client_provider is app.server_context_provider
    )
    assert app.server_runtime_service.client_provider is app.server_context_provider
    assert (
        app.server_auth_account_service.client_provider is app.server_context_provider
    )


def test_app_harness_multi_connection_databases_keep_initialized_schemas():
    app = _build_test_app()

    assert app.scheduling_service.watchlist_projection.list_jobs() == []
    assert list(app.local_research_service.list_runs()) == []
    assert app.local_writing_service.list_projects() == []


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


@pytest.mark.asyncio
async def test_app_shutdown_helper_disconnects_local_mcp_client_with_sessions():
    class FakeMCPClient:
        def __init__(self, sessions) -> None:
            self.sessions = sessions
            self.disconnect_calls = 0

        async def disconnect_all(self) -> None:
            self.disconnect_calls += 1

    client = FakeMCPClient({"srv": object()})
    app_like = SimpleNamespace(local_mcp_control_service=SimpleNamespace(client=client))

    await TldwCli._disconnect_local_mcp_client(app_like)

    assert client.disconnect_calls == 1


@pytest.mark.asyncio
async def test_app_shutdown_helper_skips_mcp_disconnect_when_no_sessions():
    class FakeMCPClient:
        def __init__(self, sessions) -> None:
            self.sessions = sessions
            self.disconnect_calls = 0

        async def disconnect_all(self) -> None:
            self.disconnect_calls += 1

    client = FakeMCPClient({})
    app_like = SimpleNamespace(local_mcp_control_service=SimpleNamespace(client=client))

    await TldwCli._disconnect_local_mcp_client(app_like)

    assert client.disconnect_calls == 0


@pytest.mark.asyncio
async def test_app_shutdown_helper_skips_mcp_disconnect_when_client_never_connected():
    """`LocalMCPControlService.client` stays `None` until a profile is
    actually connected -- must be a no-op, not an AttributeError."""
    app_like = SimpleNamespace(local_mcp_control_service=SimpleNamespace(client=None))

    await TldwCli._disconnect_local_mcp_client(app_like)  # must not raise


@pytest.mark.asyncio
async def test_app_shutdown_helper_skips_mcp_disconnect_when_service_missing():
    app_like = SimpleNamespace()

    await TldwCli._disconnect_local_mcp_client(app_like)  # must not raise


def test_app_wires_local_and_server_skills_services():
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
    assert (
        app.notifications_scope_service.local_service
        is app.client_notifications_service
    )
    assert isinstance(
        app.home_active_work_adapter, LocalNotificationHomeActiveWorkAdapter
    )
    assert (
        app.home_active_work_adapter.notification_service
        is app.client_notifications_service
    )
    assert (
        app.home_active_work_adapter.watchlist_service is app.local_watchlists_service
    )
    assert app.home_active_work_adapter.chatbook_service is app.local_chatbook_service
    assert isinstance(app.server_outputs_service, ServerOutputsService)
    assert isinstance(app.outputs_scope_service, OutputsScopeService)
    assert isinstance(app.local_research_service, LocalResearchService)
    assert (
        app.local_research_service.notification_dispatcher
        is app.notification_dispatch_service
    )
    assert app.local_research_service.notification_app is app
    assert (
        app.local_media_reading_service.notification_dispatcher
        is app.notification_dispatch_service
    )
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
    assert isinstance(
        app.server_collections_feeds_service, ServerCollectionsFeedsService
    )
    assert isinstance(app.collections_feeds_scope_service, CollectionsFeedsScopeService)
    assert (
        app.collections_feeds_scope_service.local_service
        is app.local_watchlists_service
    )
    assert isinstance(app.server_connectors_service, ServerConnectorsService)
    assert isinstance(app.connectors_scope_service, ConnectorsScopeService)
    assert isinstance(app.local_skills_service, LocalSkillsService)
    assert isinstance(app.local_skill_trust_service, SkillTrustService)
    assert app.local_skill_trust_service is app.local_skills_service.trust_service
    assert (
        app.local_skill_trust_service.skills_dir == app.local_skills_service.skills_dir
    )
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
    assert isinstance(app.local_first_sync_service, LocalFirstSyncService)
    assert isinstance(app.manual_sync_control_service, ManualSyncControlService)
    assert (
        app.manual_sync_control_service.local_first_sync_service
        is app.local_first_sync_service
    )
    assert app.media_reading_scope_service.sync_scope_service is app.sync_scope_service
    assert app.notes_scope_service.sync_scope_service is app.sync_scope_service
    assert app.research_scope_service.sync_scope_service is app.sync_scope_service
    assert isinstance(app.server_runtime_service, ServerRuntimeService)
    assert isinstance(app.server_runtime_scope_service, ServerRuntimeScopeService)
    assert isinstance(
        app.active_server_capability_service, ActiveServerCapabilityService
    )
    assert isinstance(
        app.server_credential_store,
        (KeyringServerCredentialStore, UnavailableServerCredentialStore),
    )
    assert isinstance(app.server_context_provider, RuntimeServerContextProvider)
    assert app.server_context_provider.runtime_context is app.runtime_policy
    assert app.server_context_provider.target_store is app.unified_mcp_target_store
    assert app.server_context_provider.credential_store is app.server_credential_store
    assert isinstance(
        app.local_llm_provider_catalog_service, LocalLLMProviderCatalogService
    )
    assert isinstance(
        app.server_llm_provider_catalog_service, ServerLLMProviderCatalogService
    )
    assert isinstance(
        app.llm_provider_catalog_scope_service, LLMProviderCatalogScopeService
    )
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
    assert isinstance(
        app.server_chat_conversation_service, ServerChatConversationService
    )
    assert isinstance(app.chat_conversation_scope_service, ChatConversationScopeService)
    assert isinstance(app.server_notes_workspace_service, ServerNotesWorkspaceService)
    assert isinstance(app.notes_scope_service, NotesScopeService)
    assert isinstance(
        app.server_character_persona_service, ServerCharacterPersonaService
    )
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


def test_media_screen_constructs_destination_local_runtime_state():
    app = _build_test_app()
    screen = MediaScreen(app)

    widgets = list(screen.compose_content())

    assert len(widgets) == 2  # destination header + media window
    assert not hasattr(app, "media_runtime_state")
    assert not hasattr(screen, "media_runtime_state")
    assert screen.media_window is widgets[1]
    assert (
        screen.media_window.runtime_state.runtime_backend
        == app.get_authoritative_runtime_source()
    )


@pytest.mark.asyncio
async def test_main_navigation_exposes_all_routed_primary_screens():
    from tldw_chatbook.UI.Navigation.shell_destinations import SHELL_DESTINATION_ORDER

    class TestApp(ConsolidatedCSSApp):
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
async def test_persona_buddy_app_reconcile_excludes_modal_screen():
    """An active modal never becomes a Buddy mount target."""

    from textual.screen import ModalScreen

    from tldw_chatbook.app import TldwCli

    class Modal(ModalScreen):
        async def reconcile_persona_buddy_view(self) -> None:
            raise AssertionError("modal must never receive Buddy reconciliation")

    host = type("BuddyReconcileHost", (), {"screen": Modal()})()
    await TldwCli.reconcile_persona_buddy_view(host)


@pytest.mark.asyncio
async def test_main_navigation_copy_and_order():
    expected_button_order = [
        ("nav-home", "\u23031 Home"),
        ("nav-console", "\u23032 Console"),
        ("nav-library", "\u23033 Library"),
        ("nav-artifacts", "\u23034 Artifacts"),
        ("nav-personas", "\u23035 Roleplay"),
        ("nav-watchlists_collections", "\u23036 Watchlists"),
        ("nav-schedules", "\u23037 Schedules"),
        ("nav-workflows", "\u23038 Workflows"),
        ("nav-mcp", "\u23039 MCP"),
        ("nav-acp", "\u23030 ACP"),
        ("nav-lab", "F7 Lab"),
        ("nav-logs", "F8 Logs"),
        ("nav-settings", "F9 Settings"),
    ]

    class TestApp(ConsolidatedCSSApp):
        def compose(self):
            yield MainNavigationBar(active="chat")

    app = TestApp()

    async with app.run_test(size=(160, 20)) as pilot:
        await pilot.pause(0.1)

        nav_buttons = list(app.query(".nav-button"))
        actual_button_order = [
            (button.id, str(button.label).strip()) for button in nav_buttons
        ]

        assert actual_button_order == expected_button_order
        assert str(app.query_one("#nav-console", Button).label).strip() == "\u23032 Console"
        assert nav_buttons[0].id == "nav-home"
        assert nav_buttons[1].id == "nav-console"
        assert nav_buttons[-1].id == "nav-settings"
        # TASK-2154.21 (NV-01): the static hint is now the overflow menu's
        # compact button (hidden at widths where nothing clips).
        hint = app.query_one("#nav-overflow-hint", Button)
        assert str(hint.label) == "More ▾"
        assert hint.tooltip == "All destinations"


@pytest.mark.asyncio
async def test_main_navigation_buttons_explain_compact_labels():
    from tldw_chatbook.UI.Navigation.shell_destinations import SHELL_DESTINATION_ORDER

    expected_tooltips = {
        f"nav-{destination.destination_id}": destination.tooltip
        for destination in SHELL_DESTINATION_ORDER
    }

    class TestApp(ConsolidatedCSSApp):
        def compose(self):
            yield MainNavigationBar(active="chat")

    app = TestApp()

    async with app.run_test(size=(160, 20)) as pilot:
        await pilot.pause(0.1)

        actual_tooltips = {
            button.id: str(button.tooltip) for button in app.query(".nav-button")
        }

        assert actual_tooltips == expected_tooltips


@pytest.mark.asyncio
async def test_main_navigation_route_ids_match_shell_destinations():
    from tldw_chatbook.UI.Navigation.shell_destinations import SHELL_DESTINATION_ORDER

    class TestApp(ConsolidatedCSSApp):
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
        ("watchlists_collections", "WatchlistsCollectionsScreen"),
        ("study", "StudyScreen"),
        ("stts", "STTSScreen"),
    ]

    async with app.run_test(size=(160, 40)) as pilot:
        await pilot.pause(0.1)
        # Direct handler calls simulate post-startup navigation; mark startup
        # complete so the pre-initial-screen guard lets them through.
        app._initial_screen_pushed = True

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
        _screen_name, _tab_id, screen_class = app._resolve_screen_navigation_target(
            route_id
        )
        if screen_class is None or not issubclass(screen_class, BaseAppScreen):
            offenders.append((route_id, screen_class))

    assert offenders == []


# --- Cross-visit state persistence (real save_state/restore_state) --------
#
# Screens are never cached/reused (see
# ``test_screen_navigation_always_constructs_fresh_instances`` above), so
# continuity across a visit depends entirely on ``_screen_states``
# (``save_state``/``restore_state``). These are round-trip pilots through the
# REAL navigation path -- ``NavigateToScreen`` posted, drained via bounded
# polling (the storm pilot's idiom above), real widgets mutated the way a
# user would -- not direct calls into ``save_state``/``restore_state``.


@pytest.mark.asyncio
async def test_library_screen_round_trip_restores_rag_query_and_rail_selection():
    """Select the Search/RAG rail row, type a query into the real Input
    widget, hop to Home and back, and assert both the internal state and
    the visible Input value survived on the freshly-composed instance.
    """
    from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_SEARCH

    app = _build_test_app()

    async with app.run_test(size=(170, 48)) as pilot:
        for _ in range(150):
            await pilot.pause(0.02)
            if type(app.screen).__name__ != "Screen":
                break

        app.post_message(NavigateToScreen("library"))
        for _ in range(150):
            await pilot.pause(0.02)
            if type(app.screen).__name__ == "LibraryScreen" and app.screen.query(
                "#library-row-browse-search"
            ):
                break
        assert type(app.screen).__name__ == "LibraryScreen"

        app.screen.query_one("#library-row-browse-search").press()
        for _ in range(150):
            await pilot.pause(0.02)
            if app.screen.query("#library-rag-query-input"):
                break

        app.screen.query_one("#library-rag-query-input", Input).value = "roadmap notes"
        await pilot.pause()
        await pilot.pause()

        assert app.screen._library_rag_query == "roadmap notes"
        assert app.screen._library_selected_row_id == LIBRARY_ROW_BROWSE_SEARCH

        app.post_message(NavigateToScreen("home"))
        for _ in range(150):
            await pilot.pause(0.02)
            if type(app.screen).__name__ == "HomeScreen":
                break
        assert type(app.screen).__name__ == "HomeScreen"

        app.post_message(NavigateToScreen("library"))
        for _ in range(150):
            await pilot.pause(0.02)
            if type(app.screen).__name__ == "LibraryScreen" and app.screen.query(
                "#library-rag-query-input"
            ):
                break

        restored_screen = app.screen
        assert type(restored_screen).__name__ == "LibraryScreen"
        assert restored_screen._library_rag_query == "roadmap notes"
        assert restored_screen._library_selected_row_id == LIBRARY_ROW_BROWSE_SEARCH
        query_input = restored_screen.query_one("#library-rag-query-input", Input)
        assert query_input.value == "roadmap notes"


@pytest.mark.asyncio
async def test_console_staged_live_work_launch_survives_navigate_away_and_back():
    """D3 (RAG-truth staged-evidence critique): a staged Console live-work
    launch must survive a REAL screen swap, not merely a same-screen
    refresh.

    ``ChatScreen`` is never cached/reused across navigation (see
    ``test_screen_navigation_always_constructs_fresh_instances`` above) --
    ``_create_navigation_screen`` builds a brand new instance every time, so
    ``_pending_console_launch_context`` (screen-instance state set in
    ``ChatScreen.__init__``) started life on the OLD instance and is gone
    unless ``save_state``/``restore_state`` carries it to the new one.
    Before this fix, neither method touched the launch at all, so
    navigating chat -> home -> chat silently dropped a staged live-work
    item with no error and no user-visible warning -- the live critique
    blamed Library's "Run" action for this, but Run is pure; screen
    teardown on ANY navigation away was the actual destroyer.
    """
    from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkLaunch
    from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel

    app = _build_test_app()
    app.pending_handoffs.stage(
        HandoffChannel.CONSOLE_LIVE_WORK,
        ConsoleLiveWorkLaunch.from_values(
            source="workflows",
            title="Daily digest",
            payload={"attempt": 2, "run_id": "run-1"},
            status="running",
            recovery="Workflow is starting.",
            action_label="Open workflow run",
        ),
    )

    async with app.run_test(size=(170, 48)) as pilot:
        for _ in range(150):
            await pilot.pause(0.02)
            if type(app.screen).__name__ == "ChatScreen" and app.screen.query(
                "#console-pending-launch-card"
            ):
                break
        assert type(app.screen).__name__ == "ChatScreen"
        first_screen = app.screen
        assert first_screen._pending_console_launch_context is not None
        assert first_screen._pending_console_launch_context.title == "Daily digest"
        assert not app.pending_handoffs.has_pending(HandoffChannel.CONSOLE_LIVE_WORK)

        app.post_message(NavigateToScreen("home"))
        for _ in range(150):
            await pilot.pause(0.02)
            if type(app.screen).__name__ == "HomeScreen":
                break
        assert type(app.screen).__name__ == "HomeScreen"
        assert app.screen is not first_screen

        app.post_message(NavigateToScreen("chat"))
        for _ in range(150):
            await pilot.pause(0.02)
            if type(app.screen).__name__ == "ChatScreen" and app.screen.query(
                "#console-pending-launch-card"
            ):
                break

        restored_screen = app.screen
        assert type(restored_screen).__name__ == "ChatScreen"
        # A genuinely fresh instance, not a cached/reused one.
        assert restored_screen is not first_screen
        assert restored_screen._pending_console_launch_context is not None
        assert restored_screen._pending_console_launch_context.title == "Daily digest"
        assert restored_screen.query_one("#console-pending-launch-card")
        assert (
            restored_screen.query_one("#console-live-work-title").renderable
            == "Title: Daily digest"
        )
        # The restore must not have re-claimed the handoff channel -- it was
        # already empty (consumed by `first_screen`) and stays that way.
        assert not app.pending_handoffs.has_pending(HandoffChannel.CONSOLE_LIVE_WORK)


@pytest.mark.asyncio
async def test_prompts_route_lands_on_library_with_prompts_row_selected():
    """``NavigateToScreen("prompts")`` must land on Library with the prompts
    rail row selected. The Personas "prompts" mode chip is retired (Task 7)
    and the legacy route now re-points into Library, mirroring how
    ``open_notes_workspace`` re-points "notes" via a
    ``LIBRARY_NAV_CONTEXT_MODE`` nav-context selection -- except "prompts"
    has no dedicated re-entry action to carry that context (the retired
    Personas mode chip had no equivalent workspace to return to), so the
    bare alias route itself must supply it.
    """
    from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_PROMPTS

    app = _build_test_app()

    async with app.run_test(size=(170, 48)) as pilot:
        for _ in range(150):
            await pilot.pause(0.02)
            if type(app.screen).__name__ != "Screen":
                break

        app.post_message(NavigateToScreen("prompts"))
        for _ in range(150):
            await pilot.pause(0.02)
            if type(app.screen).__name__ == "LibraryScreen" and app.screen.query(
                "#library-row-browse-prompts"
            ):
                break

        assert type(app.screen).__name__ == "LibraryScreen"
        assert app.screen._library_selected_row_id == LIBRARY_ROW_BROWSE_PROMPTS


@pytest.mark.asyncio
async def test_skills_route_lands_on_library_with_skills_row_selected():
    """``NavigateToScreen("skills")`` must land on Library with the skills
    rail row selected. The standalone Skills tab is retired (Skills
    sub-project Task 5) and the legacy route now re-points into Library,
    mirroring ``test_prompts_route_lands_on_library_with_prompts_row_selected``
    exactly -- "skills" (like "prompts") has no dedicated re-entry action to
    carry a nav-context, so the bare alias route itself must supply it via
    ``_LEGACY_ROUTE_LIBRARY_NAV_CONTEXT``.
    """
    from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_SKILLS

    app = _build_test_app()

    async with app.run_test(size=(170, 48)) as pilot:
        for _ in range(150):
            await pilot.pause(0.02)
            if type(app.screen).__name__ != "Screen":
                break

        app.post_message(NavigateToScreen("skills"))
        for _ in range(150):
            await pilot.pause(0.02)
            if type(app.screen).__name__ == "LibraryScreen" and app.screen.query(
                "#library-row-browse-skills"
            ):
                break

        assert type(app.screen).__name__ == "LibraryScreen"
        assert app.screen._library_selected_row_id == LIBRARY_ROW_BROWSE_SKILLS


@pytest.mark.asyncio
async def test_search_route_lands_on_library_rag_canvas():
    """``NavigateToScreen("search")`` must land on Library with the
    Search/RAG rail row selected. The standalone Search screen is retired
    (RAG UX v2 PR-1, Task 1) and the legacy "search" route now re-points into
    Library, mirroring ``test_prompts_route_lands_on_library_with_prompts_row_selected``
    /``test_skills_route_lands_on_library_with_skills_row_selected`` exactly --
    "search" has no dedicated re-entry action to carry a nav-context, so the
    bare alias route itself must supply it via
    ``_LEGACY_ROUTE_LIBRARY_NAV_CONTEXT``.
    """
    from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_SEARCH

    app = _build_test_app()

    async with app.run_test(size=(170, 48)) as pilot:
        for _ in range(150):
            await pilot.pause(0.02)
            if type(app.screen).__name__ != "Screen":
                break

        app.post_message(NavigateToScreen("search"))
        for _ in range(150):
            await pilot.pause(0.02)
            if type(app.screen).__name__ == "LibraryScreen" and app.screen.query(
                "#library-row-browse-search"
            ):
                break

        assert type(app.screen).__name__ == "LibraryScreen"
        assert app.screen._library_selected_row_id == LIBRARY_ROW_BROWSE_SEARCH


@pytest.mark.asyncio
async def test_study_screen_escape_returns_to_library_study_staging_canvas():
    """task-2854 AC#3: Escape on the Study screen (reached via Library's
    Study/Flashcards/Quizzes handoff rows -- "Continue in Study") must
    return to Library with the Study staging canvas ("create-study" row)
    selected. Before this fix Escape was dead on Study and the nav bar
    falsely claimed Library was still current (see
    ``test_study_screen_mounts_destination_header_and_clears_nav_highlight``
    in ``test_destination_headers.py`` for that half of the fix); this
    covers the actual round trip out.
    """
    app = _build_test_app()

    async with app.run_test(size=(170, 48)) as pilot:
        for _ in range(150):
            await pilot.pause(0.02)
            if type(app.screen).__name__ != "Screen":
                break

        app.post_message(NavigateToScreen("study"))
        for _ in range(150):
            await pilot.pause(0.02)
            if type(app.screen).__name__ == "StudyScreen":
                break
        assert type(app.screen).__name__ == "StudyScreen"

        await pilot.press("escape")
        for _ in range(150):
            await pilot.pause(0.02)
            if type(app.screen).__name__ == "LibraryScreen" and app.screen.query(
                "#library-row-create-study"
            ):
                break

        assert type(app.screen).__name__ == "LibraryScreen"
        assert app.screen._library_selected_row_id == "create-study"


@pytest.mark.asyncio
async def test_boot_with_search_default_tab_lands_on_library_rag_canvas():
    """RAG UX v2 PR-2, Task 4: booting with ``default_tab = "search"`` must
    land on Library with the Search/RAG rail row selected, not generic
    Library. Mirrors ``test_search_route_lands_on_library_rag_canvas`` above
    exactly, except it drives the BOOT path (``_push_initial_screen``, via
    ``_build_test_app(configured_default=...)``) instead of an in-app
    ``NavigateToScreen`` message -- before this fix, ``_push_initial_screen``
    never consulted ``_LEGACY_ROUTE_LIBRARY_NAV_CONTEXT``, so a configured
    "search" default tab silently degraded to the generic Library canvas
    (default rail row) instead of honoring the alias's landing promise.
    """
    from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_SEARCH

    app = _build_test_app(configured_default="search")

    async with app.run_test(size=(170, 48)) as pilot:
        for _ in range(150):
            await pilot.pause(0.02)
            if type(app.screen).__name__ == "LibraryScreen" and app.screen.query(
                "#library-row-browse-search"
            ):
                break

        assert type(app.screen).__name__ == "LibraryScreen"
        assert app.screen._library_selected_row_id == LIBRARY_ROW_BROWSE_SEARCH


@pytest.mark.asyncio
async def test_boot_with_prompts_default_tab_lands_on_library_with_prompts_row_selected():
    """Sibling of ``test_boot_with_search_default_tab_lands_on_library_rag_canvas``
    proving the boot-time fix is generic across
    ``_LEGACY_ROUTE_LIBRARY_NAV_CONTEXT`` rather than special-cased to
    "search" -- the table also carries "prompts", "skills" and "customize",
    and the fix must apply whichever pre-resolution route id
    ``_resolve_initial_shell_route()`` returns.
    """
    from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_PROMPTS

    app = _build_test_app(configured_default="prompts")

    async with app.run_test(size=(170, 48)) as pilot:
        for _ in range(150):
            await pilot.pause(0.02)
            if type(app.screen).__name__ == "LibraryScreen" and app.screen.query(
                "#library-row-browse-prompts"
            ):
                break

        assert type(app.screen).__name__ == "LibraryScreen"
        assert app.screen._library_selected_row_id == LIBRARY_ROW_BROWSE_PROMPTS


@pytest.mark.asyncio
async def test_search_all_palette_command_lands_on_library_with_honest_toast():
    """RAG UX v2 PR-1, Task 2: the "Search All Content" quick-action palette
    command dispatches through the "search" alias (Task 1), so it must
    resolve to the same Library Search/RAG canvas as
    ``test_search_route_lands_on_library_rag_canvas`` -- and its toast must
    say so honestly instead of promising the retired standalone "Search/RAG"
    screen.
    """
    from tldw_chatbook.app import QuickActionsProvider
    from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_SEARCH

    app = _build_test_app()
    notices: list[tuple[str, str]] = []

    async with app.run_test(size=(170, 48)) as pilot:
        for _ in range(150):
            await pilot.pause(0.02)
            if type(app.screen).__name__ != "Screen":
                break

        app.notify = lambda message_text, **kwargs: notices.append(
            (str(message_text), kwargs.get("severity", ""))
        )

        provider = QuickActionsProvider(screen=app.screen)
        provider.execute_quick_action("search_all")

        for _ in range(150):
            await pilot.pause(0.02)
            if type(app.screen).__name__ == "LibraryScreen" and app.screen.query(
                "#library-row-browse-search"
            ):
                break

        assert type(app.screen).__name__ == "LibraryScreen"
        assert app.screen._library_selected_row_id == LIBRARY_ROW_BROWSE_SEARCH
        assert ("Opened Library Search/RAG", "information") in notices


@pytest.mark.asyncio
async def test_media_route_round_trips_to_the_library_media_row():
    """task-2851: the legacy standalone Media Library screen (nav: Media
    Types / All Media / Analysis Review / Collections-Tags / Multi-Item
    Review) used to render UNDER the active Library tab highlight when the
    command palette's "Media & Content: Open Media Library" entry navigated
    to the bare "media" route (``MediaProvider.handle_media_action
    ("open_media")`` -> ``NavigateToScreen("media")``) -- the legacy route
    folds into the "library" shell destination for nav-bar purposes, but its
    own screen route pointed at a completely different, dead-end-duplicate
    screen. "media" is now retired the same way "search" was (RAG UX v2
    PR-1): it resolves to ``LibraryScreen`` with the Media rail row
    selected, and -- mirroring
    ``test_search_route_round_trips_to_the_library_rag_row`` exactly -- that
    selection survives a round trip through another screen rather than being
    just a first-navigation fluke of ``_LEGACY_ROUTE_LIBRARY_NAV_CONTEXT``.
    """
    from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_MEDIA

    app = _build_test_app()

    async with app.run_test(size=(170, 48)) as pilot:
        for _ in range(150):
            await pilot.pause(0.02)
            if type(app.screen).__name__ != "Screen":
                break

        app.post_message(NavigateToScreen("media"))
        for _ in range(150):
            await pilot.pause(0.02)
            if type(app.screen).__name__ == "LibraryScreen" and app.screen.query(
                "#library-row-browse-media"
            ):
                break
        assert type(app.screen).__name__ == "LibraryScreen"
        assert app.screen._library_selected_row_id == LIBRARY_ROW_BROWSE_MEDIA

        app.post_message(NavigateToScreen("home"))
        for _ in range(150):
            await pilot.pause(0.02)
            if type(app.screen).__name__ == "HomeScreen":
                break
        assert type(app.screen).__name__ == "HomeScreen"

        app.post_message(NavigateToScreen("media"))
        for _ in range(150):
            await pilot.pause(0.02)
            if type(app.screen).__name__ == "LibraryScreen" and app.screen.query(
                "#library-row-browse-media"
            ):
                break

        restored_screen = app.screen
        assert type(restored_screen).__name__ == "LibraryScreen"
        assert restored_screen._library_selected_row_id == LIBRARY_ROW_BROWSE_MEDIA


@pytest.mark.asyncio
async def test_search_route_round_trips_to_the_library_rag_row():
    """The retired standalone Search screen is folded into Library (RAG UX
    v2 PR-1, Task 1): the "search" route no longer has a runtime-state seam
    of its own, so this locks that the alias's rail-row selection survives a
    round trip through another screen and is not just a first-navigation
    fluke of ``_LEGACY_ROUTE_LIBRARY_NAV_CONTEXT``. Unlike the "library" +
    click entry point exercised by
    ``test_library_screen_round_trip_restores_rag_query_and_rail_selection``,
    entering via the bare "search" alias re-applies that legacy nav context
    on every visit rather than relying solely on restored screen state.
    """
    from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_SEARCH

    app = _build_test_app()

    async with app.run_test(size=(170, 48)) as pilot:
        for _ in range(150):
            await pilot.pause(0.02)
            if type(app.screen).__name__ != "Screen":
                break

        app.post_message(NavigateToScreen("search"))
        for _ in range(150):
            await pilot.pause(0.02)
            if type(app.screen).__name__ == "LibraryScreen" and app.screen.query(
                "#library-row-browse-search"
            ):
                break
        assert type(app.screen).__name__ == "LibraryScreen"
        assert app.screen._library_selected_row_id == LIBRARY_ROW_BROWSE_SEARCH

        app.post_message(NavigateToScreen("home"))
        for _ in range(150):
            await pilot.pause(0.02)
            if type(app.screen).__name__ == "HomeScreen":
                break
        assert type(app.screen).__name__ == "HomeScreen"

        app.post_message(NavigateToScreen("search"))
        for _ in range(150):
            await pilot.pause(0.02)
            if type(app.screen).__name__ == "LibraryScreen" and app.screen.query(
                "#library-row-browse-search"
            ):
                break

        restored_screen = app.screen
        assert type(restored_screen).__name__ == "LibraryScreen"
        assert restored_screen._library_selected_row_id == LIBRARY_ROW_BROWSE_SEARCH


# --- Media/Search unit-style save_state/restore_state contracts -----------


def test_media_screen_save_state_returns_expected_keys():
    app = _build_test_app()
    screen = MediaScreen(app)
    list(screen.compose_content())  # populate screen.media_window
    screen.media_window.active_media_type = "all-media"
    screen.media_window.selected_media_id = "media-7"
    screen.media_window.search_panel = SimpleNamespace(
        search_term="alpha", keyword_filter="beta"
    )

    state = screen.save_state()

    assert state["media_active_type"] == "all-media"
    assert state["media_selected_id"] == "media-7"
    assert state["media_search_term"] == "alpha"
    assert state["media_keyword_filter"] == "beta"


def test_media_screen_save_state_never_raises_when_window_unset():
    app = _build_test_app()
    screen = MediaScreen(app)  # compose_content never ran -- media_window is None

    state = screen.save_state()

    assert "media_active_type" not in state


def test_media_screen_restore_state_stashes_pending_dict_for_on_mount():
    """``restore_state`` runs on a fresh, not-yet-mounted instance -- the
    MediaWindow it will compose does not exist yet, so it can only stash the
    values for ``on_mount`` to apply once ``compose_content`` has run.
    """
    app = _build_test_app()
    screen = MediaScreen(app)

    screen.restore_state(
        {
            "media_active_type": "video",
            "media_selected_id": "media-9",
            "media_search_term": "q",
            "media_keyword_filter": "kw",
        }
    )

    assert screen._pending_media_restore == {
        "active_media_type": "video",
        "selected_media_id": "media-9",
        "search_term": "q",
        "keyword_filter": "kw",
    }


def test_the_retired_ingest_route_resolves_to_library():
    """The legacy ``ingest`` route must still land somewhere real.

    The standalone Ingest screen is retired (task-684.4) now that importing
    lives entirely in Library's Import media canvas, including the server-backed
    and web-clipping paths it used to own. The route id is kept as an alias
    rather than deleted, because startup configs and saved navigation state can
    still say "ingest" -- dropping it outright would dead-end them, and
    ``_ROUTABLE_LEGACY_ROUTES`` already treats it as a routable legacy id.

    Mirrors the ``notes``/``prompts``/``skills`` retirements exactly, and matches
    the Workbench route inventory, which already declared ingest -> library.
    """
    from tldw_chatbook.UI.Navigation.screen_registry import resolve_screen_target

    ingest_name, ingest_tab, ingest_class = resolve_screen_target("ingest")
    library_name, library_tab, library_class = resolve_screen_target("library")

    assert (ingest_name, ingest_tab, ingest_class) == (
        library_name,
        library_tab,
        library_class,
    )
    assert ingest_class is not None, "the ingest route must not dead-end"


def test_no_route_reaches_the_retired_ingest_screen():
    """AC#2: nothing may still resolve to the deleted screen."""
    from tldw_chatbook.UI.Navigation import screen_registry

    for route_id in screen_registry.registered_screen_route_ids():
        route = screen_registry._SCREEN_ROUTES[route_id]
        assert "media_ingest_screen" not in route.module_path, route_id
        assert route.class_name != "MediaIngestScreen", route_id


# --- Startup-failure diagnosability -----------------------------------------
# Root-caused 2026-07-27: `aiohttp` (an optional dependency) sat on the default
# chat screen's import chain via Media_Creation/swarmui_client.py. With it
# absent, `ScreenRoute.load_screen_class()` swallowed the ModuleNotFoundError
# into a warning and returned None, so the app died on start with a bare
# `RuntimeError: Unable to resolve default chat screen` -- naming neither the
# missing module nor the file that imported it. `resolve_screen_target()` keeps
# its graceful None contract (a broken optional screen must not break
# navigation), but the fatal startup site must report the underlying cause.


def test_screen_load_error_reports_underlying_import_failure(monkeypatch):
    """`screen_load_error()` must return the exception blocking a route's load.

    Args:
        monkeypatch: pytest fixture; points the chat route at a module that
            does not exist, so the load fails the way a missing dependency
            deep in the import chain does.
    """
    from tldw_chatbook.UI.Navigation import screen_registry

    route = screen_registry._SCREEN_ROUTES["chat"]
    broken = replace(route, module_path="tldw_chatbook.UI.Screens.no_such_screen_xyz")
    monkeypatch.setitem(screen_registry._SCREEN_ROUTES, "chat", broken)

    # Precondition: the route resolves to None, i.e. the masked failure mode.
    _name, _tab, screen_class = screen_registry.resolve_screen_target("chat")
    assert screen_class is None

    cause = screen_registry.screen_load_error("chat")
    assert isinstance(cause, ImportError)
    assert "no_such_screen_xyz" in str(cause)


def test_screen_load_error_returns_none_for_a_loadable_route():
    """A healthy route has no load failure to report."""
    from tldw_chatbook.UI.Navigation import screen_registry

    assert screen_registry.screen_load_error("chat") is None


def test_screen_load_error_reports_missing_optional_dependency(monkeypatch):
    """A dependency-gated route reports the gate, not a bare None.

    The gate short-circuits before the import is attempted, so there is no
    exception to surface -- but the caller still needs a reason, otherwise the
    fatal startup message stays as uninformative as the bug this guards.

    Args:
        monkeypatch: pytest fixture; gates the chat route on a dependency
            check name that `optional_deps` does not define, so
            `dependencies_available()` reports False.
    """
    from tldw_chatbook.UI.Navigation import screen_registry

    route = screen_registry._SCREEN_ROUTES["chat"]
    gated = replace(route, dependency_check="definitely_not_a_real_dep_check")
    monkeypatch.setitem(screen_registry._SCREEN_ROUTES, "chat", gated)

    cause = screen_registry.screen_load_error("chat")
    assert cause is not None
    assert "definitely_not_a_real_dep_check" in str(cause)


def test_screen_load_error_handles_unknown_route():
    """An unroutable target reports a miss rather than raising."""
    from tldw_chatbook.UI.Navigation import screen_registry

    cause = screen_registry.screen_load_error("no_such_route_xyz")
    assert cause is not None
    assert "no_such_route_xyz" in str(cause)


def test_push_initial_screen_fatal_error_names_the_underlying_cause(monkeypatch):
    """The fatal startup error must name the real blocker, not just the symptom.

    Exercises `_push_initial_screen()`'s unresolvable branch against a stub
    `self` -- the method only reads `_initial_screen_pushed` and calls two
    resolution helpers before raising, so this needs no Textual app boot.

    Driven via `asyncio.run()` rather than an `async def` test: only
    `Tests/UI/pytest.ini` sets `asyncio_mode = auto`, and there is no
    repo-root pytest.ini, so a sweep spanning Tests/UI *and* other
    directories resolves a different rootdir/config and would not collect
    this as an async test.

    Args:
        monkeypatch: pytest fixture; points the chat route at a module that
            does not exist, making the default screen unresolvable so the
            fatal branch is reached.
    """
    from tldw_chatbook.UI.Navigation import screen_registry

    route = screen_registry._SCREEN_ROUTES["chat"]
    broken = replace(route, module_path="tldw_chatbook.UI.Screens.no_such_screen_xyz")
    monkeypatch.setitem(screen_registry._SCREEN_ROUTES, "chat", broken)

    stub = SimpleNamespace(
        _initial_screen_pushed=False,
        _resolve_initial_shell_route=lambda: "chat",
        _resolve_screen_navigation_target=screen_registry.resolve_screen_target,
    )

    with pytest.raises(RuntimeError) as excinfo:
        asyncio.run(TldwCli._push_initial_screen(stub))

    message = str(excinfo.value)
    # The old message was exactly "Unable to resolve default chat screen" --
    # it named neither the failing module nor the exception type.
    assert "no_such_screen_xyz" in message, message
    assert "ModuleNotFoundError" in message, message
    # Chained, so the traceback shows the real import failure too.
    assert isinstance(excinfo.value.__cause__, ImportError)


@pytest.mark.asyncio
async def test_navigation_survives_screen_construction_failure(monkeypatch):
    """A screen whose ``__init__`` raises must not take the whole app down.

    Root cause of the reported "app crashes when clicking onto MCP": the MCP
    canvases read ``Select.NULL`` at construction time, which does not exist
    before Textual 8. ``_complete_screen_navigation`` guarded ``save_state``,
    ``restore_state`` and ``apply_navigation_context`` but ran
    ``_create_navigation_screen`` unguarded, so the AttributeError escaped the
    ``NavigateToScreen`` handler and Textual exited the app (return_code 1).

    Any screen that fails to build is a broken destination, never a dead app:
    the user must be told and left on the screen they were already using.
    """
    app = _build_test_app()
    app._initial_screen_pushed = True

    class ExplodingScreen:
        screen_name = "mcp"

        def __init__(self, app_instance):
            raise AttributeError("type object 'Select' has no attribute 'NULL'")

    def fake_resolve(target):
        return "mcp", "mcp", ExplodingScreen

    switched_screens = []

    async def fake_switch_screen(screen):
        switched_screens.append(screen)

    notifications = []

    class FakeOutgoingScreen:
        screen_name = "chat"

    # Same shim the flush/veto tests use: the handler reads self.screen for
    # the outgoing save-state step, which needs a live screen stack.
    monkeypatch.setattr(
        type(app), "screen", property(lambda self: FakeOutgoingScreen())
    )
    monkeypatch.setattr(app, "_resolve_screen_navigation_target", fake_resolve)
    monkeypatch.setattr(app, "switch_screen", fake_switch_screen)
    monkeypatch.setattr(
        app, "notify", lambda message, **kwargs: notifications.append(message)
    )

    # Must not raise: an escaping exception here is what killed the app.
    await app.handle_screen_navigation(NavigateToScreen("mcp"))

    assert switched_screens == [], "a screen that failed to build must not be switched to"
    assert notifications, "the user must be told the destination failed to open"


@pytest.mark.asyncio
async def test_navigation_survives_screen_mount_failure(monkeypatch):
    """A screen that raises while mounting must not take the whole app down.

    Sibling of the construction guard: the MCP audit canvas reads
    ``Select.NULL`` inside ``compose()``, so the same AttributeError can
    surface from ``switch_screen`` (which drives compose/mount) rather than
    from ``__init__``. Both legs must fail soft.
    """
    app = _build_test_app()
    app._initial_screen_pushed = True

    class FakeScreen:
        screen_name = "mcp"

        def __init__(self, app_instance):
            self.app_instance = app_instance

    def fake_resolve(target):
        return "mcp", "mcp", FakeScreen

    async def exploding_switch_screen(screen):
        raise AttributeError("type object 'Select' has no attribute 'NULL'")

    notifications = []

    class FakeOutgoingScreen:
        screen_name = "chat"

    # Same shim the flush/veto tests use: the handler reads self.screen for
    # the outgoing save-state step, which needs a live screen stack.
    monkeypatch.setattr(
        type(app), "screen", property(lambda self: FakeOutgoingScreen())
    )
    monkeypatch.setattr(app, "_resolve_screen_navigation_target", fake_resolve)
    monkeypatch.setattr(app, "switch_screen", exploding_switch_screen)
    monkeypatch.setattr(
        app, "notify", lambda message, **kwargs: notifications.append(message)
    )

    await app.handle_screen_navigation(NavigateToScreen("mcp"))

    assert notifications, "the user must be told the destination failed to open"


@pytest.mark.asyncio
async def test_navigation_flush_that_never_returns_does_not_freeze_the_app(monkeypatch):
    """A hung outgoing flush must not freeze the whole app forever.

    ``handle_screen_navigation`` is an ``@on`` handler on the App, so while
    it awaits, the App's own message pump processes nothing -- no clicks, no
    bindings, no further navigation. The outgoing flush reaches unbounded
    awaits (``library_screen``'s ``await worker.wait()`` with no timeout, and
    ``_run_library_service_call``'s uncancellable ``asyncio.to_thread``), so a
    save that never completes used to leave the app permanently frozen and
    unkillable rather than merely slow.

    The flush must therefore be bounded: on timeout the app fails closed
    (stays put, pending edits intact) and stays responsive.
    """
    app = _build_test_app()
    app._initial_screen_pushed = True

    class FakeTargetScreen:
        screen_name = "chat"

        def __init__(self, app_instance):
            self.app_instance = app_instance

    def fake_resolve(target):
        return "chat", "chat", FakeTargetScreen

    switched_screens = []

    async def fake_switch_screen(screen):
        switched_screens.append(screen)

    class HungOutgoingScreen:
        screen_name = "library"

        async def flush_pending_work(self):
            await asyncio.Event().wait()  # never completes

    notifications = []
    monkeypatch.setattr(app, "_resolve_screen_navigation_target", fake_resolve)
    monkeypatch.setattr(app, "switch_screen", fake_switch_screen)
    monkeypatch.setattr(
        type(app), "screen", property(lambda self: HungOutgoingScreen())
    )
    monkeypatch.setattr(
        app, "notify", lambda message, **kwargs: notifications.append(message)
    )
    monkeypatch.setattr(app, "NAVIGATION_FLUSH_TIMEOUT_SECONDS", 0.2, raising=False)

    # If the flush is unbounded this never returns and the suite hangs, so
    # bound the assertion itself rather than wedging CI.
    await asyncio.wait_for(
        app.handle_screen_navigation(NavigateToScreen("chat")), timeout=10
    )

    assert switched_screens == [], "a timed-out flush must fail closed, not switch"
    assert notifications, "the user must be told the switch was abandoned"


@pytest.mark.asyncio
async def test_navigation_timeout_does_not_cancel_the_in_flight_save(monkeypatch):
    """Timing out the flush must not cancel the save's reconciliation.

    Bounding the flush released the App message pump, but `asyncio.wait_for`
    cancels what it waits on -- and the Library File Notes save is a
    `asyncio.to_thread` write that cannot be cancelled. The thread kept
    writing while the coroutine died at the await, so `_save_draft` never ran
    the lines after it: `_save_state` stayed "saving" and `_opened.content_hash`
    kept the pre-save value.

    That is worse than the freeze it replaced. `leave_allowed` is False while
    the state is "saving", so the screen becomes permanently non-leavable and
    the next save compares against a stale hash.

    The wait must therefore be shielded: the app stops waiting, the save does
    not stop saving.
    """
    app = _build_test_app()
    app._initial_screen_pushed = True

    class FakeTargetScreen:
        screen_name = "chat"

        def __init__(self, app_instance):
            self.app_instance = app_instance

    def fake_resolve(target):
        return "chat", "chat", FakeTargetScreen

    async def fake_switch_screen(screen):
        pass

    reconciled = {"done": False, "cancelled": False}

    class SlowSavingScreen:
        screen_name = "library"

        async def flush_pending_work(self):
            try:
                # Stands in for `await asyncio.to_thread(service.save_file, ...)`:
                # slower than the navigation budget, and uncancellable in reality.
                await asyncio.sleep(0.6)
            except asyncio.CancelledError:
                reconciled["cancelled"] = True
                raise
            # The post-save reconciliation that must still run.
            reconciled["done"] = True
            return True

    notifications = []
    monkeypatch.setattr(app, "_resolve_screen_navigation_target", fake_resolve)
    monkeypatch.setattr(app, "switch_screen", fake_switch_screen)
    monkeypatch.setattr(
        type(app), "screen", property(lambda self: SlowSavingScreen())
    )
    monkeypatch.setattr(
        app, "notify", lambda message, **kwargs: notifications.append(message)
    )
    monkeypatch.setattr(app, "NAVIGATION_FLUSH_TIMEOUT_SECONDS", 0.1, raising=False)

    await app.handle_screen_navigation(NavigateToScreen("chat"))
    assert notifications, "the user must be told the switch was abandoned"

    # Give the shielded save the time it needs to finish on its own.
    await asyncio.sleep(1.0)

    assert not reconciled["cancelled"], (
        "the in-flight save was cancelled; its reconciliation never ran, so the "
        "editor is stuck in 'saving' with a stale content hash"
    )
    assert reconciled["done"], "the save must still complete after the app stops waiting"


@pytest.mark.asyncio
async def test_broken_screen_content_degrades_instead_of_killing_a_running_app():
    """Integration lock for the compose seam, in a real running Textual app.

    The focused navigation tests stub ``switch_screen``, which hid a real
    limitation: Textual composes a screen inside its own mount pipeline, so an
    exception in ``compose_content`` is never raised back to whoever called
    ``switch_screen``. Textual records it on the App and exits the process --
    the navigation handler's try/except cannot see it, so ``BaseAppScreen`` is
    the only place that can catch it. That is the path the MCP crash took.

    This drives a real ``push_screen`` + compose + mount through a live app.
    It deliberately does NOT use ``_build_test_app``: real ``switch_screen``
    does not work in that harness (navigating to a healthy screen fails there
    too), which is why the other tests stub it.
    """

    class BrokenScreen(BaseAppScreen):
        def __init__(self):
            super().__init__(app_instance=None, screen_name="mcp")

        def compose_content(self) -> ComposeResult:
            # Stands in for the MCP canvases reading Select.NULL on a Textual
            # that predates it.
            raise AttributeError("type object 'Select' has no attribute 'NULL'")
            yield  # pragma: no cover - unreachable, keeps this a generator

    class HostApp(ConsolidatedCSSApp):
        pass

    app = HostApp()
    async with app.run_test(size=(160, 40)) as pilot:
        await pilot.pause()
        await app.push_screen(BrokenScreen())
        await pilot.pause()

        assert app.is_running, "a screen that fails to compose must not exit the app"
        assert app.screen.query("#screen-content-error"), (
            "the failure must be shown, not swallowed into a blank screen"
        )
        # The message pump must still be live afterwards.
        await pilot.press("tab")
        await pilot.pause()
        assert app.is_running, "app must stay responsive after the failed compose"


# --- task-2858 (LIB-03 -> AC#1): entry routing + canvas restoration -------
#
# Direction (recorded in the task's Implementation Notes): EXPLICIT deep
# links (e.g. "Library: Import...", the "media"/"search" legacy alias
# routes) keep landing their own labeled canvas -- their own command text
# states the destination. GENERIC entries (bare ``NavigateToScreen("library")``
# with no nav-context -- the nav-bar tab button's and the "Switch to
# Library" palette command's own shape) land ONE canonical surface: the
# hub on a first visit, or the last-visited canvas on a revisit
# (restore-over-reset -- a workbench should not lose your place). These
# three pilot round trips pin that contract directly against the real
# navigation orchestration (``TldwCli.handle_screen_navigation`` /
# ``ScreenStateStore`` / ``LibraryScreen.save_state``/``restore_state``/
# ``apply_navigation_context``) rather than re-deriving it, matching
# ``test_rapid_tab_switch_storm_leaves_no_zombie_widgets``'s real-screen
# harness pattern immediately above.
async def _wait_for_initial_screen(pilot) -> None:
    """Poll until the app's own startup navigation has pushed a real screen.

    Waiting on ``app.screen``'s type alone races ``_push_initial_screen``:
    the app already composes a real (non-generic ``Screen``) instance while
    the splash screen closes, but ``handle_screen_navigation`` silently
    ignores every request until ``_initial_screen_pushed`` flips True (see
    ``_handle_screen_navigation_locked``'s startup guard) -- a direct
    ``await app.handle_screen_navigation(...)`` issued in that window is
    dropped with no error and no way to retry.
    """
    app = pilot.app
    for _ in range(150):
        await pilot.pause(0.02)
        if getattr(app, "_initial_screen_pushed", False):
            return
    raise AssertionError("app never finished pushing its initial screen")


@pytest.mark.asyncio
async def test_generic_library_entry_lands_hub_on_first_visit():
    """A GENERIC Library entry with no prior visit in this session lands the
    hub/landing canvas (``_library_selected_row_id == ""``), never a
    specific canvas by accident.
    """
    app = _build_test_app()

    async with app.run_test(size=(160, 40)) as pilot:
        await _wait_for_initial_screen(pilot)

        await app.handle_screen_navigation(NavigateToScreen("library"))

        assert type(app.screen).__name__ == "LibraryScreen"
        assert app.screen._library_selected_row_id == ""


@pytest.mark.asyncio
async def test_prompt_receipt_owner_vetoes_real_app_navigation_until_settlement():
    """The real app cannot replace the Library screen while it owns a mutation."""
    from tldw_chatbook.Prompt_Management.prompt_batch_models import (
        PromptBatchDeleteResult,
        PromptDeleteReceiptEntry,
    )

    app = _build_test_app()

    async with app.run_test(size=(160, 40)) as pilot:
        await _wait_for_initial_screen(pilot)
        await app.handle_screen_navigation(NavigateToScreen("library"))
        screen = app.screen
        assert type(screen).__name__ == "LibraryScreen"
        receipt = PromptBatchDeleteResult(
            (PromptDeleteReceiptEntry(41, "Receipt owner", "prompt", 2),)
        )
        screen._library_prompt_delete_receipt = receipt
        screen._library_prompts_mutation_in_flight = True

        await app.handle_screen_navigation(NavigateToScreen("home"))
        await pilot.pause()

        assert app.screen is screen
        assert screen._library_prompt_delete_receipt is receipt

        screen._library_prompts_mutation_in_flight = False
        await app.handle_screen_navigation(NavigateToScreen("home"))
        await pilot.pause()
        assert type(app.screen).__name__ == "HomeScreen"


@pytest.mark.asyncio
async def test_deep_link_library_route_lands_its_canvas_over_restored_state():
    """An EXPLICIT deep link (mirroring ``LibraryIngestProvider``'s
    "Library: Import..." palette command, which supplies
    ``{LIBRARY_NAV_CONTEXT_INGEST: True}``) must land its own labeled canvas
    even when a DIFFERENT canvas was left behind by a prior visit -- deep
    links state their own destination, so the generic-entry restore-over-
    reset rule does not apply to them.
    """
    from tldw_chatbook.Constants import LIBRARY_NAV_CONTEXT_INGEST
    from tldw_chatbook.Library.library_shell_state import (
        LIBRARY_ROW_BROWSE_MEDIA,
        LIBRARY_ROW_INGEST_MEDIA,
    )

    app = _build_test_app()

    async with app.run_test(size=(160, 40)) as pilot:
        await _wait_for_initial_screen(pilot)

        # A prior visit lands on Media (the "media" legacy alias route --
        # mirrors "Media & Content: Open Media Library").
        await app.handle_screen_navigation(NavigateToScreen("media"))
        assert app.screen._library_selected_row_id == LIBRARY_ROW_BROWSE_MEDIA

        # Leave -- this is what persists that Media selection as the
        # "last-visited" canvas under the canonical "library" route.
        await app.handle_screen_navigation(NavigateToScreen("home"))
        assert type(app.screen).__name__ == "HomeScreen"

        # The explicit deep link must land Import, not the restored Media row.
        await app.handle_screen_navigation(
            NavigateToScreen("library", {LIBRARY_NAV_CONTEXT_INGEST: True})
        )

        assert type(app.screen).__name__ == "LibraryScreen"
        assert app.screen._library_selected_row_id == LIBRARY_ROW_INGEST_MEDIA


@pytest.mark.asyncio
async def test_generic_reentry_restores_last_visited_library_canvas():
    """Core LIB-03 round trip: visit Search/RAG, leave to Home, then
    re-enter Library GENERICALLY (bare ``NavigateToScreen``, no context --
    the nav-bar tab button's own shape) -- the Search/RAG canvas must be
    RESTORED, not reset back to the hub or any other canvas.
    """
    from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_SEARCH

    app = _build_test_app()

    async with app.run_test(size=(160, 40)) as pilot:
        await _wait_for_initial_screen(pilot)

        # Deep-link into Search/RAG (the "search" legacy alias route --
        # mirrors "Media & Content: Search Transcripts").
        await app.handle_screen_navigation(NavigateToScreen("search"))
        assert type(app.screen).__name__ == "LibraryScreen"
        assert app.screen._library_selected_row_id == LIBRARY_ROW_BROWSE_SEARCH

        # Leave.
        await app.handle_screen_navigation(NavigateToScreen("home"))
        assert type(app.screen).__name__ == "HomeScreen"

        # Generic re-entry must restore Search/RAG.
        await app.handle_screen_navigation(NavigateToScreen("library"))

        assert type(app.screen).__name__ == "LibraryScreen"
        assert app.screen._library_selected_row_id == LIBRARY_ROW_BROWSE_SEARCH


@pytest.mark.asyncio
async def test_nav_bar_no_destination_truncates_at_160_cols():
    """NV-01 (TASK-2154.21): the strip fits all 13 destinations at 160 cols.

    The hotkey-prefixed labels (``⌃1 Home`` … ``F9 Settings``) need ~153
    cells, so the everything-fits threshold sits between 150 and 160; 160
    gives a clean margin.
    """
    from tldw_chatbook.UI.Navigation.shell_destinations import SHELL_DESTINATION_ORDER

    class TestApp(ConsolidatedCSSApp):
        def compose(self):
            yield MainNavigationBar(active="chat")

    app = TestApp()
    async with app.run_test(size=(160, 40)) as pilot:
        await pilot.pause(0.5)
        nav = pilot.app.query_one(MainNavigationBar)
        strip = nav.query_one("#nav-destination-strip")
        hint = nav.query_one("#nav-overflow-hint", Button)

        # Everything fits, so the overflow affordance hides instead of
        # re-clipping the strip (the old 14-cell static hint is what cut
        # "Settings" down to "Set").
        assert not hint.display
        assert strip.virtual_size.width <= strip.region.width
        strip_right = strip.region.x + strip.region.width
        for destination in SHELL_DESTINATION_ORDER:
            button = nav.query_one(f"#nav-{destination.destination_id}")
            assert button.region.x >= strip.region.x
            assert button.region.x + button.region.width <= strip_right, (
                f"{destination.destination_id} clips at 160 cols: {button.region}"
            )


@pytest.mark.asyncio
async def test_nav_bar_overflow_menu_reaches_undigitized_destinations():
    """NV-01 (TASK-2154.21): clipped destinations live in the overflow menu."""
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen

    class TestApp(ConsolidatedCSSApp):
        def __init__(self):
            super().__init__()
            self.nav_requests: list[str] = []

        def compose(self):
            yield MainNavigationBar(active="chat")

        def on_navigate_to_screen(self, message: NavigateToScreen) -> None:
            self.nav_requests.append(message.screen_name)

    app = TestApp()
    async with app.run_test(size=(110, 32)) as pilot:
        await pilot.pause(0.5)
        nav = pilot.app.query_one(MainNavigationBar)
        hint = nav.query_one("#nav-overflow-hint", Button)
        assert hint.display, "overflow affordance must show when the strip clips"

        hint.press()
        await pilot.pause(0.5)
        menu = pilot.app.screen_stack[-1]
        assert menu.__class__.__name__ == "NavOverflowMenu"

        # The undigitized destinations are listed with their F-key labels
        # (Lab/Logs/Settings), hotkey prefixes survive on the first ten, and
        # the active one is marked.
        assert str(menu.query_one("#nav-overflow-lab", Button).label) == "F7 Lab"
        assert str(menu.query_one("#nav-overflow-logs", Button).label) == "F8 Logs"
        assert str(menu.query_one("#nav-overflow-settings", Button).label) == "F9 Settings"
        assert str(menu.query_one("#nav-overflow-home", Button).label).startswith("⌃1 Home")
        assert "(current)" in str(
            menu.query_one("#nav-overflow-console", Button).label
        )

        menu.query_one("#nav-overflow-logs", Button).press()
        await pilot.pause(0.5)

        assert "logs" in app.nav_requests
        assert (
            pilot.app.screen_stack[-1].__class__.__name__ != "NavOverflowMenu"
        )
