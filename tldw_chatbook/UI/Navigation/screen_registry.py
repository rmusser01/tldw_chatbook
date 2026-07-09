"""Lazy screen route registry for app shell navigation."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module

from loguru import logger

from tldw_chatbook.Constants import TAB_CCP, TAB_LLM, TAB_MCP, TAB_SUBSCRIPTIONS


@dataclass(frozen=True)
class ScreenRoute:
    """Screen target metadata that defers importing the screen class."""

    screen_name: str
    canonical_tab: str
    module_path: str
    class_name: str
    dependency_check: str | None = None

    def dependencies_available(self) -> bool:
        """Return whether optional dependencies for this route are available."""

        if self.dependency_check is None:
            return True
        try:
            optional_deps = import_module("tldw_chatbook.Utils.optional_deps")
            check = getattr(optional_deps, self.dependency_check)
        except (ImportError, AttributeError) as exc:
            logger.warning(
                f"Optional dependency guard unavailable for route {self.screen_name}: {exc}"
            )
            return False
        return bool(check())

    def load_screen_class(self) -> type | None:
        """Load the screen class, returning None when an optional screen is unavailable."""

        if not self.dependencies_available():
            logger.warning(f"Screen route unavailable due to missing dependencies: {self.screen_name}")
            return None
        try:
            module = import_module(self.module_path)
            return getattr(module, self.class_name)
        except (ImportError, AttributeError) as exc:
            logger.warning(f"Screen route unavailable: {self.screen_name}: {exc}")
            return None


_SCREEN_ROUTES: dict[str, ScreenRoute] = {
    "home": ScreenRoute("home", "home", "tldw_chatbook.UI.Screens.home_screen", "HomeScreen"),
    "chat": ScreenRoute("chat", "chat", "tldw_chatbook.UI.Screens.chat_screen", "ChatScreen"),
    "library": ScreenRoute("library", "library", "tldw_chatbook.UI.Screens.library_screen", "LibraryScreen"),
    "artifacts": ScreenRoute("artifacts", "artifacts", "tldw_chatbook.UI.Screens.artifacts_screen", "ArtifactsScreen"),
    "personas": ScreenRoute("personas", "personas", "tldw_chatbook.UI.Screens.personas_screen", "PersonasScreen"),
    "watchlists_collections": ScreenRoute(
        "watchlists_collections",
        "watchlists_collections",
        "tldw_chatbook.UI.Screens.watchlists_collections_screen",
        "WatchlistsCollectionsScreen",
    ),
    "schedules": ScreenRoute("schedules", "schedules", "tldw_chatbook.UI.Screens.schedules_screen", "SchedulesScreen"),
    "workflows": ScreenRoute("workflows", "workflows", "tldw_chatbook.UI.Screens.workflows_screen", "WorkflowsScreen"),
    "mcp": ScreenRoute("mcp", TAB_MCP, "tldw_chatbook.UI.Screens.mcp_screen", "MCPScreen"),
    "acp": ScreenRoute("acp", "acp", "tldw_chatbook.UI.Screens.acp_screen", "ACPScreen"),
    "skills": ScreenRoute("skills", "skills", "tldw_chatbook.UI.Screens.skills_screen", "SkillsScreen"),
    "settings": ScreenRoute("settings", "settings", "tldw_chatbook.UI.Screens.settings_screen", "SettingsScreen"),
    "ingest": ScreenRoute("ingest", "ingest", "tldw_chatbook.UI.Screens.media_ingest_screen", "MediaIngestScreen"),
    "coding": ScreenRoute("coding", "coding", "tldw_chatbook.UI.Screens.coding_screen", "CodingScreen"),
    "conversation": ScreenRoute(
        "conversation",
        "conversation",
        "tldw_chatbook.UI.Screens.library_conversations_screen",
        "LibraryConversationsScreen",
    ),
    "ccp": ScreenRoute("ccp", "personas", "tldw_chatbook.UI.Screens.personas_screen", "PersonasScreen"),
    "media": ScreenRoute("media", "media", "tldw_chatbook.UI.Screens.media_screen", "MediaScreen"),
    "search": ScreenRoute("search", "search", "tldw_chatbook.UI.Screens.search_screen", "SearchScreen"),
    "evals": ScreenRoute("evals", "evals", "tldw_chatbook.UI.Screens.evals_screen", "EvalsScreen"),
    "tools_settings": ScreenRoute("tools_settings", TAB_MCP, "tldw_chatbook.UI.Screens.mcp_screen", "MCPScreen"),
    "llm": ScreenRoute("llm", TAB_LLM, "tldw_chatbook.UI.Screens.llm_screen", "LLMScreen"),
    "customize": ScreenRoute("customize", "customize", "tldw_chatbook.UI.Screens.customize_screen", "CustomizeScreen"),
    "logs": ScreenRoute("logs", "logs", "tldw_chatbook.UI.Screens.logs_screen", "LogsScreen"),
    "stats": ScreenRoute("stats", "stats", "tldw_chatbook.UI.Screens.stats_screen", "StatsScreen"),
    "stts": ScreenRoute("stts", "stts", "tldw_chatbook.UI.Screens.stts_screen", "STTSScreen"),
    "study": ScreenRoute("study", "study", "tldw_chatbook.UI.Screens.study_screen", "StudyScreen"),
    "writing": ScreenRoute("writing", "writing", "tldw_chatbook.UI.Screens.writing_screen", "WritingScreen"),
    "research": ScreenRoute("research", "research", "tldw_chatbook.UI.Screens.research_screen", "ResearchScreen"),
    "chatbooks": ScreenRoute("chatbooks", "chatbooks", "tldw_chatbook.UI.Screens.chatbooks_screen", "ChatbooksScreen"),
    "subscriptions": ScreenRoute(
        "subscriptions",
        TAB_SUBSCRIPTIONS,
        "tldw_chatbook.UI.Screens.subscription_screen",
        "SubscriptionScreen",
        dependency_check="check_subscriptions_deps",
    ),
}

_SCREEN_ALIASES = {
    TAB_CCP: "ccp",
    TAB_LLM: "llm",
    "subscription": "subscriptions",
    # The standalone Notes tab is retired: Notes now lives entirely inside
    # Library. Existing startup configs / callers using the legacy "notes"
    # route id still resolve to a real screen (Library) instead of erroring
    # or silently falling back to Chat.
    "notes": "library",
}


def registered_screen_route_ids() -> tuple[str, ...]:
    """Return all registered screen route ids without loading screen classes.

    Returns:
        Sorted route IDs backed by lazy screen metadata.
    """

    return tuple(sorted(_SCREEN_ROUTES))


def registered_screen_aliases() -> tuple[str, ...]:
    """Return screen route aliases without loading screen classes.

    Returns:
        Sorted alias route IDs that resolve to canonical screen routes.
    """

    return tuple(sorted(set(_SCREEN_ALIASES)))


def resolve_screen_target(target: str) -> tuple[str, str, type | None]:
    """Resolve a navigation target to a screen route without importing unrelated screens."""

    route_id = _SCREEN_ALIASES.get(target, target)
    route = _SCREEN_ROUTES.get(route_id)
    if route is None:
        return route_id, route_id, None
    return route.screen_name, route.canonical_tab, route.load_screen_class()
