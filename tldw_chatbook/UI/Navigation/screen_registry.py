"""Lazy screen route registry for app shell navigation."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module

from loguru import logger

from tldw_chatbook.Constants import TAB_CCP, TAB_LLM, TAB_MCP, TAB_MEETINGS, TAB_RESEARCH_WORKSPACE
from .shell_destinations import resolve_shell_route


@dataclass(frozen=True)
class ScreenRoute:
    """Screen target metadata that defers importing the screen class."""

    screen_name: str
    canonical_tab: str
    module_path: str
    class_name: str
    dependency_check: str | None = None
    #: TASK-24452: opt-in screen-instance reuse. A reusable route's screen is
    #: constructed once, installed (``App.install_screen``), and re-switched
    #: to on every later visit -- Textual SUSPENDS an installed screen
    #: instead of unmounting it, so the widget tree survives and warm visits
    #: skip construction/mount entirely (measured: Home-class screens drop
    #: from hundreds of ms of CPU per visit to tens). Opt-in per route
    #: because reuse changes lifecycle semantics: ``on_mount``/``on_unmount``
    #: fire once per app run instead of once per visit, so a route may only
    #: set this after auditing that (a) per-visit refresh work runs from
    #: ``on_screen_resume``, and (b) nothing load-bearing lives in
    #: ``on_unmount`` teardown (see ``_create_navigation_screen``'s history
    #: of why UNINSTALLED instances must never be reused).
    reusable: bool = False

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
            logger.warning(
                f"Screen route unavailable due to missing dependencies: {self.screen_name}"
            )
            return None
        try:
            module = import_module(self.module_path)
            return getattr(module, self.class_name)
        except (ImportError, AttributeError) as exc:
            logger.warning(f"Screen route unavailable: {self.screen_name}: {exc}")
            return None


_SCREEN_ROUTES: dict[str, ScreenRoute] = {
    "home": ScreenRoute(
        "home",
        "home",
        "tldw_chatbook.UI.Screens.home_screen",
        "HomeScreen",
        # TASK-24452 first enablement: Home has no ``on_unmount`` teardown,
        # no timers, and its per-visit refresh workers are re-triggered from
        # ``on_screen_resume`` (all ``exclusive=True`` groups, so the
        # first-visit mount+resume double-fire coalesces).
        reusable=True,
    ),
    "chat": ScreenRoute(
        "chat", "chat", "tldw_chatbook.UI.Screens.chat_screen", "ChatScreen"
    ),
    "library": ScreenRoute(
        "library", "library", "tldw_chatbook.UI.Screens.library_screen", "LibraryScreen"
    ),
    "artifacts": ScreenRoute(
        "artifacts",
        "artifacts",
        "tldw_chatbook.UI.Screens.artifacts_screen",
        "ArtifactsScreen",
    ),
    "personas": ScreenRoute(
        "personas",
        "personas",
        "tldw_chatbook.UI.Screens.personas_screen",
        "PersonasScreen",
    ),
    "watchlists_collections": ScreenRoute(
        "watchlists_collections",
        "watchlists_collections",
        "tldw_chatbook.UI.Screens.watchlists_collections_screen",
        "WatchlistsCollectionsScreen",
    ),
    "schedules": ScreenRoute(
        "schedules",
        "schedules",
        "tldw_chatbook.UI.Screens.scheduling.schedules_workbench",
        "SchedulesWorkbench",
    ),
    "workflows": ScreenRoute(
        "workflows",
        "workflows",
        "tldw_chatbook.UI.Screens.workflows_screen",
        "WorkflowsScreen",
    ),
    TAB_MEETINGS: ScreenRoute(
        "meetings", TAB_MEETINGS, "tldw_chatbook.UI.Screens.meetings_screen", "MeetingsScreen"
    ),
    "mcp": ScreenRoute(
        "mcp", TAB_MCP, "tldw_chatbook.UI.Screens.mcp_screen", "MCPScreen"
    ),
    "acp": ScreenRoute(
        "acp", "acp", "tldw_chatbook.UI.Screens.acp_screen", "ACPScreen"
    ),
    "settings": ScreenRoute(
        "settings",
        "settings",
        "tldw_chatbook.UI.Screens.settings_screen",
        "SettingsScreen",
    ),
    "conversation": ScreenRoute(
        "conversation",
        "conversation",
        "tldw_chatbook.UI.Screens.library_conversations_screen",
        "LibraryConversationsScreen",
    ),
    "ccp": ScreenRoute(
        "ccp", "personas", "tldw_chatbook.UI.Screens.personas_screen", "PersonasScreen"
    ),
    "media": ScreenRoute(
        "media", "media", "tldw_chatbook.UI.Screens.media_screen", "MediaScreen"
    ),
    "evals": ScreenRoute(
        "evals", "evals", "tldw_chatbook.UI.Screens.evals_screen", "EvalsScreen"
    ),
    "tools_settings": ScreenRoute(
        "tools_settings", TAB_MCP, "tldw_chatbook.UI.Screens.mcp_screen", "MCPScreen"
    ),
    "llm": ScreenRoute(
        "llm", TAB_LLM, "tldw_chatbook.UI.Screens.llm_screen", "LLMScreen"
    ),
    "customize": ScreenRoute(
        "customize",
        "customize",
        "tldw_chatbook.UI.Screens.customize_screen",
        "CustomizeScreen",
    ),
    "logs": ScreenRoute(
        "logs", "logs", "tldw_chatbook.UI.Screens.logs_screen", "LogsScreen"
    ),
    "stats": ScreenRoute(
        "stats", "stats", "tldw_chatbook.UI.Screens.stats_screen", "StatsScreen"
    ),
    "stts": ScreenRoute(
        "stts", "stts", "tldw_chatbook.UI.Screens.stts_screen", "STTSScreen"
    ),
    "study": ScreenRoute(
        "study", "study", "tldw_chatbook.UI.Screens.study_screen", "StudyScreen"
    ),
    "writing": ScreenRoute(
        "writing", "writing", "tldw_chatbook.UI.Screens.writing_screen", "WritingScreen"
    ),
    # task-16322 (ADR-068) re-registers the research screen: the local
    # research execution engine now drives launched local runs, so
    # ResearchWindow (the run/event observation surface) is reachable from
    # navigation again under the legacy "research" route id (still a
    # command-palette direct command via TAB_RESEARCH and valid in saved
    # startup configs). This reverses task-255's temporary library alias;
    # the Workbench migration owner stays "library"
    # (UI/Workbench/route_inventory.py).
    "research": ScreenRoute(
        "research", "research", "tldw_chatbook.UI.Screens.research_screen", "ResearchScreen"
    ),
    # Task 1 records the lazy route contract. The screen module is created by
    # the dedicated screen task, so this metadata is intentionally not
    # importable yet.
    "research_workspace": ScreenRoute(
        "research_workspace",
        TAB_RESEARCH_WORKSPACE,
        "tldw_chatbook.UI.Screens.research_workspace_screen",
        "ResearchWorkspaceScreen",
    ),
    "chatbooks": ScreenRoute(
        "chatbooks",
        "chatbooks",
        "tldw_chatbook.UI.Screens.chatbooks_screen",
        "ChatbooksScreen",
    ),
}

_SCREEN_ALIASES = {
    TAB_CCP: "ccp",
    TAB_LLM: "llm",
    "subscriptions": "watchlists_collections",
    "subscription": "watchlists_collections",
    # The standalone Notes tab is retired: Notes now lives entirely inside
    # Library. Existing startup configs / callers using the legacy "notes"
    # route id still resolve to a real screen (Library) instead of erroring
    # or silently falling back to Chat.
    "notes": "library",
    # The Personas "prompts" mode chip is retired (Task 7): prompt
    # management now lives entirely inside Library. Existing startup
    # configs / callers using the legacy "prompts" route id resolve to
    # Library instead of Personas.
    "prompts": "library",
    # The standalone Skills tab is retired (Skills sub-project Task 5):
    # skill management now lives entirely inside Library (its own Skills
    # rail row, built in Tasks 1-4). Existing startup configs / callers
    # using the legacy "skills" route id resolve to Library instead of the
    # standalone SkillsScreen -- mirrors the "notes"/"prompts" aliases
    # above exactly. ``skills_screen.py``/``SkillsScreen`` are NOT deleted:
    # the class is still directly exercised by its own destination-shell
    # test suite, and its trust passphrase modal is reused by the Library
    # skill editor's trust panel (Task 4).
    "skills": "library",
    # The standalone Ingest screen is retired (task-684.4): importing now
    # lives entirely inside Library's Import media canvas, which gained the
    # server-backed and web-clipping paths that screen used to own
    # (tasks 684.1-684.3). Existing startup configs / callers using the
    # legacy "ingest" route id resolve to Library instead of erroring --
    # mirrors the "notes"/"prompts"/"skills" aliases above, and matches the
    # route inventory, which already declared ingest -> library.
    "ingest": "library",
    # "research" is a REAL screen route again (task-16322, ADR-068) -- see
    # its ScreenRoute registration above. It is deliberately NOT an alias.
    # The standalone Search screen is retired (RAG UX v2 PR-1, critique
    # 2026-08-02T21-11-50Z): search/RAG now lives entirely inside Library's
    # Search / RAG canvas (rail row "browse-search"), with Console staging
    # via the RAG modal. Existing startup configs / callers using the
    # legacy "search" route id resolve to Library instead of dead-ending --
    # mirrors the "notes"/"prompts"/"skills"/"ingest" aliases above. The
    # route inventory already declared search -> library
    # (UI/Workbench/route_inventory.py).
    "search": "library",
    # The standalone Media Library screen is retired (task-2851, Library UAT
    # 2026-08-06): Library already reimplements full media browsing/
    # management as its own canvas (rail row "media"). The legacy "media"
    # route id used to resolve to a completely different real screen
    # (``media_screen.MediaScreen``, nav: Media Types / All Media /
    # Analysis Review / Collections-Tags / Multi-Item Review) while the
    # shell destination model already folded "media" under the "library"
    # destination for nav-bar highlighting purposes -- so the command
    # palette's "Open Media Library" entry hijacked the Library tab: the
    # nav bar showed Library active while the legacy screen's dead-end-
    # duplicate content rendered underneath. Existing startup configs /
    # callers using the legacy "media" route id now resolve to Library
    # instead, mirroring the "notes"/"prompts"/"skills"/"search" aliases
    # above. ``MediaScreen`` itself is not deleted -- its save_state/
    # restore_state contracts stay directly exercised by their own unit
    # tests, mirroring the "skills" precedent.
    "media": "library",
    # The standalone Coding screen is retired (merged into Console). Legacy
    # "coding" route ids still resolve to a real screen (Console) instead of
    # erroring; the shell destination model owns the same fold.
    "coding": "chat",
    # The standalone Customize screen is retired: theme/splash management now
    # lives inside Settings. Legacy "customize" route ids resolve to Settings.
    "customize": "settings",
}


def registered_screen_route_ids() -> tuple[str, ...]:
    """Return all registered screen route ids without loading screen classes.

    Returns:
        Sorted route IDs backed by lazy screen metadata.
    """

    return tuple(sorted(_SCREEN_ROUTES))


def registered_screen_routes() -> tuple[ScreenRoute, ...]:
    """Return every registered canonical ``ScreenRoute`` without importing.

    Unlike ``registered_screen_route_ids()`` (ids only), this exposes the
    route metadata itself -- notably ``module_path`` -- so a caller can
    dedupe routes that share one module (e.g. ``"ccp"``/``"personas"`` both
    target ``personas_screen.PersonasScreen``, ``"tools_settings"``/``"mcp"``
    both target ``mcp_screen.MCPScreen``) or call ``load_screen_class()``
    directly. Used by the app's background screen-module pre-importer
    (task-15472) to warm ``sys.modules`` after first paint.

    Returns:
        Route objects sorted by ``screen_name`` for a stable, deterministic
        iteration order.
    """

    return tuple(_SCREEN_ROUTES[route_id] for route_id in sorted(_SCREEN_ROUTES))


def registered_screen_aliases() -> tuple[str, ...]:
    """Return screen route aliases without loading screen classes.

    Returns:
        Sorted alias route IDs that resolve to canonical screen routes.
    """

    return tuple(sorted(set(_SCREEN_ALIASES)))


def resolve_screen_target(target: str) -> tuple[str, str, type | None]:
    """Resolve a navigation target to a screen route without importing unrelated screens.

    Resolution order: explicit screen aliases, then direct screen routes,
    then the shell destination model. The last leg covers destination ids
    that are not themselves screen routes (``"lab"`` -> ``"llm"``,
    ``"console"`` -> ``"chat"``) and legacy route ids the destination model
    folds onto a primary route (e.g. ``"characters"`` -> ``"personas"``).
    Unknown targets keep the ``(target, target, None)`` miss shape so the
    navigation handler logs and stays on the current screen.

    Args:
        target: The requested route id or alias.

    Returns:
        A tuple of ``(screen_name, canonical_tab, screen_class)``.
        ``screen_class`` is ``None`` when the target cannot be resolved.
    """

    route_id, route = _lookup_route(target)
    if route is None:
        return route_id, route_id, None
    return route.screen_name, route.canonical_tab, route.load_screen_class()


def resolve_screen_route(target: str) -> ScreenRoute | None:
    """Resolve a navigation target to its ``ScreenRoute`` WITHOUT importing it.

    ``resolve_screen_target()`` answers the same question but calls
    ``load_screen_class()`` as part of answering it, which is precisely the
    synchronous import a caller wanting the *metadata* is trying to avoid.
    This exposes the already-existing lazy lookup (aliases, then direct
    routes, then the shell destination model -- identical resolution, same
    helper) so a caller can decide what to import before importing it.

    task-21110: ``app.py`` uses this to learn which module the initial screen
    will need while the splash is still on screen, then warms it on the
    background thread the screen pre-importer already owns.

    Args:
        target: The requested route id or alias.

    Returns:
        The resolved ``ScreenRoute``, or ``None`` when the target is not
        routable (the same miss that ``resolve_screen_target()`` reports as a
        ``None`` screen class).
    """

    return _lookup_route(target)[1]


def _lookup_route(target: str) -> tuple[str, ScreenRoute | None]:
    """Resolve a navigation target to its route, without importing the class.

    Args:
        target: The requested route id or alias.

    Returns:
        A tuple of ``(route_id, route)``. ``route`` is ``None`` when the
        target is not routable, in which case ``route_id`` is the furthest
        the alias/shell-destination resolution got (used for the miss shape
        and for error messages).
    """

    route_id = _SCREEN_ALIASES.get(target, target)
    route = _SCREEN_ROUTES.get(route_id)
    if route is None:
        canonical_route = resolve_shell_route(route_id).canonical_route
        route_id = _SCREEN_ALIASES.get(canonical_route, canonical_route)
        route = _SCREEN_ROUTES.get(route_id)
    return route_id, route


def screen_load_error(target: str) -> BaseException | None:
    """Return the exception that prevents ``target``'s screen class loading.

    ``resolve_screen_target()`` deliberately degrades a failed route to
    ``None`` so one broken optional screen cannot break navigation as a
    whole -- but that swallows the reason. Callers for whom the failure is
    fatal (notably ``app.py``'s ``_push_initial_screen()``) use this to
    report *why* rather than emitting a bare "unable to resolve" message.

    This re-attempts the import rather than caching the original error, so
    it is only for the diagnostic/failure path, never the hot path.

    Root-caused 2026-07-27: optional ``aiohttp`` on the default chat
    screen's import chain surfaced only as ``RuntimeError: Unable to
    resolve default chat screen``, naming neither the missing module nor
    the file that imported it.

    Args:
        target: The requested route id or alias.

    Returns:
        The blocking exception, or ``None`` if the class loads cleanly.
        Unroutable targets and unavailable dependency gates -- neither of
        which raises on its own -- are reported as a synthesized
        ``LookupError``/``ImportError`` so the caller always has a reason.
    """

    route_id, route = _lookup_route(target)
    if route is None:
        return LookupError(
            f"no screen route is registered for target {target!r}"
            f" (resolution reached {route_id!r})"
        )
    if not route.dependencies_available():
        return ImportError(
            f"screen route {route.screen_name!r} is gated on optional dependency"
            f" check {route.dependency_check!r}, which reports unavailable"
        )
    try:
        module = import_module(route.module_path)
        getattr(module, route.class_name)
    except (ImportError, AttributeError) as exc:
        return exc
    return None
