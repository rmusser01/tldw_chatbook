"""Route coverage helpers for the Workbench UI migration."""

from __future__ import annotations

from dataclasses import dataclass

from tldw_chatbook.Constants import ALL_TABS
from tldw_chatbook.UI.Navigation.screen_registry import (
    registered_screen_aliases,
    registered_screen_route_ids,
)
from tldw_chatbook.UI.Navigation.shell_destinations import registered_shell_route_ids


WORKBENCH_ROUTE_OWNERS: dict[str, str] = {
    "home": "home",
    "chat": "console",
    "console": "console",
    "library": "library",
    "artifacts": "artifacts",
    "personas": "personas",
    "watchlists_collections": "watchlists_collections",
    "schedules": "schedules",
    "workflows": "workflows",
    "mcp": "mcp",
    "acp": "acp",
    "settings": "settings",
    "ingest": "library",
    "coding": "console",
    "conversation": "library",
    "ccp": "personas",
    "conversations_characters_prompts": "personas",
    "characters": "personas",
    # The Personas "prompts" mode chip is retired (Task 7): the legacy
    # "prompts" route re-points to Library, like "notes" below.
    "prompts": "library",
    # The standalone Skills tab is retired (Skills sub-project Task 5): the
    # legacy "skills" route re-points to Library, like "notes"/"prompts".
    "skills": "library",
    "media": "library",
    "notes": "library",
    "search": "library",
    "evals": "diagnostics_evals",
    "tools_settings": "mcp",
    "llm": "settings",
    "llm_management": "settings",
    "customize": "settings",
    "logs": "diagnostics_logs",
    "stats": "diagnostics_stats",
    "stts": "settings",
    "study": "library",
    "writing": "artifacts_writing",
    # The orphan "research" screen registration is removed (Task 255): the
    # route id survives only as TAB_RESEARCH plus a screen_registry alias
    # that resolves to Library, matching this owner mapping.
    "research": "library",
    "chatbooks": "artifacts",
    "subscriptions": "watchlists_collections",
    "subscription": "watchlists_collections",
}


@dataclass(frozen=True)
class WorkbenchRouteCoverage:
    """Workbench migration-owner coverage for registered route sources."""

    constant_tabs: tuple[str, ...]
    screen_routes: tuple[str, ...]
    screen_aliases: tuple[str, ...]
    shell_routes: tuple[str, ...]
    all_known_routes: tuple[str, ...]
    owner_for_route: dict[str, str]
    missing_owner_routes: tuple[str, ...]


def build_workbench_route_coverage() -> WorkbenchRouteCoverage:
    """Return route coverage for all registered navigation sources.

    Returns:
        Workbench migration-owner coverage across constants, screen routes,
        aliases, and shell routes.
    """

    constant_tabs = tuple(sorted(str(route) for route in ALL_TABS))
    screen_routes = registered_screen_route_ids()
    screen_aliases = registered_screen_aliases()
    shell_routes = registered_shell_route_ids()
    known_route_ids = (
        set(constant_tabs) | set(screen_routes) | set(screen_aliases) | set(shell_routes)
    )
    all_known_routes = tuple(sorted(known_route_ids))
    missing_owner_routes = tuple(
        route for route in all_known_routes if route not in WORKBENCH_ROUTE_OWNERS
    )

    return WorkbenchRouteCoverage(
        constant_tabs=constant_tabs,
        screen_routes=screen_routes,
        screen_aliases=screen_aliases,
        shell_routes=shell_routes,
        all_known_routes=all_known_routes,
        owner_for_route={
            route: WORKBENCH_ROUTE_OWNERS[route]
            for route in all_known_routes
            if route in WORKBENCH_ROUTE_OWNERS
        },
        missing_owner_routes=missing_owner_routes,
    )
