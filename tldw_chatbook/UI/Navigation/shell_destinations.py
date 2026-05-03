"""Master shell destination metadata and route compatibility helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class ShellDestination:
    destination_id: str
    label: str
    primary_route: str
    purpose: str
    tooltip: str
    legacy_routes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedShellRoute:
    destination_id: str
    canonical_route: str
    requested_route: str


SHELL_DESTINATION_ORDER: tuple[ShellDestination, ...] = (
    ShellDestination(
        "home",
        "Home",
        "home",
        "Dashboard, notifications, status, and next actions.",
        "Open dashboard, notifications, and active work.",
    ),
    ShellDestination(
        "console",
        "Console",
        "chat",
        "Live agent conversations, approvals, tools, RAG, and runs.",
        "Open the live agent Console.",
        ("chat",),
    ),
    ShellDestination(
        "library",
        "Library",
        "library",
        "Workspaces, source material, imports, notes, media, conversations, and Search/RAG.",
        "Browse Workspaces, imports, notes, media, search, and source material.",
        ("notes", "media", "ingest", "search", "conversation"),
    ),
    ShellDestination(
        "artifacts",
        "Artifacts",
        "artifacts",
        "Generated outputs, bundles, reports, datasets, and Chatbooks.",
        "Browse generated and portable outputs.",
        ("chatbooks",),
    ),
    ShellDestination(
        "personas",
        "Personas",
        "personas",
        "Characters, personas, prompts, dictionaries, and behavior profiles.",
        "Manage behavior profiles and persona context.",
        ("ccp", "conversations_characters_prompts", "characters", "prompts"),
    ),
    ShellDestination(
        "watchlists_collections",
        "W+C",
        "watchlists_collections",
        "Monitored sources and curated reading/content collections.",
        "Monitor feeds and curate collections.",
        ("subscriptions", "subscription"),
    ),
    ShellDestination(
        "schedules",
        "Schedules",
        "schedules",
        "When jobs, watchlists, and workflows run.",
        "Manage run timing, triggers, and recovery.",
    ),
    ShellDestination(
        "workflows",
        "Workflows",
        "workflows",
        "Reusable procedures, recipes, dry-runs, and outputs.",
        "Build and launch repeatable agent workflows.",
    ),
    ShellDestination(
        "mcp",
        "MCP",
        "mcp",
        "MCP servers, tools, permissions, auth, and audit.",
        "Configure tool and server capability plumbing.",
        ("tools_settings",),
    ),
    ShellDestination(
        "acp",
        "ACP",
        "acp",
        "Agent Client Protocol agents, sessions, runtimes, diffs, and terminals.",
        "Manage ACP agents and sessions.",
    ),
    ShellDestination(
        "skills",
        "Skills",
        "skills",
        "Agent Skills packs, discovery, validation, and attachments.",
        "Browse, import, validate, and attach skills.",
    ),
    ShellDestination(
        "settings",
        "Settings",
        "settings",
        "Global app preferences, appearance, accounts, and storage.",
        "Configure application preferences.",
        ("customize",),
    ),
)

_BY_DESTINATION_ID: Mapping[str, ShellDestination] = {
    destination.destination_id: destination for destination in SHELL_DESTINATION_ORDER
}

_ROUTABLE_LEGACY_ROUTES = {
    "chat",
    "notes",
    "media",
    "ingest",
    "search",
    "conversation",
    "chatbooks",
    "ccp",
    "subscriptions",
    "tools_settings",
    "customize",
}

_CANONICAL_ROUTE_OVERRIDES = {
    "conversations_characters_prompts": "ccp",
    "characters": "ccp",
    "prompts": "ccp",
    "subscription": "subscriptions",
}

_ROUTE_MAP: dict[str, ResolvedShellRoute] = {}

for destination in SHELL_DESTINATION_ORDER:
    _ROUTE_MAP[destination.primary_route] = ResolvedShellRoute(
        destination.destination_id,
        destination.primary_route,
        destination.primary_route,
    )
    _ROUTE_MAP[destination.destination_id] = ResolvedShellRoute(
        destination.destination_id,
        destination.primary_route,
        destination.destination_id,
    )
    for legacy_route in destination.legacy_routes:
        canonical_route = _CANONICAL_ROUTE_OVERRIDES.get(
            legacy_route,
            legacy_route if legacy_route in _ROUTABLE_LEGACY_ROUTES else destination.primary_route,
        )
        _ROUTE_MAP[legacy_route] = ResolvedShellRoute(
            destination.destination_id,
            canonical_route,
            legacy_route,
        )


def get_shell_destination(destination_id: str) -> ShellDestination:
    return _BY_DESTINATION_ID[destination_id]


def resolve_shell_route(route: str) -> ResolvedShellRoute:
    return _ROUTE_MAP.get(route, ResolvedShellRoute(route, route, route))
