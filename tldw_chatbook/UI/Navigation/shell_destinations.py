"""Master shell destination metadata and route compatibility helpers."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class ShellDestination:
    destination_id: str
    label: str
    primary_route: str
    purpose: str
    tooltip: str
    legacy_routes: tuple[str, ...] = ()
    related_routes: tuple[str, ...] = ()
    palette_aliases: tuple[str, ...] = ()
    full_label: str | None = None
    navigation_priority: int = 50

    @property
    def accessible_label(self) -> str:
        return self.full_label or self.label


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
        navigation_priority=10,
    ),
    ShellDestination(
        "console",
        "Console",
        "chat",
        "Live agent conversations, approvals, tools, RAG, and runs.",
        "Open the live agent Console.",
        # "coding" is retired as a standalone screen; old links land on Console.
        ("chat", "coding"),
        navigation_priority=20,
    ),
    ShellDestination(
        "library",
        "Library",
        "library",
        "Workspaces, source material, imports, notes, media, conversations, Study, flashcards, quizzes, and Search/RAG.",
        "Browse Workspaces, imports, notes, media, Study, flashcards, quizzes, search, and source material.",
        (
            "notes",
            "media",
            "ingest",
            "search",
            "conversation",
            "study",
            "chunking_lab",
            "prompts",
            "skills",
            "writing",
        ),
        navigation_priority=30,
    ),
    ShellDestination(
        "research",
        "Research",
        "research_workspace",
        "Grounded workspaces and durable research-run observation.",
        "Open Research Workspace for grounded research and research runs.",
        related_routes=("research",),
        palette_aliases=(
            "research workspace",
            "research runs",
            "research sessions",
            "deep research",
            "notebook",
        ),
        navigation_priority=35,
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
        # A roleplay-first newcomer finds characters from this label, so it has
        # to be readable cold. "RP&CD" could only be decoded after navigating
        # here and reading the screen title -- i.e. after already guessing right.
        # F-034: "Roleplay" is the one public name everywhere (nav, header,
        # palette); the long "Roleplay & Chat Dictionaries" form is retired.
        "Roleplay",
        "personas",
        "Characters, user profiles, dictionaries, and behavior profiles.",
        "Manage behavior profiles and user profile context.",
        ("ccp", "conversations_characters_prompts", "characters", "roleplay"),
        full_label="Roleplay",
    ),
    ShellDestination(
        "watchlists_collections",
        "Watchlists",
        "watchlists_collections",
        "Monitored sources, runs, alerts, and recovery.",
        "Open Watchlists for monitored sources, runs, alerts, and recovery.",
        ("subscriptions", "subscription"),
        full_label="Watchlists",
        navigation_priority=40,
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
        "meetings",
        "Meetings",
        "meetings",
        "Record a call or a room with a live labelled transcript, then file it in the Library.",
        "Record and transcribe a meeting.",
        palette_aliases=("meeting", "record", "transcribe"),
        navigation_priority=75,
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
        "lab",
        "Lab",
        "llm",
        "Models, speech, and evaluation runs.",
        "Manage models, speech, and evaluation runs.",
        ("llm_management", "stts", "evals"),
        navigation_priority=45,
    ),
    ShellDestination(
        "logs",
        "Logs",
        "logs",
        "Application logs and diagnostics.",
        "View application logs and diagnostics.",
    ),
    ShellDestination(
        "settings",
        "Settings",
        "settings",
        "Global app preferences, appearance, accounts, and storage.",
        "Configure application preferences.",
        ("stats",),
    ),
)

_BY_DESTINATION_ID: Mapping[str, ShellDestination] = {
    destination.destination_id: destination for destination in SHELL_DESTINATION_ORDER
}

# Shortcut ownership is a destination contract, not a position in the
# navigation strip. New destinations therefore cannot silently reassign an
# established shortcut by changing ``SHELL_DESTINATION_ORDER``.
SHELL_DESTINATION_SHORTCUTS: Mapping[str, str] = MappingProxyType(
    {
        "home": "ctrl+1",
        "console": "ctrl+2",
        "library": "ctrl+3",
        "artifacts": "ctrl+4",
        "personas": "ctrl+5",
        "watchlists_collections": "ctrl+6",
        "schedules": "ctrl+7",
        "workflows": "ctrl+8",
        "mcp": "ctrl+9",
        "acp": "ctrl+0",
        "lab": "f7",
        "logs": "f8",
        "settings": "f9",
        "research": "f10",
        "meetings": "f11",
    }
)

_ROUTABLE_LEGACY_ROUTES = {
    "chunking_lab",
    "chat",
    "notes",
    "media",
    "ingest",
    "search",
    "conversation",
    "study",
    "writing",
    "chatbooks",
    "subscriptions",
    "tools_settings",
    "stts",
    "evals",
    "stats",
    # Personas "prompts" mode chip retirement (Task 7): keep the legacy
    # route id as its own canonical route under Library, mirroring "notes".
    "prompts",
    # Standalone Skills tab retirement (Skills sub-project Task 5): keep
    # the legacy route id as its own canonical route under Library,
    # mirroring "notes"/"prompts" above.
    "skills",
}

_CANONICAL_ROUTE_OVERRIDES = {
    "subscription": "subscriptions",
    "llm_management": "llm",
}

_ROUTE_MAP: dict[str, ResolvedShellRoute] = {}

for destination in SHELL_DESTINATION_ORDER:
    _ROUTE_MAP[destination.primary_route] = ResolvedShellRoute(
        destination.destination_id,
        destination.primary_route,
        destination.primary_route,
    )
    if destination.destination_id not in destination.related_routes:
        _ROUTE_MAP[destination.destination_id] = ResolvedShellRoute(
            destination.destination_id,
            destination.primary_route,
            destination.destination_id,
        )
    for related_route in destination.related_routes:
        _ROUTE_MAP[related_route] = ResolvedShellRoute(
            destination.destination_id,
            related_route,
            related_route,
        )
    for legacy_route in destination.legacy_routes:
        canonical_route = _CANONICAL_ROUTE_OVERRIDES.get(
            legacy_route,
            legacy_route
            if legacy_route in _ROUTABLE_LEGACY_ROUTES
            else destination.primary_route,
        )
        _ROUTE_MAP[legacy_route] = ResolvedShellRoute(
            destination.destination_id,
            canonical_route,
            legacy_route,
        )


def get_shell_destination(destination_id: str) -> ShellDestination:
    return _BY_DESTINATION_ID[destination_id]


def registered_shell_route_ids() -> tuple[str, ...]:
    """Return all shell route ids and aliases known to the destination model.

    Returns:
        Sorted route IDs and compatibility aliases that resolve in the shell.
    """

    return tuple(sorted(_ROUTE_MAP))


def resolve_shell_route(route: str) -> ResolvedShellRoute:
    return _ROUTE_MAP.get(route, ResolvedShellRoute(route, route, route))
