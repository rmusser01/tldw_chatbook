from tldw_chatbook.UI.Navigation.shell_destinations import (
    SHELL_DESTINATION_ORDER,
    resolve_shell_route,
)
from Tests.UI.test_screen_navigation import _build_test_app


def test_master_shell_destination_order_matches_spec():
    assert [destination.label for destination in SHELL_DESTINATION_ORDER] == [
        "Home",
        "Console",
        "Library",
        "Artifacts",
        "Personas",
        "Watchlists",
        "Schedules",
        "Workflows",
        "MCP",
        "ACP",
        "Skills",
        "Settings",
    ]


def test_legacy_routes_resolve_to_master_destinations():
    expectations = {
        "chat": ("console", "chat"),
        "home": ("home", "home"),
        "notes": ("library", "notes"),
        "media": ("library", "media"),
        "ingest": ("library", "ingest"),
        "search": ("library", "search"),
        "study": ("library", "study"),
        "prompts": ("library", "prompts"),
        "chatbooks": ("artifacts", "chatbooks"),
        "ccp": ("personas", "personas"),
        "conversation": ("library", "conversation"),
        "conversations_characters_prompts": ("personas", "personas"),
        "subscriptions": ("watchlists_collections", "subscriptions"),
        "tools_settings": ("mcp", "tools_settings"),
        "settings": ("settings", "settings"),
    }

    for route, expected in expectations.items():
        resolved = resolve_shell_route(route)
        assert (resolved.destination_id, resolved.canonical_route) == expected


def test_ccp_legacy_routes_resolve_to_personas_destination():
    for legacy in ("ccp", "characters", "conversations_characters_prompts"):
        resolved = resolve_shell_route(legacy)
        assert resolved.destination_id == "personas"
        assert resolved.canonical_route == "personas"


def test_prompts_legacy_route_resolves_to_library_destination():
    """The Personas "prompts" mode chip is retired (Task 7): the legacy
    "prompts" route now re-points to Library, mirroring "notes" -- unlike
    the other Personas legacy routes above, it keeps its own route id as
    the canonical route rather than collapsing to "personas".
    """
    resolved = resolve_shell_route("prompts")
    assert resolved.destination_id == "library"
    assert resolved.canonical_route == "prompts"


def test_ccp_screen_route_loads_personas_screen():
    from tldw_chatbook.UI.Navigation.screen_registry import resolve_screen_target

    screen_name, _tab, screen_class = resolve_screen_target("ccp")
    assert screen_class is not None
    assert screen_class.__name__ == "PersonasScreen"


def test_each_shell_destination_has_recovery_tooltip_copy():
    for destination in SHELL_DESTINATION_ORDER:
        assert destination.tooltip
        assert destination.purpose


def test_every_shell_destination_has_readable_purpose_and_mounted_route():
    app = _build_test_app()

    for destination in SHELL_DESTINATION_ORDER:
        assert destination.purpose
        assert destination.tooltip
        _screen_name, _tab_id, screen_class = app._resolve_screen_navigation_target(destination.primary_route)
        assert screen_class is not None, destination.primary_route
