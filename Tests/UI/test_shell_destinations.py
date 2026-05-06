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
        "W+C",
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
        "chatbooks": ("artifacts", "chatbooks"),
        "ccp": ("personas", "ccp"),
        "conversation": ("library", "conversation"),
        "conversations_characters_prompts": ("personas", "ccp"),
        "subscriptions": ("watchlists_collections", "subscriptions"),
        "tools_settings": ("mcp", "tools_settings"),
        "settings": ("settings", "settings"),
    }

    for route, expected in expectations.items():
        resolved = resolve_shell_route(route)
        assert (resolved.destination_id, resolved.canonical_route) == expected


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
