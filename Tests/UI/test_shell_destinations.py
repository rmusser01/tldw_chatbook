from tldw_chatbook.UI.Navigation.shell_destinations import (
    SHELL_DESTINATION_ORDER,
    get_shell_destination,
    resolve_shell_route,
)
from Tests.UI.test_screen_navigation import _build_test_app


def test_master_shell_destination_order_matches_spec():
    assert [destination.label for destination in SHELL_DESTINATION_ORDER] == [
        "Home",
        "Console",
        "Library",
        "Artifacts",
        "RP&CD",
        "Watchlists",
        "Schedules",
        "Workflows",
        "MCP",
        "ACP",
        "Lab",
        "Logs",
        "Settings",
    ]


def test_personas_destination_renamed_to_rpcd_with_roleplay_alias():
    """The nav destination is renamed to "RP&CD" / "Roleplay & Chat Dictionaries".

    "Personas" stays reserved for the in-screen user-identity mode
    (``MODE_LABELS["personas"]``), which this test does not touch (task-435).
    """
    dest = get_shell_destination("personas")
    assert dest.label == "RP&CD"
    assert dest.full_label == "Roleplay & Chat Dictionaries"
    assert dest.accessible_label == "Roleplay & Chat Dictionaries"
    assert "roleplay" in dest.legacy_routes
    for route in (
        "personas",
        "ccp",
        "conversations_characters_prompts",
        "characters",
        "roleplay",
    ):
        assert resolve_shell_route(route).destination_id == "personas"


def test_tab_display_labels_use_rpcd_for_personas_and_ccp_tabs():
    """Both Personas-seating tab ids show the renamed destination label.

    Covers the top-level tab chrome for the RP&CD rename (task-435).
    """
    from tldw_chatbook.Constants import TAB_CCP, TAB_PERSONAS, get_tab_display_label

    assert get_tab_display_label(TAB_CCP) == "RP&CD"
    assert get_tab_display_label(TAB_PERSONAS) == "RP&CD"


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
        "skills": ("library", "skills"),
        "writing": ("library", "writing"),
        "research": ("library", "research"),
        "chatbooks": ("artifacts", "chatbooks"),
        "ccp": ("personas", "personas"),
        "conversation": ("library", "conversation"),
        "conversations_characters_prompts": ("personas", "personas"),
        "roleplay": ("personas", "personas"),
        "subscriptions": ("watchlists_collections", "subscriptions"),
        "tools_settings": ("mcp", "tools_settings"),
        "settings": ("settings", "settings"),
        # Logs is a top-level destination again; its route resolves to itself.
        "logs": ("logs", "logs"),
        "stats": ("settings", "stats"),
        "llm": ("lab", "llm"),
        "llm_management": ("lab", "llm"),
        "stts": ("lab", "stts"),
        "evals": ("lab", "evals"),
        # The retired Coding screen folds into Console: legacy "coding" links
        # land on Console's primary route.
        "coding": ("console", "chat"),
    }

    for route, expected in expectations.items():
        resolved = resolve_shell_route(route)
        assert (resolved.destination_id, resolved.canonical_route) == expected


def test_ccp_legacy_routes_resolve_to_personas_destination():
    for legacy in ("ccp", "characters", "conversations_characters_prompts", "roleplay"):
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


def test_lab_destination_id_resolves_to_models_screen():
    """NavigateToScreen("lab") must seat Lab's primary route (llm -> Models),
    not fall through the registry and leave the app on the current screen."""
    from tldw_chatbook.UI.Navigation.screen_registry import resolve_screen_target

    screen_name, _tab, screen_class = resolve_screen_target("lab")
    assert screen_name == "llm"
    assert screen_class is not None
    assert screen_class.__name__ == "LLMScreen"


def test_lab_legacy_routes_resolve_to_their_lab_screens():
    from tldw_chatbook.UI.Navigation.screen_registry import resolve_screen_target

    expectations = {
        "llm": "LLMScreen",
        "llm_management": "LLMScreen",
        "stts": "STTSScreen",
        "evals": "EvalsScreen",
    }
    for route, expected_class in expectations.items():
        _screen_name, _tab, screen_class = resolve_screen_target(route)
        assert screen_class is not None, route
        assert screen_class.__name__ == expected_class, route


def test_every_shell_destination_id_resolves_to_its_primary_screen():
    """Destination-id resolution: a destination id lands on the same screen
    as the destination's primary route (covers "lab" and "console", whose
    ids are not themselves screen routes)."""
    app = _build_test_app()

    for destination in SHELL_DESTINATION_ORDER:
        by_id = app._resolve_screen_navigation_target(destination.destination_id)
        by_primary = app._resolve_screen_navigation_target(destination.primary_route)
        assert by_id[2] is not None, destination.destination_id
        assert by_id == by_primary, destination.destination_id


def test_unknown_route_still_misses_cleanly():
    from tldw_chatbook.UI.Navigation.screen_registry import resolve_screen_target

    assert resolve_screen_target("definitely-not-a-route") == (
        "definitely-not-a-route",
        "definitely-not-a-route",
        None,
    )


def test_each_shell_destination_has_recovery_tooltip_copy():
    for destination in SHELL_DESTINATION_ORDER:
        assert destination.tooltip
        assert destination.purpose


def test_every_shell_destination_has_readable_purpose_and_mounted_route():
    app = _build_test_app()

    for destination in SHELL_DESTINATION_ORDER:
        assert destination.purpose
        assert destination.tooltip
        _screen_name, _tab_id, screen_class = app._resolve_screen_navigation_target(
            destination.primary_route
        )
        assert screen_class is not None, destination.primary_route
