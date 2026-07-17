from tldw_chatbook.Constants import (
    ALL_TABS,
    TAB_CCP,
    TAB_CHATBOOKS,
    TAB_CODING,
    TAB_CUSTOMIZE,
    TAB_EVALS,
    TAB_INGEST,
    TAB_LLM,
    TAB_LOGS,
    TAB_MCP,
    TAB_MEDIA,
    TAB_RESEARCH,
    TAB_SEARCH,
    TAB_SETTINGS,
    TAB_STATS,
    TAB_STTS,
    TAB_STUDY,
    TAB_SUBSCRIPTIONS,
    TAB_TOOLS_SETTINGS,
    TAB_WRITING,
)
from tldw_chatbook.app import TabNavigationProvider


def test_tab_navigation_provider_routes_settings_and_mcp_separately():
    assert TabNavigationProvider.route_for_tab(TAB_SETTINGS) == "settings"
    assert TabNavigationProvider.route_for_tab(TAB_MCP) == "mcp"
    assert TabNavigationProvider.route_for_tab(TAB_TOOLS_SETTINGS) == "mcp"
    assert TabNavigationProvider.route_for_tab("llm") == TAB_LLM


def test_tab_navigation_provider_includes_settings_and_mcp_shell_commands():
    tab_ids = TabNavigationProvider.navigation_tab_ids()

    assert TAB_SETTINGS in tab_ids
    assert TAB_MCP in tab_ids
    assert TAB_TOOLS_SETTINGS not in tab_ids


def test_command_palette_has_one_entry_per_shell_destination():
    from tldw_chatbook.UI.Navigation.shell_destinations import SHELL_DESTINATION_ORDER

    command_tab_ids = TabNavigationProvider.command_palette_tab_ids()

    # One labeled palette command per destination; nothing else.
    assert command_tab_ids == TabNavigationProvider.navigation_tab_ids()
    assert len(command_tab_ids) == len(SHELL_DESTINATION_ORDER) == 12

    # Legacy route ids are aliases, not separate labeled commands.
    legacy_tab_ids = set(ALL_TABS) - set(command_tab_ids)
    assert legacy_tab_ids
    assert not (legacy_tab_ids & set(command_tab_ids))
    assert "notes" not in command_tab_ids


def test_legacy_routes_are_searchable_alias_terms_on_their_destination():
    from tldw_chatbook.UI.Navigation.shell_destinations import get_shell_destination

    alias_terms = {
        destination_id: set(
            TabNavigationProvider._destination_alias_terms(get_shell_destination(destination_id))
        )
        for destination_id in (
            "console",
            "library",
            "personas",
            "watchlists_collections",
            "mcp",
            "artifacts",
            "lab",
            "settings",
        )
    }

    assert {"coding", "Coding"} <= alias_terms["console"]
    assert {"media", "Media", "search", "Search", "study", "Study"} <= alias_terms["library"]
    assert {"writing", "Writing", "research", "Research", "ingest", "Ingest"} <= alias_terms["library"]
    # Upstream retirements adopted on rebase: prompts and skills fold into Library.
    assert {"prompts", "skills"} <= alias_terms["library"]
    assert {"ccp", "conversations_characters_prompts", "characters"} <= alias_terms["personas"]
    assert "Personas" in alias_terms["personas"]  # TAB_CCP display label, deduped to one command
    assert {"subscriptions", "subscription", "Subscriptions"} <= alias_terms["watchlists_collections"]
    assert {"tools_settings", "MCP"} <= alias_terms["mcp"]
    assert {"chatbooks", "Chatbooks"} <= alias_terms["artifacts"]
    assert {"llm_management", "Models", "stts", "Speech", "evals", "Evals"} <= alias_terms["lab"]
    assert {"customize", "Customize", "logs", "Logs", "stats", "Stats"} <= alias_terms["settings"]


def test_legacy_tab_ids_still_switch_through_route_aliases():
    # switch_tab()/route_for_tab() keeps legacy tab ids navigable even though
    # they no longer appear as separate palette commands.
    representative_direct_tabs = [
        TAB_CCP,
        TAB_LLM,
        TAB_MEDIA,
        TAB_SEARCH,
        TAB_INGEST,
        TAB_SUBSCRIPTIONS,
        TAB_CHATBOOKS,
        TAB_STTS,
        TAB_EVALS,
        TAB_STUDY,
        TAB_CODING,
        TAB_LOGS,
        TAB_STATS,
        TAB_CUSTOMIZE,
        TAB_WRITING,
        TAB_RESEARCH,
    ]

    for tab_id in representative_direct_tabs:
        assert TabNavigationProvider.route_for_tab(tab_id) == tab_id
    assert TabNavigationProvider.route_for_tab(TAB_TOOLS_SETTINGS) == "mcp"
    assert TabNavigationProvider.route_for_tab("llm") == TAB_LLM


def test_tab_navigation_provider_copy_uses_shell_vocabulary():
    assert "global preferences" in TabNavigationProvider.TAB_HELP_TEXT[TAB_SETTINGS]
    assert "MCP" in TabNavigationProvider.TAB_HELP_TEXT[TAB_MCP]
    assert "MCP" in TabNavigationProvider.TAB_HELP_TEXT[TAB_TOOLS_SETTINGS]
