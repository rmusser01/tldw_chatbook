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


def test_tab_navigation_provider_preserves_all_legacy_direct_commands():
    primary_shell_ids = set(TabNavigationProvider.navigation_tab_ids())
    legacy_tab_ids = set(ALL_TABS) - primary_shell_ids

    command_tab_ids = set(TabNavigationProvider.command_palette_tab_ids())

    assert legacy_tab_ids.issubset(command_tab_ids)
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
    assert "notes" not in command_tab_ids


def test_tab_navigation_provider_copy_uses_shell_vocabulary():
    assert "global preferences" in TabNavigationProvider.TAB_HELP_TEXT[TAB_SETTINGS]
    assert "MCP" in TabNavigationProvider.TAB_HELP_TEXT[TAB_MCP]
    assert "MCP" in TabNavigationProvider.TAB_HELP_TEXT[TAB_TOOLS_SETTINGS]
