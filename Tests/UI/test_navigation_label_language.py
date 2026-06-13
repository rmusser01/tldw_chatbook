"""Regression tests for first-run-friendly navigation labels."""

from tldw_chatbook.Constants import (
    TAB_CCP,
    TAB_CHAT,
    TAB_DISPLAY_LABELS,
    TAB_INGEST,
    TAB_LLM,
    TAB_MCP,
    TAB_SETTINGS,
    TAB_STTS,
    TAB_TOOLS_SETTINGS,
    get_tab_display_label,
)
from tldw_chatbook.UI.Tab_Bar import _get_tab_label as get_tab_bar_label
from tldw_chatbook.UI.Tab_Dropdown import TabDropdown
from tldw_chatbook.UI.Tab_Links import _get_tab_label as get_tab_link_label


def test_navigation_widgets_share_tab_display_labels() -> None:
    """Visible navigation copy comes from the shared tab label map."""
    assert TAB_DISPLAY_LABELS[TAB_CCP] == "Personas"
    assert TAB_DISPLAY_LABELS[TAB_INGEST] == "Ingest"
    assert TAB_DISPLAY_LABELS[TAB_LLM] == "Models"
    assert TAB_DISPLAY_LABELS[TAB_STTS] == "Speech"

    dropdown = TabDropdown([TAB_CCP, TAB_INGEST, TAB_LLM, TAB_STTS], initial_active_tab=TAB_LLM)

    for tab_id in [TAB_CCP, TAB_INGEST, TAB_LLM, TAB_STTS]:
        expected_label = get_tab_display_label(tab_id)
        assert get_tab_link_label(tab_id) == expected_label
        assert get_tab_bar_label(tab_id) == expected_label
        assert dropdown._get_tab_label(tab_id) == expected_label


def test_legacy_ai_route_ids_use_plain_language_labels() -> None:
    """Route IDs remain stable while visible copy avoids acronyms."""
    assert get_tab_link_label(TAB_CCP) == "Personas"
    assert get_tab_link_label(TAB_LLM) == "Models"
    assert get_tab_link_label(TAB_STTS) == "Speech"
    assert get_tab_bar_label(TAB_CCP) == "Personas"
    assert get_tab_bar_label(TAB_LLM) == "Models"
    assert get_tab_bar_label(TAB_STTS) == "Speech"


def test_dropdown_uses_plain_language_without_changing_values() -> None:
    dropdown = TabDropdown([TAB_CCP, TAB_INGEST, TAB_LLM, TAB_STTS], initial_active_tab=TAB_LLM)

    assert dropdown._get_tab_label(TAB_CCP) == "Personas"
    assert dropdown._get_tab_label(TAB_INGEST) == "Ingest"
    assert dropdown._get_tab_label(TAB_LLM) == "Models"
    assert dropdown._get_tab_label(TAB_STTS) == "Speech"
    assert dropdown.tab_ids == [TAB_CCP, TAB_INGEST, TAB_LLM, TAB_STTS]


def test_legacy_chat_route_uses_console_user_label() -> None:
    assert TAB_CHAT == "chat"
    assert get_tab_display_label(TAB_CHAT) == "Console"


def test_tools_settings_label_is_mcp_not_global_settings() -> None:
    assert get_tab_display_label(TAB_MCP) == "MCP"
    assert get_tab_display_label(TAB_TOOLS_SETTINGS) == "MCP"


def test_persona_legacy_tab_label_matches_top_level_destination() -> None:
    assert get_tab_display_label(TAB_CCP) == "Personas"
    assert get_tab_display_label(TAB_SETTINGS) == "Settings"
