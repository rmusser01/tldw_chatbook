"""Regression tests for first-run-friendly navigation labels."""

from tldw_chatbook.Constants import TAB_CCP, TAB_LLM, TAB_STTS
from tldw_chatbook.UI.Tab_Bar import _get_tab_label as get_tab_bar_label
from tldw_chatbook.UI.Tab_Dropdown import TabDropdown
from tldw_chatbook.UI.Tab_Links import _get_tab_label as get_tab_link_label


def test_legacy_ai_route_ids_use_plain_language_labels() -> None:
    """Route IDs remain stable while visible copy avoids acronyms."""
    assert get_tab_link_label(TAB_CCP) == "Library"
    assert get_tab_link_label(TAB_LLM) == "Models"
    assert get_tab_link_label(TAB_STTS) == "Speech"
    assert get_tab_bar_label(TAB_CCP) == "Library"
    assert get_tab_bar_label(TAB_LLM) == "Models"
    assert get_tab_bar_label(TAB_STTS) == "Speech"


def test_dropdown_uses_plain_language_without_changing_values() -> None:
    dropdown = TabDropdown([TAB_CCP, TAB_LLM, TAB_STTS], initial_active_tab=TAB_LLM)

    assert dropdown._get_tab_label(TAB_CCP) == "Library"
    assert dropdown._get_tab_label(TAB_LLM) == "Models"
    assert dropdown._get_tab_label(TAB_STTS) == "Speech"
    assert dropdown.tab_ids == [TAB_CCP, TAB_LLM, TAB_STTS]
