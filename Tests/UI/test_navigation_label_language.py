"""Regression tests for first-run-friendly navigation labels."""

from tldw_chatbook.Constants import (
    TAB_CCP,
    TAB_DISPLAY_LABELS,
    TAB_INGEST,
    TAB_LLM,
    TAB_STTS,
    get_tab_display_label,
)
from tldw_chatbook.UI.Tab_Bar import _get_tab_label as get_tab_bar_label
from tldw_chatbook.UI.Tab_Dropdown import TabDropdown
from tldw_chatbook.UI.Tab_Links import _get_tab_label as get_tab_link_label


def test_navigation_widgets_share_tab_display_labels() -> None:
    """Visible navigation copy comes from the shared tab label map."""
    assert TAB_DISPLAY_LABELS[TAB_CCP] == "Library"
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
    assert get_tab_link_label(TAB_CCP) == "Library"
    assert get_tab_link_label(TAB_LLM) == "Models"
    assert get_tab_link_label(TAB_STTS) == "Speech"
    assert get_tab_bar_label(TAB_CCP) == "Library"
    assert get_tab_bar_label(TAB_LLM) == "Models"
    assert get_tab_bar_label(TAB_STTS) == "Speech"


def test_dropdown_uses_plain_language_without_changing_values() -> None:
    dropdown = TabDropdown([TAB_CCP, TAB_INGEST, TAB_LLM, TAB_STTS], initial_active_tab=TAB_LLM)

    assert dropdown._get_tab_label(TAB_CCP) == "Library"
    assert dropdown._get_tab_label(TAB_INGEST) == "Ingest"
    assert dropdown._get_tab_label(TAB_LLM) == "Models"
    assert dropdown._get_tab_label(TAB_STTS) == "Speech"
    assert dropdown.tab_ids == [TAB_CCP, TAB_INGEST, TAB_LLM, TAB_STTS]
