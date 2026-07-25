"""Unit tests for ``chat_approval_card``'s per-row decision-options helper (task-5).

``_options_for_row`` narrows the batch-approval card's per-row ``Select``
options: MCP rows omit ``options`` entirely and must keep getting the full
four-choice set (byte-identical to today), while a row that requests a
subset (e.g. built-in tools, session-scoped only) gets exactly that subset,
falling back to the full set if the request is empty/invalid/unknown.
"""

from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import (
    _DECISION_OPTIONS,
    _options_for_row,
)


def test_row_without_options_offers_all_four():
    assert _options_for_row({}) == _DECISION_OPTIONS


def test_row_options_filter_to_the_requested_subset():
    got = _options_for_row({"options": ["approve_once", "approve_session"]})
    assert [value for _label, value in got] == ["approve_once", "approve_session"]


def test_unknown_option_values_are_ignored_not_rendered():
    got = _options_for_row({"options": ["approve_once", "teleport"]})
    assert [value for _label, value in got] == ["approve_once"]


def test_empty_options_list_falls_back_to_all():
    # An empty subset would render a Select with no choices -- unusable.
    assert _options_for_row({"options": []}) == _DECISION_OPTIONS
