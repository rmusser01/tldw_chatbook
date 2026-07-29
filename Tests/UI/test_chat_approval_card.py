"""Unit tests for ``chat_approval_card``'s per-row decision-options helper (task-5).

``_options_for_row`` narrows the batch-approval card's per-row ``Select``
options: MCP rows omit ``options`` entirely and must keep getting the full
four-choice set (byte-identical to today), while a row that requests a
subset (e.g. built-in tools, session-scoped only) gets exactly that subset,
falling back to the full set if the request is empty/invalid/unknown.
"""

from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import (
    _DECISION_OPTIONS,
    _format_row_header,
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


# ---------------------------------------------------------------------------
# _format_row_header -- path-precheck warning badge (TASK-1231/F3 AC2)
# ---------------------------------------------------------------------------


def _row(**overrides):
    base = {
        "server_label": "Built-in",
        "tool_name": "read_file",
        "reason": "ask",
    }
    base.update(overrides)
    return base


def test_out_of_roots_path_gets_the_warning_suffix():
    header = _format_row_header(_row(path_precheck_failed=True))
    assert header == (
        "Built-in · read_file -- path outside allowed folders; "
        "will fail even if approved"
    )


def test_in_roots_path_gets_no_warning_suffix():
    header = _format_row_header(_row(path_precheck_failed=False))
    assert header == "Built-in · read_file"


def test_path_precheck_key_absent_gets_no_warning_suffix():
    # MCP rows and every pre-TASK-1231 payload never set this key at all.
    header = _format_row_header(_row())
    assert header == "Built-in · read_file"


def test_risk_floored_and_path_precheck_badges_can_combine():
    header = _format_row_header(
        _row(reason="risk_floored", path_precheck_failed=True)
    )
    assert header == (
        "Built-in · read_file (high risk) -- path outside allowed folders; "
        "will fail even if approved"
    )
