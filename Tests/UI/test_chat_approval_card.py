import pytest
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


@pytest.mark.unit
def test_card_exposes_a_first_focus_target_that_is_not_the_commit_button():
    """TASK-1845: the keyboard must not land one keystroke from approving.

    `_DEFAULT_DECISION` pre-arms every row to `approve_once`, and BOTH review
    entry points focused `#approval-submit`. So the documented keyboard route
    -- jump to the card, press Enter -- granted a tool access to a call the
    user had not read. Tools are how an agent reaches the outside world, so
    this is the egress boundary, not a confirmation nicety.

    The card must name its own first focus target, and it must never be the
    control that commits.
    """
    from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import (
        ChatApprovalCard,
    )

    assert hasattr(ChatApprovalCard, "first_focus_widget_id"), (
        "the card must own its focus-landing contract rather than leaving each "
        "caller to pick a target -- two callers already picked the commit button"
    )
    target = ChatApprovalCard.first_focus_widget_id
    assert callable(target) or isinstance(target, str)


@pytest.mark.unit
def test_no_caller_focuses_the_submit_button_directly():
    """The two review entry points must route through the card's contract.

    Both `console_status_chips` and `chat_screen` previously hardcoded
    `#approval-submit`. Pinning this here means a third caller cannot quietly
    reintroduce the one-keystroke-to-approve path.
    """
    from pathlib import Path

    offenders = []
    for rel in (
        "tldw_chatbook/Widgets/Console/console_status_chips.py",
        "tldw_chatbook/UI/Screens/chat_screen.py",
    ):
        src = Path(rel).read_text()
        if '"#approval-submit"' in src:
            offenders.append(rel)
    assert not offenders, (
        f"these focus the commit control directly: {offenders}"
    )


@pytest.mark.unit
def test_collapsed_rows_disclose_every_argument_set():
    """TASK-1845: `xN` must not hide what is being approved.

    `_collapse_pending_calls` groups by `llm_name` to match the contract that
    same-name calls in one turn share one verdict. That grouping is fine --
    but it kept only the FIRST call's arguments and incremented a counter, so
    three reads of three different targets rendered as one row showing one
    target. The user approved three things having seen one.

    Decision taken: keep one verdict per name (re-keying per call id was
    deferred), and disclose every argument set instead.
    """
    from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import (
        _collapse_pending_calls,
        _summarize_arguments,
    )

    calls = [
        {"llm_name": "read_file", "arguments": {"path": "~/notes/spec.md"}},
        {"llm_name": "read_file", "arguments": {"path": "~/notes/secrets.md"}},
        {"llm_name": "read_file", "arguments": {"path": "~/notes/todo.md"}},
    ]
    collapsed = _collapse_pending_calls(calls)
    assert len(collapsed) == 1, "grouping by name is the intended contract"
    assert collapsed[0]["count"] == 3

    rendered = _summarize_arguments(collapsed[0])
    for path in ("spec.md", "secrets.md", "todo.md"):
        assert path in rendered, (
            f"{path} is hidden behind the x3 -- the user would approve three "
            f"reads having seen: {rendered!r}"
        )


@pytest.mark.unit
def test_needs_decision_state_is_text_labelled_not_colour_only():
    """TASK-1845: PRODUCT.md forbids colour as the only carrier of meaning.

    `.approval-row.needs-decision` was a border plus a 10% tint with no text
    change, so the state was invisible in monochrome and to anyone who cannot
    distinguish the tint.
    """
    from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import (
        NEEDS_DECISION_PREFIX,
        _format_row_header,
    )

    entry = {"llm_name": "write_file", "server": "Built-in", "needs_decision": True}
    header = _format_row_header(entry)
    assert NEEDS_DECISION_PREFIX in header, (
        f"needs-decision is colour-only; header reads {header!r}"
    )
    plain = _format_row_header({"llm_name": "write_file", "server": "Built-in"})
    assert NEEDS_DECISION_PREFIX not in plain
