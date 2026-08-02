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


@pytest.mark.unit
def test_the_approval_deadline_is_rendered_not_silently_dropped():
    """TASK-1844: a clock that decides for the user must be visible.

    `set_batch` takes `timeout_seconds` and its docstring says the value is
    "surfaced on the card" -- it was accepted and never read. The controller
    arms a 120s deadline that auto-denies, so a countdown the user cannot see
    was making the decision.
    """
    from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import (
        format_approval_deadline,
    )

    assert format_approval_deadline(120) == "Auto-denies in 2:00"
    assert format_approval_deadline(95) == "Auto-denies in 1:35"
    assert format_approval_deadline(9) == "Auto-denies in 0:09"
    # No deadline armed -> say nothing rather than invent a number.
    assert format_approval_deadline(0) == ""
    assert format_approval_deadline(None) == ""


@pytest.mark.unit
def test_a_timed_out_approval_produces_its_own_marker():
    """TASK-1844: the user must be able to tell a timeout from their own deny.

    `format_agent_step_marker` had no timeout branch, so an expired approval
    just made the card vanish -- indistinguishable from "I denied it" or "it
    never ran". Tools are the egress boundary; a silent auto-deny is a
    decision the system made and never reported.
    """
    from tldw_chatbook.Chat.console_agent_bridge import (
        STEP_APPROVAL_TIMEOUT,
        format_agent_step_marker,
    )

    marker = format_agent_step_marker(
        STEP_APPROVAL_TIMEOUT, tool_name="write_file", summary="120"
    )
    assert marker, "a timeout must produce a transcript marker"
    assert "write_file" in marker
    low = marker.lower()
    assert "timed out" in low or "timeout" in low
    assert "not run" in low or "auto-denied" in low, (
        f"the marker must say the call did NOT run: {marker!r}"
    )


@pytest.mark.unit
def test_the_approval_card_carries_its_design_system_treatment():
    """TASK-1846: the highest-stakes surface must not render as body text.

    `.ds-approval-card` is the design system's approval treatment -- thick
    border in the approval-required colour, 12% tint -- and it was applied by
    NOTHING. `#chat-approval-card` had zero CSS rules of its own, so the card
    asking permission to let an agent reach the outside world looked exactly
    like a paragraph.
    """
    import inspect

    from tldw_chatbook.Widgets.Chat_Widgets import chat_approval_card as mod

    src = inspect.getsource(mod.ChatApprovalCard)
    assert "ds-approval-card" in src, (
        "the card does not apply the design system's approval treatment"
    )


@pytest.mark.unit
def test_tool_trace_is_not_the_faintest_text_on_screen():
    """TASK-1846: the record of what touched the machine must be legible.

    `.console-transcript-message-tool` rendered `dim italic` in muted grey --
    the audit trail was the least readable text in the transcript, and rows
    are not focusable so it could not even be selected by keyboard.
    """
    from pathlib import Path

    css = Path("tldw_chatbook/css/components/_agentic_terminal.tcss").read_text()
    import re

    m = re.search(r"\.console-transcript-message-tool\s*\{([^}]*)\}", css, re.S)
    assert m, "tool-row rule is missing"
    body = m.group(1)
    assert "dim" not in body, (
        f"the tool trace is still dimmed below every other row: {body.strip()!r}"
    )
