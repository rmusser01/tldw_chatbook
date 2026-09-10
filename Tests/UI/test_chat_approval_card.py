"""Unit tests for ``chat_approval_card``'s per-row decision-options helper (task-5).

``_options_for_row`` narrows the batch-approval card's per-row ``Select``
options: MCP rows omit ``options`` entirely and must keep getting the full
four-choice set (byte-identical to today), while a row that requests a
subset (e.g. built-in tools, session-scoped only) gets exactly that subset,
falling back to the full set if the request is empty/invalid/unknown.
"""

import pytest

from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import (
    _DECISION_OPTIONS,
    _default_decision_for_row,
    _format_row_header,
    _is_raw_shell_row,
    _options_for_row,
    format_approval_effects,
)


def test_fast_decision_control_copy_has_one_named_source():
    """Both construction paths must share labels, classes, and tooltips."""
    import ast
    import inspect
    import textwrap

    from tldw_chatbook.Widgets.Chat_Widgets import chat_approval_card as mod

    expected = {
        "_APPROVE_ONCE_LABEL": "Approve once",
        "_DENY_LABEL": "Deny",
        "_RAW_APPROVE_ONCE_LABEL": "Run once",
        "_FAST_APPROVE_CLASS": "approval-row-fast-approve",
        "_FAST_DENY_CLASS": "approval-row-fast-deny",
        "_FAST_APPROVE_TOOLTIP": (
            "Approve once and resume immediately (skips Select + Submit)."
        ),
        "_FAST_DENY_TOOLTIP": "Deny and resume immediately (skips Select + Submit).",
    }
    method_literals = set()
    for method in (mod.ChatApprovalCard.set_batch, mod.ChatApprovalCard._update_mounted_single_row):
        tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
        method_literals.update(
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        )
    for name, value in expected.items():
        assert getattr(mod, name) == value
        assert value not in method_literals


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


def test_raw_shell_identity_is_exact_and_defaults_to_deny():
    raw = {"server_key": "local:__local__", "tool_name": "shell_exec"}

    assert _is_raw_shell_row(raw) is True
    assert _default_decision_for_row(raw, ["approve_once", "deny"]) == "deny"
    assert _is_raw_shell_row({**raw, "server_key": "local:lookalike"}) is False
    assert _is_raw_shell_row({**raw, "tool_name": "shell_exec_extra"}) is False


def test_ordinary_rows_keep_the_approve_once_default():
    assert _default_decision_for_row({}, ["approve_once", "deny"]) == "approve_once"


def test_approval_effect_labels_are_code_owned_and_ignore_arguments():
    """Approval copy must disclose descriptor effects, never infer from args."""
    assert format_approval_effects(
        {
            "effects": (
                "private_read",
                "mutates_local",
                "network",
                "llm_spend",
            ),
            "arguments": {"path": "/private/secret", "url": "https://example"},
        }
    ) == (
        "Effects: may read private local data; may modify local data; "
        "may access the network; may incur LLM usage costs"
    )
    assert format_approval_effects({"arguments": {"url": "https://example"}}) == ""


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
    header = _format_row_header(_row(reason="risk_floored", path_precheck_failed=True))
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
    assert not offenders, f"these focus the commit control directly: {offenders}"


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
        _summarize_row_arguments,
    )

    calls = [
        {"llm_name": "read_file", "arguments": {"path": "~/notes/spec.md"}},
        {"llm_name": "read_file", "arguments": {"path": "~/notes/secrets.md"}},
        {"llm_name": "read_file", "arguments": {"path": "~/notes/todo.md"}},
    ]
    collapsed = _collapse_pending_calls(calls)
    assert len(collapsed) == 1, "grouping by name is the intended contract"
    assert collapsed[0]["count"] == 3

    rendered = _summarize_row_arguments(collapsed[0])
    for path in ("spec.md", "secrets.md", "todo.md"):
        assert path in rendered, (
            f"{path} is hidden behind the x3 -- the user would approve three "
            f"reads having seen: {rendered!r}"
        )


@pytest.mark.unit
def test_grouped_rows_take_the_first_non_empty_rationale():
    """ADR-090 (task 5): a blank-earlier rationale must not mask a later one.

    Fence-path calls collapse by `llm_name`; if the first call's `rationale`
    is blank but a later same-name call states one, the group's row must
    show that first NON-EMPTY reason -- and `description` follows the same
    rule. A group where every rationale is blank stays blank.
    """
    from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import (
        _collapse_pending_calls,
    )

    calls = [
        {"llm_name": "read_file", "rationale": "", "description": "",
         "arguments": {"path": "a.md"}},
        {"llm_name": "read_file", "rationale": "checking references",
         "description": "Reads a file", "arguments": {"path": "b.md"}},
    ]
    collapsed = _collapse_pending_calls(calls)
    assert len(collapsed) == 1
    assert collapsed[0]["rationale"] == "checking references"
    assert collapsed[0]["description"] == "Reads a file"

    blanks = [
        {"llm_name": "read_file", "rationale": "", "arguments": {"path": "a.md"}},
        {"llm_name": "read_file", "rationale": "", "arguments": {"path": "b.md"}},
    ]
    assert _collapse_pending_calls(blanks)[0]["rationale"] == ""


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


@pytest.mark.unit
def test_distinct_calls_get_their_own_row_and_verdict():
    """Per-call verdicts: the card must offer one decision per real call.

    Grouping was by `llm_name`, so two reads of two different files shared
    one row AND one verdict -- you could not allow `spec.md` and refuse
    `secrets.md`. Now that the runtime looks up by `call_id` first, the card
    can key rows per call.

    Calls that carry NO call_id (the fence path) must still collapse by name,
    because a name-keyed verdict is all the runtime can apply to them --
    splitting them into rows the verdict cannot address would be a lie.
    """
    from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import (
        _collapse_pending_calls,
    )

    distinct = [
        {"llm_name": "read_file", "call_id": "a", "arguments": {"path": "spec.md"}},
        {"llm_name": "read_file", "call_id": "b", "arguments": {"path": "secrets.md"}},
    ]
    rows = _collapse_pending_calls(distinct)
    assert len(rows) == 2, (
        f"two distinct calls must be two decisions, got {len(rows)} row(s)"
    )
    assert {r["call_id"] for r in rows} == {"a", "b"}

    # No call_id -> the runtime can only apply a name-keyed verdict, so one row.
    fence = [
        {"llm_name": "read_file", "arguments": {"path": "one.md"}},
        {"llm_name": "read_file", "arguments": {"path": "two.md"}},
    ]
    fence_rows = _collapse_pending_calls(fence)
    assert len(fence_rows) == 1, (
        "calls with no id cannot be addressed individually by the runtime, so "
        "splitting them would offer a decision that cannot be honoured"
    )
    assert fence_rows[0]["count"] == 2


def test_format_approval_deadline_hides_copy_when_no_deadline_armed():
    """ADR-067: 0/None deadline (the new shipped default) renders NO
    countdown copy. Pins the TASK-1844 behavior now that 0 is the norm."""
    from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import (
        format_approval_deadline,
    )

    assert format_approval_deadline(0) == ""
    assert format_approval_deadline(None) == ""


# ---------------------------------------------------------------------------
# task-32278: decision labels fit the closed Select and state their scope
# ---------------------------------------------------------------------------

#: What Textual's closed `Select` spends on chrome, leaving the rest for the
#: label: `SelectCurrent` is `border: tall` (1 cell each side) + `padding: 0 2`
#: (2 each side) and its `.arrow` is `width: 1` with `padding: 0 0 0 1`.
_SELECT_CHROME_CELLS = 8


def _decision_select_width() -> int:
    """Return the shipped `.approval-row-decision` width, from the stylesheet.

    Read from the source component rather than hard-coded: the label budget
    and the rule that sets it must not be able to drift apart, which is
    exactly how "Approve for session" came to render as "Approve for".
    """
    import re
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook"
        / "css"
        / "components"
        / "_agentic_terminal.tcss"
    ).read_text()
    # `;`-anchored so the `width: 1fr` in the BuddyConversationModal
    # override further down the file cannot match as "1", and the property
    # name anchored at a line start so `min-width:`/`max-width:` cannot
    # stand in for the `width` that actually sizes the closed Select.
    match = re.search(
        r"(?<![-\w ])\.approval-row-decision\s*\{[^}]*?^\s*width:\s*(\d+);",
        source,
        re.S | re.M,
    )
    assert match, "`.approval-row-decision` no longer sets an explicit width"
    return int(match.group(1))


@pytest.mark.unit
def test_every_decision_label_fits_the_closed_select():
    """AC#1: no decision label may be wider than the Select can paint.

    A label one cell too long does not ellipsize -- `SelectCurrent` is
    `height: auto` and its `Static#label` wraps, so the row's Select grows a
    line and the choice reads as two half-sentences.
    """
    from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import (
        _RAW_SHELL_DECISION_OPTIONS,
    )

    budget = _decision_select_width() - _SELECT_CHROME_CELLS
    too_long = {
        label: len(label)
        for label, _value in [*_DECISION_OPTIONS, *_RAW_SHELL_DECISION_OPTIONS]
        if len(label) > budget
    }
    assert not too_long, (
        f"these labels exceed the {budget}-cell label area of a "
        f"{_decision_select_width()}-cell Select: {too_long}"
    )


@pytest.mark.unit
def test_every_offered_decision_states_its_scope():
    """AC#2: a decision the card offers must say how long it lasts."""
    from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import (
        DECISION_SCOPE_COPY,
        _RAW_SHELL_DECISION_OPTIONS,
    )

    offered = {
        value for _label, value in [*_DECISION_OPTIONS, *_RAW_SHELL_DECISION_OPTIONS]
    }
    assert offered <= set(DECISION_SCOPE_COPY), (
        "no scope copy for: " f"{sorted(offered - set(DECISION_SCOPE_COPY))}"
    )
    assert DECISION_SCOPE_COPY["approve_once"] == "This call only."


@pytest.mark.unit
def test_persistent_decisions_name_where_to_undo_them():
    """AC#2: "remembered" is only honest if the card says where to remove it."""
    from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import (
        DECISION_SCOPE_COPY,
    )

    assert "MCP ▸ Tools" in DECISION_SCOPE_COPY["allow_matching"]
    assert "MCP ▸ Permissions" in DECISION_SCOPE_COPY["always_allow"]


@pytest.mark.unit
def test_the_high_risk_explanation_differs_for_reads_and_mutations():
    """AC#3: the reads-only sentence was also shown for `write_file`."""
    from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import (
        format_approval_reason,
    )

    read = format_approval_reason({"reason": "risk_floored"})
    mutate = format_approval_reason(
        {"reason": "risk_floored", "effects": ["mutates_local"]}
    )
    assert read == "High risk: this tool reads local data and always asks first."
    assert mutate == "High risk: this tool changes local data and always asks first."


@pytest.mark.unit
def test_a_changed_definition_explains_itself_too():
    from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import (
        format_approval_reason,
    )

    assert format_approval_reason({"reason": "config_changed"}) == (
        "Definition changed since you last allowed it; review the arguments."
    )
    assert format_approval_reason({"reason": "ask"}) == ""
    assert format_approval_reason({}) == ""


@pytest.mark.unit
def test_a_mutating_tool_is_floored_by_the_same_tag_the_effect_derives_from():
    """The card's sentence and the reason the row asks share one vocabulary.

    If `HIGH_RISK_TAGS` and the effect derivation ever key on different
    tags, a floored row can render the wrong blast radius again.
    """
    from tldw_chatbook.Agents.mcp_tool_provider import approval_effects_for_tool
    from tldw_chatbook.MCP.permission_store import HIGH_RISK_TAGS
    from tldw_chatbook.Tools.file_operation_tools import WriteFileTool

    tool = WriteFileTool()
    assert set(tool.risk_tags) & HIGH_RISK_TAGS, "write_file is no longer floored"
    assert approval_effects_for_tool(tool) == ("mutates_local",)
    # An MCP row's tool is a HubTool, which spells the same vocabulary
    # `tags` rather than `risk_tags` -- pin the REAL dataclass, since that
    # attribute-name difference is what the derivation has to bridge.
    from tldw_chatbook.MCP.hub_tool_catalog import HubTool

    def _hub(tags):
        return HubTool(
            server_key="local:srv",
            server_label="Srv",
            source="server",
            name="write",
            description="",
            input_schema=None,
            tags=tags,
            stale=False,
            executable=True,
        )

    assert approval_effects_for_tool(_hub(("mutates",))) == ("mutates_local",)
    assert approval_effects_for_tool(_hub(("reads",))) == ()


# ---------------------------------------------------------------------------
# Mounted geometry -- the card hugs its content (task-32287)
# ---------------------------------------------------------------------------
#
# Live at 200x50 a ONE-row batch rendered a 17-row card with ~10 blank rows
# between the "Approve all / Submit / Deny all" bar and the bottom border,
# and at 80x24 the bar was clipped away entirely. Cause: `#approval-batch-
# body` (a Container) and `#approval-batch-actions` (a Horizontal) had no
# CSS at all, so both kept Textual's `height: 1fr` default and grew to fill
# whatever the parent offered; `#console-task-surface` (ChatTaskCards, also
# a Container) did the same one level up, taking a 1fr share of the session
# column -- too much at 50 rows, too little at 24, where it clipped the card.

_ONE_ROW_CARD_MAX_HEIGHT = 12
_THREE_ROW_CARD_MAX_HEIGHT = 22
#: The surface's non-approval cards: a few Statics and one button row each.
_SIBLING_CARD_MAX_HEIGHT = 8


def _pending_calls(count: int) -> list[dict]:
    """Return ``count`` distinct ordinary (non-raw-shell) MCP pending calls."""
    return [
        {
            "llm_name": f"mcp__srv__tool{index}",
            "server_key": "local:srv",
            "tool_name": f"tool{index}",
            "server_label": "Srv",
            "arguments": {"query": "hello"},
            "reason": "ask",
        }
        for index in range(count)
    ]


async def _mounted_batch(app, pilot, count: int):
    """Render ``count`` rows through the real ChatTaskCards sync path."""
    from tldw_chatbook.UI.Screens.chat_screen_state import TaskResumeState
    from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import ChatApprovalCard
    from tldw_chatbook.Widgets.Chat_Widgets.chat_task_cards import ChatTaskCards

    app.query_one(ChatTaskCards).sync_state(
        TaskResumeState(
            pending_approval={
                "calls": _pending_calls(count),
                "timeout_seconds": 45.0,
            }
        )
    )
    await pilot.pause()
    card = app.query_one(ChatApprovalCard)
    assert card.display is True
    assert len(card.query(".approval-row")) == count
    return card


def _task_surface_harness():
    """An app whose only content is the production task-card surface.

    `APP_STYLESHEETS`, not `BUNDLED_STYLESHEET`: the console's rules were
    split out of the bundle into `screen_agentic_console.tcss`, which the
    real app parses on first visit to the Console (see the module docstring
    in Tests/UI/consolidated_css.py). Pinning the bundle alone would drop
    `#console-task-surface` and every other console rule on the floor and
    measure a surface production never renders.
    """
    from textual.app import ComposeResult

    from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
    from tldw_chatbook.Widgets.Chat_Widgets.chat_task_cards import ChatTaskCards

    class _TaskSurfaceHarness(ConsolidatedCSSApp):
        CSS_PATH = [str(path) for path in APP_STYLESHEETS]

        def compose(self) -> ComposeResult:
            yield ChatTaskCards(id="console-task-surface")

    return _TaskSurfaceHarness()


@pytest.mark.asyncio
async def test_one_row_approval_card_hugs_its_content():
    """A one-row card is content-height, with no reserved slack under the bar."""
    app = _task_surface_harness()
    async with app.run_test(size=(200, 50)) as pilot:
        card = await _mounted_batch(app, pilot, 1)

        actions = card.query_one("#approval-batch-actions")
        submit = card.query_one("#approval-submit")

        assert card.size.height <= _ONE_ROW_CARD_MAX_HEIGHT, (
            f"one-row approval card is {card.size.height} lines tall "
            f"(> {_ONE_ROW_CARD_MAX_HEIGHT}) -- a container inside it is "
            "still reserving blank rows"
        )
        assert actions.size.height <= submit.size.height + 1, (
            f"the action bar is {actions.size.height} lines tall for a "
            f"{submit.size.height}-line button -- it is padding the card "
            "with blank rows"
        )
        # Nothing but the card's own padding + border may sit under the bar.
        assert card.region.bottom - actions.region.bottom <= 2, (
            f"{card.region.bottom - actions.region.bottom} rows sit between "
            "the action bar and the card's bottom border"
        )


@pytest.mark.asyncio
async def test_three_row_approval_card_stays_bounded():
    """Three rows grow the card, but the rows container's cap still bounds it."""
    app = _task_surface_harness()
    async with app.run_test(size=(200, 50)) as pilot:
        card = await _mounted_batch(app, pilot, 3)

        actions = card.query_one("#approval-batch-actions")
        batch_rows = card.query_one("#approval-batch-rows")

        assert card.size.height <= _THREE_ROW_CARD_MAX_HEIGHT, (
            f"three-row approval card is {card.size.height} lines tall "
            f"(> {_THREE_ROW_CARD_MAX_HEIGHT})"
        )
        # Three 7-line rows exceed `#approval-batch-rows`' `max-height: 15`,
        # so the rows scroll and the bar sits directly under that cap.
        assert actions.region.y >= batch_rows.region.bottom, (
            "the action bar must stay below the rows it commits"
        )
        assert card.region.bottom - actions.region.bottom <= 2, (
            f"{card.region.bottom - actions.region.bottom} rows sit between "
            "the action bar and the card's bottom border"
        )


@pytest.mark.asyncio
async def test_action_bar_is_actually_visible_at_80x24_in_the_production_console():
    """AC#2: at 80x24 the Submit button is on screen AND not clipped away.

    Region alone is not evidence here: pre-fix the button reported a
    region inside the 24-row screen while `#console-task-surface`'s 1fr
    share (6 rows) clipped it, so the compositor handed those coordinates
    to the transcript's empty state instead. `get_widget_at` is the check
    that fails on the bug the live pass actually saw.
    """
    import time
    from unittest.mock import patch

    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
    from tldw_chatbook.UI.Screens.chat_screen_state import TaskResumeState

    def _settings_without_splash(section, key=None, default=None):
        if section == "splash_screen" and key == "enabled":
            return False
        return default

    app = _build_test_app()
    with patch(
        "tldw_chatbook.app.get_cli_setting", side_effect=_settings_without_splash
    ):
        async with app.run_test(size=(80, 24)) as pilot:
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline:
                screen = app.screen
                if (
                    isinstance(screen, ChatScreen)
                    and screen.is_mounted
                    and screen.query("#console-task-surface")
                ):
                    break
                await pilot.pause(0.05)
            else:
                raise AssertionError("Production Console did not finish mounting")

            screen.set_task_resume_state(
                TaskResumeState(
                    pending_approval={
                        "calls": _pending_calls(1),
                        "timeout_seconds": 45.0,
                    }
                )
            )
            deadline = time.monotonic() + 8.0
            while time.monotonic() < deadline:
                cards = screen.query("#chat-approval-card")
                if cards and cards.first().display and cards.first().query(
                    ".approval-row"
                ):
                    break
                await pilot.pause(0.05)
            else:
                raise AssertionError("Approval batch did not finish rendering")
            # An approval can only reach a Console the user has already set
            # up, so the first-run modal is never up at the same time; left
            # covering the workbench it would be the widget every hit test
            # below reported, measuring nothing about the card. `display =
            # False` is not enough -- the screen re-syncs the modal during
            # the pause that follows, so it has to go.
            await screen.query("#console-setup-modal").remove()
            await pilot.pause()

            submit = screen.query_one("#approval-submit")
            x, y = submit.region.center
            assert submit.region in app.screen.region, (
                f"Submit at {submit.region} is off an 80x24 screen"
            )
            hit, _region = app.screen.get_widget_at(int(x), int(y))
            assert hit.id == "approval-submit", (
                "the Submit button's own coordinates render "
                f"{hit.id or type(hit).__name__} instead -- the approval card "
                "is clipped by its task surface at 80x24"
            )


@pytest.mark.asyncio
async def test_every_task_surface_card_hugs_its_content():
    """The surface's other cards must not re-inflate it (task-32287 review).

    `#console-task-surface { height: auto }` is only half a fix while a
    card inside it still carries Textual's `height: 1fr` default: the
    fraction resolves against the whole offered box, so the surface grows
    right back. All four cards it hosts measured 50 rows in this 50-row
    harness before their rules landed (the two skill cards through their
    unstyled `Horizontal` button rows, one level down).
    """
    from tldw_chatbook.UI.Screens.chat_screen_state import TaskResumeState
    from tldw_chatbook.Widgets.Chat_Widgets.chat_task_cards import ChatTaskCards

    revealed = {
        "#chat-resume-panel": TaskResumeState(
            summary="Refactor the parser",
            last_step="ran the tests",
            diff_summary="3 files changed",
            next_action="review the diff",
        ),
        "#chat-skill-install-card": TaskResumeState(
            pending_skill_install={
                "url": "https://example.com/demo.zip",
                "timeout_seconds": 120.0,
                "request_id": "r1",
            }
        ),
        "#chat-skill-script-card": TaskResumeState(
            pending_skill_script={
                "skill_name": "demo",
                "script_path": "run.sh",
                "request_id": "r1",
            }
        ),
    }

    for selector, state in revealed.items():
        app = _task_surface_harness()
        async with app.run_test(size=(200, 50)) as pilot:
            await pilot.pause()
            surface = app.query_one(ChatTaskCards)
            surface.sync_state(state)
            await pilot.pause()
            await pilot.pause()

            card = app.query_one(selector)
            assert card.display is True, f"{selector} did not reveal"
            assert card.size.height <= _SIBLING_CARD_MAX_HEIGHT, (
                f"{selector} is {card.size.height} lines tall "
                f"(> {_SIBLING_CARD_MAX_HEIGHT}) -- it is still filling the "
                "surface instead of hugging its content"
            )
            visible = sum(
                child.region.height for child in surface.children if child.display
            )
            assert surface.size.height == visible, (
                f"the task surface is {surface.size.height} lines tall for "
                f"{visible} lines of visible cards"
            )
