"""TASK-1846 AC#2: the approval row's arguments get full width.

The row laid header, arguments, decision Select and (single-row) two fast
buttons out on ONE `Horizontal`. The controls are fixed-width -- 26 + 14 + 14
= 54 cells -- so header and arguments split whatever remains, measured at
**10 cells each on an 80-column terminal**.

Ten cells shows `{"path":"~/` of `{"path":"~/notes/secrets.md"}`. Since
TASK-1861 the card offers one decision per TARGET, so telling `spec.md` from
`secrets.md` is the entire point of the row -- and at 80 columns it was
impossible.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import ChatApprovalCard

BUNDLE = (
    Path(__file__).resolve().parents[2]
    / "tldw_chatbook"
    / "css"
    / "tldw_cli_modular.tcss"
)

#: Every `.approval-row*` rule lives in the app stylesheet, not in a
#: `DEFAULT_CSS` -- a bare App harness measures an UNSTYLED row (header and
#: args both reported the full 80 cells, the Select 1) and would have passed
#: this test while the shipped card stayed unreadable.
class _StyledCardHarness(App[None]):
    CSS_PATH = str(BUNDLE)

    def compose(self) -> ComposeResult:
        yield ChatApprovalCard()


async def _show_batch(app, pilot, calls: list[dict]) -> None:
    """Render `calls` on the card and wait for real geometry, by CONDITION.

    `#approval-batch-body` starts hidden. Before task-17500 that hide was
    DEFERRED mount work (`on_mount` -> `call_after_refresh`), and a fixed
    number of `pilot.pause()` calls RACED it: land `set_batch` first and
    the hide ran afterwards, leaving every region at 0 -- a machine-speed-
    dependent failure, the exact shape of the flaky test filed as
    TASK-1900, and (recognised much later) the exact mechanism of the
    task-17500 production bug, where a real terminal's slow first paint
    made the hide land last and unrender a headless round's card. The card
    is now hidden at CONSTRUCTION, so the first wait below is satisfied
    immediately; it is kept as a cheap contract check. The second wait --
    rows actually laid out -- is still load-bearing.

    Args:
        app: The mounted harness app.
        pilot: Its `Pilot`, used to let the event loop settle between checks.
        calls: Pending calls to render, in `set_batch` shape.

    Raises:
        AssertionError: If either state is not reached within the deadline.
    """
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        await pilot.pause()
        body = app.query("#approval-batch-body")
        if body and not body.first().display:
            break
    else:
        raise AssertionError("card never finished its deferred mount hide")

    app.query_one(ChatApprovalCard).set_batch(calls, timeout_seconds=45.0)
    while time.monotonic() < deadline:
        await pilot.pause()
        rows = list(app.query(".approval-row"))
        if len(rows) == len(calls) and all(r.region.width > 0 for r in rows):
            return
    raise AssertionError("approval rows never received a non-zero layout")


async def _row_geometry(cols: int, calls: list[dict]):
    app = _StyledCardHarness()
    async with app.run_test(size=(cols, 40)) as pilot:
        await _show_batch(app, pilot, calls)
        row = app.query_one(".approval-row")
        args = app.query_one(".approval-row-args", Static)
        # The SVG export is what the compositor actually painted. `render()`
        # returns the renderable and reports the whole string even when the
        # widget is 10 cells wide -- it passed against the broken layout.
        painted = app.export_screenshot()
        return row.region.width, args.region.width, painted


@pytest.mark.asyncio
@pytest.mark.parametrize("cols", [80, 120, 212])
async def test_arguments_get_full_row_width_at_every_supported_size(cols: int):
    """The row's arguments span it, at every width the Console supports.

    Args:
        cols: Terminal width under test -- 80 is the classic floor where the
            54 fixed cells of controls hurt most, 212 a full-screen session.
    """
    row_w, args_w, _text = await _row_geometry(
        cols,
        [{"llm_name": "read_file", "arguments": {"path": "~/notes/secrets.md"}}],
    )
    assert args_w >= row_w - 2, (
        f"at {cols} columns the arguments got {args_w} of {row_w} cells; the "
        "fixed-width controls are still eating the row"
    )


@pytest.mark.asyncio
async def test_the_distinguishing_part_of_a_path_is_visible_at_80_columns():
    """The security-relevant case: which file is this call about?"""
    _row_w, _args_w, text = await _row_geometry(
        80,
        [{"llm_name": "read_file", "arguments": {"path": "~/notes/secrets.md"}}],
    )
    assert "secrets" in text, (
        "the filename under approval is never painted at 80 columns -- the "
        "row shows the first few characters of the JSON and stops"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("rows", [1, 4, 10])
async def test_the_action_bar_stays_reachable_on_a_short_terminal(rows: int):
    """The commit controls must never be pushed off screen by row count.

    Args:
        rows: Number of pending calls in the batch. 4 is the count this
            change would have broken without the row cap; 10 is well past
            what fits, so the rows must scroll rather than grow.

    The card is `height: auto` inside a plain Container (`ChatTaskCards`), so
    a long batch simply grew past the viewport and took Submit / Approve-all /
    Deny-all with it. This was ALREADY broken before TASK-1846 -- on an 80x24
    terminal five rows put Submit at y=24 -- and giving arguments their own
    line costs a line per row, which would have moved the cliff to four rows.
    `#approval-batch-rows` is now capped and scrolls instead.

    A user who cannot reach Submit cannot answer the card, and the run stays
    blocked until the 120s auto-deny fires.
    """
    app = _StyledCardHarness()
    async with app.run_test(size=(80, 24)) as pilot:
        await _show_batch(
            app,
            pilot,
            [
                {
                    "llm_name": "read_file",
                    "arguments": {"path": f"~/notes/file{i}.md"},
                    "call_id": f"c{i}",
                }
                for i in range(rows)
            ],
        )

        region = app.query_one("#approval-submit").region
        assert region.y + region.height <= 24, (
            f"with {rows} rows on an 80x24 terminal Submit sits at y={region.y}, "
            "off the bottom of the screen"
        )


@pytest.mark.asyncio
async def test_a_row_hugs_its_content_instead_of_ballooning():
    """`.approval-row-controls` is a Horizontal, and those default to 1fr.

    That is the fr-inside-flex trap this stylesheet block already documents
    for `.approval-row` itself. Left at the default the controls row grows to
    14 lines and the row to 15 (measured), so two pending calls would fill an
    80x24 terminal with one visible row. Caught only by measuring: the args
    width and action-bar tests both still pass while the row is 3x too tall.
    """
    app = _StyledCardHarness()
    async with app.run_test(size=(80, 24)) as pilot:
        await _show_batch(
            app,
            pilot,
            [{"llm_name": "read_file", "arguments": {"path": "~/notes/secrets.md"}}],
        )

        row = app.query_one(".approval-row")
        # 7 = header + arguments + the 4-line Select's controls row + the
        # task-32278 scope line. The balloon this pins measured 15.
        assert row.region.height <= 7, (
            f"a one-argument row is {row.region.height} lines tall; the "
            "headline is claiming an fr share instead of hugging its content"
        )


# ---------------------------------------------------------------------------
# task-32278: the decision's scope, and the risk reason, are on the card
# ---------------------------------------------------------------------------

MCP_ROW = {
    "server_key": "srv",
    "server_label": "Notes",
    "tool_name": "search",
    "llm_name": "search",
    "arguments": {"query": "roadmap"},
}


def _text(widget) -> str:
    """Return a Static's rendered text as a plain string."""
    return str(widget.renderable)


def _painted(app) -> str:
    """Return what the compositor painted, as plain text.

    `export_screenshot` returns SVG, where every space is `&#160;` and each
    styled run is its own `<text>` element -- so a raw `in` check for a
    sentence fails against a screen that paints it perfectly.
    """
    import html
    import re

    runs = re.findall(r">([^<>]*)</text>", app.export_screenshot())
    return html.unescape("".join(runs)).replace("\xa0", " ")


@pytest.mark.asyncio
async def test_the_scope_line_states_the_selected_decision_and_follows_it():
    """AC#2: the card must say how long the highlighted grant lasts."""
    from textual.widgets import Select

    from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import (
        DECISION_SCOPE_COPY,
    )

    app = _StyledCardHarness()
    async with app.run_test(size=(80, 40)) as pilot:
        await _show_batch(app, pilot, [dict(MCP_ROW)])

        scope = app.query_one(".approval-row-scope", Static)
        assert _text(scope) == DECISION_SCOPE_COPY["approve_once"]

        app.query_one(".approval-row-decision", Select).value = "always_allow"
        await pilot.pause()
        assert _text(scope) == DECISION_SCOPE_COPY["always_allow"]
        assert DECISION_SCOPE_COPY["always_allow"] in _painted(app), (
            "the scope line is not painted at 80 columns"
        )


@pytest.mark.asyncio
async def test_a_high_risk_row_explains_itself_without_hover():
    """AC#3: "(high risk)" used to explain itself only in a tooltip.

    The literal sentence, not `format_approval_reason(entry)` -- comparing
    the widget to the function that filled it asserts nothing.
    """
    expected = "High risk: this tool changes local data and always asks first."

    app = _StyledCardHarness()
    async with app.run_test(size=(80, 40)) as pilot:
        entry = {**MCP_ROW, "reason": "risk_floored", "effects": ["mutates_local"]}
        await _show_batch(app, pilot, [entry])

        assert _text(app.query_one(".approval-row-reason", Static)) == expected
        assert expected in _painted(app)
        # The header tooltip was the hover-only version of this line.
        assert app.query_one(".approval-row-header", Static).tooltip is None


@pytest.mark.asyncio
async def test_the_longest_decision_label_paints_on_one_line():
    """AC#1: measured, not counted -- a too-long label wraps the Select."""
    from textual.widgets import Select

    from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import (
        _DECISION_OPTIONS,
    )

    longest = max((label for label, _v in _DECISION_OPTIONS), key=len)
    value = next(v for label, v in _DECISION_OPTIONS if label == longest)
    app = _StyledCardHarness()
    async with app.run_test(size=(80, 40)) as pilot:
        await _show_batch(app, pilot, [dict(MCP_ROW)])
        select = app.query_one(".approval-row-decision", Select)
        select.value = value
        await pilot.pause()
        assert select.region.height <= 3, (
            f"{longest!r} wraps the closed Select to "
            f"{select.region.height} lines"
        )
        assert longest in _painted(app), (
            f"{longest!r} is clipped by the closed Select"
        )


@pytest.mark.asyncio
async def test_the_reused_single_row_keeps_a_live_scope_line_below_its_controls():
    """`_update_mounted_single_row` rebuilds the controls in place.

    It appends the replacement `Horizontal`, so an unhandled scope line ends
    up ABOVE the controls it annotates -- and bound to the previous round's
    Select, which no longer exists.
    """
    from textual.containers import Horizontal
    from textual.widgets import Select

    from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import (
        ChatApprovalCard,
        DECISION_SCOPE_COPY,
    )

    app = _StyledCardHarness()
    async with app.run_test(size=(80, 40)) as pilot:
        await _show_batch(app, pilot, [{**MCP_ROW, "call_id": "a"}])
        first_row = id(app.query_one(".approval-row"))
        app.query_one(".approval-row-decision", Select).value = "deny"
        await pilot.pause()

        card = app.query_one(ChatApprovalCard)
        card.set_batch(
            [{**MCP_ROW, "call_id": "b", "arguments": {"query": "budget"}}],
            timeout_seconds=45.0,
        )
        await pilot.pause()

        row = app.query_one(".approval-row")
        assert id(row) == first_row, "the in-place update path was not exercised"
        kinds = [
            "controls" if isinstance(child, Horizontal) else child.classes
            for child in row.children
        ]
        controls_at = kinds.index("controls")
        scope_at = next(
            i for i, k in enumerate(kinds) if k != "controls" and "approval-row-scope" in k
        )
        assert scope_at > controls_at, f"scope line above the controls: {kinds}"

        scope = app.query_one(".approval-row-scope", Static)
        assert _text(scope) == DECISION_SCOPE_COPY["approve_once"]
        app.query_one(".approval-row-decision", Select).value = "approve_session"
        await pilot.pause()
        assert _text(scope) == DECISION_SCOPE_COPY["approve_session"]
