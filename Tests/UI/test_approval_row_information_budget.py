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
        assert row.region.height <= 6, (
            f"a one-argument row is {row.region.height} lines tall; the "
            "headline is claiming an fr share instead of hugging its content"
        )
