"""TASK-33003.20: the production approval batch geometry is a plain CI gate."""

import pytest
from textual.widgets import Select, Static

from Tests.private_profile import private_profile_test
from Tests.UI.test_console_mcp_approval import (
    _build_ready_production_console_app,
    _sample_calls,
    _show_production_approval_batch,
)


@pytest.mark.asyncio
@private_profile_test
async def test_batch_row_widgets_have_nonzero_geometry_and_do_not_overlap_under_bundled_css(
    request: pytest.FixtureRequest,
) -> None:
    """Without an explicit width, `_conversations.tcss`'s bare `Select {
    width: 100%; }` rule (retired in TASK-33003.1) sized a row's decision
    Select to the FULL row width (not just its own share), overlapping/
    clipping it behind the header and args Statics laid out before it in the
    row's Horizontal -- verified empirically before landing the fix. Asserts all three
    per-row widgets render with real size AND stay within the row's own
    bounds in left-to-right order, under the real bundled stylesheet.

    T9 (MCP Hub Phase 5): also asserts HEIGHT bounds: each `.approval-row`
    stays compact (height <= 6 -- three stacked lines since TASK-1846 split
    header/args/controls, plus slack for a multi-arg collapsed row; a row
    that lost `height: auto` balloons to ~15), the `#approval-batch-rows`
    container doesn't balloon (height <= rows*3 + slack), and the
    `#approval-batch-actions` bar sits close after the rows (region.y within
    a few rows of the last row's bottom), matching the audit-mode geometry
    tests' discipline so all Horizontals/Verticals in the bundle stay compact."""
    # Persist before building: Console readiness re-reads the admitted profile,
    # so an in-memory splash override is lost after config cache invalidation.
    from tldw_chatbook.config import save_settings_to_cli_config

    assert save_settings_to_cli_config({"splash_screen": {"enabled": False}})
    app = _build_ready_production_console_app()
    async with app.run_test(size=(200, 40)) as pilot:
        card = await _show_production_approval_batch(app, pilot, _sample_calls())

        rows = list(card.query(".approval-row"))
        assert len(rows) == 2
        for row in rows:
            header = row.query_one(".approval-row-header", Static)
            args = row.query_one(".approval-row-args", Static)
            select = row.query_one(".approval-row-decision", Select)

            assert header.size.width > 0 and header.size.height > 0, (
                "approval row header collapsed to zero size under bundled CSS"
            )
            assert args.size.width > 0 and args.size.height > 0, (
                "approval row args summary collapsed to zero size under bundled CSS"
            )
            assert select.size.width > 0 and select.size.height > 0, (
                "approval row decision Select collapsed to zero size under bundled CSS"
            )
            # The decision Select must not claim the row's FULL width (the
            # actual bug this CSS fixes) -- it gets a definite, bounded
            # share instead.
            assert select.size.width < row.size.width, (
                f"decision Select width {select.size.width} claimed the "
                f"entire row width {row.size.width} under bundled CSS"
            )
            # task-32278: 27 = the 19-cell longest label ("Always · these
            # args") + 8 cells of Textual Select chrome. The closed Select
            # does not ellipsize -- it WRAPS and grows -- so this number
            # and `_DECISION_OPTIONS` move together.
            assert select.size.width == 27, (
                f"decision Select width {select.size.width} != pinned 27"
            )
            # TASK-1846: the row is three stacked lines now -- header,
            # arguments, then `.approval-row-controls` -- so neither text
            # widget shares a line with a fixed-width control. The
            # left-to-right ordering this used to assert (`args.x >=
            # header.right`) no longer describes the layout, so the
            # guarantee it was protecting -- nothing overlaps anything --
            # is asserted directly instead.
            assert select.region.right <= row.region.right
            assert args.region.y >= header.region.bottom, (
                "the arguments did not drop below the header"
            )
            assert select.region.y >= args.region.bottom, (
                "the controls did not drop below the arguments"
            )
            # The whole point of the split: arguments get the row, not a
            # leftover share of it after 54 cells of fixed-width controls.
            assert args.region.width >= row.region.width - 2, (
                f"arguments got {args.region.width} of {row.region.width} "
                "cells -- the controls are still eating the row"
            )
            for a, b in ((header, args), (select, args), (header, select)):
                assert not (
                    a.region.x < b.region.right
                    and b.region.x < a.region.right
                    and a.region.y < b.region.bottom
                    and b.region.y < a.region.bottom
                ), f"{a.classes} overlaps {b.classes}: {a.region} vs {b.region}"

            # T9: height bounds -- each row must stay compact (height: auto;
            # min-height: 1) instead of ballooning to 1fr (which would balloon
            # to fill the card height and push the actions bar far down).
            # Empirically measured before this fix: rows ballooning to height 9-10.
            # TASK-1846: 4 -> 6. The row gained a line when the
            # arguments moved to their own, and a collapsed `xN` row may
            # legitimately render several argument sets. A row that has
            # lost `height: auto` balloons to 15, so this still catches it.
            # task-32278: 6 -> 8. Every row gained the scope line under
            # its controls, and the `config_changed` row in
            # `_sample_calls` gained the reason line that used to be a
            # header tooltip.
            assert row.size.height <= 8, (
                f"approval row ballooned to height {row.size.height} under "
                "bundled CSS -- height: auto; min-height: 1; is not winning"
            )

        # T9: container height bound -- the Vertical wrapping all rows must
        # also stay compact (height: auto; min-height: 0) instead of balloning
        # to 1fr and claiming the full card height, which would push the
        # #approval-batch-actions bar far down. Empirically measured before
        # this fix: container ballooning to height 19, actions pushed to y=20.
        batch_rows = card.query_one("#approval-batch-rows")
        # task-32278: this was a per-row CONSTANT (3, then 6), which had
        # to be re-bumped every time a row gained a line -- and each bump
        # loosened it. Bounded by the rows' ACTUAL heights instead: the
        # bug it guards is the container claiming space its rows do not
        # need, which this states directly and needs no future bumping.
        assert batch_rows.size.height <= sum(r.size.height for r in rows) + 2, (
            f"approval-batch-rows container ballooned to height "
            f"{batch_rows.size.height} over {len(rows)} rows totalling "
            f"{sum(r.size.height for r in rows)} under bundled CSS "
            "-- height: auto; min-height: 0; is not winning"
        )

        # T9: action bar positioning -- must sit close after the rows,
        # not far below due to container ballooning. Within a few rows'
        # worth of lines from the last row's bottom edge.
        batch_actions = card.query_one("#approval-batch-actions")
        last_row = rows[-1]
        max_y_gap = 3  # generous slack: a few rows worth of lines
        assert batch_actions.region.y <= last_row.region.bottom + max_y_gap, (
            f"approval-batch-actions bar at y={batch_actions.region.y} is too far "
            f"below last row's bottom ({last_row.region.bottom}) -- should be "
            f"within {max_y_gap} lines"
        )
