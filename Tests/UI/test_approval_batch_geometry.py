"""TASK-33003.20: the production approval batch geometry is a plain CI gate."""

from html import unescape
from itertools import combinations

import pytest
from textual.widgets import Button, Select, Static

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
    """Protect bounded rows and pinned actions before and after real disclosure.

    The retired bare Select width rule let a decision claim the entire row.
    More options now deliberately hides that Select initially, so check its
    geometry after opening it, alongside the unchanged text, overlap and
    container bounds. Both states must keep their complete actions reachable.
    """
    from tldw_chatbook.config import save_settings_to_cli_config

    # Persist splash-off: Console readiness re-reads the admitted profile.
    assert save_settings_to_cli_config({"splash_screen": {"enabled": False}})
    app = _build_ready_production_console_app()
    async with app.run_test(size=(200, 40)) as pilot:
        card = await _show_production_approval_batch(app, pilot, _sample_calls())
        rows = list(card.query(".approval-row"))
        assert len(rows) == 2

        for disclosed in (False, True):
            if disclosed:
                more = card.query_one("#approval-more-options-batch", Button)
                assert await pilot.click(more)
                await pilot.pause()
            assert card.has_class("approval-options-open") == disclosed

            for row in rows:
                header = row.query_one(".approval-row-header", Static)
                args = row.query_one(".approval-row-args", Static)
                scope = row.query_one(".approval-row-scope", Static)
                select = row.query_one(".approval-row-decision", Select)
                visible = [header, args, scope]
                assert select.display == disclosed
                if disclosed:
                    assert (
                        select.size.width > 0 and select.size.height > 0
                    ), "disclosed decision Select collapsed under bundled CSS"
                    # Approved $ds-approval-decision-width: the longest choice
                    # must fit with Textual's chrome, without claiming the row.
                    assert select.size.width == 28
                    assert select.size.width < row.size.width
                    assert select.region.y >= args.region.bottom
                    visible.append(select)
                else:
                    assert select.size.width == 0 and select.size.height == 0

                for widget in visible:
                    assert (
                        widget.size.width > 0 and widget.size.height > 0
                    ), f"{widget.classes} collapsed under bundled CSS"
                    assert row.region.contains_region(widget.region), (
                        widget.classes,
                        widget.region,
                        row.region,
                    )
                assert args.region.y >= header.region.bottom
                assert (
                    args.region.width >= row.region.width - 2
                ), "fixed controls are starving the argument preview"
                for first, second in combinations(visible, 2):
                    assert not first.region.overlaps(second.region), (
                        f"{first.classes} overlaps {second.classes}: "
                        f"{first.region} vs {second.region}"
                    )
                # A lost height:auto balloons these sample rows to 15;
                # retain the established compact bound in both states.
                assert (
                    row.size.height <= 8
                ), f"approval row ballooned to {row.size.height}"

            batch_rows = card.query_one("#approval-batch-rows")
            assert (
                batch_rows.size.height <= sum(row.size.height for row in rows) + 2
            ), "row container claimed height its children do not need"
            batch_actions = card.query_one("#approval-batch-actions")
            assert (
                batch_actions.region.y <= rows[-1].region.bottom + 3
            ), "actions were pushed away from the request rows"
            action_ids = (
                "#approval-more-options-batch",
                "#approval-submit" if disclosed else "#approval-approve-all",
                "#approval-deny-all",
            )
            actions = [card.query_one(selector, Button) for selector in action_ids]
            for action in actions:
                assert action.display and not action.disabled
                assert action.region.width >= len(action.label.plain)
                assert action.region.height > 0
                assert card.content_region.contains_region(action.region)
                assert app.screen.region.contains_region(action.region)
                hit, _ = app.screen.get_widget_at(*action.region.center)
                assert hit is action, (action.id, action.region, hit)
                action.focus()
                await pilot.pause()
                assert app.focused is action
            for first, second in combinations(actions, 2):
                assert not first.region.overlaps(second.region)

        # Exercise a real painted scope choice, then close it without committing.
        select = rows[0].query_one(".approval-row-decision", Select)
        select.focus()
        await pilot.press("enter", "down", "enter")
        await pilot.pause()
        assert select.value == "approve_session"
        assert "Until Chatbook exits" in unescape(app.export_screenshot()).replace(
            "\xa0", " "
        )
        await pilot.press("escape")
        await pilot.pause()
        assert not select.display
        assert select.value == "approve_session"
        assert app.focused is card.query_one("#approval-more-options-batch", Button)
