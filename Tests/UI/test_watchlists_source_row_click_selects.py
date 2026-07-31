"""Clicking a row in a Watchlists table must select THAT row — TASK-1100/1105.

Two stacked defects, both found against real feeds.

TASK-1100. `SourcesPane` handled `RowSelected`/`CellSelected`, which Textual
fires on *activation* — Enter, or a second click — not when a click merely
moves the cursor onto a row. So clicking a source left `selected_source` at
`None`, `Preview` and `Check now` stayed disabled, and pressing `Check now`
returned silently because `handle_check_now_requested` early-returns on
`entity is None`. Fixed by handling `RowHighlighted`/`CellHighlighted`.

TASK-1105. With that in place the *default* highlight of row 0 selected a
source, but a click still never moved the cursor, so every action ran against
row 0. The cause is `*:focus { outline: solid $ds-focus-accent; }`
(core/_reset.tcss): Textual's `outline` is painted OVER the widget's outermost
rendered lines rather than around them, and the segments it overwrites lose
the `{"row", "column"}` metadata `DataTable._on_click` reads. `_on_click`
bails on `"row" not in meta`, so the bottom-most visible row of any focused
`DataTable` is unclickable. The first click is what gives the table focus, so
the very click being resolved is already evaluated against the outlined
render. In the Watchlists workbench the Sources table is three rows tall
(header + two rows), which left row 0 as the only reachable row.

The scrape backend was never at fault: driven directly it fetched a real feed
and ingested 10 items in 268ms.
"""
from __future__ import annotations

import pytest
from textual.widgets import Button, DataTable

from Tests.UI.test_destination_shells import StaticWatchlistsScopeService
from Tests.UI.test_destination_visual_parity_correction import (
    _active_destination_screen,
    _visual_destination_harness,
)
from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.UI.Watchlists_Modules.items_pane import ItemsPane
from tldw_chatbook.UI.Watchlists_Modules.notifications_pane import NotificationsPane
from tldw_chatbook.UI.Watchlists_Modules.rules_pane import RulesPane
from tldw_chatbook.UI.Watchlists_Modules.runs_pane import RunsPane
from tldw_chatbook.UI.Watchlists_Modules.sources_pane import SourcesPane

# TWO sources, and every click targets the SECOND row. Populating the table
# highlights row 0, so a single-source fixture would let these assertions pass
# even if click-to-select regressed entirely -- the default selection would
# stand in for the click. Row 1 can only be selected by the click itself.
SOURCES = [
    {"id": "local:subscription:1", "source_id": 1, "name": "Summit Route",
     "source_type": "rss", "active": True},
    {"id": "local:subscription:2", "source_id": 2, "name": "Darknet Diaries",
     "source_type": "rss", "active": True},
]
SECOND = SOURCES[1]

# Row 0 sits one line below the header, so the second row is at y-offset 2.
SECOND_ROW_OFFSET = (4, 2)


async def _sources_pane(pilot, host):
    screen = _active_destination_screen(host)
    screen.active_section = "sources"
    await pilot.pause(0.3)
    pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
    pane.sources = list(SOURCES)
    await pilot.pause(0.2)
    return screen, pane


@pytest.mark.asyncio
async def test_clicking_a_source_row_selects_that_row():
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")
    async with host.run_test(size=(235, 52)) as pilot:
        screen, pane = await _sources_pane(pilot, host)

        # Row 0 is selected by default after populate; row 1 is not.
        assert pane.selected_source is not None
        assert pane.selected_source["id"] == SOURCES[0]["id"]

        await pilot.click("#sources-table", offset=SECOND_ROW_OFFSET)
        await pilot.pause(0.3)

        assert pane.selected_source is not None, (
            "clicking a source row must select it; without this Preview and "
            "Check now can never be armed by mouse"
        )
        assert pane.selected_source["id"] == SECOND["id"], (
            "the click must move the selection to the row that was clicked, "
            "not leave it on the default row 0 (TASK-1105)"
        )
        assert screen.selected_source is not None
        assert screen.selected_source["id"] == SECOND["id"]
        assert not pane.query_one("#sources-check-now-button", Button).disabled


@pytest.mark.asyncio
async def test_check_now_acts_on_the_clicked_source():
    """The whole point: a click must make Check now run against THAT source."""
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")
    async with host.run_test(size=(235, 52)) as pilot:
        screen, pane = await _sources_pane(pilot, host)

        calls: list = []

        async def fake_check_now(*, runtime_backend, source_id):
            calls.append(source_id)
            return {"status": "completed"}

        screen._controller.check_now = fake_check_now

        await pilot.click("#sources-table", offset=SECOND_ROW_OFFSET)
        await pilot.pause(0.3)
        pane.query_one("#sources-check-now-button", Button).press()
        for _ in range(20):
            await pilot.pause()
            if calls:
                break

        assert calls, "Check now never reached the controller after a row click"
        assert calls[0] == SECOND["id"], (
            "Check now must act on the source the user clicked, not on row 0"
        )


@pytest.mark.asyncio
async def test_preview_acts_on_the_clicked_source():
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")
    async with host.run_test(size=(235, 52)) as pilot:
        screen, pane = await _sources_pane(pilot, host)

        calls: list = []

        async def fake_preview(*, runtime_backend, source_config):
            calls.append(source_config.get("id"))
            return {"items": [], "log_text": "ok"}

        screen._controller.preview_source = fake_preview

        await pilot.click("#sources-table", offset=SECOND_ROW_OFFSET)
        await pilot.pause(0.3)
        pane.query_one("#sources-preview-button", Button).press()
        for _ in range(20):
            await pilot.pause()
            if calls:
                break

        assert calls, "Preview never reached the controller after a row click"
        assert calls[0] == SECOND["id"]


@pytest.mark.asyncio
async def test_delete_acts_on_the_clicked_source():
    """`Delete` lives on the Inspector, which is fed by the same selection."""
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")
    async with host.run_test(size=(235, 52)) as pilot:
        screen, pane = await _sources_pane(pilot, host)

        await pilot.click("#sources-table", offset=SECOND_ROW_OFFSET)
        await pilot.pause(0.3)

        assert screen.selected_entity is not None
        assert screen.selected_entity["id"] == SECOND["id"], (
            "the Inspector -- and therefore Delete -- must target the clicked "
            "source"
        )


@pytest.mark.asyncio
async def test_keyboard_cursor_movement_selects_the_row_it_lands_on():
    """AC#4: arrowing onto a row selects it the same way a click does."""
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")
    async with host.run_test(size=(235, 52)) as pilot:
        screen, pane = await _sources_pane(pilot, host)
        table = pane.query_one("#sources-table", DataTable)
        table.focus()
        await pilot.pause(0.2)

        await pilot.press("down")
        await pilot.pause(0.3)

        assert pane.selected_source is not None
        assert pane.selected_source["id"] == SECOND["id"]

        await pilot.press("up")
        await pilot.pause(0.3)
        assert pane.selected_source["id"] == SOURCES[0]["id"]


@pytest.mark.asyncio
async def test_focused_table_keeps_click_metadata_on_its_last_row():
    """The root cause, asserted directly.

    `DataTable._on_click` resolves the clicked row from the Rich style meta at
    the pointer. A full-box `outline` repaints the widget's outermost lines
    and drops that meta, which is why the bottom row of a focused table stopped
    responding to the mouse at all. Assert the meta survives focus rather than
    only asserting the symptom, so a future focus-ring change cannot silently
    reintroduce it.
    """
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")
    async with host.run_test(size=(235, 52)) as pilot:
        screen, pane = await _sources_pane(pilot, host)
        table = pane.query_one("#sources-table", DataTable)
        table.focus()
        await pilot.pause(0.3)

        # The last data row of the visible table.
        last_row_y = table.region.height - 1
        x, y = table.region.offset + (4, last_row_y)
        meta = pilot.app.screen.get_style_at(x, y).meta

        assert meta.get("row") == last_row_y - 1, (
            "the focused table's last row lost its DataTable click metadata "
            f"(meta={meta!r}); a focus indicator must not repaint over the "
            "table's own content"
        )


# Every centre section that lists rows. Each is populated with TWO rows and
# the SECOND is clicked, for the same reason the Sources fixture has two.
OTHER_TABLES = (
    (
        "items",
        "#watchlists-items-pane",
        ItemsPane,
        "items",
        "#items-table",
        "selected_item",
        [
            {"id": "local:watchlist_item:1", "item_id": 1, "title": "First post",
             "source_name": "Summit Route", "status": "new"},
            {"id": "local:watchlist_item:2", "item_id": 2, "title": "Second post",
             "source_name": "Summit Route", "status": "new"},
        ],
    ),
    (
        "runs",
        "#watchlists-runs-pane",
        RunsPane,
        "runs",
        "#runs-table",
        "selected_run",
        [
            {"id": "local:run:1", "source_title": "Summit Route", "status": "completed",
             "found_count": 1, "processed_count": 1},
            {"id": "local:run:2", "source_title": "Darknet Diaries", "status": "completed",
             "found_count": 2, "processed_count": 2},
        ],
    ),
    (
        "rules",
        "#watchlists-rules-pane",
        RulesPane,
        "rules",
        "#rules-table",
        "selected_rule",
        [
            {"id": "local:rule:1", "rule_id": 1, "name": "Keyword",
             "condition_type": "keyword", "severity": "info", "enabled": True},
            {"id": "local:rule:2", "rule_id": 2, "name": "Volume",
             "condition_type": "volume", "severity": "warning", "enabled": True},
        ],
    ),
    (
        "notifications",
        "#watchlists-notifications-pane",
        NotificationsPane,
        "notifications",
        "#notifications-table",
        "selected_notification",
        [
            {"id": 1, "entity_kind": "client_notification", "title": "First alert",
             "category": "watchlist", "severity": "info", "is_read": False},
            {"id": 2, "entity_kind": "client_notification", "title": "Second alert",
             "category": "watchlist", "severity": "info", "is_read": False},
        ],
    ),
)


@pytest.mark.parametrize(
    "section,pane_id,pane_type,rows_attr,table_id,selected_attr,rows",
    OTHER_TABLES,
    ids=[entry[0] for entry in OTHER_TABLES],
)
@pytest.mark.asyncio
async def test_every_watchlists_table_selects_the_clicked_row(
    section, pane_id, pane_type, rows_attr, table_id, selected_attr, rows
):
    """AC#5: Runs, Items, Rules and Notifications had the same defect."""
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    host = _visual_destination_harness(app, "watchlists_collections")
    async with host.run_test(size=(235, 52)) as pilot:
        screen = _active_destination_screen(host)
        screen.active_section = section
        await pilot.pause(0.3)
        pane = screen.query_one(pane_id, pane_type)
        setattr(pane, rows_attr, list(rows))
        await pilot.pause(0.3)

        await pilot.click(table_id, offset=SECOND_ROW_OFFSET)
        await pilot.pause(0.4)

        selected = getattr(pane, selected_attr)
        assert selected is not None, f"{section}: clicking a row selected nothing"
        assert str(selected["id"]) == str(rows[1]["id"]), (
            f"{section}: the click must select the row it landed on, not row 0"
        )
