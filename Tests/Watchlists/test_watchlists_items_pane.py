"""Tests for the Watchlists items pane."""

import pytest
from rich.markup import escape as escape_markup
from textual.app import App, ComposeResult
from textual.widgets import Button, DataTable, Input, Select, Static

from tldw_chatbook.UI.Watchlists_Modules.items_pane import (
    ItemSelected,
    ItemsPane,
    NextUnreadRequested,
    RefreshItemsRequested,
)

# Marked so CI actually runs this file (whole-branch review fix 5): the unit
# job selects `pytest -m unit`, and an unmarked file in `Tests/Watchlists`
# is invisible to it. Matches the convention sibling files in this
# directory already use (`test_watchlist_name_and_copy.py`,
# `test_region_layout_store.py`, `test_watchlist_dialogs_escape.py`).
pytestmark = pytest.mark.unit


class ItemsPaneHarness(App):
    def __init__(self):
        super().__init__()
        self.captured_messages = []

    def compose(self) -> ComposeResult:
        yield ItemsPane()

    def on_item_selected(self, message: ItemSelected) -> None:
        self.captured_messages.append(("item_selected", message.item))

    def on_refresh_items_requested(self, message: RefreshItemsRequested) -> None:
        self.captured_messages.append(("refresh_items_requested", None))

    def on_next_unread_requested(self, message: NextUnreadRequested) -> None:
        self.captured_messages.append(("next_unread_requested", None))


@pytest.fixture
def sample_items():
    return [
        {
            "id": "local:watchlist_item:1",
            "item_id": 1,
            "title": "AI Breakthrough",
            "source_name": "AI News RSS",
            "status": "new",
            "created_at": "2026-07-18",
        },
        {
            "id": "local:watchlist_item:2",
            "item_id": 2,
            "title": "Tech Roundup",
            "source_name": "Tech Atom Feed",
            "status": "reviewed",
            "created_at": "2026-07-17",
        },
    ]


@pytest.mark.asyncio
async def test_items_pane_renders_table_and_toolbar():
    app = ItemsPaneHarness()
    async with app.run_test(size=(120, 40)):
        pane = app.query_one(ItemsPane)
        assert pane.query_one("#items-refresh-button", Button)
        assert pane.query_one("#items-search-input", Input)
        assert pane.query_one("#items-status-select", Select)
        assert pane.query_one("#items-table", DataTable)


@pytest.mark.asyncio
async def test_the_queued_column_carries_a_discoverable_legend():
    """TASK-2313, AC#6: the Queued column was a bare glyph or blank cell
    with no discoverable meaning anywhere on screen."""
    app = ItemsPaneHarness()
    async with app.run_test(size=(120, 40)):
        pane = app.query_one(ItemsPane)
        legend = pane.query_one("#items-queued-legend", Static)
        text = str(legend.renderable)
        assert pane._QUEUED_GLYPH in text
        assert "queued for the next briefing" in text


@pytest.mark.asyncio
async def test_status_select_carries_a_visible_label():
    """TASK-2310: the status filter must not paint as a bare "All statuses"
    with nothing naming what it filters."""
    app = ItemsPaneHarness()
    async with app.run_test(size=(120, 40)):
        pane = app.query_one(ItemsPane)
        row = pane.query_one("#items-toolbar")
        children = list(row.children)
        index = next(
            i for i, child in enumerate(children)
            if child.id == "items-status-select"
        )
        label = children[index - 1]
        assert isinstance(label, Static)
        assert str(label.renderable) == "Status"


@pytest.mark.asyncio
async def test_markup_shaped_item_text_is_escaped_at_the_datatable_boundary(sample_items):
    """`DataTable` markup-parses `str` cells, and item title / source name are
    remote feed content -- so `[bold red]BREAKING[/]` in a feed title would be
    interpreted as Rich markup rather than shown as text (TASK-1348 AC#1). The
    escape at the `add_row` boundary keeps the markup delimiters as data. This
    test fails if that escape is removed: the raw tag form would sit in the
    cell instead of the escaped one.

    Args:
        sample_items: Two normalized item dicts (the module fixture); the
            first is overwritten here with markup-shaped title/source_name.
    """
    hostile_title = "[bold red]BREAKING[/] news"
    hostile_source = "[link=http://evil.test]Feed[/link]"
    items = [dict(sample_items[0], title=hostile_title, source_name=hostile_source)]

    app = ItemsPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ItemsPane)
        pane.items = items
        await pilot.pause()

        table = pane.query_one("#items-table", DataTable)
        row = table.get_row(str(items[0]["id"]))
        # The cell holds the escaped form (delimiters survive as literal data,
        # never consumed as Rich tags) -- and NOT the raw markup that would be
        # parsed if the boundary escape were gone.
        assert row[0] == escape_markup(hostile_title)
        assert row[1] == escape_markup(hostile_source)
        assert row[0] != hostile_title
        assert row[1] != hostile_source


@pytest.mark.asyncio
async def test_status_repaint_escapes_at_the_update_cell_boundary(sample_items):
    """`update_item_status_cell` repaints a single Status cell via
    `DataTable.update_cell`, which markup-parses its value exactly as
    `add_row` does -- so the sibling write site must escape too, or it
    reopens the sink `compose()` closed (TASK-1348, Qodo finding 3). Status is
    an app-controlled enum today, but a markup-shaped value proves the
    boundary holds; fails if the repaint escape is removed.

    Args:
        sample_items: Two normalized item dicts (the module fixture); the
            first row's Status cell is repainted with a markup-shaped value.
    """
    items = [dict(sample_items[0])]
    app = ItemsPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ItemsPane)
        pane.items = items
        await pilot.pause()

        hostile_status = "[blink]ingested[/]"
        pane.update_item_status_cell(items[0]["id"], hostile_status)
        await pilot.pause()

        table = pane.query_one("#items-table", DataTable)
        cell = table.get_row(str(items[0]["id"]))[2]
        assert cell == escape_markup(hostile_status)
        assert cell != hostile_status


@pytest.mark.asyncio
async def test_items_pane_populates_table(sample_items):
    app = ItemsPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ItemsPane)
        pane.items = sample_items
        await pilot.pause()

        table = pane.query_one("#items-table", DataTable)
        assert table.row_count == 2


@pytest.mark.asyncio
async def test_items_pane_filters_by_status(sample_items):
    app = ItemsPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ItemsPane)
        pane.items = sample_items
        pane.status_filter = "reviewed"
        await pilot.pause()

        table = pane.query_one("#items-table", DataTable)
        assert table.row_count == 1
        assert "Tech Roundup" in str(table.get_row_at(0)[0])


@pytest.mark.asyncio
async def test_items_pane_refresh_posts_request():
    app = ItemsPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ItemsPane)
        pane.query_one("#items-refresh-button", Button).press()
        await pilot.pause()

        assert app.captured_messages == [("refresh_items_requested", None)]


@pytest.mark.asyncio
async def test_items_pane_selects_item_and_posts_message(sample_items):
    app = ItemsPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ItemsPane)
        pane.items = sample_items
        await pilot.pause()

        pane.select_item_by_id("local:watchlist_item:1")
        await pilot.pause()

        assert pane.selected_item == sample_items[0]
        assert app.captured_messages == [("item_selected", sample_items[0])]


# --- Spec #2 phase 1, task 5: the queued-for-briefing indicator column ----


@pytest.mark.asyncio
async def test_queued_indicator_renders_from_the_normalized_flag_on_load(sample_items):
    """Requirement 5: a pre-queued item shows the glyph after a plain load --
    pinning the read path (Task 1's `queued_for_briefing`) end to end, with
    no button press involved."""
    items = [dict(sample_items[0], queued_for_briefing=True), dict(sample_items[1])]
    app = ItemsPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ItemsPane)
        pane.items = items
        await pilot.pause()

        table = pane.query_one("#items-table", DataTable)
        assert table.get_row(str(items[0]["id"]))[4] == ItemsPane._QUEUED_GLYPH, (
            "a queued item must show the glyph as soon as it is loaded"
        )
        assert table.get_row(str(items[1]["id"]))[4] == "", (
            "an item the flag was never set on must show nothing"
        )


@pytest.mark.asyncio
async def test_update_item_queued_cell_repaints_in_place_without_recompose(sample_items):
    """Mirrors `update_item_status_cell`'s own contract: the same instances
    (pane AND table) must survive the repaint -- the Phase D pattern this
    stream keeps re-verifying (a recompose once destroyed the live table)."""
    app = ItemsPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ItemsPane)
        pane.items = [dict(item) for item in sample_items]
        await pilot.pause()
        table = pane.query_one("#items-table", DataTable)
        row_key = str(sample_items[0]["id"])

        pane.update_item_queued_cell(row_key, True)
        await pilot.pause()

        assert pane.query_one("#items-table", DataTable) is table, (
            "repainting a cell must not recompose the pane"
        )
        assert table.get_row(row_key)[4] == ItemsPane._QUEUED_GLYPH

        pane.update_item_queued_cell(row_key, False)
        await pilot.pause()
        assert table.get_row(row_key)[4] == "", "toggling back must clear the glyph"


# --- TASK-2308: the Items table shows PUBLISH date, not ingest time -------


def test_item_published_text_prefers_the_real_publish_date():
    """AC#2: the column reads `published_date`, not `created_at` -- the
    UAT's exact defect (every row from one check carrying the same
    microsecond-identical ingest time under a "Published" heading)."""
    text = ItemsPane.item_published_text({
        "published_date": "2026-08-04T17:55:00+00:00",
        "created_at": "2026-08-04T18:15:22.123456+00:00",
    })
    from tldw_chatbook.UI.Watchlists_Modules.humane_time import humane_timestamp

    assert text == humane_timestamp("2026-08-04T17:55:00+00:00")
    # And it must actually differ from what the ingest time would render as
    # -- pinning the UAT's own repro (17:55 reader byline vs. 18:15 table).
    assert text != humane_timestamp("2026-08-04T18:15:22.123456+00:00")


def test_item_published_text_falls_back_honestly_when_the_feed_omits_one():
    """When a feed supplies no publish date (`_parse_date` returns `None`
    rather than defaulting to "now" -- see `monitoring_engine.py`), the cell
    must say it is showing ingest time, never present it silently as a
    publish date under a "Published" heading."""
    from tldw_chatbook.UI.Watchlists_Modules.humane_time import humane_timestamp

    text = ItemsPane.item_published_text({
        "published_date": None,
        "created_at": "2026-08-04T18:15:22+00:00",
    })
    assert text == f"added {humane_timestamp('2026-08-04T18:15:22+00:00')}"
    assert "Published" not in text  # no false claim of being a publish date


def test_item_published_text_with_neither_field_shows_the_dash():
    assert ItemsPane.item_published_text({}) == "-"
    assert ItemsPane.item_published_text(
        {"published_date": None, "created_at": None}
    ) == "-"


@pytest.mark.asyncio
async def test_the_published_column_header_and_cells_are_wired_end_to_end():
    """The column mapping at the pane level: header says "Published", and
    the cell for a real item comes from `published_date` through
    `item_published_text`/`humane_timestamp`, not the raw ingest column."""
    from tldw_chatbook.UI.Watchlists_Modules.humane_time import humane_timestamp

    items = [
        {
            "id": "local:watchlist_item:9",
            "item_id": 9,
            "title": "Published item",
            "source_name": "Feed",
            "status": "new",
            "published_date": "2026-08-04T17:55:00+00:00",
            "created_at": "2026-08-04T18:15:22.123456+00:00",
        }
    ]
    app = ItemsPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ItemsPane)
        pane.items = items
        await pilot.pause()

        table = pane.query_one("#items-table", DataTable)
        assert [str(col.label) for col in table.columns.values()][3] == "Published"
        cell = table.get_row(str(items[0]["id"]))[3]
        assert cell == escape_markup(humane_timestamp("2026-08-04T17:55:00+00:00"))
        assert cell != escape_markup("2026-08-04T18:15:22.123456+00:00")


# --- task-2513 Task 10: `space` = next unread, pane-bound -------------------
#
# `space` is bound on the PANE, not the screen: the rail's tree is made of
# Buttons and a screen-level space binding would fire while the rail has
# focus; Input consumes printable keys before any binding; DataTable has no
# space binding, so space with the table focused bubbles up to here.


@pytest.mark.asyncio
async def test_space_on_the_items_table_posts_next_unread_requested(sample_items):
    app = ItemsPaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ItemsPane)
        pane.items = sample_items
        await pilot.pause()
        pane.query_one("#items-table", DataTable).focus()
        await pilot.press("space")
        await pilot.pause()

        assert ("next_unread_requested", None) in app.captured_messages
