"""TASK-1035: the Watchlists create-source form, driven the way a user drives it.

The 2026-07-28 experienced-user UAT reported that the create-source form
"cannot be filled in": opening it and typing, and opening it and pressing
`Tab` five times then typing, both left `Name` and `URL` empty.

Everything here runs against the **production stylesheet in the full shell**
(`_visual_destination_harness`), because the form's real failure is a
geometry/focus interaction that a bare `App` with no CSS cannot reproduce --
the same reason `test_destination_visual_parity_correction.py` exists.

What the UAT saw, established in code:

* Pressing `New Source` sets `SourcesPane.show_create_form`, which is
  `reactive(..., recompose=True)`. The recompose tears down and remounts
  every child of the pane -- including the `New Source` `Button` that was
  holding focus. Textual does not re-home focus after a descendant
  recompose, so `Screen.focused` lands on `None` and the form opens with
  nothing focused anywhere on the screen. Typing goes to the void.
* From `Screen.focused is None`, `Tab` restarts at the head of the screen's
  focus chain, which is the top navigation bar. Measured at 235x52 the first
  form field is **37** tabs away, so five tabs land on `nav-personas` and
  typing still goes nowhere.
* Clicking the `Name` input does focus it and typing does land, on every one
  of its three rows including its borders -- see
  `test_clicking_any_row_of_the_name_input_focuses_it`. That part of the UAT
  report did not reproduce and is recorded here so the next run does not
  re-file it.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from textual.widgets import Button, Input

from Tests.UI.full_app_destination_context import (
    StaticWatchlistsScopeService,
    active_destination_screen as _active_destination_screen,
    full_app_destination_context as _visual_destination_harness,
)
from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.UI.Watchlists_Modules.sources_pane import SourcesPane

# 235x52 is the size the UAT ran at, so the geometry here is the geometry it
# saw. 160x42 is the small end the rest of the Watchlists parity suite covers,
# and it is the size that actually constrains this form: the Sources pane is
# 16 rows there and its toolbar takes two, so a form of five full-height rows
# puts `Create`/`Cancel` past the bottom edge -- present in the DOM, reported
# by `.region`, and unreachable.
SIZES = [(160, 42), (235, 52)]

CREATE_FIELD_ORDER = [
    "sources-create-name",
    "sources-create-url",
    "sources-create-type",
    "sources-create-active",
    "sources-create-tags",
    "sources-create-frequency",
    # TASK-1362: the noise control. Focusable, so it is part of the Tab walk,
    # and subject to every geometry assertion below -- the form had exactly
    # zero spare rows at 160x42 before it was added.
    "sources-create-ignore-selectors",
    "sources-create-submit",
    "sources-create-cancel",
]


def _watchlists_host():
    app = _build_test_app()
    app.watchlist_scope_service = StaticWatchlistsScopeService([])
    return _visual_destination_harness(app, "watchlists_collections")


async def _open_sources_create_form(pilot, host):
    """Open the form exactly as a user does: click through `New Source`."""
    screen = _active_destination_screen(host)
    screen.active_section = "sources"
    await pilot.pause(0.2)
    pane = screen.query_one("#watchlists-sources-pane", SourcesPane)
    screen.query_one("#sources-new-button", Button).press()
    await pilot.pause(0.3)
    assert pane.query("#sources-create-form"), "the create form never opened"
    return screen, pane


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.asyncio
async def test_create_form_focuses_its_first_field_when_it_opens(size):
    """AC#3: match `WatchlistNameDialog`, which focuses its input `on_mount`."""
    host = _watchlists_host()
    async with host.run_test(size=size) as pilot:
        screen, _pane = await _open_sources_create_form(pilot, host)

        focused = screen.focused
        assert focused is not None, (
            "the create form opened with nothing focused anywhere on the "
            "screen: the recompose that mounts the form destroyed the "
            "`New Source` Button that held focus, and nothing took its place"
        )
        assert focused.id == "sources-create-name", (
            f"expected the Name field to be focused on open, got {focused.id!r}"
        )


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.asyncio
async def test_typing_straight_after_opening_the_form_lands_in_name(size):
    """AC#5, the headline UAT step: open the form and type. No clicking."""
    host = _watchlists_host()
    async with host.run_test(size=size) as pilot:
        _screen, pane = await _open_sources_create_form(pilot, host)

        await pilot.press(*"Morning")
        await pilot.pause(0.1)

        assert pane.query_one("#sources-create-name", Input).value == "Morning", (
            "typing straight after opening the create form put the text "
            "nowhere -- exactly what the UAT reported"
        )


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.asyncio
async def test_tab_walks_the_create_form_in_visual_order(size):
    """AC#4: from the focused first field, `Tab` follows what the eye sees."""
    host = _watchlists_host()
    async with host.run_test(size=size) as pilot:
        screen, pane = await _open_sources_create_form(pilot, host)

        assert screen.focused is not None, "nothing focused; see AC#3 test"
        seen = [screen.focused.id]
        for _ in range(len(CREATE_FIELD_ORDER) - 1):
            await pilot.press("tab")
            await pilot.pause(0.05)
            seen.append(screen.focused.id if screen.focused else None)

        assert seen == CREATE_FIELD_ORDER, (
            f"Tab order through the create form is {seen}, expected "
            f"{CREATE_FIELD_ORDER}"
        )

        # "Visual order" is not just DOM order: every step must move down the
        # form, or right along the same row, and must stay inside the pane.
        previous = None
        for field_id in CREATE_FIELD_ORDER:
            widget = pane.query_one(f"#{field_id}")
            region = widget.region
            assert region.x >= pane.region.x and region.right <= pane.region.right, (
                f"#{field_id} is outside the Sources pane horizontally: "
                f"{region} vs {pane.region}"
            )
            if previous is not None:
                assert (region.y, region.x) >= previous, (
                    f"#{field_id} at {region} comes before the field that "
                    f"precedes it in Tab order (previous top-left {previous})"
                )
            previous = (region.y, region.x)


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.asyncio
async def test_create_and_cancel_sit_side_by_side_like_the_dialog(size):
    """AC#6: `WatchlistNameDialog` pairs its buttons in one `.dialog-buttons`
    row. The form stacked them with blank rows in between."""
    host = _watchlists_host()
    async with host.run_test(size=size) as pilot:
        screen, pane = await _open_sources_create_form(pilot, host)

        submit = pane.query_one("#sources-create-submit", Button)
        cancel = pane.query_one("#sources-create-cancel", Button)
        assert submit.region.y == cancel.region.y, (
            f"Create and Cancel are on different rows: {submit.region} vs "
            f"{cancel.region} -- the New-watchlist dialog puts its pair on one"
        )
        assert cancel.region.x > submit.region.x, (
            "Cancel should sit to the right of Create, as in the dialog"
        )

        # Regions alone would not catch a pair that is laid out on one row and
        # then clipped, so require both labels on the same painted row.
        strips = screen._compositor.render_strips()
        painted = "".join(seg.text for seg in strips[submit.region.y])
        assert "Create" in painted and "Cancel" in painted, (
            f"row {submit.region.y} paints {painted.strip()!r}; both buttons "
            "must reach the screen on one row"
        )


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.asyncio
async def test_the_whole_create_form_fits_inside_the_sources_pane(size):
    """Every control has to be reachable, not merely present.

    The `Grid` this form used to be took `height: 1fr` and spread seven
    controls over 23 rows. `Vertical` + `height: auto` fixes the spread, but
    an auto-height form can just as easily grow past the bottom of a pane
    that is 16 rows tall at 160x42 -- and a `Create` button below the fold
    is exactly as unusable as one that was never focusable. Asserted on both
    the geometry and the paint, because a widget clipped by an ancestor
    still reports a perfectly sensible `.region`.
    """
    host = _watchlists_host()
    async with host.run_test(size=size) as pilot:
        screen, pane = await _open_sources_create_form(pilot, host)

        strips = screen._compositor.render_strips()
        for field_id in CREATE_FIELD_ORDER:
            region = pane.query_one(f"#{field_id}").region
            assert region.bottom <= pane.region.bottom, (
                f"#{field_id} at {region} hangs below the Sources pane "
                f"{pane.region} at {size} -- the user cannot reach it"
            )
            assert region.y >= pane.region.y, (
                f"#{field_id} at {region} starts above the Sources pane "
                f"{pane.region}"
            )

        # The table must survive too: the form is transient, the list is not.
        table_region = pane.query_one("#sources-table").region
        assert table_region.height >= 1 and table_region.bottom <= pane.region.bottom, (
            f"#sources-table is {table_region} inside {pane.region} at {size}"
        )
        header = "".join(seg.text for seg in strips[table_region.y])
        assert "Name" in header, (
            f"the sources table header never reaches the screen at {size}: "
            f"row {table_region.y} paints {header.strip()!r}"
        )


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.asyncio
async def test_a_source_can_be_created_end_to_end_through_the_form(size):
    """AC#2: open, type, tab, type, submit -- no widget `.value =` anywhere."""
    host = _watchlists_host()
    async with host.run_test(size=size) as pilot:
        screen, pane = await _open_sources_create_form(pilot, host)
        created = AsyncMock(return_value={"id": 1, "name": "Morning"})
        screen._controller.create_source = created

        await pilot.press(*"Morning")
        await pilot.press("tab")
        await pilot.pause(0.05)
        await pilot.press(*"https://example.com/feed")
        await pilot.pause(0.05)

        assert pane.query_one("#sources-create-name", Input).value == "Morning"
        assert (
            pane.query_one("#sources-create-url", Input).value
            == "https://example.com/feed"
        )

        pane.query_one("#sources-create-submit", Button).press()
        await pilot.pause(0.3)

        assert created.await_count == 1, (
            "pressing Create never reached the controller"
        )
        payload = created.await_args.kwargs["payload"]
        assert payload["name"] == "Morning"
        assert payload["url"] == "https://example.com/feed"
        assert not pane.query("#sources-create-form"), (
            "the form should close once the source is submitted"
        )


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.asyncio
async def test_clicking_any_row_of_the_name_input_focuses_it(size):
    """AC#1, the part of the UAT report that did NOT reproduce.

    The UAT clicked "the `Name` input's own row" and reported nothing landed.
    A bordered `Input` is three rows, and only the middle one is content, so
    the natural suspicion was that the click hit a border row. It does not
    matter: `Screen.get_widget_at` returns the `Input` for all three rows and
    Textual focuses on mouse-down, so a click anywhere in the widget focuses
    it and typing lands. Pinned so the next UAT does not re-file this.
    """
    host = _watchlists_host()
    async with host.run_test(size=size) as pilot:
        screen, pane = await _open_sources_create_form(pilot, host)

        name_region = pane.query_one("#sources-create-name", Input).region
        column = name_region.x + 10
        for row in range(name_region.y, name_region.bottom):
            screen.set_focus(None)
            await pilot.pause(0.05)
            await pilot.click(offset=(column, row))
            await pilot.pause(0.1)
            assert screen.focused is not None and (
                screen.focused.id == "sources-create-name"
            ), f"clicking ({column},{row}) did not focus Name"
            await pilot.press("x")
            await pilot.pause(0.05)
            assert pane.query_one("#sources-create-name", Input).value.endswith("x"), (
                f"typing after clicking ({column},{row}) did not land"
            )
            pane.query_one("#sources-create-name", Input).value = ""
            await pilot.pause(0.05)


@pytest.mark.asyncio
async def test_creating_a_source_refreshes_the_table_and_the_tree_counts():
    """TASK-1040: creation only refreshed the view that read the data directly.

    `_create_source` called `_refresh_local_wc_snapshot` and
    `_refresh_overview_data` but never `_load_sources` or `_load_tree_data`,
    so `#sources-table` kept the list from before and the rail's counts stayed
    behind. Measured live: the rail read `All sources  0` while the centre read
    `Feeds in All sources (1)` -- the same thing, disagreeing on one screen.

    A first-time user is told nothing happened, and reasonably tries again.
    """
    host = _watchlists_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen, _pane = await _open_sources_create_form(pilot, host)

        created: list[dict] = []

        async def fake_create(*, runtime_backend, payload):
            created.append(payload)

        async def fake_list(*, runtime_backend, limit=100):
            return [
                {"id": 1, "name": "AI News RSS", "source_type": "rss", "active": True}
            ] if created else []

        screen._controller.create_source = fake_create
        screen._controller.list_sources = fake_list

        reloaded: list[str] = []
        real_sources = screen._load_sources
        real_tree = screen._load_tree_data

        async def watch_sources():
            reloaded.append("sources")
            await real_sources()

        def watch_tree():
            reloaded.append("tree")
            return real_tree()

        screen._load_sources = watch_sources
        screen._load_tree_data = watch_tree

        await screen._create_source({"name": "AI News RSS", "url": "https://x/f"})
        for _ in range(20):
            await pilot.pause()
            if {"sources", "tree"} <= set(reloaded):
                break

        assert "sources" in reloaded, (
            "creating a source must reload the sources table; without it the "
            "table keeps the old list until the user leaves and comes back"
        )
        assert "tree" in reloaded, (
            "creating a source must reload the tree counts; without it the "
            "rail says 0 while the centre says 1"
        )
