import pytest
from textual.app import App, ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Button, Static

from tldw_chatbook.UI.Watchlists_Modules.article_list import ArticleListPane
from tldw_chatbook.UI.Watchlists_Modules.region_layout import Region, RegionLayout
from tldw_chatbook.UI.Watchlists_Modules.watchlists_workbench import (
    REGION_TITLES,
    RegionToggled,
    WatchlistsWorkbench,
)


class _WorkbenchApp(App):
    def __init__(self, layout: RegionLayout) -> None:
        super().__init__()
        self._layout = layout
        self.toggles: list[Region] = []

    def compose(self) -> ComposeResult:
        yield WatchlistsWorkbench(self._layout, id="wl-workbench")

    def on_region_toggled(self, message: RegionToggled) -> None:
        self.toggles.append(message.region)


def _article(item_id: int) -> dict:
    return {
        "id": f"local:watchlist_item:{item_id}",
        "item_id": item_id,
        "title": f"Article {item_id:02d}",
        "source_name": "Geometry Feed",
        "status": "new",
        "published_date": "2026-08-13T12:00:00+00:00",
        "created_at": "2026-08-13T12:00:00+00:00",
        "content_preview": f"Preview for article {item_id:02d}.",
        "queued_for_briefing": False,
        "is_flagged": False,
    }


class _ReadGeometryApp(App):
    """Production-CSS harness for the Read centre's nested scroll owners."""

    from pathlib import Path

    CSS_PATH = str(
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook"
        / "css"
        / "tldw_cli_modular.tcss"
    )

    def __init__(
        self,
        item_count: int,
        *,
        read_mode: bool = True,
        layout: RegionLayout = RegionLayout(),
    ) -> None:
        super().__init__()
        self.item_count = item_count
        self.read_mode = read_mode
        self.initial_layout = layout

    def _items_pane(self) -> ArticleListPane:
        pane = ArticleListPane(id="watchlists-detail-pane")
        pane.items = [_article(index) for index in range(self.item_count)]
        return pane

    def compose(self) -> ComposeResult:
        classes = "watchlists-read-mode" if self.read_mode else ""
        yield WatchlistsWorkbench(
            self.initial_layout,
            content={
                Region.LEFT_RAIL: lambda: Static("fixed-left", id="left-probe"),
                Region.ITEMS: self._items_pane,
                Region.CONTENT: lambda: Vertical(
                    *[Static(f"content-{row:02d}") for row in range(30)],
                    id="watchlists-content-pane",
                ),
                Region.RIGHT_RAIL: lambda: Static("fixed-right", id="right-probe"),
            },
            hidden=(frozenset() if self.read_mode else frozenset({Region.CONTENT})),
            header=lambda: Static("Read status", id="wl-centre-status"),
            id="wl-workbench",
            classes=classes,
        )


def test_region_titles_cover_exactly_the_live_regions():
    # No FEEDS entry left over from the five-region era (task-2513), and no
    # live region missing a title -- `_region_widget` would KeyError.
    assert set(REGION_TITLES) == set(Region)


@pytest.mark.asyncio
async def test_only_the_centre_is_the_workbench_scroll_viewport():
    app = _WorkbenchApp(RegionLayout())
    async with app.run_test():
        workbench = app.query_one(WatchlistsWorkbench)
        centre = app.query_one("#wl-centre")

        assert isinstance(centre, VerticalScroll)
        assert list(workbench.children) == [
            app.query_one("#wl-region-left_rail"),
            centre,
            app.query_one("#wl-region-right_rail"),
        ]


@pytest.mark.parametrize(
    ("item_count", "expected_relation"),
    [(0, "floor"), (3, "natural"), (50, "cap")],
)
@pytest.mark.asyncio
async def test_read_items_region_grows_from_ten_rows_to_a_fifty_row_cap(
    item_count, expected_relation
):
    app = _ReadGeometryApp(item_count)
    async with app.run_test(size=(180, 100)) as pilot:
        await pilot.pause()
        items = app.query_one("#wl-region-items")

        assert 10 <= items.region.height <= 50, items.region
        if expected_relation == "floor":
            assert items.styles.min_height.value == 10
            assert items.region.height <= 12, items.region
        elif expected_relation == "natural":
            assert 10 < items.region.height < 50, items.region
        else:
            assert items.region.height == 50, items.region


@pytest.mark.parametrize("size", [(120, 36), (180, 50)])
@pytest.mark.asyncio
async def test_outer_centre_reaches_content_without_moving_either_rail(size):
    app = _ReadGeometryApp(50)
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        centre = app.query_one("#wl-centre", VerticalScroll)
        left = app.query_one("#wl-region-left_rail")
        right = app.query_one("#wl-region-right_rail")
        content = app.query_one("#wl-region-content")
        rail_regions = (left.region, right.region)

        assert centre.max_scroll_y > 0
        centre.scroll_end(animate=False)
        await pilot.pause()
        await pilot.pause()

        assert centre.scroll_y == centre.max_scroll_y
        assert (left.region, right.region) == rail_regions
        assert content.region.intersection(centre.content_region).height > 0
        screenshot = app.export_screenshot()
        assert "Content" in screenshot and "content-00" in screenshot


@pytest.mark.asyncio
async def test_non_read_items_pane_keeps_its_fill_layout_above_fifty_rows():
    app = _ReadGeometryApp(0, read_mode=False)
    async with app.run_test(size=(180, 90)) as pilot:
        await pilot.pause()
        items = app.query_one("#wl-region-items")

        assert not app.query_one(WatchlistsWorkbench).has_class(
            "watchlists-read-mode"
        )
        assert items.region.height > 50, items.region


@pytest.mark.asyncio
async def test_solo_items_lifts_caps_and_restore_rebounds_without_replacing_list():
    app = _ReadGeometryApp(50)
    async with app.run_test(size=(180, 90)) as pilot:
        await pilot.pause()
        workbench = app.query_one(WatchlistsWorkbench)
        pane = app.query_one(ArticleListPane)
        table = app.query_one("#items-table")
        bounded_items = app.query_one("#wl-region-items")

        assert bounded_items.region.height == 50
        assert table.region.height < bounded_items.region.height
        bounded_table_height = table.region.height
        table.focus()

        workbench.region_layout = RegionLayout().solo(Region.ITEMS)
        await pilot.pause()
        await pilot.pause()

        solo_items = app.query_one("#wl-region-items")
        solo_table = app.query_one("#items-table")
        assert solo_items.region.height > 50, solo_items.region
        assert solo_table.region.height > bounded_table_height
        assert app.query_one(ArticleListPane) is pane
        assert solo_table is table and table.has_focus

        workbench.region_layout = RegionLayout()
        await pilot.pause()
        await pilot.pause()

        assert app.query_one("#wl-region-items").region.height == 50
        assert app.query_one("#items-table").region.height < 50
        assert app.query_one(ArticleListPane) is pane


@pytest.mark.asyncio
async def test_all_regions_render_expanded_by_default():
    app = _WorkbenchApp(RegionLayout())
    async with app.run_test():
        for region in Region:
            assert app.query(f"#wl-region-{region.value}")
            assert not app.query(f"#wl-header-{region.value}")


@pytest.mark.asyncio
async def test_collapsed_region_renders_a_header_instead_of_a_body():
    app = _WorkbenchApp(RegionLayout(collapsed=frozenset({Region.CONTENT})))
    async with app.run_test():
        assert app.query("#wl-header-content")
        assert not app.query("#wl-region-content")


@pytest.mark.asyncio
async def test_collapsed_header_shows_suffix():
    """task-2513 Task 9: a collapsed rail advertises what it hides.

    NNW's sidebar toggle keeps the unread total visible while collapsed;
    here the collapsed left rail's header carries "N unread" so the user
    never loses the number that drives daily triage.
    """

    class _App(App):
        def compose(self) -> ComposeResult:
            yield WatchlistsWorkbench(
                RegionLayout(collapsed=frozenset({Region.LEFT_RAIL})),
                collapsed_suffixes={Region.LEFT_RAIL: "12 unread"},
                id="wl-workbench",
            )

    app = _App()
    async with app.run_test():
        header = app.query_one("#wl-header-left_rail", Button)
        assert str(header.label) == "▸ Watchlists  12 unread"


@pytest.mark.asyncio
async def test_set_collapsed_suffixes_repaints_mounted_collapsed_header():
    """Counts refresh while the rail stays collapsed: repaint in place.

    A full recompose for a number is exactly what `refresh_region_content`
    exists to avoid for bodies; expanded regions (no header mounted) are a
    no-op.
    """

    class _App(App):
        def compose(self) -> ComposeResult:
            yield WatchlistsWorkbench(
                RegionLayout(collapsed=frozenset({Region.LEFT_RAIL})),
                id="wl-workbench",
            )

    app = _App()
    async with app.run_test():
        header = app.query_one("#wl-header-left_rail", Button)
        assert str(header.label) == "▸ Watchlists"
        workbench = app.query_one(WatchlistsWorkbench)
        workbench.set_collapsed_suffixes({Region.LEFT_RAIL: "7 unread"})
        assert str(header.label) == "▸ Watchlists  7 unread"
        # Expanded region: nothing mounted to repaint, and no error.
        workbench.set_collapsed_suffixes({Region.ITEMS: "ignored"})


@pytest.mark.asyncio
async def test_collapsed_header_is_focusable_so_collapse_is_not_one_way():
    app = _WorkbenchApp(RegionLayout(collapsed=frozenset({Region.ITEMS})))
    async with app.run_test():
        header = app.query_one("#wl-header-items")
        assert header.focusable, "a collapsed region must be reachable by keyboard"


@pytest.mark.asyncio
async def test_clicking_a_collapsed_header_posts_region_toggled():
    app = _WorkbenchApp(RegionLayout(collapsed=frozenset({Region.CONTENT})))
    async with app.run_test() as pilot:
        await pilot.click("#wl-header-content")
        await pilot.pause()
        assert app.toggles == [Region.CONTENT]


@pytest.mark.asyncio
async def test_updating_the_layout_reactive_re_renders():
    # NOTE: the reactive is exposed as `region_layout`, not `layout`. Widget
    # already defines a read-only `layout` property (the compositor's arrange
    # strategy; see textual/widget.py and textual/_arrange.py:97) that a
    # same-named reactive here would shadow, crashing every render with
    # AttributeError: '...' object has no attribute 'arrange'. Verified
    # empirically with a minimal Horizontal subclass before renaming.
    app = _WorkbenchApp(RegionLayout())
    async with app.run_test() as pilot:
        workbench = app.query_one(WatchlistsWorkbench)
        workbench.region_layout = RegionLayout(collapsed=frozenset({Region.LEFT_RAIL}))
        await pilot.pause()
        assert app.query("#wl-header-left_rail")
        assert not app.query("#wl-region-left_rail")


@pytest.mark.asyncio
async def test_every_centre_region_may_be_collapsed_at_once():
    app = _WorkbenchApp(RegionLayout(collapsed=frozenset(CENTRE := {Region.ITEMS, Region.CONTENT})))
    async with app.run_test():
        for region in CENTRE:
            assert app.query(f"#wl-header-{region.value}")
        # The rails survive, so the screen is never empty.
        assert app.query("#wl-region-left_rail")
        assert app.query("#wl-region-right_rail")


@pytest.mark.asyncio
async def test_a_hidden_centre_region_is_unmounted_not_collapsed():
    """The screen gates CONTENT off every non-Read tab this way
    (`WatchlistsCollectionsScreen._hidden_centre_regions`, TASK-1344 AC#4):
    hidden means no DOM presence at all -- no body, not even a one-line
    header."""
    class _App(App):
        def compose(self) -> ComposeResult:
            yield WatchlistsWorkbench(
                RegionLayout(),
                hidden=frozenset({Region.CONTENT}),
                id="wl-workbench",
            )

    app = _App()
    async with app.run_test():
        assert app.query("#wl-region-left_rail")
        assert app.query("#wl-region-items")
        assert app.query("#wl-region-right_rail")
        assert not app.query("#wl-region-content")
        assert not app.query("#wl-header-content")


@pytest.mark.asyncio
async def test_supplied_content_is_mounted_into_its_region():
    from textual.widgets import Label

    # `content` holds FACTORIES, not instances — see the empirical finding
    # documented on `WatchlistsWorkbench.__init__` and
    # `test_supplied_content_with_nested_children_survives_recompose` below.
    class _App(App):
        def compose(self) -> ComposeResult:
            yield WatchlistsWorkbench(
                RegionLayout(),
                content={Region.ITEMS: lambda: Label("real items table", id="my-real-items")},
                id="wl-workbench",
            )

    app = _App()
    async with app.run_test():
        assert app.query("#my-real-items"), "supplied content should be mounted"
        region = app.query_one("#wl-region-items")
        assert app.query_one("#my-real-items") in region.walk_children()


@pytest.mark.asyncio
async def test_a_region_without_supplied_content_renders_no_stub_copy():
    """Whole-branch review: `REGION_PLACEHOLDERS` is gone.

    Every region the screen builds supplies a factory, so the only thing the
    "... arrives in the next slice." stubs still did was read, to anyone
    grepping the module, like shipped product copy for an unfinished feature.
    A region with no factory now renders its title and nothing else.
    """
    class _App(App):
        def compose(self) -> ComposeResult:
            yield WatchlistsWorkbench(RegionLayout(), content={}, id="wl-workbench")

    app = _App()
    async with app.run_test():
        assert app.query("#wl-region-content"), "the region body still renders"
        assert not app.query(".watchlists-region-placeholder")
        rendered = " ".join(
            str(getattr(node, "renderable", "")) for node in app.query("Static")
        )
        assert "arrives in the next slice" not in rendered


@pytest.mark.asyncio
async def test_supplied_content_with_nested_children_survives_recompose():
    """Regression test for an empirically-found bug: passing an already-
    constructed CONTAINER widget (with its own constructor-supplied children,
    e.g. `Vertical(Static(...), Static(...))`) as `content` works on the
    FIRST render, but its grandchildren vanish after ANY subsequent
    recompose — `region_layout` is `recompose=True`, so toggling even an
    unrelated region unmounts and rebuilds every region. A widget's
    constructor-supplied children are only mounted on that instance's first
    mount; the same instance remounted a second time comes back childless.
    `content` must therefore hold factories that build a fresh instance on
    every call, not pre-built instances — this test would fail against the
    pre-fix, instance-based implementation.
    """
    from textual.containers import Vertical as _Vertical
    from textual.widgets import Static as _Static

    def build_nested():
        return _Vertical(
            _Static("outer-title", id="outer-title"),
            _Static("inner-content", id="inner-content"),
            id="my-nested-pane",
        )

    class _App(App):
        def compose(self) -> ComposeResult:
            yield WatchlistsWorkbench(
                RegionLayout(), content={Region.ITEMS: build_nested}, id="wl-workbench"
            )

    app = _App()
    async with app.run_test() as pilot:
        assert app.query("#inner-content")

        workbench = app.query_one(WatchlistsWorkbench)
        # Toggle an UNRELATED region. Because the reactive recomposes the
        # whole workbench, this rebuilds ITEMS too, even though ITEMS itself
        # was never toggled.
        workbench.region_layout = RegionLayout(collapsed=frozenset({Region.LEFT_RAIL}))
        await pilot.pause()

        assert app.query("#my-nested-pane"), "the region body should still be rebuilt"
        assert app.query(
            "#inner-content"
        ), "nested content must survive an unrelated region's recompose"


@pytest.mark.asyncio
async def test_a_region_with_supplied_content_does_not_double_title():
    from textual.widgets import Static

    def factory():
        return Static("Items", classes="pane-title")

    class _App(App):
        def compose(self) -> ComposeResult:
            yield WatchlistsWorkbench(
                RegionLayout(), content={Region.ITEMS: factory}, id="wl-workbench"
            )

    app = _App()
    async with app.run_test():
        titles = [str(n.renderable) for n in app.query(".watchlists-region-title")]
        assert "Items" not in titles, (
            "a region whose content supplies its own heading should not add a second one"
        )


@pytest.mark.asyncio
async def test_left_rail_keeps_its_heading_when_its_content_supplies_none():
    """Fix round 3, Finding 1: title suppression was keyed on
    factory-presence, but the rule is "suppress it where the pane supplies
    its own heading". LEFT_RAIL is where those two signals diverge — it
    supplies content (`WatchlistTree`) that composes only navigation buttons
    and no heading widget — so the expanded rail rendered as an unlabelled
    bordered box while its collapsed header still read "▸ Watchlists".
    `SELF_HEADED_REGIONS` now carries the real rule.
    """
    from textual.widgets import Label

    class _App(App):
        def compose(self) -> ComposeResult:
            yield WatchlistsWorkbench(
                RegionLayout(),
                content={
                    # Stands in for `WatchlistTree`: real content, no heading.
                    Region.LEFT_RAIL: lambda: Label("All sources  0", id="headingless"),
                    Region.ITEMS: lambda: Label("Items", id="self-headed"),
                },
                id="wl-workbench",
            )

    app = _App()
    async with app.run_test():
        assert app.query("#headingless"), "supplied rail content should be mounted"
        titles = [str(n.renderable) for n in app.query(".watchlists-region-title")]
        assert "Watchlists" in titles, (
            "the left rail's content supplies no heading of its own, so the "
            f"region must still label it: {titles}"
        )
        assert "Items" not in titles, (
            "a region whose pane DOES supply its own heading must not add a "
            f"second one: {titles}"
        )


@pytest.mark.asyncio
async def test_a_soloed_centre_region_is_marked_for_css():
    """Fix round 3, Finding 3: `RegionLayout.solo` only collapses the soloed
    region's *siblings*, so nothing in the DOM distinguished a soloed region
    from an ordinarily-expanded one — and a capped region stayed pinned at
    its `max-height` with the rest of the centre blank. This class is the
    hook `.watchlists-region-sole-centre` keys off (see `_watchlists.tcss`);
    the geometry it produces is asserted against the real stylesheet in
    `Tests/UI/test_destination_visual_parity_correction.py::
    test_watchlists_soloed_centre_region_fills_the_centre`.
    """
    app = _WorkbenchApp(RegionLayout().solo(Region.CONTENT))
    async with app.run_test():
        content = app.query_one("#wl-region-content")
        assert content.has_class("watchlists-region-sole-centre"), sorted(content.classes)
        # The rails are still expanded, and solo never applies to them.
        for rail in ("left_rail", "right_rail"):
            rail_region = app.query_one(f"#wl-region-{rail}")
            assert not rail_region.has_class("watchlists-region-sole-centre")

    # Reaching the same DOM by hand must get the same treatment: `z` on ITEMS
    # leaves CONTENT just as alone as `Z` on CONTENT does.
    manual = _WorkbenchApp(RegionLayout(collapsed=frozenset({Region.ITEMS})))
    async with manual.run_test():
        assert manual.query_one("#wl-region-content").has_class(
            "watchlists-region-sole-centre"
        )

    # ... and an ordinary expanded layout must NOT get it, or the cap never
    # applies at all.
    ordinary = _WorkbenchApp(RegionLayout())
    async with ordinary.run_test():
        for region in Region:
            assert not ordinary.query_one(f"#wl-region-{region.value}").has_class(
                "watchlists-region-sole-centre"
            )


@pytest.mark.asyncio
async def test_content_height_is_capped_and_scrollable_when_content_overflows():
    """`.watchlists-region-content` is the one capped region left (FEEDS is
    gone, task-2513): `height: auto` + `max-height: 12` + `overflow-y: auto`
    in `_watchlists.tcss`, so the reader grows to fit small content, stops at
    the cap, and scrolls past it rather than either clipping silently or
    displacing ITEMS.

    Unlike this file's other tests, this one loads the REAL production
    stylesheet (`CSS_PATH`) -- a bare `App` with no CSS at all would never
    exercise `max-height`/`overflow-y`, so the assertions below would be
    vacuous against the pre-fix code as much as the post-fix code.
    """
    from pathlib import Path

    from textual.containers import Vertical
    from textual.widgets import Static

    css_path = str(
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook"
        / "css"
        / "tldw_cli_modular.tcss"
    )

    def overflowing_reader() -> Vertical:
        # 40 rows: far more than any reasonable cap, standing in for a long
        # article opened in the reader.
        body = Vertical(
            *[Static(f"paragraph-{i:02d}") for i in range(40)],
            id="content-overflow-probe",
        )
        # Mirrors the production companion fix on `#watchlists-content-pane`
        # in `_watchlists.tcss`: a bare `Vertical` defaults to
        # `height: 1fr`, which is circular inside CONTENT's `height: auto`
        # region -- it must size to its own content instead.
        body.styles.height = "auto"
        return body

    class _App(App):
        CSS_PATH = css_path

        def compose(self) -> ComposeResult:
            yield WatchlistsWorkbench(
                RegionLayout(),
                content={Region.CONTENT: overflowing_reader},
                id="wl-workbench",
            )

    app = _App()
    async with app.run_test(size=(100, 40)) as pilot:
        content = app.query_one("#wl-region-content")
        items = app.query_one("#wl-region-items")

        assert content.region.height <= 12, (
            f"CONTENT should stop growing at the cap once its content "
            f"overflows it: {content.region}"
        )
        assert items.region.height > content.region.height, (
            f"ITEMS must stay the taller reading area even when CONTENT's "
            f"content would otherwise dwarf it: items={items.region} "
            f"content={content.region}"
        )

        # Confirm it SCROLLS rather than clips: all 40 supplied rows must
        # be reachable, not silently cut off past the cap.
        def painted_rows() -> list[str]:
            strips = content.screen._compositor.render_strips()
            region = content.region
            return [
                "".join(segment.text for segment in strips[y])[
                    region.x : region.x + region.width
                ]
                for y in range(region.y, region.y + region.height)
            ]

        rows = painted_rows()
        assert any("paragraph-00" in row for row in rows), (
            f"expected the first supplied row on screen initially: {rows!r}"
        )
        # Fix round 3, Finding 2: the region owns the border AND the scroll,
        # so the box must stay closed at both scroll extremes. When the pane
        # inside owned the border instead, the border rows were part of the
        # scrolled content — at scroll top the bottom edge was off-screen and
        # at scroll end the top edge was.
        assert rows[0].startswith("╭") and rows[0].endswith("╮"), rows[0]
        assert rows[-1].startswith("╰") and rows[-1].endswith("╯"), rows[-1]

        content.scroll_end(animate=False)
        await pilot.pause()
        await pilot.pause()

        rows = painted_rows()
        assert any("paragraph-39" in row for row in rows), (
            f"the last supplied row should be reachable by scrolling, not "
            f"lost past the cap: {rows!r}"
        )
        assert rows[0].startswith("╭") and rows[0].endswith("╮"), (
            f"scrolling must not carry the region's own top border away: {rows[0]!r}"
        )
        assert rows[-1].startswith("╰") and rows[-1].endswith("╯"), (
            f"scrolling must not carry the region's own bottom border away: "
            f"{rows[-1]!r}"
        )


# --- `refresh_region_content` ---------------------------------------------
#
# `WatchlistsCollectionsScreen` needs a region to follow new data without
# recomposing the whole workbench (a full recompose would also replace the
# Inspector, breaking its "same instance, updated in place" contract -- see
# `WatchlistsCollectionsScreen.watch_selected_scope`). These pin the
# primitive it relies on, independent of the screen.


@pytest.mark.asyncio
async def test_refresh_region_content_rebuilds_only_the_named_region():
    from textual.widgets import Label

    calls = {"content": 0}

    def content_factory():
        calls["content"] += 1
        return Label(f"content-{calls['content']}", id="reader-content")

    items_widget_ids: list[int] = []

    def items_factory():
        label = Label("items", id="items-content")
        items_widget_ids.append(id(label))
        return label

    class _App(App):
        def compose(self) -> ComposeResult:
            yield WatchlistsWorkbench(
                RegionLayout(),
                content={Region.CONTENT: content_factory, Region.ITEMS: items_factory},
                id="wl-workbench",
            )

    app = _App()
    async with app.run_test() as pilot:
        workbench = app.query_one(WatchlistsWorkbench)
        first_content = app.query_one("#reader-content", Label)
        first_items = app.query_one("#items-content", Label)
        assert str(first_content.renderable) == "content-1"

        await workbench.refresh_region_content(Region.CONTENT)
        await pilot.pause()

        refreshed_content = app.query_one("#reader-content", Label)
        assert str(refreshed_content.renderable) == "content-2", (
            "the factory should run again, reflecting whatever changed"
        )
        assert refreshed_content is not first_content, (
            "the old content widget should be replaced, not mutated in place"
        )

        still_items = app.query_one("#items-content", Label)
        assert still_items is first_items, (
            "an unrelated region's content must not be touched"
        )
        assert calls["content"] == 2
        assert len(items_widget_ids) == 1, "ITEMS's factory must not run again"


@pytest.mark.asyncio
async def test_refresh_region_content_is_a_noop_when_the_region_is_collapsed():
    from textual.widgets import Label

    calls = {"content": 0}

    def content_factory():
        calls["content"] += 1
        return Label("content", id="reader-content")

    class _App(App):
        def compose(self) -> ComposeResult:
            yield WatchlistsWorkbench(
                RegionLayout(collapsed=frozenset({Region.CONTENT})),
                content={Region.CONTENT: content_factory},
                id="wl-workbench",
            )

    app = _App()
    async with app.run_test():
        workbench = app.query_one(WatchlistsWorkbench)
        assert calls["content"] == 0, "a collapsed region should not build its content at all"

        await workbench.refresh_region_content(Region.CONTENT)

        assert calls["content"] == 0, "refreshing a collapsed region must be a no-op"


@pytest.mark.asyncio
async def test_refresh_region_content_keeps_a_non_self_headed_regions_title():
    """Fix round 1, Finding 3: `refresh_region_content` removed *all* of the
    region body's children and remounted only `factory()`.

    LEFT_RAIL supplies content (the tree) but is NOT in
    `SELF_HEADED_REGIONS`, so `_region_widget` prepends the generic
    "Watchlists" heading for it -- which the blanket remove then threw away,
    leaving an unlabelled bordered rail until the next region toggle happened
    to rebuild it. That is the exact defect `SELF_HEADED_REGIONS`' own
    comment records having shipped once already; both of the original tests
    used self-headed regions, so neither could see it.
    """
    from textual.widgets import Label

    def rail_factory():
        return Label("tree", id="rail-content")

    class _App(App):
        def compose(self) -> ComposeResult:
            yield WatchlistsWorkbench(
                RegionLayout(),
                content={Region.LEFT_RAIL: rail_factory},
                id="wl-workbench",
            )

    app = _App()
    async with app.run_test() as pilot:
        workbench = app.query_one(WatchlistsWorkbench)
        rail = app.query_one("#wl-region-left_rail")
        titles = [
            child for child in rail.children if child.has_class("watchlists-region-title")
        ]
        assert len(titles) == 1, "precondition: the rail starts with its heading"

        await workbench.refresh_region_content(Region.LEFT_RAIL)
        await pilot.pause()

        rail = app.query_one("#wl-region-left_rail")
        titles = [
            child for child in rail.children if child.has_class("watchlists-region-title")
        ]
        assert len(titles) == 1, (
            "refreshing a non-self-headed region must not strip its heading; "
            f"children are {[type(c).__name__ for c in rail.children]}"
        )
        assert str(titles[0].renderable) == REGION_TITLES[Region.LEFT_RAIL]
        assert app.query_one("#rail-content", Label), "the content was rebuilt"
        assert rail.children[0] is titles[0], (
            "the heading must stay above the content, not be remounted below it"
        )


@pytest.mark.asyncio
async def test_refresh_region_content_never_leaves_the_region_empty_on_a_build_failure():
    """Fix round 1, Finding 3 (companion): build the replacement *before*
    detaching the old content, so a factory that raises leaves the mounted
    pane standing instead of a bordered empty box.
    """
    from textual.widgets import Label

    calls = {"n": 0}

    def content_factory():
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("scope resolution blew up")
        return Label("content", id="reader-content")

    class _App(App):
        def compose(self) -> ComposeResult:
            yield WatchlistsWorkbench(
                RegionLayout(),
                content={Region.CONTENT: content_factory},
                id="wl-workbench",
            )

    app = _App()
    async with app.run_test() as pilot:
        workbench = app.query_one(WatchlistsWorkbench)

        with pytest.raises(RuntimeError):
            await workbench.refresh_region_content(Region.CONTENT)
        await pilot.pause()

        assert app.query_one("#reader-content", Label), (
            "a failed rebuild must leave the previous content mounted"
        )


# --- task-1344 fix wave (Qodo correctness): `refresh_header_content` -------
#
# The workbench's `header=` carries the section tab strip plus the snapshot
# summary on EVERY tab (`WatchlistsCollectionsScreen._build_centre_status_
# header` is wired unconditionally since task-2513). Nothing used to rebuild
# that header in place when its content went stale (e.g. the tree scope
# moving) -- these pin the primitive independent of the screen, the same way
# the `refresh_region_content` tests above do for a region.


@pytest.mark.asyncio
async def test_refresh_header_content_rebuilds_the_header_in_place():
    from textual.widgets import Label

    calls = {"n": 0}

    def header_factory():
        calls["n"] += 1
        return Label(f"header-{calls['n']}", id="wl-centre-status")

    class _App(App):
        def compose(self) -> ComposeResult:
            yield WatchlistsWorkbench(
                RegionLayout(),
                content={Region.ITEMS: lambda: Label("items", id="items-content")},
                hidden=frozenset({Region.CONTENT}),
                header=header_factory,
                id="wl-workbench",
            )

    app = _App()
    async with app.run_test() as pilot:
        workbench = app.query_one(WatchlistsWorkbench)
        first = app.query_one("#wl-centre-status", Label)
        assert str(first.renderable) == "header-1"

        await workbench.refresh_header_content()
        await pilot.pause()

        refreshed = app.query_one("#wl-centre-status", Label)
        assert str(refreshed.renderable) == "header-2", (
            "the factory should run again, reflecting whatever changed"
        )
        assert refreshed is not first, (
            "the old header widget should be replaced, not mutated in place"
        )
        assert calls["n"] == 2
        # An unrelated region's content must survive the header refresh.
        assert app.query_one("#items-content", Label)


@pytest.mark.asyncio
async def test_refresh_header_content_is_a_noop_without_a_header_factory():
    class _App(App):
        def compose(self) -> ComposeResult:
            yield WatchlistsWorkbench(RegionLayout(), id="wl-workbench")

    app = _App()
    async with app.run_test():
        workbench = app.query_one(WatchlistsWorkbench)
        assert not app.query("#wl-centre-status")

        await workbench.refresh_header_content()

        assert not app.query("#wl-centre-status"), (
            "refreshing with no header factory must not mount one"
        )
