"""Focused contracts for the permanent-centre Watchlists workbench."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.containers import Horizontal, HorizontalScroll, Vertical, VerticalScroll
from textual.widgets import Button, Label, Static

from tldw_chatbook.UI.Watchlists_Modules.article_list import ArticleListPane
from tldw_chatbook.UI.Watchlists_Modules.content_pane import ContentPane
from tldw_chatbook.UI.Watchlists_Modules.pane_grip import (
    RegionToggled,
    WatchlistsPaneGrip,
)
from tldw_chatbook.UI.Watchlists_Modules.region_layout import Region, RegionLayout
from tldw_chatbook.UI.Watchlists_Modules.watchlists_workbench import (
    REGION_TITLES,
    WatchlistsWorkbench,
)


class _WorkbenchApp(App[None]):
    def __init__(self, layout: RegionLayout, *, read_mode: bool = True) -> None:
        super().__init__()
        self.layout_state = layout
        self.read_mode = read_mode
        self.toggles: list[Region] = []

    def compose(self) -> ComposeResult:
        yield WatchlistsWorkbench(
            self.layout_state,
            read_mode=self.read_mode,
            id="wl-workbench",
        )

    def on_region_toggled(self, message: RegionToggled) -> None:
        self.toggles.append(message.region)


def _direct_child_ids(widget) -> list[str | None]:
    return [child.id for child in widget.children]


def test_region_titles_cover_exactly_the_live_regions() -> None:
    assert set(REGION_TITLES) == set(Region)


@pytest.mark.parametrize(
    ("selector", "target", "minimum"),
    [
        (".watchlists-region-left_rail", 28, 24),
        (".watchlists-read-mode .watchlists-region-items", 40, 32),
        (".watchlists-region-right_rail", 34, 30),
    ],
)
def test_side_pane_css_keeps_approved_target_and_minimum_widths(
    selector: str, target: int, minimum: int
) -> None:
    css_path = (
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook"
        / "css"
        / "features"
        / "_watchlists.tcss"
    )
    block = css_path.read_text().split(f"{selector} {{", 1)[1].split("}", 1)[0]

    assert f"\n    width: {target};" in block
    assert f"\n    min-width: {minimum};" in block


@pytest.mark.asyncio
async def test_read_mounts_header_above_exact_horizontal_body_order() -> None:
    class _App(App[None]):
        def compose(self) -> ComposeResult:
            yield WatchlistsWorkbench(
                RegionLayout(),
                read_mode=True,
                header=lambda: Static("status", id="wl-centre-status"),
                id="wl-workbench",
            )

    app = _App()
    async with app.run_test():
        workbench = app.query_one(WatchlistsWorkbench)
        body = app.query_one("#wl-workbench-body", Horizontal)

        assert _direct_child_ids(workbench) == [
            "wl-centre-status",
            "wl-workbench-body",
        ]
        assert _direct_child_ids(body) == [
            "wl-region-left_rail",
            "wl-grip-left_rail",
            "wl-region-items",
            "wl-grip-items",
            "wl-region-content",
            "wl-grip-right_rail",
            "wl-region-right_rail",
        ]


@pytest.mark.parametrize(
    ("region", "body_id", "grip_id"),
    [
        (Region.LEFT_RAIL, "wl-region-left_rail", "wl-grip-left_rail"),
        (Region.ITEMS, "wl-region-items", "wl-grip-items"),
        (Region.RIGHT_RAIL, "wl-region-right_rail", "wl-grip-right_rail"),
    ],
)
@pytest.mark.asyncio
async def test_read_collapses_only_side_body_and_keeps_its_grip(
    region: Region, body_id: str, grip_id: str
) -> None:
    app = _WorkbenchApp(RegionLayout(collapsed=frozenset({region})))
    async with app.run_test():
        assert not app.query(f"#{body_id}")
        grip = app.query_one(f"#{grip_id}", WatchlistsPaneGrip)
        assert grip.expanded is False
        assert app.query("#wl-region-content")


@pytest.mark.asyncio
async def test_legacy_content_collapse_cannot_unmount_reader_in_read() -> None:
    app = _WorkbenchApp(
        RegionLayout(collapsed=frozenset({Region.CONTENT}))
    )
    async with app.run_test():
        assert app.query("#wl-region-content")
        assert not app.query("#wl-grip-content")


@pytest.mark.asyncio
async def test_hidden_compatibility_selects_management_horizontal_body() -> None:
    class _App(App[None]):
        def compose(self) -> ComposeResult:
            yield WatchlistsWorkbench(
                RegionLayout(collapsed=frozenset({Region.ITEMS})),
                hidden=frozenset({Region.CONTENT}),
                header=lambda: Static("status", id="wl-centre-status"),
                id="wl-workbench",
            )

    app = _App()
    async with app.run_test():
        workbench = app.query_one(WatchlistsWorkbench)
        body = app.query_one("#wl-workbench-body", Horizontal)
        assert _direct_child_ids(workbench) == [
            "wl-centre-status",
            "wl-workbench-body",
        ]
        assert _direct_child_ids(body) == [
            "wl-region-left_rail",
            "wl-grip-left_rail",
            "wl-region-items",
            "wl-grip-right_rail",
            "wl-region-right_rail",
        ]
        assert workbench.read_mode is False
        assert app.query("#wl-region-items")
        assert not app.query("#wl-grip-items")
        assert not app.query("#wl-region-content")


@pytest.mark.asyncio
async def test_side_toggle_preserves_unaffected_bodies_all_grips_and_reader() -> None:
    app = _WorkbenchApp(RegionLayout())
    async with app.run_test() as pilot:
        workbench = app.query_one(WatchlistsWorkbench)
        preserved = {
            selector: app.query_one(selector)
            for selector in (
                "#wl-region-items",
                "#wl-region-content",
                "#wl-region-right_rail",
                "#wl-grip-left_rail",
                "#wl-grip-items",
                "#wl-grip-right_rail",
            )
        }

        workbench.region_layout = RegionLayout(
            collapsed=frozenset({Region.LEFT_RAIL})
        )
        await pilot.pause()

        assert not app.query("#wl-region-left_rail")
        for selector, instance in preserved.items():
            assert app.query_one(selector) is instance
        assert preserved["#wl-grip-left_rail"].expanded is False


@pytest.mark.asyncio
async def test_expanding_inspector_mounts_body_after_permanent_grip() -> None:
    app = _WorkbenchApp(
        RegionLayout(collapsed=frozenset({Region.RIGHT_RAIL}))
    )
    async with app.run_test() as pilot:
        workbench = app.query_one(WatchlistsWorkbench)
        grip = app.query_one("#wl-grip-right_rail")
        reader = app.query_one("#wl-region-content")

        workbench.region_layout = RegionLayout()
        await pilot.pause()

        body = app.query_one("#wl-workbench-body", Horizontal)
        assert _direct_child_ids(body)[-3:] == [
            "wl-region-content",
            "wl-grip-right_rail",
            "wl-region-right_rail",
        ]
        assert app.query_one("#wl-grip-right_rail") is grip
        assert app.query_one("#wl-region-content") is reader


@pytest.mark.asyncio
async def test_grip_activation_uses_shared_region_toggled_message() -> None:
    app = _WorkbenchApp(RegionLayout())
    async with app.run_test() as pilot:
        await pilot.click("#wl-grip-items")
        await pilot.pause()
    assert app.toggles == [Region.ITEMS]


@pytest.mark.asyncio
async def test_collapsed_suffix_compatibility_is_a_noop() -> None:
    class _App(App[None]):
        def compose(self) -> ComposeResult:
            yield WatchlistsWorkbench(
                RegionLayout(collapsed=frozenset({Region.LEFT_RAIL})),
                collapsed_suffixes={Region.LEFT_RAIL: "12 unread"},
                id="wl-workbench",
            )

    app = _App()
    async with app.run_test():
        workbench = app.query_one(WatchlistsWorkbench)
        grip = app.query_one("#wl-grip-left_rail", WatchlistsPaneGrip)
        label = str(grip.label)
        workbench.set_collapsed_suffixes({Region.LEFT_RAIL: "7 unread"})
        assert app.query_one("#wl-grip-left_rail") is grip
        assert str(grip.label) == label


@pytest.mark.asyncio
async def test_refresh_region_content_rebuilds_only_named_factory_output() -> None:
    calls = {"items": 0, "content": 0}

    def factory(name: str):
        def build() -> Label:
            calls[name] += 1
            return Label(f"{name}-{calls[name]}", id=f"{name}-content")

        return build

    class _App(App[None]):
        def compose(self) -> ComposeResult:
            yield WatchlistsWorkbench(
                RegionLayout(),
                content={
                    Region.ITEMS: factory("items"),
                    Region.CONTENT: factory("content"),
                },
                id="wl-workbench",
            )

    app = _App()
    async with app.run_test() as pilot:
        workbench = app.query_one(WatchlistsWorkbench)
        old_items = app.query_one("#items-content")
        old_content = app.query_one("#content-content")

        await workbench.refresh_region_content(Region.CONTENT)
        await pilot.pause()

        assert app.query_one("#items-content") is old_items
        assert app.query_one("#content-content") is not old_content
        assert calls == {"items": 1, "content": 2}


@pytest.mark.asyncio
async def test_refresh_region_content_preserves_heading_on_factory_failure() -> None:
    calls = 0

    def factory() -> Label:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("build failed")
        return Label("tree", id="rail-content")

    class _App(App[None]):
        def compose(self) -> ComposeResult:
            yield WatchlistsWorkbench(
                RegionLayout(),
                content={Region.LEFT_RAIL: factory},
                id="wl-workbench",
            )

    app = _App()
    async with app.run_test():
        workbench = app.query_one(WatchlistsWorkbench)
        region = app.query_one("#wl-region-left_rail")
        heading = region.query_one(".watchlists-region-title")
        content = app.query_one("#rail-content")

        with pytest.raises(RuntimeError, match="build failed"):
            await workbench.refresh_region_content(Region.LEFT_RAIL)

        assert app.query_one("#wl-region-left_rail") is region
        assert region.query_one(".watchlists-region-title") is heading
        assert app.query_one("#rail-content") is content


@pytest.mark.asyncio
async def test_refresh_header_replaces_only_header_and_is_failure_safe() -> None:
    calls = 0
    fail = False

    def header() -> Label:
        nonlocal calls
        calls += 1
        if fail:
            raise RuntimeError("header failed")
        return Label(f"header-{calls}", id="wl-centre-status")

    class _App(App[None]):
        def compose(self) -> ComposeResult:
            yield WatchlistsWorkbench(
                RegionLayout(),
                header=header,
                id="wl-workbench",
            )

    app = _App()
    async with app.run_test() as pilot:
        workbench = app.query_one(WatchlistsWorkbench)
        body = app.query_one("#wl-workbench-body")
        old_header = app.query_one("#wl-centre-status")
        await workbench.refresh_header_content()
        await pilot.pause()
        fresh_header = app.query_one("#wl-centre-status")
        assert fresh_header is not old_header
        assert app.query_one("#wl-workbench-body") is body

        fail = True
        with pytest.raises(RuntimeError, match="header failed"):
            await workbench.refresh_header_content()
        assert app.query_one("#wl-centre-status") is fresh_header
        assert app.query_one("#wl-workbench-body") is body


@pytest.mark.asyncio
async def test_apply_section_view_rebuilds_only_required_centres() -> None:
    calls = {"items": 0, "content": 0}

    def factory(name: str):
        def build() -> Label:
            calls[name] += 1
            return Label(f"{name}-{calls[name]}", id=f"{name}-content")

        return build

    class _App(App[None]):
        def compose(self) -> ComposeResult:
            yield WatchlistsWorkbench(
                RegionLayout(),
                content={
                    Region.ITEMS: factory("items"),
                    Region.CONTENT: factory("content"),
                },
                id="wl-workbench",
            )

    app = _App()
    async with app.run_test() as pilot:
        workbench = app.query_one(WatchlistsWorkbench)
        left = app.query_one("#wl-region-left_rail")
        right = app.query_one("#wl-region-right_rail")
        left_grip = app.query_one("#wl-grip-left_rail")
        right_grip = app.query_one("#wl-grip-right_rail")
        reader = app.query_one("#content-content")

        await workbench.apply_section_view(
            read_mode=False,
            layout=RegionLayout(),
            rebuild_regions=(Region.ITEMS,),
        )
        await pilot.pause()

        assert app.query_one(WatchlistsWorkbench) is workbench
        assert app.query_one("#wl-region-left_rail") is left
        assert app.query_one("#wl-region-right_rail") is right
        assert app.query_one("#wl-grip-left_rail") is left_grip
        assert app.query_one("#wl-grip-right_rail") is right_grip
        assert not app.query("#wl-grip-items")
        assert not app.query("#wl-region-content")
        assert calls == {"items": 2, "content": 1}

        await workbench.apply_section_view(
            hidden=frozenset(),
            layout=RegionLayout(),
            rebuild_regions=(Region.ITEMS,),
        )
        await pilot.pause()

        assert app.query_one("#wl-region-left_rail") is left
        assert app.query_one("#wl-region-right_rail") is right
        assert app.query("#wl-grip-items")
        assert app.query_one("#content-content") is not reader
        assert calls == {"items": 3, "content": 2}


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


class _ReadGeometryApp(App[None]):
    """Production-CSS harness for Reader-local containment."""

    CSS_PATH = str(
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook"
        / "css"
        / "tldw_cli_modular.tcss"
    )

    def _items_pane(self) -> Vertical:
        pane = ArticleListPane(id="watchlists-items-pane")
        pane.items = [_article(index) for index in range(3)]
        return Vertical(
            Static("Read", id="watchlists-detail-title"),
            pane,
            id="watchlists-detail-pane",
        )

    def _content_pane(self) -> ContentPane:
        pane = ContentPane(id="watchlists-content-pane")
        pane.item = {
            "id": "local:watchlist_item:reader",
            "item_id": "reader",
            "title": "Geometry article",
            "source_name": "Geometry Feed",
            "content_kind": "article",
            "content_format": "text",
            "content": "\n\n".join(f"paragraph-{row:02d}" for row in range(80)),
        }
        pane.position = "1 of 1"
        return pane

    def compose(self) -> ComposeResult:
        yield WatchlistsWorkbench(
            RegionLayout(),
            content={
                Region.LEFT_RAIL: lambda: Static("fixed-left"),
                Region.ITEMS: self._items_pane,
                Region.CONTENT: self._content_pane,
                Region.RIGHT_RAIL: lambda: Static("fixed-right"),
            },
            read_mode=True,
            id="wl-workbench",
            classes="watchlists-read-mode",
        )


@pytest.mark.asyncio
async def test_reader_body_scroll_preserves_local_actions_footer_and_neighbours() -> None:
    app = _ReadGeometryApp()
    async with app.run_test(size=(180, 50)) as pilot:
        await pilot.pause()
        body_scroll = app.query_one("#content-body-scroll", VerticalScroll)
        actions = app.query_one("#content-actions")
        footer = app.query_one("#content-footer")
        left = app.query_one("#wl-region-left_rail")
        items = app.query_one("#wl-region-items")
        right = app.query_one("#wl-region-right_rail")
        fixed = (actions.region, footer.region, left.region, items.region, right.region)

        assert body_scroll.max_scroll_y > 0
        body_scroll.scroll_end(animate=False)
        await pilot.pause()

        assert body_scroll.scroll_y == body_scroll.max_scroll_y
        assert (actions.region, footer.region, left.region, items.region, right.region) == fixed


@pytest.mark.asyncio
async def test_reader_actions_and_footer_remain_inside_reader() -> None:
    app = _ReadGeometryApp()
    async with app.run_test(size=(180, 50)) as pilot:
        await pilot.pause()
        reader = app.query_one("#wl-region-content")
        actions = app.query_one("#content-actions", HorizontalScroll)
        footer = app.query_one("#content-footer")

        assert reader.content_region.contains_region(actions.region)
        assert reader.content_region.contains_region(footer.region)
        assert actions.region.height == 1
        for button in actions.query(Button):
            assert actions.content_region.contains_region(button.region)


@pytest.mark.asyncio
async def test_reader_focus_order_keeps_body_between_actions_and_footer() -> None:
    app = _ReadGeometryApp()
    async with app.run_test(size=(180, 50)) as pilot:
        await pilot.pause()
        body_scroll = app.query_one("#content-body-scroll", VerticalScroll)
        first_action = app.query_one("#content-mark-unread-button")
        footer_action = app.query_one("#content-next-unread-button")
        focus_chain = app.screen.focus_chain

        assert focus_chain.index(first_action) < focus_chain.index(body_scroll)
        assert focus_chain.index(body_scroll) < focus_chain.index(footer_action)
