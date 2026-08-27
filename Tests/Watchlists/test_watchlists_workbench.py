"""Focused contracts for the permanent-centre Watchlists workbench."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
from textual.app import App, ComposeResult
from textual.containers import Horizontal, HorizontalScroll, Vertical, VerticalScroll
from textual.widget import Widget
from textual.widgets import Button, Label, Static

from tldw_chatbook.UI.Watchlists_Modules import (
    watchlists_workbench as workbench_module,
)
from tldw_chatbook.UI.Watchlists_Modules.article_list import ArticleListPane
from tldw_chatbook.UI.Watchlists_Modules.content_pane import ContentPane
from tldw_chatbook.UI.Watchlists_Modules.pane_grip import (
    RegionToggled,
    WatchlistsPaneGrip,
)
from tldw_chatbook.UI.Watchlists_Modules.region_layout import (
    Region,
    RegionLayout,
    resolve_effective_layout,
)
from tldw_chatbook.UI.Watchlists_Modules.watchlists_workbench import (
    REGION_TITLES,
    RegionLayoutApplied,
    RegionLayoutApplyFailed,
    WatchlistsWorkbench,
)


class _WorkbenchApp(App[None]):
    def __init__(self, layout: RegionLayout, *, read_mode: bool = True) -> None:
        super().__init__()
        self.layout_state = layout
        self.read_mode = read_mode
        self.toggles: list[Region] = []
        self.layout_events: list[RegionLayoutApplied | RegionLayoutApplyFailed] = []

    def compose(self) -> ComposeResult:
        yield WatchlistsWorkbench(
            self.layout_state,
            read_mode=self.read_mode,
            id="wl-workbench",
        )

    def on_region_toggled(self, message: RegionToggled) -> None:
        self.toggles.append(message.region)

    def on_region_layout_applied(self, message: RegionLayoutApplied) -> None:
        self.layout_events.append(message)

    def on_region_layout_apply_failed(
        self, message: RegionLayoutApplyFailed
    ) -> None:
        self.layout_events.append(message)


def _direct_child_ids(widget) -> list[str | None]:
    return [child.id for child in widget.children]


_REAL_CSS_PATH = str(
    Path(__file__).resolve().parents[2]
    / "tldw_chatbook"
    / "css"
    / "tldw_cli_modular.tcss"
)


class _BoundaryGeometryApp(App[None]):
    """Real-bundle harness for responsive workbench geometry."""

    CSS_PATH = _REAL_CSS_PATH

    def __init__(self, width: int, *, read_mode: bool) -> None:
        super().__init__()
        self.read_mode = read_mode
        self.layout = resolve_effective_layout(
            RegionLayout(),
            width=width,
            read_mode=read_mode,
            article_focus=False,
            priority_target=None,
        )

    def compose(self) -> ComposeResult:
        yield WatchlistsWorkbench(
            self.layout,
            content={
                Region.LEFT_RAIL: lambda: Static("navigation"),
                Region.ITEMS: lambda: Static("management or feed items"),
                Region.CONTENT: lambda: Static("permanent reader"),
                Region.RIGHT_RAIL: lambda: Static("inspector"),
            },
            read_mode=self.read_mode,
            id="wl-workbench",
        )


def _expected_boundary_child_ids(
    layout: RegionLayout, *, read_mode: bool
) -> list[str]:
    expected: list[str] = []
    if not layout.is_collapsed(Region.LEFT_RAIL):
        expected.append("wl-region-left_rail")
    expected.append("wl-grip-left_rail")
    if read_mode:
        if not layout.is_collapsed(Region.ITEMS):
            expected.append("wl-region-items")
        expected.extend(("wl-grip-items", "wl-region-content"))
    else:
        expected.append("wl-region-items")
    expected.append("wl-grip-right_rail")
    if not layout.is_collapsed(Region.RIGHT_RAIL):
        expected.append("wl-region-right_rail")
    return expected


def test_region_titles_cover_exactly_the_live_regions() -> None:
    assert set(REGION_TITLES) == set(Region)


@pytest.mark.parametrize(
    ("width", "centre_width"),
    [(161, 44), (206, 89), (220, 103), (224, 107)],
)
@pytest.mark.asyncio
async def test_read_real_bundle_fills_wide_body_exactly(
    width: int, centre_width: int
) -> None:
    app = _BoundaryGeometryApp(width, read_mode=True)

    async with app.run_test(size=(width, 24)) as pilot:
        await pilot.pause()
        assert app.layout.collapsed == frozenset()
        expected = {
            ".watchlists-region-left_rail": (28, 24),
            ".watchlists-read-mode .watchlists-region-items": (40, 32),
            ".watchlists-region-right_rail": (34, 30),
        }
        for selector, (target, minimum) in expected.items():
            pane = app.query_one(selector)
            assert pane.styles.min_width is not None
            assert pane.styles.min_width.value == minimum
            assert pane.styles.max_width is not None
            assert pane.styles.max_width.value == target
            assert pane.region.width == target
        _assert_real_bundle_geometry(
            app,
            width=width,
            read_mode=True,
            expected_widths={
                "wl-region-left_rail": 28,
                "wl-region-items": 40,
                "wl-region-content": centre_width,
                "wl-region-right_rail": 34,
            },
        )


def _assert_real_bundle_geometry(
    app: _BoundaryGeometryApp,
    *,
    width: int,
    read_mode: bool,
    expected_widths: dict[str, int],
) -> None:
    """Assert one mounted real-bundle frame is contained and non-overlapping."""
    body = app.query_one("#wl-workbench-body", Horizontal)
    workbench = app.query_one("#wl-workbench", WatchlistsWorkbench)
    children = list(body.children)
    expected_order = _expected_boundary_child_ids(app.layout, read_mode=read_mode)

    assert _direct_child_ids(body) == expected_order
    assert workbench.content_region.contains_region(body.region)
    assert body.region.width == workbench.content_region.width == width
    assert body.region.height == workbench.content_region.height
    assert body.max_scroll_x == 0
    assert body.virtual_size.width == body.content_region.width
    assert children[-1].region.right == body.content_region.right
    assert all(body.content_region.contains_region(child.region) for child in children)
    assert all(
        left.region.right <= right.region.x
        for left, right in zip(children, children[1:])
    )

    grips = list(body.query(WatchlistsPaneGrip))
    assert len(grips) == (3 if read_mode else 2)
    assert all(grip.outer_size.width == grip.region.width == 5 for grip in grips)
    assert all(grip.region.height == body.content_region.height for grip in grips)

    centre_id = "wl-region-content" if read_mode else "wl-region-items"
    centre = app.query_one(f"#{centre_id}")
    assert centre.styles.min_width is not None
    side_regions = (
        (Region.LEFT_RAIL, Region.ITEMS, Region.RIGHT_RAIL)
        if read_mode
        else (Region.LEFT_RAIL, Region.RIGHT_RAIL)
    )
    has_expanded_side_pane = any(
        not app.layout.is_collapsed(region) for region in side_regions
    )
    assert workbench.has_class("watchlists-has-expanded-side-pane") is (
        has_expanded_side_pane
    )
    assert centre.styles.min_width.value == (
        44 if has_expanded_side_pane else 0
    )
    assert centre.region.width == body.content_region.width - sum(
        child.region.width for child in children if child is not centre
    )
    assert centre.region.width > 0
    assert centre.region.height == body.content_region.height
    actual_widths = {child.id: child.region.width for child in children}
    assert {
        node_id: actual_widths[node_id] for node_id in expected_widths
    } == expected_widths


@pytest.mark.parametrize(
    ("width", "collapsed", "expected_widths"),
    [
        (
            145,
            frozenset(),
            {
                "wl-region-left_rail": 24,
                "wl-region-items": 32,
                "wl-region-content": 44,
                "wl-region-right_rail": 30,
            },
        ),
        (144, frozenset({Region.RIGHT_RAIL}), {}),
        (
            115,
            frozenset({Region.RIGHT_RAIL}),
            {
                "wl-region-left_rail": 24,
                "wl-region-items": 32,
                "wl-region-content": 44,
            },
        ),
        (114, frozenset({Region.LEFT_RAIL, Region.RIGHT_RAIL}), {}),
        (
            91,
            frozenset({Region.LEFT_RAIL, Region.RIGHT_RAIL}),
            {"wl-region-items": 32, "wl-region-content": 44},
        ),
        (90, frozenset({Region.LEFT_RAIL, Region.ITEMS, Region.RIGHT_RAIL}), {}),
        (60, frozenset({Region.LEFT_RAIL, Region.ITEMS, Region.RIGHT_RAIL}), {}),
        (
            40,
            frozenset({Region.LEFT_RAIL, Region.ITEMS, Region.RIGHT_RAIL}),
            {"wl-region-content": 25},
        ),
    ],
)
@pytest.mark.asyncio
async def test_read_real_bundle_threshold_geometry(
    width: int,
    collapsed: frozenset[Region],
    expected_widths: dict[str, int],
) -> None:
    app = _BoundaryGeometryApp(width, read_mode=True)

    async with app.run_test(size=(width, 24)) as pilot:
        await pilot.pause()
        assert app.layout.collapsed == collapsed
        _assert_real_bundle_geometry(
            app,
            width=width,
            read_mode=True,
            expected_widths=expected_widths,
        )


@pytest.mark.parametrize(
    ("width", "collapsed", "expected_widths"),
    [
        (
            108,
            frozenset(),
            {
                "wl-region-left_rail": 24,
                "wl-region-items": 44,
                "wl-region-right_rail": 30,
            },
        ),
        (107, frozenset({Region.RIGHT_RAIL}), {}),
        (
            78,
            frozenset({Region.RIGHT_RAIL}),
            {"wl-region-left_rail": 24, "wl-region-items": 44},
        ),
        (77, frozenset({Region.LEFT_RAIL, Region.RIGHT_RAIL}), {}),
        (76, frozenset({Region.LEFT_RAIL, Region.RIGHT_RAIL}), {}),
        (
            40,
            frozenset({Region.LEFT_RAIL, Region.RIGHT_RAIL}),
            {"wl-region-items": 30},
        ),
    ],
)
@pytest.mark.asyncio
async def test_management_real_bundle_threshold_geometry(
    width: int,
    collapsed: frozenset[Region],
    expected_widths: dict[str, int],
) -> None:
    app = _BoundaryGeometryApp(width, read_mode=False)

    async with app.run_test(size=(width, 24)) as pilot:
        await pilot.pause()
        assert app.layout.collapsed == collapsed
        _assert_real_bundle_geometry(
            app,
            width=width,
            read_mode=False,
            expected_widths=expected_widths,
        )


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
async def test_explicit_management_mode_selects_management_horizontal_body() -> None:
    class _App(App[None]):
        def compose(self) -> ComposeResult:
            yield WatchlistsWorkbench(
                RegionLayout(collapsed=frozenset({Region.ITEMS})),
                read_mode=False,
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

        workbench.request_region_layout(
            RegionLayout(collapsed=frozenset({Region.LEFT_RAIL})), token=1
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

        workbench.request_region_layout(RegionLayout(), token=1)
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
async def test_expansion_factory_failure_keeps_collapsed_grip_and_dom(
    monkeypatch,
) -> None:
    fail_rail = True
    logged: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        workbench_module.logger,
        "bind",
        lambda **context: SimpleNamespace(
            exception=lambda message: logged.append((message, context))
        ),
    )

    def rail_factory() -> Widget:
        if fail_rail:
            raise RuntimeError("rail build failed")
        return Label("navigation")

    class _App(App[None]):
        def compose(self) -> ComposeResult:
            yield WatchlistsWorkbench(
                RegionLayout(
                    collapsed=frozenset(
                        {Region.LEFT_RAIL, Region.ITEMS, Region.RIGHT_RAIL}
                    )
                ),
                content={Region.LEFT_RAIL: rail_factory},
                read_mode=True,
                id="wl-workbench",
            )

    app = _App()
    async with app.run_test() as pilot:
        workbench = app.query_one(WatchlistsWorkbench)
        grip = app.query_one("#wl-grip-left_rail", WatchlistsPaneGrip)
        reader = app.query_one("#wl-region-content")
        assert not workbench.has_class("watchlists-has-expanded-side-pane")

        expanded_left = RegionLayout(
            collapsed=frozenset({Region.ITEMS, Region.RIGHT_RAIL})
        )
        workbench.request_region_layout(expanded_left, token=1)
        await pilot.pause()

        assert workbench.region_layout.is_collapsed(Region.LEFT_RAIL)
        assert app.query_one("#wl-grip-left_rail") is grip
        assert grip.expanded is False
        assert not app.query("#wl-region-left_rail")
        assert app.query_one("#wl-region-content") is reader
        assert not workbench.has_class("watchlists-has-expanded-side-pane")
        assert app.is_running
        assert logged == [
            (
                "Watchlists pane expansion factory failed",
                {"token": 1, "read_mode": True, "regions": ("left_rail",)},
            )
        ]

        fail_rail = False
        workbench.request_region_layout(expanded_left, token=2)
        await pilot.pause()
        assert app.query("#wl-region-left_rail")
        assert grip.expanded is True
        assert workbench.has_class("watchlists-has-expanded-side-pane")


@pytest.mark.asyncio
async def test_expansion_mount_failure_logs_request_context(monkeypatch) -> None:
    collapsed = RegionLayout(collapsed=frozenset({Region.LEFT_RAIL}))
    logged: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        workbench_module.logger,
        "bind",
        lambda **context: SimpleNamespace(
            exception=lambda message: logged.append((message, context))
        ),
    )
    app = _WorkbenchApp(collapsed)
    async with app.run_test() as pilot:
        workbench = app.query_one(WatchlistsWorkbench)

        async def fail_mount(*_widgets, **_kwargs) -> None:
            raise RuntimeError("mount failed")

        monkeypatch.setattr(app.query_one("#wl-workbench-body"), "mount", fail_mount)
        workbench.request_region_layout(RegionLayout(), token=9)
        await pilot.pause()

        assert workbench.region_layout == collapsed
        assert logged == [
            (
                "Watchlists pane expansion mount failed",
                {"token": 9, "read_mode": True, "regions": ("left_rail",)},
            )
        ]


@pytest.mark.asyncio
async def test_layout_requests_acknowledge_same_layout_with_exact_token() -> None:
    layout = RegionLayout()
    app = _WorkbenchApp(layout)
    async with app.run_test() as pilot:
        workbench = app.query_one(WatchlistsWorkbench)

        workbench.request_region_layout(layout, token=41)
        workbench.request_region_layout(layout, token=42)
        await pilot.pause()

        applied = [
            event for event in app.layout_events if isinstance(event, RegionLayoutApplied)
        ]
        assert [event.token for event in applied] == [41, 42]
        assert all(event.previous == layout == event.layout for event in applied)


@pytest.mark.asyncio
async def test_failed_layout_request_reports_its_exact_token() -> None:
    def broken_factory() -> Widget:
        raise RuntimeError("rail build failed")

    collapsed = RegionLayout(collapsed=frozenset({Region.LEFT_RAIL}))

    class _App(_WorkbenchApp):
        def compose(self) -> ComposeResult:
            yield WatchlistsWorkbench(
                collapsed,
                content={Region.LEFT_RAIL: broken_factory},
                read_mode=True,
                id="wl-workbench",
            )

    app = _App(collapsed)
    async with app.run_test() as pilot:
        workbench = app.query_one(WatchlistsWorkbench)

        workbench.request_region_layout(RegionLayout(), token=73)
        await pilot.pause()

        failures = [
            event
            for event in app.layout_events
            if isinstance(event, RegionLayoutApplyFailed)
        ]
        assert len(failures) == 1
        assert failures[0].token == 73
        assert failures[0].attempted == RegionLayout()
        assert failures[0].fallback == collapsed


@pytest.mark.asyncio
async def test_grip_activation_uses_shared_region_toggled_message() -> None:
    app = _WorkbenchApp(RegionLayout())
    async with app.run_test() as pilot:
        await pilot.click("#wl-grip-items")
        await pilot.pause()
    assert app.toggles == [Region.ITEMS]


def test_transitional_workbench_adapters_are_removed() -> None:
    constructor = inspect.signature(WatchlistsWorkbench.__init__)
    apply_view = inspect.signature(WatchlistsWorkbench.apply_section_view)
    assert "hidden" not in constructor.parameters
    assert "collapsed_suffixes" not in constructor.parameters
    assert "hidden" not in apply_view.parameters
    assert not hasattr(WatchlistsWorkbench, "set_collapsed_suffixes")


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
                read_mode=True,
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
                read_mode=True,
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
                read_mode=True,
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
                read_mode=True,
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
            token=1,
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
            read_mode=True,
            layout=RegionLayout(),
            token=2,
            rebuild_regions=(Region.ITEMS,),
        )
        await pilot.pause()

        assert app.query_one("#wl-region-left_rail") is left
        assert app.query_one("#wl-region-right_rail") is right
        assert app.query("#wl-grip-items")
        assert app.query_one("#content-content") is not reader
        assert calls == {"items": 3, "content": 2}


@pytest.mark.asyncio
async def test_mode_switch_factory_failure_preserves_previous_read_view(
    monkeypatch,
) -> None:
    fail_items = False
    observed_read_classes: list[bool] = []
    logged: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        workbench_module.logger,
        "bind",
        lambda **context: SimpleNamespace(
            exception=lambda message: logged.append((message, context))
        ),
    )

    def items_factory() -> Label:
        if fail_items:
            observed_read_classes.append(
                app.query_one(WatchlistsWorkbench).has_class("watchlists-read-mode")
            )
            raise RuntimeError("management centre failed")
        return Label("feed items", id="items-content")

    class _App(App[None]):
        def compose(self) -> ComposeResult:
            yield WatchlistsWorkbench(
                RegionLayout(),
                content={
                    Region.ITEMS: items_factory,
                    Region.CONTENT: lambda: Label("reader", id="reader-content"),
                },
                read_mode=True,
                id="wl-workbench",
            )

    app = _App()
    async with app.run_test() as pilot:
        workbench = app.query_one(WatchlistsWorkbench)
        body = app.query_one("#wl-workbench-body", Horizontal)
        previous_ids = _direct_child_ids(body)
        items = app.query_one("#items-content")
        reader = app.query_one("#reader-content")
        fail_items = True

        applied = await workbench.apply_section_view(
            read_mode=False,
            layout=RegionLayout(),
            token=7,
            rebuild_regions=(Region.ITEMS,),
        )
        await pilot.pause()

        assert applied is False
        assert workbench.read_mode is True
        assert workbench.has_class("watchlists-read-mode")
        assert _direct_child_ids(body) == previous_ids
        assert app.query_one("#items-content") is items
        assert app.query_one("#reader-content") is reader
        assert app.is_running
        assert observed_read_classes == [False], (
            "the target mode class must be active while its body is reconciled, "
            "then rolled back with the previous view when the factory fails"
        )
        assert logged == [
            (
                "Watchlists section-view factory failed",
                {"token": 7, "read_mode": False, "regions": ("items",)},
            )
        ]

        fail_items = False
        applied = await workbench.apply_section_view(
            read_mode=False,
            layout=RegionLayout(),
            token=8,
            rebuild_regions=(Region.ITEMS,),
        )
        assert applied is True
        assert workbench.read_mode is False
        assert not workbench.has_class("watchlists-read-mode")
        assert not app.query("#reader-content")


@pytest.mark.asyncio
async def test_read_mode_class_tracks_incremental_mode_switches() -> None:
    rails_collapsed = RegionLayout(
        collapsed=frozenset({Region.LEFT_RAIL, Region.RIGHT_RAIL})
    )
    app = _WorkbenchApp(rails_collapsed, read_mode=True)
    async with app.run_test():
        workbench = app.query_one(WatchlistsWorkbench)
        assert workbench.has_class("watchlists-read-mode")
        assert workbench.has_class("watchlists-has-expanded-side-pane")

        await workbench.apply_section_view(
            read_mode=False,
            layout=rails_collapsed,
            token=1,
        )
        assert not workbench.has_class("watchlists-read-mode")
        assert not workbench.has_class("watchlists-has-expanded-side-pane")

        await workbench.apply_section_view(
            read_mode=True,
            layout=rails_collapsed,
            token=2,
        )
        assert workbench.has_class("watchlists-read-mode")
        assert workbench.has_class("watchlists-has-expanded-side-pane")


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
        grips = tuple(app.query(WatchlistsPaneGrip))
        fixed = (
            actions.region,
            footer.region,
            left.region,
            items.region,
            right.region,
            *(grip.region for grip in grips),
        )

        assert body_scroll.max_scroll_y > 0
        body_scroll.scroll_end(animate=False)
        await pilot.pause()

        assert body_scroll.scroll_y == body_scroll.max_scroll_y
        assert (
            actions.region,
            footer.region,
            left.region,
            items.region,
            right.region,
            *(grip.region for grip in grips),
        ) == fixed


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
async def test_reader_empty_state_uses_the_approved_copy() -> None:
    class _App(App[None]):
        def compose(self) -> ComposeResult:
            yield ContentPane(id="watchlists-content-pane")

    app = _App()
    async with app.run_test():
        empty = app.query_one("#content-empty", Static)

        assert str(empty.renderable) == "Select a feed to display it here."


@pytest.mark.asyncio
async def test_reader_exposes_only_core_actions_in_the_approved_order() -> None:
    app = _ReadGeometryApp()
    async with app.run_test(size=(180, 50)):
        actions = app.query_one("#content-actions", HorizontalScroll)
        footer = app.query_one("#content-footer", Horizontal)

        assert _direct_child_ids(actions) == [
            "content-star-button",
            "content-mark-unread-button",
            "content-open-button",
        ]
        assert _direct_child_ids(footer) == [
            "content-position",
            "content-next-unread-button",
        ]


@pytest.mark.asyncio
async def test_reader_focus_order_keeps_body_between_actions_and_footer() -> None:
    app = _ReadGeometryApp()
    async with app.run_test(size=(180, 50)) as pilot:
        await pilot.pause()
        body_scroll = app.query_one("#content-body-scroll", VerticalScroll)
        first_action = app.query_one("#content-star-button")
        footer_action = app.query_one("#content-next-unread-button")
        focus_chain = app.screen.focus_chain

        assert focus_chain.index(first_action) < focus_chain.index(body_scroll)
        assert focus_chain.index(body_scroll) < focus_chain.index(footer_action)
