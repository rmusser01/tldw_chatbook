"""Production compositor contract for Console's edge-owned workbench rails."""

from __future__ import annotations

import pytest
from textual.widget import Widget

from Tests.UI.test_console_shell_regions import make_console_pilot


_LEFT_OWNERS = ("#console-left-rail", "#console-context-rail-handle")
_RIGHT_OWNERS = ("#console-right-rail", "#console-inspector-rail-handle")


def _visible(screen, selectors: tuple[str, ...]) -> Widget:
    candidates = [screen.query_one(selector) for selector in selectors]
    visible = [candidate for candidate in candidates if candidate.display]
    assert len(visible) == 1
    return visible[0]


def _border_kinds(widget: Widget) -> tuple[str, str, str, str]:
    return tuple(
        kind or "none"
        for kind in (
            widget.styles.border_top[0],
            widget.styles.border_right[0],
            widget.styles.border_bottom[0],
            widget.styles.border_left[0],
        )
    )


def _hit_belongs_to(screen, x: int, y: int, owner: Widget) -> bool:
    hit = screen.get_widget_at(x, y)[0]
    return hit is owner or owner in hit.ancestors


def _geometry_snapshot(screen) -> dict[str, object]:
    selectors = (
        "#console-workspace-grid",
        "#console-left-rail",
        "#console-context-rail-handle",
        "#console-main-column",
        "#console-transcript-region",
        "#console-right-rail",
        "#console-inspector-rail-handle",
    )
    return {
        selector: (
            screen.query_one(selector).region,
            screen.query_one(selector).content_region,
            screen.query_one(selector).display,
        )
        for selector in selectors
    }


def _assert_edge_owned_workbench(screen) -> None:
    grid = screen.query_one("#console-workspace-grid")
    left = _visible(screen, _LEFT_OWNERS)
    main = screen.query_one("#console-main-column")
    transcript = screen.query_one("#console-transcript-region")
    right = _visible(screen, _RIGHT_OWNERS)

    assert grid.content_region.x == screen.region.x == 0
    assert grid.content_region.right == screen.region.right
    assert left.region.x == grid.content_region.x
    assert right.region.right == grid.content_region.right
    assert left.region.right == main.region.x
    assert main.region.right == right.region.x
    assert transcript.region.x == main.region.x
    assert transcript.region.right == main.region.right

    assert _border_kinds(grid) == ("solid", "none", "solid", "none")
    assert _border_kinds(left) == ("none", "solid", "none", "none")
    assert _border_kinds(transcript) == ("none", "none", "none", "none")
    assert _border_kinds(right) == ("none", "none", "none", "solid")

    sample_y = grid.content_region.y + (grid.content_region.height - 1) // 2
    assert _hit_belongs_to(screen, screen.region.x, sample_y, left)
    assert _hit_belongs_to(screen, screen.region.right - 1, sample_y, right)

    # Each boundary spends exactly one painted divider cell, owned by its rail.
    assert _hit_belongs_to(screen, left.region.right - 1, sample_y, left)
    assert _hit_belongs_to(screen, main.region.x, sample_y, transcript)
    assert _hit_belongs_to(screen, main.region.right - 1, sample_y, transcript)
    assert _hit_belongs_to(screen, right.region.x, sample_y, right)
    for x in (
        transcript.region.x,
        transcript.region.x + transcript.region.width // 2,
        transcript.region.right - 1,
    ):
        assert _hit_belongs_to(screen, x, sample_y, transcript)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "size",
    [
        pytest.param((100, 30), id="context-floor"),
        pytest.param((120, 30), id="inspector-priority"),
        pytest.param((150, 45), id="both-rails-eligible"),
        pytest.param((160, 24), id="short-height"),
    ],
)
async def test_production_workspace_regions_touch_application_edges(
    size: tuple[int, int],
) -> None:
    async with make_console_pilot(size=size) as pilot:
        _assert_edge_owned_workbench(pilot.app.screen)


@pytest.mark.asyncio
async def test_expanded_and_collapsed_rail_owners_keep_one_divider() -> None:
    async with make_console_pilot(size=(160, 45)) as pilot:
        screen = pilot.app.screen
        assert await pilot.click("#console-inspector-rail-open")
        await pilot.pause()
        _assert_edge_owned_workbench(screen)

    async with make_console_pilot(size=(160, 45)) as pilot:
        screen = pilot.app.screen
        assert await pilot.click("#console-context-rail-collapse")
        await pilot.pause()
        _assert_edge_owned_workbench(screen)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("focus_selector", "owner_selector", "divider_edge"),
    [
        ("#console-context-rail-collapse", "#console-left-rail", "right"),
        ("#console-inspector-rail-collapse", "#console-right-rail", "left"),
    ],
)
async def test_focus_repaints_only_owned_divider_without_moving_geometry(
    focus_selector: str,
    owner_selector: str,
    divider_edge: str,
) -> None:
    async with make_console_pilot(size=(160, 45)) as pilot:
        screen = pilot.app.screen
        assert await pilot.click("#console-inspector-rail-open")
        await pilot.pause()
        before = _geometry_snapshot(screen)
        control = screen.query_one(focus_selector)
        control.focus()
        await pilot.pause()

        assert pilot.app.focused is control
        assert _geometry_snapshot(screen) == before
        _assert_edge_owned_workbench(screen)
        owner = screen.query_one(owner_selector)
        assert owner.has_class("console-edge-region-focused")
        assert getattr(owner.styles, f"border_{divider_edge}")[0] == "solid"
        assert control.styles.text_style.bold
        assert control.styles.text_style.underline
