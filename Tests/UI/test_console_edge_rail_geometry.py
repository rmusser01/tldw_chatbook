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


def _hit_snapshot(screen) -> tuple[Widget, ...]:
    grid = screen.query_one("#console-workspace-grid")
    transcript = screen.query_one("#console-transcript-region")
    sample_y = grid.content_region.y + (grid.content_region.height - 1) // 2
    return tuple(
        screen.get_widget_at(x, sample_y)[0]
        for x in (
            screen.region.x,
            transcript.region.x,
            transcript.region.right - 1,
            screen.region.right - 1,
        )
    )


def _assert_edge_owned_workbench(screen) -> tuple[int, int]:
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
    return screen.size.width, screen.size.height


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


@pytest.mark.asyncio
async def test_f6_transcript_focus_marks_only_stable_title_without_geometry_change() -> (
    None
):
    """Transcript focus remains visible without relying on a color delta."""
    async with make_console_pilot(size=(160, 45)) as pilot:
        screen = pilot.app.screen
        title = screen.query_one("#console-transcript-title")
        region = screen.query_one("#console-transcript-region")
        transcript = screen.query_one("#console-native-transcript")
        collapse = screen.query_one("#console-context-rail-collapse")
        before = _geometry_snapshot(screen)
        title_region = title.region

        collapse.focus()
        await pilot.pause()
        await pilot.press("f6")
        await pilot.pause()

        assert pilot.app.focused is transcript
        assert "focus-within" in region.get_pseudo_classes()
        assert region in title.ancestors
        assert title.styles.text_style.bold
        assert title.styles.text_style.underline
        assert not transcript.styles.text_style.underline
        # Monochrome oracle: erase the focus palette delta; typography alone
        # must continue to identify the focused pane.
        title.styles.background = transcript.styles.background
        title.styles.color = transcript.styles.color
        assert title.styles.background == transcript.styles.background
        assert title.styles.color == transcript.styles.color
        assert title.styles.text_style.bold
        assert title.styles.text_style.underline
        assert title.region == title_region
        assert _geometry_snapshot(screen) == before
        _assert_edge_owned_workbench(screen)

        await pilot.press("f6")
        await pilot.pause()

        assert pilot.app.focused is not transcript
        assert not title.styles.text_style.bold
        assert not title.styles.text_style.underline
        assert title.region == title_region
        assert _geometry_snapshot(screen) == before
        _assert_edge_owned_workbench(screen)


@pytest.mark.asyncio
async def test_live_resize_reasserts_edge_ownership_and_preserves_rail_intent() -> None:
    """One mounted Console retains physical ownership through every band."""
    async with make_console_pilot(size=(150, 30)) as pilot:
        screen = pilot.app.screen
        assert await pilot.click("#console-inspector-rail-open")
        await pilot.pause()
        initial_state = screen._last_console_rail_state
        assert initial_state is not None
        assert initial_state.preferred_left_open
        assert initial_state.preferred_right_open

        collapse = screen.query_one("#console-context-rail-collapse")
        collapse.focus()
        await pilot.pause()
        checks = [_assert_edge_owned_workbench(screen)]

        for width, height in ((120, 30), (100, 30), (100, 24)):
            await pilot.resize_terminal(width, height)
            await pilot.pause(0.2)
            await pilot.pause()
            checks.append(_assert_edge_owned_workbench(screen))

            state = screen._last_console_rail_state
            assert state is not None
            assert state.preferred_left_open
            assert state.preferred_right_open
            transcript = screen.query_one("#console-native-transcript")
            assert transcript.display
            assert transcript.region.width > 0
            assert transcript.can_focus

        reveal = screen.query_one("#console-context-rail-open")
        assert reveal.display
        assert pilot.app.focused is reveal
        assert checks == [(150, 30), (120, 30), (100, 30), (100, 24)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("collapse_selector", "control_selector", "owner_selector", "divider_edge"),
    [
        (
            "#console-context-rail-collapse",
            "#console-context-rail-open",
            "#console-context-rail-handle",
            "right",
        ),
        (
            None,
            "#console-inspector-rail-open",
            "#console-inspector-rail-handle",
            "left",
        ),
    ],
)
async def test_collapsed_handle_focus_is_non_color_and_dimension_stable(
    collapse_selector: str | None,
    control_selector: str,
    owner_selector: str,
    divider_edge: str,
) -> None:
    async with make_console_pilot(size=(160, 45)) as pilot:
        screen = pilot.app.screen
        if collapse_selector is not None:
            assert await pilot.click(collapse_selector)
            await pilot.pause()
        control = screen.query_one(control_selector)
        owner = screen.query_one(owner_selector)
        transcript = screen.query_one("#console-native-transcript")
        before = _geometry_snapshot(screen)

        control.focus()
        await pilot.pause()

        assert pilot.app.focused is control
        assert owner.has_class("console-edge-region-focused")
        assert getattr(owner.styles, f"border_{divider_edge}")[0] == "solid"
        assert control.styles.text_style.bold
        assert control.styles.text_style.underline
        assert _geometry_snapshot(screen) == before
        _assert_edge_owned_workbench(screen)

        transcript.focus()
        await pilot.pause()

        assert pilot.app.focused is transcript
        assert not owner.has_class("console-edge-region-focused")
        assert not control.styles.text_style.bold
        assert not control.styles.text_style.underline
        assert _geometry_snapshot(screen) == before
        _assert_edge_owned_workbench(screen)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("focus_selector", "owner_selector", "cue_selector", "focus_class", "edge"),
    [
        (
            "#console-native-transcript",
            "#console-transcript-region",
            "#console-transcript-title",
            "console-transcript-region-focused",
            None,
        ),
        (
            "#console-context-rail-collapse",
            "#console-left-rail",
            "#console-context-rail-collapse",
            "console-edge-region-focused",
            "right",
        ),
        (
            "#console-inspector-rail-open",
            "#console-inspector-rail-handle",
            "#console-inspector-rail-open",
            "console-edge-region-focused",
            "left",
        ),
    ],
)
async def test_focus_paint_clears_when_focus_becomes_none(
    focus_selector: str,
    owner_selector: str,
    cue_selector: str,
    focus_class: str,
    edge: str | None,
) -> None:
    async with make_console_pilot(size=(160, 45)) as pilot:
        screen = pilot.app.screen
        target = screen.query_one(focus_selector)
        owner = screen.query_one(owner_selector)
        cue = screen.query_one(cue_selector)
        geometry = _geometry_snapshot(screen)
        hits = _hit_snapshot(screen)
        rest_background = cue.styles.background
        rest_color = cue.styles.color
        rest_divider = (
            getattr(owner.styles, f"border_{edge}") if edge is not None else None
        )

        target.focus()
        await pilot.pause()

        assert pilot.app.focused is target
        assert owner.has_class(focus_class)
        assert cue.styles.text_style.bold
        assert cue.styles.text_style.underline

        screen.set_focus(None)
        await pilot.pause()

        assert pilot.app.focused is None
        assert not list(screen.query(".console-edge-region-focused"))
        assert not list(screen.query(".console-transcript-region-focused"))
        assert not cue.styles.text_style.bold
        assert not cue.styles.text_style.underline
        assert cue.styles.background == rest_background
        assert cue.styles.color == rest_color
        if edge is not None:
            assert getattr(owner.styles, f"border_{edge}") == rest_divider
        assert _geometry_snapshot(screen) == geometry
        assert _hit_snapshot(screen) == hits
        _assert_edge_owned_workbench(screen)


@pytest.mark.asyncio
async def test_native_widget_blur_repaints_the_actual_focus_successor() -> None:
    """Widget.blur advances focus; paint must follow rather than clear blindly."""
    async with make_console_pilot(size=(160, 45)) as pilot:
        screen = pilot.app.screen
        reveal = screen.query_one("#console-inspector-rail-open")
        handle = screen.query_one("#console-inspector-rail-handle")
        transcript = screen.query_one("#console-native-transcript")
        transcript_region = screen.query_one("#console-transcript-region")
        title = screen.query_one("#console-transcript-title")
        geometry = _geometry_snapshot(screen)
        hits = _hit_snapshot(screen)

        reveal.focus()
        await pilot.pause()
        reveal.blur()
        await pilot.pause()

        assert pilot.app.focused is transcript
        assert not handle.has_class("console-edge-region-focused")
        assert not reveal.styles.text_style.bold
        assert not reveal.styles.text_style.underline
        assert transcript_region.has_class("console-transcript-region-focused")
        assert title.styles.text_style.bold
        assert title.styles.text_style.underline
        assert _geometry_snapshot(screen) == geometry
        assert _hit_snapshot(screen) == hits
        _assert_edge_owned_workbench(screen)


@pytest.mark.asyncio
async def test_rapid_focus_transfer_keeps_only_the_current_edge_painted() -> None:
    async with make_console_pilot(size=(160, 45)) as pilot:
        screen = pilot.app.screen
        context = screen.query_one("#console-context-rail-collapse")
        inspector = screen.query_one("#console-inspector-rail-open")
        left = screen.query_one("#console-left-rail")
        right_handle = screen.query_one("#console-inspector-rail-handle")
        left_rest = left.styles.border_right
        right_rest = right_handle.styles.border_left
        geometry = _geometry_snapshot(screen)
        hits = _hit_snapshot(screen)

        context.focus()
        inspector.focus()
        await pilot.pause()

        assert pilot.app.focused is inspector
        assert not left.has_class("console-edge-region-focused")
        assert right_handle.has_class("console-edge-region-focused")
        assert not context.styles.text_style.bold
        assert not context.styles.text_style.underline
        assert inspector.styles.text_style.bold
        assert inspector.styles.text_style.underline
        assert left.styles.border_right == left_rest
        assert right_handle.styles.border_left != right_rest
        assert _geometry_snapshot(screen) == geometry
        assert _hit_snapshot(screen) == hits
        _assert_edge_owned_workbench(screen)
