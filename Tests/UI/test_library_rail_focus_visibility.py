"""Task-32598: rail keyboard targets must paint above the docked scroll cue."""

from __future__ import annotations

import pytest
from textual.widgets import Button

from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _build_test_app,
    _seed_conversations,
    _two_conversations,
    _wait_for_library_shell,
)
from tldw_chatbook.Widgets.Library.library_rail import LibraryRail


def _host() -> LibraryProductionCSSHarness:
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    return LibraryProductionCSSHarness(app)


def _assert_focus_paints(host: LibraryProductionCSSHarness, button: Button) -> None:
    """Containment alone misses a docked sibling covering the focused glyph."""
    assert button.has_focus
    region = button.region
    strips = host.screen._compositor.render_strips()
    cropped = [
        strips[y].crop(region.x, region.right)
        for y in range(max(0, region.y), min(len(strips), region.bottom))
    ]
    glyph = button.label.plain
    assert glyph in "\n".join(strip.text for strip in cropped), (
        button.id,
        region,
        host.screen.can_view_entire(button),
        [strip.text for strip in cropped],
    )
    assert any(
        segment.style and segment.style.underline
        for strip in cropped
        for segment in strip
        if glyph in segment.text
    ), "The visible toggle must retain its underline focus cue."


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
async def test_tab_reveals_create_above_the_docked_cue(theme: str) -> None:
    """A focus move must scroll even when the screen calls a covered chip visible."""
    host = _host()
    async with host.run_test(size=(80, 24)) as pilot:
        host.theme = theme
        screen = host.screen
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-search", Button).focus()
        await pilot.pause()
        await pilot.press("tab")
        await pilot.pause()
        button = screen.query_one("#console-rail-section-toggle-library-create", Button)
        _assert_focus_paints(host, button)
        cue = screen.query_one("#library-rail-fold-cue")
        assert cue.display
        assert button.region.bottom <= cue.region.y
        await pilot.press("enter")
        await pilot.pause()
        button = screen.query_one("#console-rail-section-toggle-library-create", Button)
        assert button.label.plain == "▸"
        _assert_focus_paints(host, button)


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
async def test_all_section_toggles_paint_through_reverse_traversal_and_resize(
    theme: str,
) -> None:
    """Deep rail sections stay visible without navigating away from Library."""
    host = _host()
    async with host.run_test(size=(120, 45)) as pilot:
        host.theme = theme
        screen = host.screen
        await _wait_for_library_shell(screen, pilot)
        screen._set_library_rail_section("details", True)
        await pilot.pause()
        await pilot.pause()
        route = screen._library_selected_row_id
        prefix = "console-rail-section-toggle-library-"
        expected = {
            prefix + name
            for name in (
                "browse",
                "create",
                "study",
                "ingest",
                "details",
                "details-diagnostics",
            )
        }
        screen.query_one("#library-search-input").focus()
        await pilot.resize_terminal(80, 24)
        await pilot.pause()
        for key in ("tab", "shift+tab"):
            seen = set()
            for _ in range(70):
                await pilot.press(key)
                await pilot.pause()
                focused = host.focused
                if isinstance(focused, Button) and focused.id in expected:
                    _assert_focus_paints(host, focused)
                    seen.add(focused.id)
                    if seen == expected:
                        break
            assert seen == expected, (key, expected - seen)
            assert screen._library_selected_row_id == route

        screen.query_one(f"#{prefix}details-diagnostics", Button).focus()
        await pilot.pause()
        for size in ((120, 45), (80, 24), (120, 45)):
            await pilot.resize_terminal(*size)
            await pilot.pause()
            await pilot.pause()
            assert isinstance(host.focused, Button)
            _assert_focus_paints(host, host.focused)

        rail = screen.query_one("#library-rail", LibraryRail)
        cue = screen.query_one("#library-rail-fold-cue")
        assert cue.display and rail.max_scroll_y > 0
