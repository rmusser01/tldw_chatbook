"""Settings workbench layout regression tests (task-1342).

UAT (headless, real stylesheet) found three layout defects:

1. ``Button.settings-category-button`` forced ``height: 1; max-height: 1``
   while Textual 8 Buttons reserve one ``line-pad`` cell on each side of the
   label, so ``Providers & Models`` / ``Privacy & Security`` wrapped and the
   second line (and the dirty ``*`` marker) was clipped at <=120 cols.
2. At 80x24 the category pane's fixed header overhead consumed every row of
   the list -- the category list vanished and the 3:6:2 split squeezed the
   detail pane to a 3-line banner while its controls stayed focusable.
3. The scope inspector folded long words mid-word ("Selecte d / categor /
   y:") at narrow widths.

These tests load the REAL application stylesheet (``tldw_cli_modular.tcss``)
-- the shared ``DestinationHarness`` runs without it and therefore cannot see
these defects.
"""

from pathlib import Path

import pytest

import tldw_chatbook
from Tests.UI.test_destination_shells import DestinationHarness, _active_destination_screen
from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId

CSS_PATH = str(Path(tldw_chatbook.__file__).parent / "css" / "tldw_cli_modular.tcss")

class _SettingsCssHarness(DestinationHarness):
    """DestinationHarness with the real application stylesheet loaded."""

    CSS_PATH = CSS_PATH


def _rendered_lines(screen) -> list[str]:
    strips = screen._compositor.render_strips()
    return ["".join(segment.text for segment in strip) for strip in strips]


def _region_rows(screen, widget) -> list[str]:
    lines = _rendered_lines(screen)
    region = widget.region
    rows = []
    for y in range(region.y, region.y + region.height):
        if 0 <= y < len(lines):
            rows.append(lines[y][region.x : region.x + region.width])
    return rows


async def _scrolled_region_rows(pilot, widget) -> list[str]:
    """Collect rendered rows of a scrollable widget across its full scroll range."""
    screen = widget.screen
    widget.scroll_to(y=0, animate=False)
    await pilot.pause()
    rows: list[str] = []
    seen_y = -1
    # Step by the CONTENT height: scrolling by the full region height would
    # skip the two rows hidden behind the horizontal border lines.
    step = max(1, widget.content_region.height)
    while widget.scroll_offset.y != seen_y:
        seen_y = widget.scroll_offset.y
        rows.extend(_region_rows(screen, widget))
        if seen_y >= widget.max_scroll_y:
            break
        widget.scroll_to(y=min(seen_y + step, widget.max_scroll_y), animate=False)
        await pilot.pause()
    return rows


async def _settle(pilot) -> None:
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()


async def _scroll_to_category(pilot, screen, category_value: str) -> None:
    button = screen.query_one(f"#settings-category-{category_value}")
    screen.query_one("#settings-category-list").scroll_to_widget(button, animate=False)
    await pilot.pause()


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(120, 35), (100, 30)])
async def test_all_category_labels_render_fully(size):
    """AC1: all category labels render in full (no clipping)."""
    app = _build_test_app()
    host = _SettingsCssHarness(app, "settings")
    async with host.run_test(size=size) as pilot:
        await _settle(pilot)
        screen = _active_destination_screen(host)
        # Upstream's Domain Defaults rail starts collapsed; open it so every
        # category label is actually rendered.
        screen._domain_group_expanded = True
        screen._apply_category_search_filter()
        await pilot.pause()
        category_list = screen.query_one("#settings-category-list")
        assert category_list.region.height >= 3, (
            f"Category list has no visible rows at {size}: {category_list.region}"
        )
        for summary in screen._category_summaries():
            title = summary.title
            category_value = summary.category.value
            await _scroll_to_category(pilot, screen, category_value)
            pane_text = "\n".join(
                _region_rows(screen, screen.query_one("#settings-category-pane"))
            )
            assert title in pane_text, (
                f"Category label {title!r} clipped at {size}: {pane_text!r}"
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(120, 35), (100, 30)])
async def test_dirty_marker_renders_fully(size):
    """AC1: the dirty ``*`` marker is not clipped off edited categories."""
    app = _build_test_app()
    host = _SettingsCssHarness(app, "settings")
    async with host.run_test(size=size) as pilot:
        await _settle(pilot)
        screen = _active_destination_screen(host)
        screen._domain_group_expanded = True
        screen._apply_category_search_filter()
        await pilot.pause()
        original = screen._category_has_unsaved_changes
        screen._category_has_unsaved_changes = (
            lambda category: category is SettingsCategoryId.PROVIDERS_MODELS
        )
        try:
            screen._refresh_category_button_label(SettingsCategoryId.PROVIDERS_MODELS)
            await pilot.pause()
            await _scroll_to_category(pilot, screen, "providers-models")
            pane_text = "\n".join(
                _region_rows(screen, screen.query_one("#settings-category-pane"))
            )
            assert "Providers & Models *" in pane_text, (
                f"Dirty marker clipped at {size}: {pane_text!r}"
            )
        finally:
            screen._category_has_unsaved_changes = original
            screen._refresh_category_button_label(SettingsCategoryId.PROVIDERS_MODELS)


@pytest.mark.asyncio
async def test_narrow_width_layout_80x24():
    """AC2: at 80x24 the screen offers a working narrow layout.

    Categories stay visible (fixed-width compact sidebar), the filter input
    keeps its hint placeholder, and the detail pane keeps a usable width
    instead of squeezing all three panes into slivers.
    """
    app = _build_test_app()
    host = _SettingsCssHarness(app, "settings")
    async with host.run_test(size=(80, 24)) as pilot:
        await _settle(pilot)
        screen = _active_destination_screen(host)

        category_pane = screen.query_one("#settings-category-pane")
        assert category_pane.display and category_pane.region.width >= 24, (
            f"Category pane unusable at 80x24: {category_pane.region}"
        )
        category_list = screen.query_one("#settings-category-list")
        assert category_list.region.height >= 3, (
            f"Category list has no visible rows at 80x24: {category_list.region}"
        )
        detail_pane = screen.query_one("#settings-detail-pane")
        assert detail_pane.region.width >= 30, (
            f"Detail pane squeezed at 80x24: {detail_pane.region}"
        )

        screen_text = "\n".join(_rendered_lines(screen))
        # The active category is visible in the list (categories stay visible).
        assert "Overview" in "\n".join(_region_rows(screen, category_pane)), (
            f"No visible category rows at 80x24: {screen_text!r}"
        )
        # Filter affordance survives the narrow layout.
        search = screen.query_one("#settings-category-search")
        assert search.display and search.region.height >= 1


@pytest.mark.asyncio
async def test_console_capture_controls_are_reachable_at_80x24() -> None:
    app = _build_test_app()
    host = _SettingsCssHarness(app, "settings")
    async with host.run_test(size=(80, 24)) as pilot:
        await _settle(pilot)
        screen = _active_destination_screen(host)
        screen._select_category(SettingsCategoryId.CONSOLE_BEHAVIOR)
        await pilot.pause()

        enabled = screen.query_one("#settings-console-exchange-capture-enabled")
        pii = screen.query_one("#settings-console-trace-pii-redaction")
        viewer = screen.query_one("#settings-console-trace-viewer-profile")
        status = screen.query_one("#settings-console-exchange-capture-status")
        scroll = screen.query_one("#settings-detail-pane-body")
        scroll.scroll_to_widget(status, animate=False)
        await pilot.pause()

        assert enabled.region.height > 0
        assert pii.region.height > 0
        assert viewer.region.height > 0
        assert status.region.height > 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("size", "category_value", "expected_words"),
    [
        # Overview inspector copy (upstream wording). Only 120x35: upstream's
        # pinned inspector header (task-1716) fills the whole impact pane at
        # 30 rows, leaving the scrollable guidance body a single line.
        ((120, 35), "overview", ("Read-only:", "Recovery:", "Boundary:")),
    ],
)
async def test_inspector_does_not_wrap_mid_word(size, category_value, expected_words):
    """AC3: inspector text wraps on whitespace, never mid-word."""
    app = _build_test_app()
    host = _SettingsCssHarness(app, "settings")
    async with host.run_test(size=size) as pilot:
        await _settle(pilot)
        screen = _active_destination_screen(host)
        button = screen.query_one(f"#settings-category-{category_value}")
        screen.query_one("#settings-category-list").scroll_to_widget(
            button, animate=False
        )
        await pilot.pause()
        await pilot.click(f"#settings-category-{category_value}")
        await _settle(pilot)
        screen = _active_destination_screen(host)
        # Upstream split the impact pane into a pinned header and the
        # scrollable -body; scroll the body for the full content sweep.
        impact_pane = screen.query_one("#settings-impact-pane-body")
        rows = await _scrolled_region_rows(pilot, impact_pane)
        for word in expected_words:
            assert any(word in row for row in rows), (
                f"Inspector word {word!r} split mid-word at {size} "
                f"(category {category_value}): {rows!r}"
            )
