"""Contracts for compact Console rail handle presentation."""

from __future__ import annotations

from pathlib import Path

import pytest
from rich.cells import cell_len
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, Static

import tldw_chatbook.Widgets.Console.console_rail_handle as console_rail_handle
from tldw_chatbook.Chat.console_glyphs import (
    GLYPH_COLLAPSE_LEFT,
    GLYPH_COLLAPSE_RIGHT,
    GLYPH_COLLAPSED,
)
from tldw_chatbook.Chat.console_rail_state import (
    CONSOLE_RAIL_CONTEXT_LABEL,
    CONSOLE_RAIL_INSPECTOR_LABEL,
)
from tldw_chatbook.Widgets.Console.console_rail_handle import ConsoleRailHandle


# TASK-25812: the console handle rules live in the split console sheet;
# the harness loads the same app-tier set the running app ends up with.
from Tests.UI.consolidated_css import APP_STYLESHEETS as _APP_STYLESHEETS


class _VerticalRailHandleHarness(App[None]):
    """Small live layout for asserting bundled vertical-handle geometry."""

    CSS_PATH = [str(path) for path in _APP_STYLESHEETS]

    def compose(self) -> ComposeResult:
        yield Horizontal(
            _handle(
                label=CONSOLE_RAIL_CONTEXT_LABEL,
                button_id="vertical-context-button",
                badge_id="vertical-context-badge",
                side="left",
                vertical=True,
                id="vertical-context-handle",
            ),
            _handle(
                label=CONSOLE_RAIL_INSPECTOR_LABEL,
                badge="1 approval",
                button_id="vertical-inspector-button",
                badge_id="vertical-inspector-badge",
                side="right",
                vertical=True,
                id="vertical-inspector-handle",
            ),
            id="vertical-rail-handle-harness",
        )


def _handle(
    *,
    label: str,
    badge: str = "",
    side: str = "left",
    vertical: bool = False,
    id: str | None = None,
    button_id: str = "test-console-rail-button",
    badge_id: str = "test-console-rail-badge",
) -> ConsoleRailHandle:
    """Build a handle with stable IDs for focused presentation tests."""
    return ConsoleRailHandle(
        label=label,
        badge=badge,
        button_id=button_id,
        badge_id=badge_id,
        side=side,
        vertical=vertical,
        id=id,
    )


def _assert_content_column_contained(handle, child) -> None:
    """Assert a child content column is exactly one cell inside its handle."""
    content = handle.content_region
    child_content = child.content_region

    assert child_content.width == ConsoleRailHandle.VERTICAL_CONTENT_WIDTH
    assert child_content.x >= content.x
    assert child_content.x + child_content.width <= content.x + content.width
    assert child_content.y >= content.y
    assert child_content.y + child_content.height <= content.y + content.height


@pytest.mark.asyncio
async def test_vertical_handles_use_bundled_full_height_geometry_and_keep_badge_visible() -> (
    None
):
    """Bundled TCSS makes both vertical rail sides narrow, tall, and contained."""
    app = _VerticalRailHandleHarness()

    async with app.run_test(size=(32, 20)) as pilot:
        await pilot.pause()
        host = app.query_one("#vertical-rail-handle-harness", Horizontal)
        left = app.query_one("#vertical-context-handle", ConsoleRailHandle)
        right = app.query_one("#vertical-inspector-handle", ConsoleRailHandle)
        left_button = app.query_one("#vertical-context-button", Button)
        right_button = app.query_one("#vertical-inspector-button", Button)
        right_badge = app.query_one("#vertical-inspector-badge", Static)

        assert left.region.width == ConsoleRailHandle.VERTICAL_WIDTH
        assert right.region.width == ConsoleRailHandle.VERTICAL_WIDTH
        assert left.region.height == host.content_region.height
        assert right.region.height == host.content_region.height
        assert left.content_region.width == ConsoleRailHandle.VERTICAL_CONTENT_WIDTH
        assert right.content_region.width > ConsoleRailHandle.VERTICAL_CONTENT_WIDTH
        for button in (left_button, right_button):
            assert button.styles.min_height.value == 7
            assert button.styles.max_height.value == 100
            assert button.styles.max_height.unit.name == "HEIGHT"
        assert left.styles.border.top[0] == "solid"
        assert right.styles.border.top[0] in {"", "none"}
        _assert_content_column_contained(left, left_button)
        _assert_content_column_contained(right, right_button)
        _assert_content_column_contained(right, right_badge)
        assert left_button.region.height >= len(left._display_label().splitlines())
        assert right_button.region.height >= len(right._display_label().splitlines())
        assert right_badge.region.height >= len(right._display_badge().splitlines())
        assert (
            right_button.region.y + right_button.region.height == right_badge.region.y
        )
        assert right_badge.region.y >= right.content_region.y
        assert right_badge.region.y + right_badge.region.height == (
            right.content_region.y + right.content_region.height
        )
        assert left_button.tooltip == "Open Context rail"
        assert right_button.tooltip == "Open Inspector rail"
        assert right_badge.tooltip == "1 approval"


@pytest.mark.asyncio
async def test_vertical_buttons_paint_a_single_centered_column() -> None:
    """Button paint stays inside the centered one-cell layout contract."""
    app = _VerticalRailHandleHarness()

    async with app.run_test(size=(32, 20)) as pilot:
        await pilot.pause()
        pairs = (
            (
                app.query_one("#vertical-context-handle", ConsoleRailHandle),
                app.query_one("#vertical-context-button", Button),
            ),
            (
                app.query_one("#vertical-inspector-handle", ConsoleRailHandle),
                app.query_one("#vertical-inspector-button", Button),
            ),
        )

        for handle, button in pairs:
            expected_x = (
                handle.region.x
                + (handle.region.width - ConsoleRailHandle.VERTICAL_CONTENT_WIDTH) // 2
            )
            painted_lines = [
                strip
                for y in range(button.region.height)
                if (strip := button.render_line(y)).text.strip()
            ]

            assert painted_lines
            assert all(
                cell_len(strip.text) == button.region.width for strip in painted_lines
            )
            assert button.region.x == expected_x


def test_vertical_context_label_stacks_without_direction_glyph() -> None:
    handle = _handle(label=CONSOLE_RAIL_CONTEXT_LABEL, vertical=True)

    assert handle._display_label() == "C\no\nn\nt\ne\nx\nt"
    assert "▸" not in handle._display_label()


def test_vertical_inspector_label_stacks_without_direction_glyph() -> None:
    handle = _handle(
        label=CONSOLE_RAIL_INSPECTOR_LABEL,
        side="right",
        vertical=True,
    )

    assert handle._display_label() == "I\nn\ns\np\ne\nc\nt\no\nr"
    assert "◂" not in handle._display_label()


def test_horizontal_canonical_labels_use_inward_console_button_copy() -> None:
    context = _handle(label=CONSOLE_RAIL_CONTEXT_LABEL)
    inspector = _handle(label=CONSOLE_RAIL_INSPECTOR_LABEL, side="right")

    assert context._display_label() == "Context ▸"
    assert inspector._display_label() == "◂ Inspect"


def test_horizontal_noncanonical_left_label_is_unchanged() -> None:
    handle = _handle(label="Sources")

    assert handle._display_label() == "Sources"


def test_horizontal_noncanonical_right_label_is_unchanged() -> None:
    handle = _handle(label="Review", side="right")

    assert handle._display_label() == "Review"


def test_vertical_generic_label_normalizes_whitespace_before_stacking() -> None:
    handle = _handle(label="  Review\n queue  ", vertical=True)

    assert handle._display_label() == "R\ne\nv\ni\ne\nw\n \nq\nu\ne\nu\ne"


def test_vertical_right_badge_stacks_compact_semantic_display() -> None:
    handle = _handle(badge="1 approval", side="right", vertical=True, label="Inspector")

    assert handle._display_badge() == "1\n \na\np\np\nr"


def test_vertical_known_label_normalizes_before_glyph_removal() -> None:
    handle = _handle(label=f"  \n{CONSOLE_RAIL_CONTEXT_LABEL}\n  ", vertical=True)

    assert handle._display_label() == "C\no\nn\nt\ne\nx\nt"


def test_vertical_canonical_context_label_derives_visible_text_from_constant(
    monkeypatch,
) -> None:
    canonical_label = f"Orbit {GLYPH_COLLAPSED}"
    monkeypatch.setattr(
        console_rail_handle, "CONSOLE_RAIL_CONTEXT_LABEL", canonical_label
    )
    handle = _handle(label=canonical_label, vertical=True)

    assert handle._display_label() == "O\nr\nb\ni\nt"


def test_vertical_compose_marks_handle_and_children_and_uses_content_width() -> None:
    handle = _handle(
        label=CONSOLE_RAIL_INSPECTOR_LABEL,
        badge="1 approval",
        side="right",
        vertical=True,
    )

    button, badge = list(handle.compose())

    assert handle.has_class("console-rail-handle-vertical")
    assert isinstance(button, Button)
    assert button.has_class("console-rail-handle-button-vertical")
    assert button.styles.width.value == ConsoleRailHandle.VERTICAL_CONTENT_WIDTH
    assert button.styles.height.value == 1
    assert button.styles.height.unit.name == "FRACTION"
    assert isinstance(badge, Static)
    assert badge.has_class("console-rail-handle-badge-vertical")
    assert badge.styles.width.value == ConsoleRailHandle.VERTICAL_CONTENT_WIDTH


# --- TASK-31665 AC#4: one arrow vocabulary, open and collapsed --------------


def test_collapsed_handles_speak_the_same_arrow_vocabulary_as_the_open_rails():
    """AC#4. The compact forms used to spell their arrows in ASCII
    (`Context->` / `<-Inspect`) while the OPEN rails' own collapse controls
    used the glyph vocabulary (`Context ◂` in left_rail, `▸ Inspect` in
    right_rail) -- so one rail spoke two arrow languages depending on
    whether it was open."""
    context = _handle(label=CONSOLE_RAIL_CONTEXT_LABEL)
    inspector = _handle(label=CONSOLE_RAIL_INSPECTOR_LABEL, side="right")

    for label in (context._display_label(), inspector._display_label()):
        assert "->" not in label and "<-" not in label, (
            f"{label!r} still spells its arrow in ASCII"
        )
        assert any(glyph in label for glyph in (GLYPH_COLLAPSE_LEFT, GLYPH_COLLAPSE_RIGHT))


def test_collapsed_handle_arrows_survive_ascii_glyph_mode(monkeypatch) -> None:
    """AC#4: the unification must not cost the ASCII fallback. `resolve_glyph`
    maps ◂/▸ to </>, so the label stays nine cells wide either way -- which is
    what the collapsed handle's fixed geometry and TASK-31663's right-edge
    focus carrier both depend on."""
    from tldw_chatbook.Widgets import glyph_fallback

    glyph_fallback.set_ascii_glyph_mode(True)
    try:
        context = _handle(label=CONSOLE_RAIL_CONTEXT_LABEL)
        inspector = _handle(label=CONSOLE_RAIL_INSPECTOR_LABEL, side="right")
        assert context._display_label() == "Context >"
        assert inspector._display_label() == "< Inspect"
    finally:
        glyph_fallback.set_ascii_glyph_mode(False)

    assert cell_len(_handle(label=CONSOLE_RAIL_CONTEXT_LABEL)._display_label()) == 9
    assert cell_len(
        _handle(label=CONSOLE_RAIL_INSPECTOR_LABEL, side="right")._display_label()
    ) == 9
