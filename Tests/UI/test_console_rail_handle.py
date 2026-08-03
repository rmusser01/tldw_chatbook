"""Contracts for compact Console rail handle presentation."""

from __future__ import annotations

from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_rail_state import (
    CONSOLE_RAIL_CONTEXT_LABEL,
    CONSOLE_RAIL_INSPECTOR_LABEL,
)
from tldw_chatbook.Widgets.Console.console_rail_handle import ConsoleRailHandle


def _handle(
    *,
    label: str,
    badge: str = "",
    side: str = "left",
    vertical: bool = False,
) -> ConsoleRailHandle:
    """Build a handle with stable IDs for focused presentation tests."""
    return ConsoleRailHandle(
        label=label,
        badge=badge,
        button_id="test-console-rail-button",
        badge_id="test-console-rail-badge",
        side=side,
        vertical=vertical,
    )


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


def test_horizontal_defaults_preserve_existing_visible_labels() -> None:
    context = _handle(label=CONSOLE_RAIL_CONTEXT_LABEL)
    inspector = _handle(label=CONSOLE_RAIL_INSPECTOR_LABEL, side="right")

    assert context._display_label() == CONSOLE_RAIL_CONTEXT_LABEL
    assert inspector._display_label() == "Inspector"


def test_vertical_generic_label_normalizes_whitespace_before_stacking() -> None:
    handle = _handle(label="  Review\n queue  ", vertical=True)

    assert handle._display_label() == "R\ne\nv\ni\ne\nw\n \nq\nu\ne\nu\ne"


def test_vertical_right_badge_stacks_compact_semantic_display() -> None:
    handle = _handle(badge="1 approval", side="right", vertical=True, label="Inspector")

    assert handle._display_badge() == "1\n \na\np\np\nr"


def test_vertical_known_label_normalizes_before_glyph_removal() -> None:
    handle = _handle(label=f"  \n{CONSOLE_RAIL_CONTEXT_LABEL}\n  ", vertical=True)

    assert handle._display_label() == "C\no\nn\nt\ne\nx\nt"


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
