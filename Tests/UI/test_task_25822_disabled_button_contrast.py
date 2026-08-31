"""TASK-25822: a disabled control must read as unavailable, not absent.

`Button:disabled` stacked `opacity: 50%` on top of an already-dim
`$text-disabled`. Measured live in the Console Library-access modal, the
disabled primary painted 1.39:1 against its background while the adjacent
Cancel painted ~14:1 -- so the dialog appeared to offer only its dismissive
action. The repo had already hit this once and patched a single button
(`Button.model-import:disabled`, "must remain readable"); this pins the
readable floor app-wide instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button

import tldw_chatbook.css as css_pkg

_CSS_DIR = Path(css_pkg.__file__).parent
_WCAG_LARGE_TEXT_FLOOR = 3.0


def _relative_luminance(triplet) -> float:
    def _channel(value: int) -> float:
        srgb = value / 255
        return srgb / 12.92 if srgb <= 0.04045 else ((srgb + 0.055) / 1.055) ** 2.4

    return (
        0.2126 * _channel(triplet.red)
        + 0.7152 * _channel(triplet.green)
        + 0.0722 * _channel(triplet.blue)
    )


def _contrast(first, second) -> float:
    lighter, darker = sorted(
        (_relative_luminance(first), _relative_luminance(second)), reverse=True
    )
    return (lighter + 0.05) / (darker + 0.05)


class _DisabledButtonHarness(App):
    CSS_PATH = [
        str(_CSS_DIR / "screen_css_scoped.tcss"),
        str(_CSS_DIR / "tldw_cli_modular.tcss"),
        str(_CSS_DIR / "screen_css_self.tcss"),
    ]

    def compose(self) -> ComposeResult:
        yield Button("Save", id="probe-save", variant="primary", disabled=True)
        yield Button("Later", id="probe-default", disabled=True)
        yield Button("Import", id="probe-import", classes="model-import", disabled=True)


def _painted(app: App, widget) -> tuple[object, object]:
    """Return the first painted glyph's compositor foreground/background."""
    strips = app.screen._compositor.render_strips()
    for y in range(widget.region.y, widget.region.bottom):
        cursor = 0
        for segment in strips[y]:
            next_cursor = cursor + segment.cell_length
            overlaps = cursor < widget.region.right and next_cursor > widget.region.x
            if overlaps and segment.text.strip() and segment.style is not None:
                style = segment.style
                if style.color is not None and style.bgcolor is not None:
                    return style.color.triplet, style.bgcolor.triplet
            cursor = next_cursor
    raise AssertionError("no painted glyph found for the disabled button")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "probe_id",
    [
        "probe-save",  # primary variant -- the Console modal's Save
        "probe-default",  # stock variant
        "probe-import",  # previously carried its own readability opt-out
    ],
)
async def test_disabled_button_text_stays_readable(probe_id: str) -> None:
    app = _DisabledButtonHarness()
    async with app.run_test(size=(60, 12)):
        button = app.query_one(f"#{probe_id}", Button)
        foreground, background = _painted(app, button)
        ratio = _contrast(foreground, background)

    assert ratio >= _WCAG_LARGE_TEXT_FLOOR, (
        f"disabled Button {probe_id} painted {ratio:.2f}:1 "
        f"(fg={foreground}, bg={background}); a disabled control must stay "
        "legible enough to read as unavailable rather than absent"
    )
