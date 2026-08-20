"""Focused compact `Input`/`Checkbox` content invisible -- TASK-17961.

`core/_reset.tcss` carries the app-wide keyboard-focus fallback::

    *:focus { outline: solid $ds-focus-accent; }

Textual paints an ``outline`` OVER a widget's outermost rendered lines rather
than around them -- the same mechanism TASK-1160 (`DataTable:focus`,
`components/_lists.tcss`) and TASK-2300 (`Select.-textual-compact:focus`,
also `components/_lists.tcss`) already fixed for their own widgets. This is
the third member of the family:

* Textual pins ``border: none !important`` on ``Input.-textual-compact``
  (`textual/widgets/_input.py`), so a compact `Input` is exactly ONE row
  tall and that row IS its value. The outline overwrites it outright.
* `ToggleButton` (the base class behind `Checkbox`/`RadioButton`) has its
  OWN focused-content recolour baked into Textual's DEFAULT_CSS
  (``&:focus > .toggle--label { background: $block-cursor-background; ...
  }``) -- but the app-wide outline still paints over the perimeter row(s)
  Textual's own focus style does not reach, colliding with this app's
  ``ToggleButton:focus`` border/background rule the same way it collided
  with `DataTable`'s cursor.

These tests deliberately load the **production** bundle
(`tldw_cli_modular.tcss`) on a bare, id-only harness -- a plain `App` with no
CSS at all cannot reproduce the defect, since the outline rule itself is the
cause. Painted-frame reads go through the real compositor
(`Screen._compositor.render_strips()`), not `Widget.renderable`/`.value`,
because the defect is specifically about what lands on screen after CSS
paints over it -- see `Tests/UI/test_datatable_focus_outline_click.py` and
`Tests/UI/test_console_session_tab_strip.py::_rendered_text` for the same
pattern.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest
from textual.app import App, ComposeResult
from textual.widget import Widget
from textual.widgets import Checkbox, Input

import tldw_chatbook

_BUNDLED_CSS_PATH = Path(tldw_chatbook.__file__).parent / "css" / "tldw_cli_modular.tcss"


def _rendered_text(app: App) -> str:
    """Join every compositor strip's segment text into one blob.

    Same helper as `Tests/UI/test_console_session_tab_strip.py`: Textual
    8.2.7 has no `App.export_text()`, so `Screen._compositor.render_strips()`
    is the only way to read what was ACTUALLY painted (post-CSS, post-clip),
    as opposed to inferring it from the un-clipped source string.
    """
    strips = app.screen._compositor.render_strips()
    return "\n".join("".join(segment.text for segment in strip) for strip in strips)


class ProductionCssWidgetHarness(App):
    """A single focusable widget under the real app stylesheet.

    No screen-local id/class rules can reach a bare widget like this, so
    whatever these tests observe is what EVERY compact `Input` and EVERY
    `Checkbox`/`RadioButton` gets by default. `AUTO_FOCUS` is disabled so
    the widget starts genuinely blurred -- a real `pilot.press("tab")` is
    what moves focus onto it, mirroring how a user actually reaches it.
    """

    AUTO_FOCUS = None
    CSS_PATH = str(_BUNDLED_CSS_PATH)

    def __init__(self, widget_factory: Callable[[], Widget]) -> None:
        super().__init__()
        self._widget_factory = widget_factory

    def compose(self) -> ComposeResult:
        yield self._widget_factory()


@pytest.mark.asyncio
async def test_compact_input_blurred_frame_shows_its_value() -> None:
    """Baseline: a blurred compact Input renders its value fine."""
    app = ProductionCssWidgetHarness(
        lambda: Input(value="hello-world", compact=True, id="probe-compact-input")
    )
    async with app.run_test(size=(60, 10)) as pilot:
        widget = app.query_one("#probe-compact-input", Input)
        await pilot.pause()
        assert not widget.has_focus

        frame = _rendered_text(app)
        assert "hello-world" in frame, (
            f"blurred compact Input did not render its value; frame:\n{frame!r}"
        )


@pytest.mark.asyncio
async def test_compact_input_focused_frame_shows_its_value() -> None:
    """AC#1/#3. The regression: a compact Input's ONE row is its value row,
    and the app-wide `*:focus{outline:solid ...}` fallback paints straight
    over it -- Textual pins `border: none !important` on
    `Input.-textual-compact`, so there is no border row to absorb the
    outline the way a tall `Input` has."""
    app = ProductionCssWidgetHarness(
        lambda: Input(value="hello-world", compact=True, id="probe-compact-input")
    )
    async with app.run_test(size=(60, 10)) as pilot:
        widget = app.query_one("#probe-compact-input", Input)
        await pilot.press("tab")
        await pilot.pause()
        assert widget.has_focus

        frame = _rendered_text(app)
        assert "hello-world" in frame, (
            "a focused compact Input's value is not on screen -- the global "
            f"focus outline is overwriting its only rendered row; frame:\n{frame!r}"
        )


@pytest.mark.asyncio
async def test_compact_checkbox_focused_frame_shows_its_label() -> None:
    """AC#1/#3, `Checkbox`/`RadioButton` half of the same family. Mirrors
    the ACTUAL originally-reported widget: `workspace_create_modal.py`'s
    "Switch to this workspace" Checkbox, which is `compact=True`.

    Deliberately NOT the non-compact "Show archived" Checkbox in Settings >
    Workspaces the task brief also named: investigation (measured against
    both the unfixed and fixed bundle) found that widget is squeezed to
    ZERO content rows by a separate, pre-existing, focus-INDEPENDENT bug --
    `Checkbox { width: 100%; height: 2; }`, an unscoped rule in features/
    _conversations.tcss, collides with `ToggleButton`'s own `border: tall`
    (2 rows) even while BLURRED, leaving no row for the label at all. That
    is a different root cause than this task's outline-over-content-row
    family and out of this fix's bounded scope; a compact Checkbox isolates
    the family this task actually fixes (blurred renders fine, only focus
    breaks it) and is the widget the original bug report was filed against.
    """
    app = ProductionCssWidgetHarness(
        lambda: Checkbox(
            "Switch to this workspace",
            True,
            id="probe-checkbox",
            compact=True,
        )
    )
    async with app.run_test(size=(60, 10)) as pilot:
        widget = app.query_one("#probe-checkbox", Checkbox)
        await pilot.pause()
        assert not widget.has_focus
        blurred_frame = _rendered_text(app)
        assert "Switch to this workspace" in blurred_frame, (
            f"blurred compact Checkbox did not render its label; frame:\n{blurred_frame!r}"
        )

        await pilot.press("tab")
        await pilot.pause()
        assert widget.has_focus

        frame = _rendered_text(app)
        assert "Switch to this workspace" in frame, (
            "a focused compact Checkbox's label is not on screen -- the global "
            f"focus outline is overwriting its rendered row(s); frame:\n{frame!r}"
        )


@pytest.mark.asyncio
async def test_plain_input_focused_frame_still_shows_its_value() -> None:
    """Guard: a plain (non-compact) Input keeps today's outline+border
    behavior -- it was never touched by this fix. `Input:focus` in
    `components/_forms.tcss` recolours its OWN `border: tall` (3 rows), so
    the value lives on the middle row, off the perimeter the outline paints
    over -- unlike the compact case above."""
    app = ProductionCssWidgetHarness(
        lambda: Input(value="plain-value", id="probe-plain-input")
    )
    async with app.run_test(size=(60, 10)) as pilot:
        widget = app.query_one("#probe-plain-input", Input)
        await pilot.press("tab")
        await pilot.pause()
        assert widget.has_focus

        frame = _rendered_text(app)
        assert "plain-value" in frame, (
            f"a focused plain Input lost its value; frame:\n{frame!r}"
        )
