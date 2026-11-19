"""Text never folds at a zero fold width (TASK-38740).

Textual 8.2.8 folds a line with ``range(0, length, width)`` (no wrap) or
Rich's ``chop_cells`` (wrap). Both raise ``ValueError: range() arg 3 must not
be zero`` at a fold width of 0, and a raise in layout exits the app. A Button
reaches it when its content box is exactly its two ``line-pad`` cells (the
Settings Workspaces and Console behavior rows at 44-46 columns, the File Notes
toolbar in a 4-cell grid column); an OptionList option when its box is 0
(Settings > Theme). No stylesheet rule can rule those widths out, so the
package's Textual shims guard the one method every height and paint calls.

These tests pin that guard: it is installed, nothing raises at any width in
either wrap mode, the zero fold width renders one line inside the box, and
the upstream methods still route through it (an upgrade that stops calling
``_wrap_and_format`` fails here instead of quietly dropping the guard).
TASK-38952 widened it: a wrapped box at or below its line-pad (0 to
``2 * line_pad``) renders the same cropped line, and a wrapped ``ellipsis`` or
``clip`` line wider than the box is cut to fit before its pad. Widths whose
lines already fit render exactly as upstream does. Gate-free: no app boots.
"""

from __future__ import annotations

import inspect

import pytest
from rich.cells import cell_len
from textual.content import Content

import tldw_chatbook.Widgets  # noqa: F401  (installs the shims, as the app does)
from tldw_chatbook import ZERO_WIDTH_FOLD_GUARD

_LABELS = ("Refresh", "Back to session", "x", "")


def _lines(wrap_and_format, label, width, *, no_wrap, line_pad):
    lines = wrap_and_format(Content(label), width, no_wrap=no_wrap, line_pad=line_pad)
    return [(line.content.plain, line.x, line.y) for line in lines]


def test_every_height_and_paint_goes_through_the_guard():
    assert getattr(Content._wrap_and_format, ZERO_WIDTH_FOLD_GUARD, False)
    for method in (Content.get_height, Content.render_strips):
        assert "._wrap_and_format(" in inspect.getsource(method), method.__name__


@pytest.mark.parametrize("no_wrap", (False, True), ids=("wrap", "nowrap"))
@pytest.mark.parametrize("line_pad", (0, 1), ids=("pad0", "button-pad1"))
def test_no_width_raises_and_lines_that_fit_render_as_upstream(no_wrap, line_pad):
    guarded = Content._wrap_and_format
    upstream = getattr(guarded, "__wrapped__", guarded)
    zero = 0 if no_wrap else 2 * line_pad
    rules = {"line_pad": line_pad, "text_wrap": "nowrap" if no_wrap else "wrap"}
    for label in _LABELS:
        for width in range(-1, 7):
            Content(label).get_height(rules, width)  # layout: a grid's auto row
            got = _lines(guarded, label, width, no_wrap=no_wrap, line_pad=line_pad)
            # TASK-38952, rewritten on purpose: a wrapped box narrower than its
            # line-pad crops the same way (it painted the pad past the box).
            if width == zero or (not no_wrap and line_pad and 0 <= width < zero):
                if label and width == zero:
                    with pytest.raises(ValueError):
                        _lines(
                            upstream, label, width, no_wrap=no_wrap, line_pad=line_pad
                        )
                # One line, inside the box: no line-pad added on top.
                assert [cell_len(text) <= width for text, _x, _y in got] == [True]
                continue
            want = _lines(upstream, label, width, no_wrap=no_wrap, line_pad=line_pad)
            assert got == want, (label, width)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("button_css", "box"), (("width: 1; padding: 0;", 1), ("width: 3; padding: 0 1;", 3))
)
async def test_a_one_cell_label_paints_inside_its_button(button_css, box):
    """TASK-38952 AC#1: at a 1-cell content box the wrapped label was cut to 1
    cell and then given its line-pad, so ``Save`` painted 3 cells (5 with
    ``padding: 0 1``) and pushed the widget beside it 2 cells right."""
    from textual.app import App
    from textual.containers import Horizontal
    from textual.widgets import Button, Static

    class _Row(App):
        CSS = f"""
        Horizontal {{ height: 1; }}
        Button {{ {button_css} min-width: 0; height: 1; border: none; }}
        Static {{ width: 1fr; }}
        """

        def compose(self):
            with Horizontal():
                yield Button("Save")
                yield Static("XXXX")

    app = _Row()
    async with app.run_test(size=(40, 6)) as pilot:
        await pilot.pause()
        button = app.query_one(Button)
        static = app.query_one(Static)
        assert button.region.width == box
        painted = [
            cell_len(strip.text)
            for strip in button.render_lines(button.region.reset_offset)
        ]
        assert max(painted) <= box, painted
        row = app.screen._compositor.render_strips()[static.region.y].text
        assert row[static.region.x : static.region.x + 4] == "XXXX", row


@pytest.mark.parametrize("overflow", ("ellipsis", "clip"))
def test_a_wrapped_line_never_paints_past_its_box(overflow):
    """TASK-38952: an unbreakable word was cut to the full width and then
    padded, so it painted 2 cells past its box (``<---`` in 4 cells)."""
    for label in ("<---", "Refresh", "Back to session", "a  b"):
        for width in range(12):
            lines = Content(label)._wrap_and_format(
                width, overflow=overflow, line_pad=1
            )
            assert all(
                line.content.cell_length <= width for line in lines
            ), (label, width, [line.content.plain for line in lines])
