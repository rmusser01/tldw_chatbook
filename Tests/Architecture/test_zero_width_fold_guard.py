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
either wrap mode, the zero width renders one line inside the box, every other
width renders exactly as upstream does, and the upstream methods still route
through it (an upgrade that stops
calling ``_wrap_and_format`` fails here instead of quietly dropping the
guard). Gate-free: no app boots.
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
def test_no_width_raises_and_only_the_zero_fold_width_changes(no_wrap, line_pad):
    guarded = Content._wrap_and_format
    upstream = getattr(guarded, "__wrapped__", guarded)
    zero = 0 if no_wrap else 2 * line_pad
    rules = {"line_pad": line_pad, "text_wrap": "nowrap" if no_wrap else "wrap"}
    for label in _LABELS:
        for width in range(-1, 7):
            Content(label).get_height(rules, width)  # layout: a grid's auto row
            got = _lines(guarded, label, width, no_wrap=no_wrap, line_pad=line_pad)
            if width == zero:
                if label:
                    with pytest.raises(ValueError):
                        _lines(
                            upstream, label, width, no_wrap=no_wrap, line_pad=line_pad
                        )
                # One line, inside the box: no line-pad added on top.
                assert [cell_len(text) <= width for text, _x, _y in got] == [True]
                continue
            want = _lines(upstream, label, width, no_wrap=no_wrap, line_pad=line_pad)
            assert got == want, (label, width)
