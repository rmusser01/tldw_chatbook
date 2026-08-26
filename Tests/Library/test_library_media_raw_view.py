"""Tests for the virtualized Raw content view (TASK-22500).

Renders through :class:`WrapIndex` -- render cost must not scale with
document size, and virtual height must track the wrap index exactly.
"""

import pytest
from textual.app import App, ComposeResult
from textual.events import MouseMove
from textual.geometry import Offset
from textual.selection import Selection

from tldw_chatbook.Widgets.Library.library_media_raw_view import VirtualizedRawContent

DOC = "\n".join(f"line {i} " + "alpha beta gamma " * 4 for i in range(2000))


class _Harness(App):
    def __init__(self, content: str) -> None:
        super().__init__()
        self._content = content

    def compose(self) -> ComposeResult:
        yield VirtualizedRawContent(
            content=self._content, query="", match_index=0, id="raw"
        )


@pytest.mark.asyncio
async def test_renders_only_visible_rows_regardless_of_document_size():
    """The AC's guard: render cost must not scale with the document."""
    small = "\n".join(f"line {i}" for i in range(50))
    counts = {}
    for label, doc in (("small", small), ("large", DOC)):
        app = _Harness(doc)
        async with app.run_test(size=(100, 40)) as pilot:
            widget = app.query_one("#raw", VirtualizedRawContent)
            widget.RENDER_LINE_CALLS["n"] = 0
            widget.refresh()
            await pilot.pause()
            counts[label] = widget.RENDER_LINE_CALLS["n"]
    assert counts["small"] <= 60
    assert counts["large"] <= 60, f"large document rendered {counts['large']} rows"


@pytest.mark.asyncio
async def test_virtual_height_reflects_wrapped_rows():
    app = _Harness("x" * 250)
    async with app.run_test(size=(100, 40)) as pilot:
        widget = app.query_one("#raw", VirtualizedRawContent)
        await pilot.pause()
        assert widget.wrap_index is not None
        assert widget.virtual_size.height == widget.wrap_index.virtual_height
        assert widget.virtual_size.height >= 3


@pytest.mark.asyncio
async def test_short_document_stays_compact_and_long_document_is_capped():
    """CSS gives the body height:auto/max-height:18; the widget must not
    request its virtual height or the pane balloons to tens of thousands
    of rows."""
    app = _Harness("one\ntwo\nthree")
    async with app.run_test(size=(100, 40)) as pilot:
        widget = app.query_one("#raw", VirtualizedRawContent)
        await pilot.pause()
        assert widget.styles.height.value == 3
    app = _Harness(DOC)
    async with app.run_test(size=(100, 40)) as pilot:
        widget = app.query_one("#raw", VirtualizedRawContent)
        await pilot.pause()
        assert widget.styles.height.value == 18


@pytest.mark.asyncio
async def test_selection_across_a_wrap_boundary_returns_source_text():
    """Static gave drag-select for free; a custom widget loses it silently
    unless get_selection is implemented."""
    app = _Harness("alpha " * 40 + "\nsecond line here")
    async with app.run_test(size=(40, 20)) as pilot:
        widget = app.query_one("#raw", VirtualizedRawContent)
        await pilot.pause()
        assert widget.allow_select is True
        got = widget.get_selection(Selection(Offset(0, 0), Offset(5, 2)))
        assert got is not None
        selected, _ = got
        assert selected.startswith("alpha")
        assert "\n" not in selected.rstrip("\n") or selected.count("\n") <= 1


@pytest.mark.asyncio
async def test_square_brackets_render_literally():
    """markup=False parity: an unescaped [Imported] must not vanish."""
    app = _Harness("prefix [Imported] suffix")
    async with app.run_test(size=(60, 10)) as pilot:
        widget = app.query_one("#raw", VirtualizedRawContent)
        await pilot.pause()
        strip = widget.render_line(0)
        assert "[Imported]" in strip.text


@pytest.mark.asyncio
async def test_selection_all_returns_whole_document_with_real_line_breaks():
    """``Selection(None, None)`` is Textual's SELECT_ALL. It must return the
    full document, with a newline only at real source-line boundaries, not
    at every wrapped row."""
    content = "alpha " * 40 + "\nsecond line here"
    app = _Harness(content)
    async with app.run_test(size=(40, 20)) as pilot:
        widget = app.query_one("#raw", VirtualizedRawContent)
        await pilot.pause()
        got = widget.get_selection(Selection(None, None))
        assert got is not None
        selected, _ = got
        assert selected == content


@pytest.mark.asyncio
async def test_real_mouse_drag_within_a_wrapped_line_selects_contiguous_text():
    """A REAL drag (as opposed to a hand-built Selection) only reaches
    ``get_selection`` with a meaningful per-cell offset if ``render_line``
    embeds content-offset meta into its Strips. Without that, Textual's
    compositor resolves every point on this widget to a `None` content
    offset and silently downgrades the drag to "select the whole widget"
    (``Selection(None, None)``) -- this is the failure mode the unit-level
    ``get_selection`` test alone cannot see."""
    content = "alpha " * 60
    app = _Harness(content)
    async with app.run_test(size=(40, 20)) as pilot:
        widget = app.query_one("#raw", VirtualizedRawContent)
        await pilot.pause()
        assert widget.wrap_index.virtual_height >= 2

        await pilot.mouse_down(widget, offset=(0, 0))
        await pilot._post_mouse_events([MouseMove], widget=widget, offset=(3, 1))
        await pilot.mouse_up(widget, offset=(3, 1))
        await pilot.pause()

        selected = app.screen.get_selected_text()
        assert selected == "alpha alpha alpha alpha alpha alpha alph"
        assert "\n" not in selected
        assert content.startswith(selected)


@pytest.mark.asyncio
async def test_real_mouse_drag_across_two_source_lines_inserts_exactly_one_newline():
    """Same real-drag path as above, but the drag crosses an actual source
    line boundary -- exactly one newline must appear, at the right place."""
    content = "first line\nsecond line"
    app = _Harness(content)
    async with app.run_test(size=(40, 20)) as pilot:
        widget = app.query_one("#raw", VirtualizedRawContent)
        await pilot.pause()

        await pilot.mouse_down(widget, offset=(0, 0))
        await pilot._post_mouse_events([MouseMove], widget=widget, offset=(6, 1))
        await pilot.mouse_up(widget, offset=(6, 1))
        await pilot.pause()

        selected = app.screen.get_selected_text()
        assert selected == "first line\nsecond "
        assert selected.count("\n") == 1


@pytest.mark.asyncio
async def test_real_mouse_drag_paints_the_selection_style_only_over_dragged_cells():
    """``get_selection`` returning the right text is not the same as the drag
    being visible. A hand-rolled ``render_line`` does not get Static's
    automatic selection highlight (from ``Visual.to_strips`` reading
    ``widget.text_selection``) for free -- it must apply
    ``screen.get_component_rich_style("screen--selection")`` itself, the
    same component style ``Static``/``Log`` use. This asserts the RENDERED
    strips carry that style over the covered cells and nowhere else, which a
    ``get_selection``-only test cannot see."""
    content = "alpha " * 60
    app = _Harness(content)
    async with app.run_test(size=(40, 20)) as pilot:
        widget = app.query_one("#raw", VirtualizedRawContent)
        await pilot.pause()

        def bgcolor_at(strip, x):
            return list(strip.crop(x, x + 1))[0].style.bgcolor

        # Before any drag: nothing is styled.
        assert bgcolor_at(widget.render_line(0), 0) is None

        # Drag from row 0 col 3 to row 1 col 2 -- a wrap-boundary selection
        # that is partial on BOTH rendered rows of this single wrapped
        # source line.
        await pilot.mouse_down(widget, offset=(3, 0))
        await pilot._post_mouse_events([MouseMove], widget=widget, offset=(2, 1))
        await pilot.mouse_up(widget, offset=(2, 1))
        await pilot.pause()

        selection_bg = app.screen.get_component_rich_style("screen--selection").bgcolor
        assert selection_bg is not None

        row0 = widget.render_line(0)
        row1 = widget.render_line(1)
        row5 = widget.render_line(5)  # far outside the drag

        # Row 0 (first row): covered from column 3 onward, NOT before it.
        assert bgcolor_at(row0, 0) is None
        assert bgcolor_at(row0, 1) is None
        assert bgcolor_at(row0, 2) is None
        assert bgcolor_at(row0, 3) == selection_bg
        assert bgcolor_at(row0, 7) == selection_bg

        # Row 1 (last row): covered up to column 3, NOT after it.
        assert bgcolor_at(row1, 0) == selection_bg
        assert bgcolor_at(row1, 1) == selection_bg
        assert bgcolor_at(row1, 2) == selection_bg
        assert bgcolor_at(row1, 3) is None
        assert bgcolor_at(row1, 4) is None

        # An undragged row elsewhere in the same document: untouched.
        for x in range(4):
            assert bgcolor_at(row5, x) is None


@pytest.mark.asyncio
async def test_select_all_highlights_every_visible_row():
    """``Selection(None, None)`` (SELECT_ALL) must highlight every row, not
    just the ones a partial drag would touch."""
    content = "one\ntwo\nthree"
    app = _Harness(content)
    async with app.run_test(size=(40, 20)) as pilot:
        widget = app.query_one("#raw", VirtualizedRawContent)
        await pilot.pause()
        app.screen.selections = {widget: Selection(None, None)}
        await pilot.pause()

        selection_bg = app.screen.get_component_rich_style("screen--selection").bgcolor
        for row in range(3):
            strip = widget.render_line(row)
            bgcolor = list(strip.crop(0, 1))[0].style.bgcolor
            assert bgcolor == selection_bg
