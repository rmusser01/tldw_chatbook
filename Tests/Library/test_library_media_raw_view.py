"""Tests for the virtualized Raw content view (TASK-22500).

Renders through :class:`WrapIndex` -- render cost must not scale with
document size, and virtual height must track the wrap index exactly.
"""

import time

import pytest
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.events import MouseMove
from textual.geometry import Offset
from textual.selection import Selection
from textual.widgets import Static

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
async def test_render_line_self_time_does_not_scale_with_document_size():
    """Coverage gap closed (task-22500 task 9): the call-count guard above
    cannot see a regression that keeps the call count flat but reintroduces
    O(document) work INSIDE each call -- a reviewer measured 0.26 ms cached
    vs 364 ms if a call rebuilt the index from scratch, a ~1400x difference
    invisible to a count-only assertion. This asserts on render_line's own
    per-call wall time instead.

    Measured DIRECTLY: ``render_line`` is called and timed with
    ``time.perf_counter()`` outside of ``pilot.pause()`` entirely --
    ``pilot.pause()`` costs ~30 ms/call from event-loop scheduling alone,
    which would swamp the sub-millisecond cost this test needs to see. The
    only use of the pilot here is to reach a real first paint so the wrap
    index exists.
    """
    app = _Harness(DOC)
    async with app.run_test(size=(100, 40)) as pilot:
        widget = app.query_one("#raw", VirtualizedRawContent)
        await pilot.pause()
        assert widget.wrap_index is not None
        # Warm the segment cache exactly like a real repaint would (the
        # first touch of a line populates WrapIndex._segment_cache); an
        # unwarmed first call is not what "one more repaint" costs.
        visible_rows = range(min(40, widget.wrap_index.virtual_height))
        for y in visible_rows:
            widget.render_line(y)

        t0 = time.perf_counter()
        repeats = 25
        for _ in range(repeats):
            for y in visible_rows:
                widget.render_line(y)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        calls = repeats * len(visible_rows)
        per_call_ms = elapsed_ms / calls

        # A real (cached) call costs a fraction of a millisecond; rebuilding
        # any O(document) structure inside the call costs tens to hundreds
        # of ms per call on this 2000-line fixture. 2 ms/call is generous
        # headroom over real cost while still being far below what any
        # document-sized rebuild inside the call would cost.
        assert per_call_ms < 2.0, (
            f"render_line averaged {per_call_ms:.3f} ms/call over {calls} calls "
            f"({elapsed_ms:.1f} ms total) -- self-time must stay a small "
            "constant, not scale with document size"
        )


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
    of rows.

    The cap is the room the PARENT actually has, never a constant: the
    container's max-height is an outer bound that its border rows come out
    of, so pinning a literal here is what let the widget overflow its
    parent by exactly the border and strand the tail of the document
    (see test_the_last_row_is_reachable_under_production_css).
    """
    app = _Harness("one\ntwo\nthree")
    async with app.run_test(size=(100, 40)) as pilot:
        widget = app.query_one("#raw", VirtualizedRawContent)
        await pilot.pause()
        assert widget.styles.height.value == 3
    app = _Harness(DOC)
    async with app.run_test(size=(100, 40)) as pilot:
        widget = app.query_one("#raw", VirtualizedRawContent)
        await pilot.pause()
        available = widget.parent.content_region.height
        assert widget.wrap_index.virtual_height > available, "fixture too short to cap"
        assert widget.styles.height.value == available


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


class _StaticHarness(App):
    """Hosts the ``Static`` widget being replaced, for equivalence checks.

    Applies the same empty-content substitution the production body does
    before ``Static`` ever sees the string (``library_media_content.py``'s
    ``content or "No stored content."``) -- ``VirtualizedRawContent``
    performs this substitution internally, so leaving it out here would
    compare "No stored content." against "" for a reason that has nothing
    to do with rendering fidelity.
    """

    def __init__(self, content: str) -> None:
        super().__init__()
        self._content = content

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Static(self._content or "No stored content.", id="old", markup=False)


@pytest.mark.parametrize(
    "content",
    [
        "plain short line",
        "wrapping " * 40,
        "unicode ✓ wide 日本語 text " * 10,
        "tabbed\tcolumns\there",
        "trailing newline\n",
        "",
        "[Imported] literal brackets",
    ],
    ids=["short", "wrapped", "unicode", "tabs", "trailing", "empty", "markup"],
)
@pytest.mark.asyncio
async def test_first_rows_match_the_static_they_replace(content):
    """The widget must paint what Static painted for the same document."""
    old_rows = []
    app = _StaticHarness(content)
    async with app.run_test(size=(40, 12)) as pilot:
        await pilot.pause()
        widget = app.query_one("#old", Static)
        old_rows = [widget.render_line(y).text.rstrip() for y in range(4)]
    new_rows = []
    app = _Harness(content)
    async with app.run_test(size=(40, 12)) as pilot:
        await pilot.pause()
        widget = app.query_one("#raw", VirtualizedRawContent)
        new_rows = [widget.render_line(y).text.rstrip() for y in range(4)]
    assert new_rows == old_rows


async def _drag_and_get_selected_text(app_cls, content, down, up, widget_id, widget_type):
    """Perform a real mouse drag and read back the resulting selection.

    Shared by the Static-vs-VirtualizedRawContent tab comparison tests below
    so both widgets undergo the exact same down/move/up sequence.
    """
    app = app_cls(content)
    async with app.run_test(size=(40, 12)) as pilot:
        await pilot.pause()
        widget = app.query_one(widget_id, widget_type)
        await pilot.mouse_down(widget, offset=down)
        await pilot._post_mouse_events([MouseMove], widget=widget, offset=up)
        await pilot.mouse_up(widget, offset=up)
        await pilot.pause()
        return app.screen.get_selected_text()


@pytest.mark.asyncio
async def test_full_line_drag_over_tabs_matches_static():
    """A drag spanning the whole (tab-bearing) line must copy literal tabs,
    identically to Static -- not the spaces the wrap/paint path uses
    internally to match Static's rendered columns."""
    content = "tabbed\tcolumns\there"
    static_selected = await _drag_and_get_selected_text(
        _StaticHarness, content, (0, 0), (30, 0), "#old", Static
    )
    raw_selected = await _drag_and_get_selected_text(
        _Harness, content, (0, 0), (30, 0), "#raw", VirtualizedRawContent
    )
    assert raw_selected == static_selected
    assert "\t" in raw_selected


@pytest.mark.asyncio
async def test_partial_drag_across_a_tab_matches_static():
    """A drag that starts and ends mid-line, straddling a tab, must produce
    byte-identical text to Static -- including Static's own quirk where a
    release landing on the tab's second expanded display cell already reads
    one raw character past the tab (the drag lands at column 7, inside the
    tab's 2-column expanded span that starts at column 6)."""
    content = "tabbed\tcolumns\there"
    static_selected = await _drag_and_get_selected_text(
        _StaticHarness, content, (0, 0), (7, 0), "#old", Static
    )
    raw_selected = await _drag_and_get_selected_text(
        _Harness, content, (0, 0), (7, 0), "#raw", VirtualizedRawContent
    )
    assert raw_selected == static_selected
    assert raw_selected == "tabbed\tc"


@pytest.mark.asyncio
async def test_a_resize_burst_reindexes_once():
    """Re-indexing costs ~125-155 ms on a 2.5 MB document; a drag-resize
    must not pay it per event (TASK-22211's hysteresis precedent)."""
    app = _Harness(DOC)
    async with app.run_test(size=(100, 40)) as pilot:
        widget = app.query_one("#raw", VirtualizedRawContent)
        await pilot.pause()
        builds = {"n": 0}
        original = widget._build_index_now

        def counting(width):
            builds["n"] += 1
            return original(width)

        widget._build_index_now = counting
        # End the burst on the width the widget is actually painted at.
        # render_line re-arms a rebuild whenever the indexed width differs
        # from the real render width (that convergence is what stops a
        # scrollbar appearing mid-life from truncating every row), so a
        # burst ending on a fictional width would legitimately cost a
        # second, corrective rebuild and mask what this test measures.
        painted = widget.scrollable_content_region.width or widget.size.width
        for width in (painted + 4, painted + 3, painted + 2, painted + 1, painted):
            widget._request_reindex(width)
        await pilot.pause(0.3)
        assert builds["n"] == 1, f"re-indexed {builds['n']} times for one burst"


@pytest.mark.asyncio
async def test_mount_indexes_synchronously_without_debounce():
    """First paint must not be delayed: on_mount must have a wrap index
    available immediately after mount, before the debounce interval would
    ever elapse."""
    app = _Harness(DOC)
    async with app.run_test(size=(100, 40)):
        widget = app.query_one("#raw", VirtualizedRawContent)
        # No pause at all: on_mount must have already built the index
        # synchronously, not scheduled it behind REINDEX_DEBOUNCE_SECONDS.
        assert widget.wrap_index is not None


@pytest.mark.asyncio
async def test_pending_reindex_does_not_fire_after_unmount_via_textual_cleanup():
    """A debounce timer armed just before unmount must not touch the
    widget once it has been removed from the DOM, via the ordinary path.

    NOTE: this alone does not prove `_fire_pending_reindex`'s own
    `is_attached` guard (or `on_unmount`'s `timer.stop()`) does anything --
    Textual's `MessagePump._close_messages` keeps every `set_timer` timer in
    a `WeakSet` and auto-cancels them on widget shutdown, independent of
    this widget's code, so this test stays green even with both guards
    deleted. It is kept as a test of the ordinary, real-world path; the
    guard itself is proven by
    `test_fire_pending_reindex_noops_when_called_directly_after_unmount`
    below, which bypasses Textual's timer cleanup entirely.
    """
    app = _Harness(DOC)
    async with app.run_test(size=(100, 40)) as pilot:
        widget = app.query_one("#raw", VirtualizedRawContent)
        await pilot.pause()
        builds = {"n": 0}
        original = widget._build_index_now

        def counting(width):
            builds["n"] += 1
            return original(width)

        widget._build_index_now = counting
        widget._request_reindex(42)
        await widget.remove()
        await pilot.pause(0.3)
        assert builds["n"] == 0, "reindex fired into a detached widget"


@pytest.mark.asyncio
async def test_fire_pending_reindex_noops_when_called_directly_after_unmount():
    """Proves `_fire_pending_reindex`'s own `is_attached` guard, not
    Textual's timer cleanup.

    Textual's `MessagePump._close_messages` (`message_pump.py`) auto-cancels
    every `set_timer` timer on widget shutdown via a `WeakSet`, entirely
    independent of this widget's code -- so a test that only removes the
    widget and waits for the REAL timer to (not) fire can pass even with
    both of this widget's own defenses (`on_unmount`'s `timer.stop()` and
    `_fire_pending_reindex`'s `is_attached` check) deleted. This test
    bypasses that cleanup by calling `_fire_pending_reindex()` directly
    after removal -- exactly what would happen if a timer ever fired
    through a path Textual itself did not clean up -- so it reds if the
    `is_attached` guard is ever removed.
    """
    app = _Harness(DOC)
    async with app.run_test(size=(100, 40)) as pilot:
        widget = app.query_one("#raw", VirtualizedRawContent)
        await pilot.pause()
        builds = {"n": 0}
        original = widget._build_index_now

        def counting(width):
            builds["n"] += 1
            return original(width)

        widget._build_index_now = counting
        widget._request_reindex(42)
        await widget.remove()
        widget._fire_pending_reindex()
        assert builds["n"] == 0, "reindex fired into a detached widget"


@pytest.mark.asyncio
async def test_partial_drag_starting_and_ending_mid_line_across_a_tab_matches_static():
    """Same acid test as above, but with BOTH endpoints mid-line (neither at
    column 0 nor past the end), which is the more common real-world drag."""
    content = "tabbed\tcolumns\there"
    static_selected = await _drag_and_get_selected_text(
        _StaticHarness, content, (3, 0), (10, 0), "#old", Static
    )
    raw_selected = await _drag_and_get_selected_text(
        _Harness, content, (3, 0), (10, 0), "#raw", VirtualizedRawContent
    )
    assert raw_selected == static_selected
    assert "\t" in raw_selected


# ---------------------------------------------------------------------------
# FINAL WHOLE-BRANCH REVIEW FINDING 1: a match straddling a wrap boundary
# used to lose its highlight entirely (the old code searched each RENDERED
# segment independently instead of the SOURCE line, so a needle split across
# two rows matched neither segment's own substring). These tests pin the
# fix against the retired Static-backed highlighter's exact contract: one
# `str.find` per SOURCE line, first occurrence only, styled `reverse` (or
# `reverse bold` for the active match) -- reconstructed here since
# `build_raw_content_highlight_plan` itself was deleted when the raw view
# was virtualized (commit 75a3bfc01a).
# ---------------------------------------------------------------------------


def _highlighted_static_text(content: str, query: str) -> Text:
    """Rebuild the retired ``build_raw_content_highlight_plan``'s output.

    One ``str.find`` per SOURCE line (case-insensitive), first occurrence
    only, always styled as the ACTIVE match (``reverse bold``) -- matching
    what ``RawContentHighlightPlan.renderable(match_index)`` produced for
    whichever line is the current match. Every fixture below has exactly
    one matching line, so "the current match" and "the only match" are the
    same line and there is no plain-vs-active distinction to reconstruct.

    Args:
        content: Source text to display.
        query: Case-insensitive search text to highlight.

    Returns:
        A Rich ``Text`` with the first per-line occurrence of ``query``
        styled ``"reverse bold"``.
    """
    needle = query.strip().lower()
    text = Text()
    for index, line in enumerate(content.split("\n")):
        if index:
            text.append("\n")
        hit = line.lower().find(needle) if needle else -1
        if hit < 0:
            text.append(line)
            continue
        text.append(line[:hit])
        text.append(line[hit : hit + len(needle)], style="reverse bold")
        text.append(line[hit + len(needle) :])
    return text


class _QueryStaticHarness(App):
    """Hosts the pre-virtualization Static, highlighted like the retired plan did."""

    def __init__(self, content: str, query: str) -> None:
        super().__init__()
        self._content = content
        self._query = query

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Static(
                _highlighted_static_text(self._content, self._query),
                id="old",
                markup=False,
            )


class _QueryHarness(App):
    """Hosts the virtualized widget with a live query and a single active match."""

    def __init__(self, content: str, query: str) -> None:
        super().__init__()
        self._content = content
        self._query = query

    def compose(self) -> ComposeResult:
        yield VirtualizedRawContent(
            content=self._content, query=self._query, match_index=0, id="raw"
        )


def _style_run(strip) -> list[tuple[bool, bool]]:
    """Expand a Strip's segments into one ``(reverse, bold)`` pair per cell."""
    cells: list[tuple[bool, bool]] = []
    for segment in strip._segments:
        style = segment.style
        reverse = bool(style and style.reverse)
        bold = bool(style and style.bold)
        cells.extend([(reverse, bold)] * len(segment.text))
    return cells


async def _rendered_rows(app_cls, content, query, widget_id, widget_type, width, rows):
    app = app_cls(content, query)
    async with app.run_test(size=(width, rows + 4)) as pilot:
        await pilot.pause()
        widget = app.query_one(widget_id, widget_type)
        if widget_type is VirtualizedRawContent:
            widget.set_match_lines((0,))
            widget.sync_search(query, 0)
        return [widget.render_line(y) for y in range(rows)]


def _assert_same_highlight(static_strip, raw_strip, *, row: int) -> None:
    """Compare per-cell (reverse, bold) styling, ignoring trailing padding.

    ``VirtualizedRawContent.render_line`` pads every Strip out to the full
    viewport width via ``adjust_cell_length`` (so the compositor always gets
    a full-width row); ``Static`` does not -- it only emits cells for its
    actual (wrapped) text. That padding is a display detail unrelated to
    highlight fidelity, so this truncates to Static's own (unpadded) length
    before comparing, then confirms none of the raw view's extra padding
    cells are styled either (a real regression could not "hide" a wrongly
    painted cell out there without failing this).
    """
    static_cells = _style_run(static_strip)
    raw_cells = _style_run(raw_strip)
    assert raw_cells[: len(static_cells)] == static_cells, (
        f"row {row} highlight mismatch: static={static_cells} "
        f"raw={raw_cells[: len(static_cells)]}"
    )
    assert not any(reverse for reverse, _bold in raw_cells[len(static_cells) :]), (
        f"row {row}: padding past the real text must never carry the match style"
    )


@pytest.mark.asyncio
async def test_highlight_spans_a_wrap_boundary_matching_static():
    """The exact proof case from the review: a needle split across a wrap
    boundary must be painted -- partially, where clipped -- on BOTH rows it
    covers, exactly like Static painted it when Rich wrapped one styled
    span across two rows on its own."""
    line = "x" * 40 + "NEEDLE" + "y" * 20
    width = 42
    query = "NEEDLE"

    static_rows = await _rendered_rows(
        _QueryStaticHarness, line, query, "#old", Static, width, 2
    )
    raw_rows = await _rendered_rows(
        _QueryHarness, line, query, "#raw", VirtualizedRawContent, width, 2
    )

    for y in range(2):
        _assert_same_highlight(static_rows[y], raw_rows[y], row=y)

    # Guard against a vacuous pass: both rows must actually carry SOME
    # reversed cells, or this would trivially agree by painting nothing.
    assert any(reverse for reverse, _bold in _style_run(static_rows[0]))
    assert any(reverse for reverse, _bold in _style_run(static_rows[1]))
    # "NE" on row 0, "EDLE" on row 1 -- the split the review reported.
    assert sum(reverse for reverse, _bold in _style_run(static_rows[0])) == 2
    assert sum(reverse for reverse, _bold in _style_run(static_rows[1])) == 4


@pytest.mark.asyncio
async def test_highlight_styles_only_the_first_occurrence_on_a_wrapped_line():
    """Second divergence from the review: a wrapped line with TWO
    occurrences of the query must only highlight the FIRST -- matching
    ``build_raw_content_highlight_plan``'s single ``str.find`` per line --
    not both, even though the second occurrence renders on its own,
    otherwise-unstyled row."""
    query = "needle"
    line = "needle" + "z" * 40 + "needle"  # first at [0:6], second at [46:52]
    width = 42

    static_rows = await _rendered_rows(
        _QueryStaticHarness, line, query, "#old", Static, width, 2
    )
    raw_rows = await _rendered_rows(
        _QueryHarness, line, query, "#raw", VirtualizedRawContent, width, 2
    )

    for y in range(2):
        _assert_same_highlight(static_rows[y], raw_rows[y], row=y)

    # Row 0 carries the first occurrence; row 1 (the second occurrence's
    # own row) must carry NOTHING -- proving only the first was styled.
    assert any(reverse for reverse, _bold in _style_run(static_rows[0]))
    assert not any(reverse for reverse, _bold in _style_run(static_rows[1]))
    assert not any(reverse for reverse, _bold in _style_run(raw_rows[1]))


# --- Geometry against the real CSS box (TASK-22500 review: C1/C2/C3) ---------
#
# The fidelity test above compares rstripped TEXT, which is structurally blind
# to three regressions the review found: rows wider than the widget (wrong
# Strip cell length), an unreachable last row (widget taller than its parent's
# CONTENT box), and rows silently truncated (index built at a width the widget
# is not painted at). These mount the widget in the production container's box
# -- `height: auto; max-height: 18; border: solid; padding: 0 1` -- and assert
# the properties that box actually has to satisfy.

_PRODUCTION_BOX_CSS = """
#body {
    height: auto;
    max-height: 18;
    min-height: 3;
    border: solid white;
    padding: 0 1;
}
#raw { width: 100%; overflow-x: hidden; }
"""


class _BoxedHarness(App):
    """Mirrors #library-media-viewer-content's real box around the widget."""

    CSS = _PRODUCTION_BOX_CSS

    def __init__(self, content: str) -> None:
        super().__init__()
        self._content = content

    def compose(self) -> ComposeResult:
        from textual.containers import Container

        with Container(id="body"):
            yield VirtualizedRawContent(
                content=self._content, query="", match_index=0, id="raw"
            )


def _visible_text(widget: VirtualizedRawContent) -> list[str]:
    height = widget.scrollable_content_region.height
    return [widget.render_line(y).text.rstrip() for y in range(height)]


@pytest.mark.asyncio
async def test_the_last_row_is_reachable_under_production_css():
    """The container's max-height is an OUTER bound: its two border rows are
    not available to the child. A child that claims the full max-height
    overflows by exactly the border, and ScrollView still computes
    max_scroll_y against the height it thinks it has -- so the final rows of
    every long document become unreachable by scrolling."""
    doc = "\n".join(f"line {i}" for i in range(200))
    app = _BoxedHarness(doc)
    async with app.run_test(size=(100, 40)) as pilot:
        widget = app.query_one("#raw", VirtualizedRawContent)
        await pilot.pause()
        body = app.query_one("#body")
        assert widget.size.height <= body.content_region.height, (
            f"widget claims {widget.size.height} rows inside a "
            f"{body.content_region.height}-row content box"
        )
        widget.scroll_end(animate=False)
        await pilot.pause()
        assert "line 199" in "\n".join(_visible_text(widget))


@pytest.mark.asyncio
async def test_no_row_is_truncated_under_production_css():
    """The index is built from the width measured before scrollbars exist.
    If the widget later paints narrower than it indexed, every wrapped row
    loses the difference off its end -- cut, not re-flowed."""
    doc = "\n".join("abcdefghij" * 20 for _ in range(200))
    app = _BoxedHarness(doc)
    async with app.run_test(size=(60, 40)) as pilot:
        widget = app.query_one("#raw", VirtualizedRawContent)
        await pilot.pause(0.3)  # let any convergence rebuild settle
        painted = widget.scrollable_content_region.width
        assert widget._indexed_width == painted, (
            f"indexed at {widget._indexed_width}, painted at {painted}"
        )
        first_segment = widget.wrap_index.segments(0)[0]
        rendered = widget.render_line(0).text.rstrip()
        assert rendered == first_segment.rstrip()


@pytest.mark.asyncio
async def test_wide_glyph_rows_declare_their_true_cell_length():
    """Strip's second argument is a CELL count. Passing a CHARACTER count
    under-declares any 2-cell glyph, and adjust_cell_length then PADS
    instead of truncating -- emitting rows wider than the widget."""
    from rich.segment import Segment

    doc = "\n".join("unicode wide " + "日本語" * 6 for _ in range(20))
    app = _BoxedHarness(doc)
    async with app.run_test(size=(60, 40)) as pilot:
        widget = app.query_one("#raw", VirtualizedRawContent)
        await pilot.pause(0.3)
        width = widget.scrollable_content_region.width
        for y in range(widget.scrollable_content_region.height):
            strip = widget.render_line(y)
            real = Segment.get_line_length(strip._segments)
            assert strip.cell_length == real, f"row {y} declares {strip.cell_length}, is {real}"
            assert real == width, f"row {y} is {real} cells in a {width}-cell widget"


# --- Wide-glyph selection parity (TASK-22500, Qodo review) ------------------
#
# Textual hands a selection its columns in CELLS, while the raw line is
# indexed in CHARACTERS, and a CJK glyph is 2 cells wide but 1 character
# long. These drive real mouse drags over wide-glyph documents and require
# byte-identical output to the Static this widget replaces -- Static resolves
# the same coordinates through the compositor, so matching it is the whole
# contract. The text-only fidelity test above cannot see this: it never
# performs a selection.


@pytest.mark.asyncio
async def test_drag_over_wide_glyphs_matches_static():
    """A drag across CJK text must copy exactly what Static copies."""
    content = "日本語のテキストです and some ascii"
    static_selected = await _drag_and_get_selected_text(
        _StaticHarness, content, (0, 0), (20, 0), "#old", Static
    )
    raw_selected = await _drag_and_get_selected_text(
        _Harness, content, (0, 0), (20, 0), "#raw", VirtualizedRawContent
    )
    assert raw_selected == static_selected


@pytest.mark.asyncio
async def test_partial_drag_landing_inside_a_wide_glyph_matches_static():
    """Releasing on the SECOND cell of a 2-cell glyph is the ambiguous case:
    whatever Static resolves it to, this widget must resolve identically."""
    content = "ab日本語cd"
    for release_col in (3, 4, 5, 6):
        static_selected = await _drag_and_get_selected_text(
            _StaticHarness, content, (0, 0), (release_col, 0), "#old", Static
        )
        raw_selected = await _drag_and_get_selected_text(
            _Harness, content, (0, 0), (release_col, 0), "#raw", VirtualizedRawContent
        )
        assert raw_selected == static_selected, (
            f"release at cell {release_col}: {raw_selected!r} != {static_selected!r}"
        )


@pytest.mark.asyncio
async def test_select_all_over_wide_glyphs_matches_static():
    """SELECT_ALL must return the document verbatim, wide glyphs included."""
    content = "日本語 wide\nsecond 行 line\nplain ascii"
    app = _Harness(content)
    async with app.run_test(size=(40, 12)) as pilot:
        widget = app.query_one("#raw", VirtualizedRawContent)
        await pilot.pause()
        selected, _ = widget.get_selection(Selection(None, None))
        assert selected == content
