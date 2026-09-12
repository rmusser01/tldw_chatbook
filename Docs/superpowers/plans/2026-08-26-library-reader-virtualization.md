# Library Reader Raw-Body Virtualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Library media reader's Raw view render only the rows in view, so first paint and every search/match repaint stop costing a full-document render.

**Architecture:** A pure wrap-index module (Rich only, no Textual) maps source lines to virtual rows using exact word-wrapping. A `ScrollView` subclass renders single rows on demand through that index. `LibraryMediaContentBody` becomes a container that hosts the virtualized scroller for Raw and today's `VerticalScroll` + `Markdown` for Rendered, exposing one `scroller` property so callers stop guessing the type.

**Tech Stack:** Python 3.12, Textual 8.2.8 (`ScrollView`, `Strip`, `render_line`), Rich (`rich._wrap.divide_line` behind an adapter, `rich.text.Text`), pytest.

**Spec:** `Docs/superpowers/specs/2026-08-26-library-reader-virtualization-design.md`

## Global Constraints

- Worktree: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/t22500`, branch `feat/task-22500-reader-virtualization`, base dev `732105c2d`.
- Run pytest as `.venv/bin/python -m pytest` **from the worktree root**; confirm `.venv/bin/python -c "import tldw_chatbook; print(tldw_chatbook.__file__)"` resolves inside this worktree before trusting any result.
- **Never assert performance on harness wall time.** `pilot.pause()` costs ~30 ms per call and dominates it (300 ms per 10 scroll steps in BOTH arms). Assert `render_line` self-time or call counts.
- The pure index module goes under `tldw_chatbook/Utils/`, never `tldw_chatbook/Library/`: `Library/__init__` eagerly imports a 66-module service stack (TASK-22223).
- `markup=False` semantics are preserved — a literal `[Imported]` must render as `[Imported]` (documented repo trap).
- Do not move the Library recompose ratchet, pinned at **74** in `Tests/UI/test_library_recompose_ratchet.py`.
- Existing reader gates must stay green: `Tests/UI/test_library_media_reader_traversal_t22207.py`, `test_library_media_reader_no_change_sync_t22208.py`, `test_library_media_reader_match_nav_t22209.py`.
- Run `./scripts/preflight.sh` before any PR; if the diagnostic inventory drifts, read the named rows before regenerating.
- Commit after every task. Never `git add -A`; stage explicit paths.

---

## File Structure

| File | Responsibility |
|---|---|
| `tldw_chatbook/Utils/text_wrap_index.py` (create) | Pure wrap arithmetic: the `divide_line` adapter, `WrapIndex` (row<->line mapping, virtual height), line-segment splitting. No Textual import. |
| `tldw_chatbook/Widgets/Library/library_media_raw_view.py` (create) | `VirtualizedRawContent(ScrollView)`: `render_line`, `virtual_size`, height rule, selection, highlight styling, `scroll_to_source_line`. |
| `tldw_chatbook/Widgets/Library/library_media_content.py` (modify) | `LibraryMediaContentBody` becomes a container with a `scroller` property; raw mode hosts the new widget; `RawContentHighlightPlan` loses its whole-document `Text` build. |
| `tldw_chatbook/UI/Screens/library_screen.py` (modify) | Three scroller lookups resolve `body.scroller`; `_scroll_library_media_content_to_line` delegates to the widget's exact mapping. |
| `tldw_chatbook/css/components/_agentic_terminal.tcss` (modify) | Height rule for the inner raw scroller; bundle regenerated. |
| `Tests/Utils/test_text_wrap_index.py` (create) | Wrap-index unit tests incl. the `divide_line` vs `Text.wrap` agreement pin. |
| `Tests/Library/test_library_media_raw_view.py` (create) | Widget behaviour: rows rendered, selection, highlight, height rule, perf guards. |
| `Tests/Library/test_library_media_content.py` (modify) | `raw.renderable` assertions convert to rendered-line assertions. |
| `Tests/UI/test_library_media_reader_scroller_resolution.py` (create) | The three call sites actually find their scroller (they fail silently today). |

---

### Task 1: Pure wrap index

**Files:**
- Create: `tldw_chatbook/Utils/text_wrap_index.py`
- Test: `Tests/Utils/test_text_wrap_index.py`

**Interfaces:**
- Consumes: nothing (leaf module; `rich` only).
- Produces: `divide_source_line(line: str, width: int) -> list[int]` (break offsets);
  `class WrapIndex` with `WrapIndex.build(lines: Sequence[str], width: int) -> WrapIndex`,
  `.virtual_height: int`, `.row_to_line(row: int) -> tuple[int, int]` returning
  `(line_index, segment_index)`, `.line_start_row(line_index: int) -> int`,
  `.segments(line_index: int) -> list[str]` (cached).

- [ ] **Step 1: Write the failing test**

```python
# Tests/Utils/test_text_wrap_index.py
import pytest
from rich.console import Console
from rich.text import Text

from tldw_chatbook.Utils.text_wrap_index import WrapIndex, divide_source_line


def test_divide_source_line_agrees_with_public_text_wrap():
    """Pins the private rich._wrap.divide_line against the public API.

    If a Rich upgrade moves or changes divide_line, this fails loudly here
    instead of silently changing how every document wraps.
    """
    console = Console(width=40)
    lines = [
        "short",
        "the quick brown fox jumps over the lazy dog and keeps on running forever",
        "supercalifragilisticexpialidocious " * 3,
        "",
    ]
    for line in lines:
        ours = len(divide_source_line(line, 40)) + 1
        theirs = max(1, len(Text(line).wrap(console, 40)))
        assert ours == theirs, f"row count disagreed for {line!r}"


def test_index_maps_rows_to_lines_and_segments():
    lines = ["a" * 10, "b" * 25, "c"]
    index = WrapIndex.build(lines, width=10)
    assert index.virtual_height == 1 + 3 + 1
    assert index.row_to_line(0) == (0, 0)
    assert index.row_to_line(1) == (1, 0)
    assert index.row_to_line(3) == (1, 2)
    assert index.row_to_line(4) == (2, 0)
    assert index.line_start_row(2) == 4


def test_segments_round_trip_the_source_line():
    index = WrapIndex.build(["hello world " * 5], width=17)
    assert "".join(index.segments(0)) == "hello world " * 5


def test_empty_document_has_one_row():
    index = WrapIndex.build([""], width=20)
    assert index.virtual_height == 1
    assert index.row_to_line(0) == (0, 0)
    assert index.segments(0) == [""]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest Tests/Utils/test_text_wrap_index.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tldw_chatbook.Utils.text_wrap_index'`

- [ ] **Step 3: Write minimal implementation**

```python
# tldw_chatbook/Utils/text_wrap_index.py
"""Exact wrap arithmetic for virtualized text views (TASK-22500).

Pure: Rich only, no Textual, no Library package import (``Library/__init__``
eagerly pulls a 66-module service stack -- TASK-22223).
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence

from rich.console import Console
from rich.text import Text

try:  # pragma: no cover - exercised by the agreement test
    from rich._wrap import divide_line as _rich_divide_line
except ImportError:  # pragma: no cover - fallback path
    _rich_divide_line = None

_FALLBACK_CONSOLE = Console(width=80)


def divide_source_line(line: str, width: int) -> list[int]:
    """Return the offsets at which ``line`` wraps at ``width``.

    Uses Rich's ``divide_line`` when available -- it is private, so the
    fallback re-derives the same breaks through the public ``Text.wrap``
    and ``Tests/Utils/test_text_wrap_index.py`` pins that the two agree.
    """
    if width <= 0 or not line:
        return []
    if _rich_divide_line is not None:
        return list(_rich_divide_line(line, width))
    lines = Text(line).wrap(_FALLBACK_CONSOLE, width)
    offsets: list[int] = []
    running = 0
    for segment in lines[:-1]:
        running += len(segment.plain)
        offsets.append(running)
    return offsets


class WrapIndex:
    """Maps virtual rows to (source line, wrapped segment) at one width."""

    __slots__ = ("_lines", "_width", "_starts", "virtual_height", "_segment_cache")

    _SEGMENT_CACHE_LIMIT = 512

    def __init__(self, lines: Sequence[str], width: int, starts: list[int], height: int):
        self._lines = lines
        self._width = width
        self._starts = starts
        self.virtual_height = height
        self._segment_cache: dict[int, list[str]] = {}

    @classmethod
    def build(cls, lines: Sequence[str], width: int) -> "WrapIndex":
        starts: list[int] = []
        running = 0
        for line in lines:
            starts.append(running)
            running += len(divide_source_line(line, width)) + 1
        return cls(lines, width, starts, max(running, 1))

    def row_to_line(self, row: int) -> tuple[int, int]:
        line_index = bisect_right(self._starts, row) - 1
        if line_index < 0:
            return (0, 0)
        return (line_index, row - self._starts[line_index])

    def line_start_row(self, line_index: int) -> int:
        if not self._starts:
            return 0
        clamped = max(0, min(line_index, len(self._starts) - 1))
        return self._starts[clamped]

    def segments(self, line_index: int) -> list[str]:
        cached = self._segment_cache.get(line_index)
        if cached is not None:
            return cached
        line = self._lines[line_index]
        breaks = divide_source_line(line, self._width)
        segments: list[str] = []
        start = 0
        for offset in (*breaks, len(line)):
            segments.append(line[start:offset])
            start = offset
        if not segments:
            segments = [""]
        # Bounded: one pathological 500k-character line costs ~9.4 ms per
        # divide_line call, which render_line would otherwise pay per row.
        if len(self._segment_cache) >= self._SEGMENT_CACHE_LIMIT:
            self._segment_cache.clear()
        self._segment_cache[line_index] = segments
        return segments
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest Tests/Utils/test_text_wrap_index.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Add the exactness guard that justifies this module**

```python
# append to Tests/Utils/test_text_wrap_index.py
import random


def test_exact_index_beats_character_division_on_ragged_text():
    """The cheap approximation was measured wrong on 12.4% of ragged lines.

    Character division (cell_len // width) drifts virtual height ~2.9%,
    which is a visibly wrong scrollbar and a match jump that lands in the
    wrong place. This pins that WrapIndex is exact where that is not.
    """
    from rich.cells import cell_len

    random.seed(7)
    words = ["a", "to", "the", "quick", "extraordinarily", "fox", "internationalization"]
    lines = [
        " ".join(random.choice(words) for _ in range(random.randint(20, 60)))
        for _ in range(500)
    ]
    console = Console(width=100)
    index = WrapIndex.build(lines, width=100)
    exact = sum(max(1, len(Text(line).wrap(console, 100))) for line in lines)
    approx = sum(max(1, -(-cell_len(line) // 100)) for line in lines)
    assert index.virtual_height == exact
    assert approx != exact, "fixture no longer exercises the approximation's error"
```

- [ ] **Step 6: Run it**

Run: `.venv/bin/python -m pytest Tests/Utils/test_text_wrap_index.py -v`
Expected: PASS (5 tests)

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Utils/text_wrap_index.py Tests/Utils/test_text_wrap_index.py
git commit -m "feat(library): exact wrap index for the virtualized reader (TASK-22500)"
```

---

### Task 2: The virtualized widget

**Files:**
- Create: `tldw_chatbook/Widgets/Library/library_media_raw_view.py`
- Test: `Tests/Library/test_library_media_raw_view.py`

**Interfaces:**
- Consumes: `WrapIndex`, `divide_source_line` from Task 1.
- Produces: `class VirtualizedRawContent(ScrollView)` with
  `__init__(*, content: str, query: str, match_index: int, max_visible_rows: int = 18, **kwargs)`,
  `.sync_search(query: str, match_index: int) -> None`,
  `.scroll_to_source_line(line_index: int) -> None`,
  `.source_lines: list[str]`, `.wrap_index: WrapIndex | None`,
  and class attribute `RENDER_LINE_CALLS: dict[str, int]` used only by tests.

- [ ] **Step 1: Write the failing test**

```python
# Tests/Library/test_library_media_raw_view.py
import pytest
from textual.app import App, ComposeResult

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest Tests/Library/test_library_media_raw_view.py -v`
Expected: FAIL — `ModuleNotFoundError: ... library_media_raw_view`

- [ ] **Step 3: Write minimal implementation**

```python
# tldw_chatbook/Widgets/Library/library_media_raw_view.py
"""Virtualized Raw content view for the Library media reader (TASK-22500).

Renders only the rows in view. The whole-document ``Static`` this replaces
cost 1051 ms at first paint and 684 ms on every ``update()`` -- that is,
on every search keystroke and every match-navigation click.
"""

from __future__ import annotations

from typing import Any

from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual.geometry import Size
from textual.scroll_view import ScrollView
from textual.strip import Strip

from tldw_chatbook.Utils.text_wrap_index import WrapIndex

MATCH_STYLE = Style(reverse=True)
ACTIVE_MATCH_STYLE = Style(reverse=True, bold=True)
EMPTY_CONTENT_MESSAGE = "No stored content."


class VirtualizedRawContent(ScrollView):
    """Scrollable raw text that renders one row at a time."""

    # Test-only instrumentation; asserting harness wall time is meaningless
    # because pilot.pause() costs ~30 ms per call.
    RENDER_LINE_CALLS: dict[str, int] = {"n": 0}

    def __init__(
        self,
        *,
        content: str,
        query: str,
        match_index: int,
        max_visible_rows: int = 18,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.source_lines = (content or EMPTY_CONTENT_MESSAGE).split("\n")
        self.wrap_index: WrapIndex | None = None
        self._indexed_width: int | None = None
        self._query = query.strip()
        self._match_index = match_index
        self._max_visible_rows = max_visible_rows
        self._match_lines: tuple[int, ...] = ()

    def on_resize(self, _event: Any = None) -> None:
        self._reindex_if_width_changed()

    def on_mount(self) -> None:
        self._reindex_if_width_changed()

    def _reindex_if_width_changed(self) -> None:
        width = self.scrollable_content_region.width or self.size.width
        if width <= 0 or width == self._indexed_width:
            return
        self.wrap_index = WrapIndex.build(self.source_lines, width)
        self._indexed_width = width
        self.virtual_size = Size(width, self.wrap_index.virtual_height)
        self.styles.height = min(self.wrap_index.virtual_height, self._max_visible_rows)
        self.refresh()

    def sync_search(self, query: str, match_index: int) -> None:
        """Restyle the visible rows for a new query or active match."""
        self._query = query.strip()
        self._match_index = match_index
        self.refresh()

    def scroll_to_source_line(self, line_index: int) -> None:
        """Scroll so a SOURCE line is visible, mapping through the index.

        The screen previously scrolled to the source-line index as if it
        were a screen row, which drifts once any line wraps.
        """
        if self.wrap_index is None:
            return
        self.scroll_to(y=self.wrap_index.line_start_row(line_index), animate=False)

    def render_line(self, y: int) -> Strip:
        type(self).RENDER_LINE_CALLS["n"] += 1
        width = self.scrollable_content_region.width or self.size.width
        if self.wrap_index is None or width <= 0:
            return Strip.blank(max(width, 0))
        row = y + int(self.scroll_offset.y)
        if row < 0 or row >= self.wrap_index.virtual_height:
            return Strip.blank(width)
        line_index, segment_index = self.wrap_index.row_to_line(row)
        segments = self.wrap_index.segments(line_index)
        piece = segments[segment_index] if segment_index < len(segments) else ""
        text = Text(piece, no_wrap=True, end="")
        if self._query:
            hit = piece.lower().find(self._query.lower())
            if hit >= 0:
                active = (
                    self._match_lines
                    and line_index
                    == self._match_lines[self._match_index % len(self._match_lines)]
                )
                text.stylize(
                    ACTIVE_MATCH_STYLE if active else MATCH_STYLE,
                    hit,
                    hit + len(self._query),
                )
        rendered = list(text.render(self.app.console))
        return Strip(rendered, len(piece)).adjust_cell_length(width)

    def set_match_lines(self, match_lines: tuple[int, ...]) -> None:
        """Record which SOURCE lines match, for active-match styling."""
        self._match_lines = match_lines
        self.refresh()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest Tests/Library/test_library_media_raw_view.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Library/library_media_raw_view.py Tests/Library/test_library_media_raw_view.py
git commit -m "feat(library): virtualized raw content widget (TASK-22500)"
```

---

### Task 3: Selection and rendering fidelity

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_media_raw_view.py`
- Test: `Tests/Library/test_library_media_raw_view.py`

**Interfaces:**
- Consumes: `VirtualizedRawContent` from Task 2.
- Produces: `VirtualizedRawContent.get_selection(selection) -> tuple[str, str] | None`.

- [ ] **Step 1: Write the failing test**

```python
# append to Tests/Library/test_library_media_raw_view.py
from textual.geometry import Offset
from textual.selection import Selection


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest Tests/Library/test_library_media_raw_view.py -k "selection or brackets" -v`
Expected: FAIL — `get_selection` returns the base-class result (None) for the wrap case

- [ ] **Step 3: Write minimal implementation**

```python
# add to VirtualizedRawContent in library_media_raw_view.py
    def get_selection(self, selection: Any) -> tuple[str, str] | None:
        """Map a screen selection back to SOURCE text.

        Rows are wrapped segments, so a selection spanning a wrap boundary
        must re-join the segments it covers rather than inserting newlines
        the document does not contain.
        """
        if self.wrap_index is None:
            return None
        top = selection.start or Offset(0, 0)
        bottom = selection.end or Offset(0, 0)
        first_row = min(top.y, bottom.y) + int(self.scroll_offset.y)
        last_row = max(top.y, bottom.y) + int(self.scroll_offset.y)
        collected: list[str] = []
        previous_line: int | None = None
        for row in range(first_row, last_row + 1):
            if row >= self.wrap_index.virtual_height:
                break
            line_index, segment_index = self.wrap_index.row_to_line(row)
            segments = self.wrap_index.segments(line_index)
            piece = segments[segment_index] if segment_index < len(segments) else ""
            if previous_line is not None and line_index != previous_line:
                collected.append("\n")
            collected.append(piece)
            previous_line = line_index
        return ("".join(collected), "")
```

Note: import `Offset` at the top of the module — `from textual.geometry import Offset, Size`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest Tests/Library/test_library_media_raw_view.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Library/library_media_raw_view.py Tests/Library/test_library_media_raw_view.py
git commit -m "feat(library): preserve drag-select and literal markup in the raw view (TASK-22500)"
```

---

### Task 4: Equivalence against today's Static output

**Files:**
- Test: `Tests/Library/test_library_media_raw_view.py`

**Interfaces:**
- Consumes: `VirtualizedRawContent`.
- Produces: nothing (guard only).

- [ ] **Step 1: Write the equivalence test**

```python
# append to Tests/Library/test_library_media_raw_view.py
from textual.containers import VerticalScroll
from textual.widgets import Static


class _StaticHarness(App):
    def __init__(self, content: str) -> None:
        super().__init__()
        self._content = content

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Static(self._content, id="old", markup=False)


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
```

- [ ] **Step 2: Run it and fix what it catches**

Run: `.venv/bin/python -m pytest Tests/Library/test_library_media_raw_view.py -k match_the_static -v`
Expected: any failure here is a real fidelity gap (tab expansion, empty-document message, wide glyphs). Fix `VirtualizedRawContent` until all 7 parameterizations pass; do not weaken the test.

- [ ] **Step 3: Commit**

```bash
git add Tests/Library/test_library_media_raw_view.py tldw_chatbook/Widgets/Library/library_media_raw_view.py
git commit -m "test(library): pin raw-view output against the Static it replaces (TASK-22500)"
```

---

### Task 5: Wire the widget into the content body

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_media_content.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Test: `Tests/Library/test_library_media_content.py`

**Interfaces:**
- Consumes: `VirtualizedRawContent`.
- Produces: `LibraryMediaContentBody.scroller -> ScrollableContainer` (the active
  scroller for the current mode); `LibraryMediaContentBody.raw_view -> VirtualizedRawContent | None`.
  `sync_mode`/`sync_search` signatures are unchanged.

- [ ] **Step 1: Write the failing test**

```python
# append to Tests/Library/test_library_media_content.py
@pytest.mark.asyncio
async def test_body_exposes_the_active_scroller_per_mode():
    from textual.containers import ScrollableContainer

    from tldw_chatbook.Widgets.Library.library_media_raw_view import (
        VirtualizedRawContent,
    )

    body = LibraryMediaContentBody(
        content="# Heading\n\nbody text",
        is_markdown=True,
        mode="raw",
        query="",
        match_index=0,
        id="library-media-viewer-content",
    )
    async with BodyHarness(body).run_test() as pilot:  # helper already in this file
        assert isinstance(body.scroller, ScrollableContainer)
        assert isinstance(body.raw_view, VirtualizedRawContent)
        assert body.scroller is body.raw_view
        await body.sync_mode("rendered")
        await pilot.pause()
        assert body.scroller is not body.raw_view
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest Tests/Library/test_library_media_content.py -k active_scroller -v`
Expected: FAIL — `AttributeError: 'LibraryMediaContentBody' object has no attribute 'scroller'`

- [ ] **Step 3: Implement the container**

Change `LibraryMediaContentBody` to subclass `Container` (import from `textual.containers`); in
`compose`, yield `VirtualizedRawContent(content=self.content, query=self._query,
match_index=self._match_index, id="library-media-viewer-content-text")` for raw mode, and for
rendered mode yield a `VerticalScroll(id="library-media-viewer-content-markdown-scroll")`
containing today's `Markdown`, storing that scroller on `self._markdown_scroll` (initialise
`self._markdown_scroll: VerticalScroll | None = None` in `__init__` alongside
`self._raw_widget`/`self._markdown_widget`). Replace `_build_raw_widget` with the new
construction, keep `_build_markdown_widget` as-is. Add:

```python
    @property
    def raw_view(self) -> VirtualizedRawContent | None:
        return self._raw_widget

    @property
    def scroller(self) -> ScrollableContainer:
        """The scroller for the CURRENT mode.

        Callers used to query this container as a VerticalScroll inside
        try/except; when the type stopped matching they silently no-opped
        and the reader quietly lost scroll restoration.
        """
        if self._desired_mode == "raw" and self._raw_widget is not None:
            return self._raw_widget
        if self._markdown_scroll is not None:
            return self._markdown_scroll
        return self
```

`sync_search` forwards to `self._raw_widget.sync_search(query, match_index)`.

- [ ] **Step 4: Add the CSS rule for the inner scroller**

In `tldw_chatbook/css/components/_agentic_terminal.tcss`, directly after the
`#library-media-viewer-content` block, add:

```css
#library-media-viewer-content-text {
    /* task-22500: the widget sets its own height from the wrap index
       (min(virtual rows, 18)); width must fill so the index is built at
       the real render width. */
    width: 100%;
}
```

Then regenerate: `.venv/bin/python tldw_chatbook/css/build_css.py`

- [ ] **Step 5: Run the body tests**

Run: `.venv/bin/python -m pytest Tests/Library/test_library_media_content.py -v`
Expected: PASS. Convert any test asserting `raw.renderable` to assert on
`body.raw_view.render_line(0).text` instead — do not delete the assertion.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Widgets/Library/library_media_content.py \
        tldw_chatbook/css/components/_agentic_terminal.tcss \
        tldw_chatbook/css/tldw_cli_modular.tcss \
        Tests/Library/test_library_media_content.py
git commit -m "feat(library): host the virtualized raw view in the content body (TASK-22500)"
```

---

### Task 6: Fix the three silent scroller lookups

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` (three sites near lines 35208, 35326, 35583)
- Test: `Tests/UI/test_library_media_reader_scroller_resolution.py`

**Interfaces:**
- Consumes: `LibraryMediaContentBody.scroller`, `VirtualizedRawContent.scroll_to_source_line`.
- Produces: nothing.

- [ ] **Step 1: Write the failing test**

```python
# Tests/UI/test_library_media_reader_scroller_resolution.py
"""These three lookups sit inside try/except: when the type stops matching
they no-op silently, so the reader loses scroll capture/restore and match
scrolling with every test still green."""
import pytest

from tldw_chatbook.Widgets.Library.library_media_content import LibraryMediaContentBody


@pytest.mark.asyncio
async def test_each_scroller_site_finds_its_target():
    # Build the screen exactly as Tests/UI/test_library_media_reader_match_nav_t22209.py
    # does: it has no fixture, it has module-level helpers. Copy (do not import)
    # `_document`, `_seed_row_document` and `_load_row_with_document` from that file.
    screen, pilot, service = await _open_reader_with_document(_document(400))
    body = screen.query_one("#library-media-viewer-content", LibraryMediaContentBody)
    assert body.scroller is not None

    screen._capture_library_media_loaded_scroll()
    captured = dict(screen._library_media_read_scroll_by_id)
    assert captured, "scroll capture resolved no scroller"

    screen._scroll_library_media_content_to_line(120)
    assert body.scroller.scroll_offset.y > 0, "match scroll resolved no scroller"
```

`_open_reader_with_document` is a thin local wrapper you write over the copied
`_load_row_with_document(screen, pilot, service, index, content)` helper. That file also has
`_raw_static(screen) -> Static`, which after Task 5 no longer returns a `Static` -- repoint it
at `body.raw_view` in the same commit as Task 5 so the 22209 gates keep passing.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest Tests/UI/test_library_media_reader_scroller_resolution.py -v`
Expected: FAIL — capture is empty and/or scroll offset stays 0

- [ ] **Step 3: Update the three call sites**

At each of the three sites, replace

```python
content = self.query_one("#library-media-viewer-content", VerticalScroll)
```

with

```python
body = self.query_one("#library-media-viewer-content", LibraryMediaContentBody)
content = body.scroller
```

and in `_scroll_library_media_content_to_line`, replace `content_scroll.scroll_to(y=line_index, ...)` with:

```python
        raw_view = body.raw_view
        if raw_view is not None:
            # task-22500: source line -> virtual row through the wrap index.
            # The old call scrolled to the source-line index as if it were a
            # screen row, which drifts once any line wraps.
            raw_view.scroll_to_source_line(line_index)
            return
        body.scroller.scroll_to(y=line_index, animate=False)
```

Import `LibraryMediaContentBody` at the top of `library_screen.py` if not already imported.

- [ ] **Step 4: Run the test and the reader gates**

Run:
```bash
.venv/bin/python -m pytest Tests/UI/test_library_media_reader_scroller_resolution.py \
  Tests/UI/test_library_media_reader_match_nav_t22209.py \
  Tests/UI/test_library_media_reader_traversal_t22207.py \
  Tests/UI/test_library_media_reader_no_change_sync_t22208.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/library_screen.py Tests/UI/test_library_media_reader_scroller_resolution.py
git commit -m "fix(library): resolve the reader scroller explicitly and scroll to matches exactly (TASK-22500)"
```

---

### Task 7: Retire the whole-document highlight Text

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_media_content.py`
- Test: `Tests/Library/test_library_media_content.py`

**Interfaces:**
- Consumes: `VirtualizedRawContent.set_match_lines`.
- Produces: `build_raw_content_match_lines(content: str, query: str) -> tuple[int, ...]`
  replacing `build_raw_content_highlight_plan`'s Text construction.
  `build_raw_content_renderable` is removed; its only remaining callers are tests.

- [ ] **Step 1: Write the failing test**

```python
# append to Tests/Library/test_library_media_content.py
def test_match_lines_are_derived_without_building_a_document_text():
    from tldw_chatbook.Widgets.Library import library_media_content as module

    assert not hasattr(module, "build_raw_content_renderable")
    lines = module.build_raw_content_match_lines("alpha\nbudget here\nomega\nbudget", "budget")
    assert lines == (1, 3)
    assert module.build_raw_content_match_lines("alpha", "") == ()
    assert module.build_raw_content_match_lines("", "x") == ()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest Tests/Library/test_library_media_content.py -k match_lines -v`
Expected: FAIL — `build_raw_content_renderable` still exists

- [ ] **Step 3: Implement**

```python
def build_raw_content_match_lines(content: str, query: str) -> tuple[int, ...]:
    """Return the SOURCE line indexes whose text contains ``query``.

    task-22500: the virtualized view styles matches per rendered row, so
    the whole-document ``Text`` this used to build (an O(document) pass on
    every query change) is gone. Only the line list survives -- which is
    all navigation and the "N of M" status ever consumed.
    """
    normalized = query.strip().lower()
    if not normalized or not content:
        return ()
    return tuple(
        index
        for index, line in enumerate(content.split("\n"))
        if normalized in line.lower()
    )
```

Delete `RawContentHighlightPlan`, `build_raw_content_highlight_plan` and
`build_raw_content_renderable`. Update `_raw_content_renderable` callers: the body now calls
`self._raw_widget.set_match_lines(build_raw_content_match_lines(self.content, query))` from
`sync_search`, then `self._raw_widget.sync_search(query, match_index)`.

- [ ] **Step 4: Run the content and match-nav suites**

Run:
```bash
.venv/bin/python -m pytest Tests/Library/test_library_media_content.py \
  Tests/UI/test_library_media_reader_match_nav_t22209.py -v
```
Expected: PASS. TASK-22209's probes assert match-navigation behaviour, not the removed
functions; if one imports a deleted name, repoint it at `build_raw_content_match_lines`.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Library/library_media_content.py Tests/Library/test_library_media_content.py
git commit -m "refactor(library): drop the whole-document highlight Text (TASK-22500)"
```

---

### Task 8: Resize coalescing

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_media_raw_view.py`
- Test: `Tests/Library/test_library_media_raw_view.py`

**Interfaces:**
- Consumes: `VirtualizedRawContent`.
- Produces: `VirtualizedRawContent.REINDEX_DEBOUNCE_SECONDS: float = 0.12`.

- [ ] **Step 1: Write the failing test**

```python
# append to Tests/Library/test_library_media_raw_view.py
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
        for width in (99, 98, 97, 96, 95):
            widget._request_reindex(width)
        await pilot.pause(0.3)
        assert builds["n"] == 1, f"re-indexed {builds['n']} times for one burst"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest Tests/Library/test_library_media_raw_view.py -k resize_burst -v`
Expected: FAIL — `AttributeError: ... _request_reindex`

- [ ] **Step 3: Implement**

Split `_reindex_if_width_changed` into `_request_reindex(width)` (stores the pending width and
arms a single `set_timer(self.REINDEX_DEBOUNCE_SECONDS, ...)`, cancelling any previous timer)
and `_build_index_now(width)` (the body from Task 2). `on_resize` calls `_request_reindex`;
`on_mount` calls `_build_index_now` directly so first paint is not delayed. Guard the timer
against firing after unmount by checking `self.is_attached`.

- [ ] **Step 4: Run it**

Run: `.venv/bin/python -m pytest Tests/Library/test_library_media_raw_view.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Library/library_media_raw_view.py Tests/Library/test_library_media_raw_view.py
git commit -m "perf(library): coalesce raw-view re-indexing across a resize burst (TASK-22500)"
```

---

### Task 9: Measure, verify, and record

**Files:**
- Modify: `backlog/tasks/task-22500 - Virtualize-the-Library-media-reader-body---it-repaints-every-line-of-the-document.md`
- Modify: `Docs/User_Guide/library.md` (Verified-against stamp)

**Interfaces:**
- Consumes: everything above.
- Produces: the measured before/after numbers for the PR body.

- [ ] **Step 1: Measure at three document sizes**

Write a throwaway probe under the session scratchpad (not the repo) that mounts the reader
body with 100 KB, 1 MB and 2.5 MB documents and reports, for each: index build time, first
paint `render_line` call count and self-time, and one `sync_search` repaint's `render_line`
self-time. Compare against dev by checking out `origin/dev`'s
`library_media_content.py` into a scratch copy — do not measure harness wall time.

- [ ] **Step 2: Run the full affected surface**

```bash
.venv/bin/python -m pytest Tests/Library/test_library_media_content.py \
  Tests/Library/test_library_media_raw_view.py Tests/Utils/test_text_wrap_index.py \
  Tests/UI/test_library_media_reader_traversal_t22207.py \
  Tests/UI/test_library_media_reader_no_change_sync_t22208.py \
  Tests/UI/test_library_media_reader_match_nav_t22209.py \
  Tests/UI/test_library_media_reader_scroller_resolution.py \
  Tests/UI/test_library_recompose_ratchet.py -q 2>&1 | tee /tmp/t22500-final.txt
```
Read the counts from the tee. Any red in `test_library_shell.py`'s media slice must be
A/B'd against `origin/dev` before being called yours — that file has a known pre-existing
red set from the #2064 reader redesign.

- [ ] **Step 3: Mutation-test the guards**

Break each in turn, confirm the named test reds, then Edit-restore:
1. `WrapIndex.build` uses `cell_len // width` instead of exact breaks -> `test_exact_index_beats_character_division_on_ragged_text` reds.
2. `render_line` renders the whole document -> `test_renders_only_visible_rows_regardless_of_document_size` reds.
3. `get_selection` returns `None` -> the selection test reds.
4. `scroll_to_source_line` uses `line_index` directly as the row -> the scroller-resolution test reds.

- [ ] **Step 4: Measure the Markdown view (the scope decision)**

Mount the same three documents in `mode="rendered"` and record first-paint time and mount
count. If it is materially slow, extend this task with a further plan; otherwise file a
follow-up task carrying the numbers. Record which you did in the task notes either way.

- [ ] **Step 5: Preflight and finish the task file**

```bash
./scripts/preflight.sh
```
Tick the four ACs, add Implementation Notes (approach, the corrected numbers, trade-offs,
files), set status Done, and stamp `Docs/User_Guide/library.md`.

- [ ] **Step 6: Commit**

```bash
git add "backlog/tasks/task-22500 - Virtualize-the-Library-media-reader-body---it-repaints-every-line-of-the-document.md" Docs/User_Guide/library.md
git commit -m "docs(library): record TASK-22500's measurements and close the task"
```
