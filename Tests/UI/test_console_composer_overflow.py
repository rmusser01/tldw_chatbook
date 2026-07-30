"""Two root-caused Console composer rendering bugs (windowing + wrap width).

Bug 1: `_visible_draft_line_slices` prepends `"... "` to the first visible row
without re-budgeting it. A whitespace-flush boundary row that already fills
the wrap width becomes `width + 4` characters once prefixed; the visible-draft
`Text` isn't `no_wrap`, so Rich rewraps that one row into two physical rows at
paint time, pushing the fixed 4-row window's true last row (the freshly
inserted/dictated text) off screen without any visible sign anything is wrong.

Bug 2: `_wrap_draft_line_slices` wraps by character count (`textwrap`), but
terminal cells are what actually get painted. Double-width text (CJK, emoji)
can be well under the wrap width in characters while over it in cells, so the
row-count budget under-counts and the tail gets clipped at paint time.

Both are reproduced here at the exact numbers found in diagnosis: a mounted
Console composer at app size 120x30 computes a visible-draft wrap width of 57
cells (`ConsoleComposerBar._draft_render_width()`), which is also used
directly in the pure-function tests below.
"""

from __future__ import annotations

from rich.cells import cell_len
import pytest
from textual.widgets import Static

from Tests.UI.test_console_dictation import _mounted_console, _ready_host
from tldw_chatbook.Widgets.Console import ConsoleComposerBar

WIDTH = 57

# --- Bug 1 fixture: a whitespace-flush row landing exactly at the wrap width
# ---------------------------------------------------------------------------
# `_LEAD` pushes enough rows above the boundary row that windowing kicks in
# (MAX_DRAFT_ROWS == 4); `_BODY` wraps, at width 57, so its second physical
# row ("lambda mu ... by ") is exactly 57 characters -- the whitespace-flush
# boundary case named in diagnosis; `_TAIL` supplies the sentinel that must
# survive the fixed 4-row window.
_LEAD = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
_BODY = (
    "the quick brown fox jumps over the lazy dog by the winding river "
    "the quick brown fox jumps over the lazy dog by the winding river"
)
_SENTINEL = "SENTINEL_TAIL_MARKER"
_TAIL = f" and then the {_SENTINEL} row must survive on screen for sure"
_BOUNDARY_TEXT = _LEAD + _BODY + _TAIL

# --- Bug 2 fixture: 40 double-width CJK characters + a short ASCII tail ----
# 40 * "個" + " ZEBRA4" is 47 *characters* (under width 57 by character
# count) but 87 *cells* ("個" is cell-width 2), so a char-counted wrap
# under-counts it as a single row when it actually needs two.
_CJK_TAIL = "ZEBRA4"
_CJK_TEXT = "個" * 40 + f" {_CJK_TAIL}"


# ---------------------------------------------------------------------------
# Bug 1: pure-function reproduction (no app mount)
# ---------------------------------------------------------------------------


def test_boundary_row_is_exactly_width_before_any_prefix_is_applied():
    """Pin the fixture's premise: the row bug 1 prefixes is whitespace-flush
    at exactly the wrap width, and the source text needs windowing at all."""
    wrapped = ConsoleComposerBar._wrap_draft_line_slices(_BOUNDARY_TEXT, WIDTH)
    assert len(wrapped) > ConsoleComposerBar.MAX_DRAFT_ROWS
    boundary_row = wrapped[1]
    assert cell_len(boundary_row.text) == WIDTH
    assert boundary_row.text.rstrip() != boundary_row.text  # whitespace-flush


def test_every_unfocused_visible_row_fits_the_wrap_width_and_the_tail_survives():
    """RED reproduction, `cursor_index=None` branch (unfocused/dictation)."""
    visible = ConsoleComposerBar._visible_draft_line_slices(_BOUNDARY_TEXT, WIDTH)
    assert len(visible) == ConsoleComposerBar.MAX_DRAFT_ROWS

    for row_index, line_slice in enumerate(visible):
        assert cell_len(line_slice.text) <= WIDTH, (
            f"row {row_index} ({line_slice.text!r}) is "
            f"{cell_len(line_slice.text)} cells, wider than {WIDTH}"
        )

    assert _SENTINEL in "".join(line_slice.text for line_slice in visible)


def test_every_caret_following_visible_row_fits_the_wrap_width_and_the_tail_survives():
    """RED reproduction, caret-following branch with the caret on the final row.

    Diagnosis: "reachable from BOTH the cursor_index=None ... branch and the
    caret-following branch when the caret sits on the final wrapped row" --
    caret at the very end of the draft makes `first_visible` identical to the
    unfocused case, so this exercises the same overflow through the other path.
    """
    caret_index = len(_BOUNDARY_TEXT)
    visible = ConsoleComposerBar._visible_draft_line_slices(
        _BOUNDARY_TEXT, WIDTH, cursor_index=caret_index
    )
    assert len(visible) == ConsoleComposerBar.MAX_DRAFT_ROWS

    for row_index, line_slice in enumerate(visible):
        assert cell_len(line_slice.text) <= WIDTH, (
            f"row {row_index} ({line_slice.text!r}) is "
            f"{cell_len(line_slice.text)} cells, wider than {WIDTH}"
        )

    assert _SENTINEL in "".join(line_slice.text for line_slice in visible)


def test_prefixed_row_offsets_still_map_into_the_source_text():
    """The trimmed prefix row's `start`/`end` must still describe real source
    text: every character trimmed to make room for `"... "` (the whitespace
    `lstrip()` already dropped, plus the newly trimmed leading characters)
    advances `start` by exactly that much, so the row's displayed text past
    the synthetic prefix is *exactly* the source slice `[start:end)`.
    """
    visible = ConsoleComposerBar._visible_draft_line_slices(_BOUNDARY_TEXT, WIDTH)
    prefixed = visible[0]
    assert prefixed.text.startswith("... ")
    assert 0 <= prefixed.start <= prefixed.end <= len(_BOUNDARY_TEXT)
    displayed_tail = prefixed.text[prefixed.synthetic_prefix_columns :]
    assert _BOUNDARY_TEXT[prefixed.start : prefixed.end] == displayed_tail


# ---------------------------------------------------------------------------
# Bug 2: pure-function reproduction (no app mount)
# ---------------------------------------------------------------------------


def test_cjk_text_measures_wider_in_cells_than_in_characters():
    """Pin the fixture's premise: char count and cell count diverge."""
    assert len(_CJK_TEXT) == 47
    assert cell_len(_CJK_TEXT) == 87
    assert cell_len(_CJK_TEXT) > WIDTH > len(_CJK_TEXT)


def test_cell_aware_row_count_is_two_for_the_cjk_case_not_one():
    """RED reproduction: char-counted wrap reports 1 row; cells need 2."""
    row_count = ConsoleComposerBar._visible_draft_row_count(_CJK_TEXT, WIDTH)
    assert row_count == 2


def test_cell_aware_wrap_keeps_every_row_within_the_cell_width_and_keeps_the_tail():
    wrapped = ConsoleComposerBar._wrap_draft_line_slices(_CJK_TEXT, WIDTH)
    assert len(wrapped) == 2
    for line_slice in wrapped:
        assert cell_len(line_slice.text) <= WIDTH
    assert _CJK_TAIL in "".join(line_slice.text for line_slice in wrapped)


# ---------------------------------------------------------------------------
# Bug 2: single-width text must wrap identically to before (pinned values)
# ---------------------------------------------------------------------------
# These expected values were computed from `textwrap.wrap(text, width=width,
# break_long_words=True, break_on_hyphens=False, drop_whitespace=False,
# replace_whitespace=False) or [""]` -- the exact call the cell-aware wrapper
# replaces -- and hardcoded here so this pins the wrapper's own output, not a
# comparison against the old implementation.


@pytest.mark.parametrize(
    ("text", "width", "expected"),
    [
        (
            "the quick brown fox jumps over the lazy dog",
            20,
            ["the quick brown fox ", "jumps over the lazy ", "dog"],
        ),
        (
            "supercalifragilisticexpialidocious",
            10,
            ["supercalif", "ragilistic", "expialidoc", "ious"],
        ),
        (
            "multiple   spaces   between   words",
            12,
            ["multiple   ", "spaces   ", "between   ", "words"],
        ),
        ("short", 80, ["short"]),
        ("", 80, [""]),
    ],
)
def test_cell_wrap_line_matches_pinned_single_width_output(text, width, expected):
    assert ConsoleComposerBar._cell_wrap_line(text, width) == expected


def test_cell_wrap_line_matches_pinned_output_for_the_boundary_fixture():
    """The exact wrap of `_BODY`'s home paragraph at width 57 (single-width),
    pinned so the windowing tests above rest on a known-good wrap, not a
    self-referential one.
    """
    expected = [
        "alpha beta gamma delta epsilon zeta eta theta iota kappa ",
        "lambda mu the quick brown fox jumps over the lazy dog by ",
        "the winding river the quick brown fox jumps over the lazy",
        " dog by the winding river and then the ",
        "SENTINEL_TAIL_MARKER row must survive on screen for sure",
    ]
    assert ConsoleComposerBar._cell_wrap_line(_BOUNDARY_TEXT, WIDTH) == expected


# ---------------------------------------------------------------------------
# Painted-state reproductions (mounted app, real Static widget)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bug1_unfocused_insert_keeps_the_sentinel_row_painted():
    """Mounted reproduction of the `cursor_index=None` (unfocused/dictation)
    windowing bug: insert prose ending in a sentinel while the composer is
    *not* focused (as it would be right after a dictation insertion lands
    while the mic button, not the draft, holds focus), and require the
    sentinel to actually paint somewhere in the fixed-height visible draft.
    """
    _, host = _ready_host()
    async with host.run_test(size=(120, 30)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        await pilot.pause()
        composer.blur()
        await pilot.pause()
        assert composer.has_focus_within is False

        composer.insert_text(_BOUNDARY_TEXT)
        await pilot.pause()

        visible_draft = composer.query_one("#console-command-visible-text", Static)
        painted_rows = [
            visible_draft.render_line(row).text
            for row in range(visible_draft.size.height)
        ]
        assert any(_SENTINEL in row for row in painted_rows), painted_rows


@pytest.mark.asyncio
async def test_bug2_cjk_insert_keeps_the_tail_painted():
    """Mounted reproduction of the char-vs-cell wrap bug: 40 double-width CJK
    characters plus a short ASCII tail must not clip the tail at paint time.
    """
    _, host = _ready_host()
    async with host.run_test(size=(120, 30)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        await pilot.pause()
        composer.blur()
        await pilot.pause()

        composer.insert_text(_CJK_TEXT)
        await pilot.pause()

        visible_draft = composer.query_one("#console-command-visible-text", Static)
        painted_rows = [
            visible_draft.render_line(row).text
            for row in range(visible_draft.size.height)
        ]
        assert any(_CJK_TAIL in row for row in painted_rows), painted_rows
