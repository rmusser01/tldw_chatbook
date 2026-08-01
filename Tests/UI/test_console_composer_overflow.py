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

import signal
from pathlib import Path

from rich.cells import cell_len
import pytest
from textual.widgets import Static

from Tests.UI.test_console_dictation import _mounted_console, _ready_host
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Widgets.Console import ConsoleComposerBar


class _CssTrueConsoleHarness(ConsoleHarness):
    """ConsoleHarness that loads the real app CSS bundle.

    The shared harness is a bare ``App`` -- none of the app's stylesheet
    applies under it, so geometry/clipping assertions made there are void
    (the live-gate crop shipped through two review rounds because of it).
    """

    CSS_PATH = str(
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook"
        / "css"
        / "tldw_cli_modular.tcss"
    )

# PURE-FUNCTION width parameter only -- pass it to `_wrap_draft_line_slices`/
# `_cell_wrap_line`/etc. below, where an explicit width argument is exactly
# what's under test. Do NOT assume it equals the mounted widget's real
# `_draft_render_width()`: a post-rebase composer-actions-button width change
# (task-1680's "☰" overflow button) silently drifted the real render width
# away from this literal (57 -> 52) and broke tests elsewhere in the suite
# that had copied this constant to predict mounted/painted geometry. Mounted
# tests must read `composer._draft_render_width()` themselves.
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

# --- Fix-round-2 fixtures: review findings (MEDIUM caret-at-start regression,
# LOW ZWJ/grapheme cell-arithmetic leaks) -------------------------------------
# Built from explicit codepoints, never typed/pasted literals: a ZWJ
# (U+200D) or variation selector is invisible in an editor and silently
# corrupts on copy-paste, which would quietly turn a targeted regression
# test into a vacuous one.
_ZWJ_WORD = "".join(
    chr(c) for c in (0x4B, 0x200D, 0x54, 0x43, 0x43, 0x4F, 0x74, 0x52)
)  # "K<ZWJ>TCCOtR" -- reviewer's Finding 2 counterexample (cluster-cells=6,
#    per-character cell_len sum=7); exercises the prefix-trim's cell math.
_ZWJ_EMOJI_STRING = "".join(
    chr(c) for c in (0x74, 0x52, 0x45, 0x6D, 0x1F600, 0x20, 0x200D, 0x200D, 0x65, 0x79)
)  # "tREm<emoji> <ZWJ><ZWJ>ey" -- reviewer's Finding 3 counterexample.
# A genuine width>=8 (the production floor) counterexample for the
# join-boundary cell-arithmetic bug in `_cell_wrap_line`'s hard-break path,
# found by differential fuzzing during the fix. (The reviewer's Finding 3
# string above was an over-width row EMITTED by the round-1 code while
# windowing a larger fuzz draft, not a standalone input -- fed back in
# directly it happens to split legally, which is why this separate
# counterexample was needed; see the review's round-2 adjudication.)
# A run of short chunks ending in a ZWJ followed by a chunk starting with
# a variation selector + emoji.
_ZWJ_JOIN_BOUNDARY_TEXT = "".join(
    chr(c)
    for c in (
        0x23, 0x5A, 0x58, 0x59, 0x20, 0x62, 0x39, 0x200D, 0x20,
        0xFE0F, 0x1F600, 0x62, 0x63, 0x63, 0x33, 0x59, 0x59,
        0x1F600, 0x33, 0x58, 0x5A, 0x58, 0x59, 0x23, 0x39, 0x30,
        0x30, 0x59, 0x63,
    )
)


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
# Fix round 2 -- review findings
# ---------------------------------------------------------------------------


def test_home_row_is_never_prefixed_or_trimmed_when_nothing_is_scrolled_above():
    """MEDIUM RED reproduction: caret at draft offset 0 (Home) on a >4-row
    draft must not delete real leading content or the caret glyph.

    `first_visible == 0` means row 0 IS the draft's true first row -- nothing
    is scrolled off above it, so the "... " prefix-and-trim must not run at
    all. Splices the caret glyph at offset 0 exactly as `_draft_renderable`
    does for a focused composer, then windows with `cursor_index=0` (the
    caret-following branch, caret on the draft's very first row).
    """
    render_text = ConsoleComposerBar.CURSOR_GLYPH + _BOUNDARY_TEXT
    visible = ConsoleComposerBar._visible_draft_line_slices(
        render_text, WIDTH, cursor_index=0
    )
    assert len(visible) == ConsoleComposerBar.MAX_DRAFT_ROWS

    first_row = visible[0]
    assert not first_row.text.startswith("...")
    assert first_row.text.startswith(ConsoleComposerBar.CURSOR_GLYPH)
    # Row 0 is exactly the untouched first wrapped row -- not merely
    # "contains the caret somewhere", but byte-identical to what an
    # unwindowed wrap would have produced for this row.
    unwindowed = ConsoleComposerBar._wrap_draft_line_slices(render_text, WIDTH)
    assert first_row.text == unwindowed[0].text
    assert first_row.start == unwindowed[0].start
    assert first_row.end == unwindowed[0].end


def test_prefix_trim_uses_the_zwj_fuzz_counterexample_and_stays_within_width():
    """LOW 1 RED reproduction: a ZWJ grapheme cluster whose per-character
    `cell_len` sum (7) exceeds its own whole-cluster `cell_len` (6) must not
    make the old per-character-decrement trim exit early and leave the
    prefixed row still over budget.
    """
    lead = "aa bb cc dd ee ff gg hh "
    text = lead + _ZWJ_WORD + " ii jj kk ll mm SENTINEL"
    width = 8

    visible = ConsoleComposerBar._visible_draft_line_slices(text, width)
    assert len(visible) == ConsoleComposerBar.MAX_DRAFT_ROWS
    for row_index, line_slice in enumerate(visible):
        assert cell_len(line_slice.text) <= width, (
            f"row {row_index} ({line_slice.text!r}) is "
            f"{cell_len(line_slice.text)} cells, wider than {width}"
        )
    assert "SENTINEL" in "".join(line_slice.text for line_slice in visible)


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
# Fix round 2 -- LOW 2/LOW 3: `_cell_wrap_line`'s own grapheme-cluster and
# join-boundary cell arithmetic (independent of the prefix-trim above)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "width"),
    [
        (_ZWJ_WORD, 8),
        (_ZWJ_EMOJI_STRING, 8),
    ],
)
def test_cell_wrap_line_stays_within_width_for_the_reviewers_fuzz_strings(text, width):
    """Reviewer's Finding 2/3 counterexample strings, pinned as an explicit
    invariant check. Both were over-width rows EMITTED by the reviewed
    commit's code while windowing larger fuzz drafts (the findings were
    real); fed back in as standalone inputs they happen to split legally
    under both the reviewed commit and this fix -- see the review's
    round-2 adjudication of exactly this input-vs-emitted-row distinction.
    So this pins "stays correct", not a RED reproduction.
    `test_prefix_trim_uses_the_zwj_fuzz_counterexample_...` above,
    `test_cell_wrap_line_does_not_hang_at_width_one_...` below, and
    `test_cell_wrap_line_hard_break_pin_against_the_reviewed_commit` are
    the ones that actually fail against the reviewed commit.
    """
    rows = ConsoleComposerBar._cell_wrap_line(text, width)
    for row in rows:
        assert cell_len(row) <= width, (row, cell_len(row))


def test_cell_wrap_line_stays_within_width_at_the_join_boundary():
    """Join-boundary regression pin for `_cell_wrap_line`'s hard-break path.

    `cell_len` itself is not additive across every join -- a trailing ZWJ
    silently absorbs the character that would follow it in a longer string,
    so a numeric budget derived from `cell_len(current_text)` in isolation
    (checked against a hard-break piece's own isolated `cell_len`) can
    disagree with the true joined width. This does NOT fail against the
    reviewed commit (its `chop_cells`-based hard-break happens to split this
    particular string at a different, still-valid point); it pins a defect
    introduced by, and fixed within, this same fix round's first attempt at
    LOW 2/3 (a numeric-budget version of `_extend_fitting_cells`), caught by
    differential fuzzing during development before it was ever committed.
    Kept as a regression guard because reverting `_extend_fitting_cells` to
    that numeric-budget shape is exactly the mistake it would silently
    repeat. Reproduces at width 9, inside the production floor (>= 8).
    """
    width = 9
    rows = ConsoleComposerBar._cell_wrap_line(_ZWJ_JOIN_BOUNDARY_TEXT, width)
    for row in rows:
        assert cell_len(row) <= width, (row, cell_len(row))
    # The row-joining round-trips to the source content -- no characters
    # silently dropped by the hard-break's forced-progress path.
    assert "".join(rows) == _ZWJ_JOIN_BOUNDARY_TEXT.expandtabs(8)


# "<ZWJ><ZWJ>dExfvMbzW" -- found by the reviewer's round-2 differential
# search between the reviewed and current `_cell_wrap_line`. Codepoints,
# per this file's invisible-character convention.
_ZWJ_HARD_BREAK_PIN = "".join(
    chr(c)
    for c in (0x200D, 0x200D, 0x64, 0x45, 0x78, 0x66, 0x76, 0x4D, 0x62, 0x7A, 0x57)
)


def test_cell_wrap_line_hard_break_pin_against_the_reviewed_commit():
    """Committed regression pin for the join-boundary fix at the production
    width floor (8), not only the synthetic width-1 hang case.

    The reviewed commit's `chop_cells`-based hard break emits
    `['<ZWJ><ZWJ>dExfvMbzW']` -- one row of 9 cells at width 8 -- because
    `cell_len` is not additive across a trailing ZWJ. The current
    join-aware hard break splits it within budget. Reverting
    `_cell_wrap_line` to the reviewed implementation fails here.
    """
    width = 8
    rows = ConsoleComposerBar._cell_wrap_line(_ZWJ_HARD_BREAK_PIN, width)
    for row in rows:
        assert cell_len(row) <= width, (row, cell_len(row))
    assert "".join(rows) == _ZWJ_HARD_BREAK_PIN


def test_cell_wrap_line_does_not_hang_at_width_one_with_double_width_text():
    """LOW 3 RED reproduction: a chunk whose leading grapheme alone exceeds
    the wrap width (only possible at width < 2 with double-width content --
    unreachable via either production call site, both of which floor at 8,
    but `_cell_wrap_line` doesn't itself assume that floor) must still make
    forward progress every call, not spin forever re-appending an empty row.
    A hard 3s alarm turns a hang into a test failure instead of a stuck
    suite.
    """

    def _on_alarm(signum, frame):
        raise AssertionError(
            "_cell_wrap_line('個個個', 1) did not return within 3s -- "
            "infinite loop regression"
        )

    previous_handler = signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(3)
    try:
        rows = ConsoleComposerBar._cell_wrap_line("個個個", 1)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)

    assert rows == ["個", "個", "個"]


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
async def test_the_expanded_row_grows_with_the_draft_so_the_screen_shows_all_rows():
    """Live-gate RED reproduction: the parent row crops the grown draft.

    `#console-composer-expanded` is pinned to `height: 1` in
    `_agentic_terminal.tcss`, so when `_apply_draft_height` grows the
    visible-draft Static to 4 rows and the composer bar to 8, the parent
    Horizontal crops the paint to ONE screen row, vertically centered in
    the bar with blank rows around it. Every earlier test here asserted
    `visible_draft.render_line(...)` -- the widget's own paint -- which a
    parent's crop never touches, so the whole suite stayed green while the
    real screen showed one line. This test asserts the DISPLAY CHAIN: the
    parent row must be at least as tall as the draft it contains, and the
    draft's rows must lie inside the parent's screen region.
    """
    app, host = _ready_host()
    # The shared ConsoleHarness is a bare App: it never loads the real CSS
    # bundle, so `#console-composer-expanded { height: 1 }` -- the rule that
    # causes this bug -- silently doesn't apply under it and this test would
    # pass against broken code. Run under a bundle-loading harness instead.
    host = _CssTrueConsoleHarness(app)
    async with host.run_test(size=(120, 30)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        await pilot.pause()

        composer.insert_text(_BOUNDARY_TEXT)
        await pilot.pause()
        await pilot.pause()

        visible_draft = composer.query_one("#console-command-visible-text", Static)
        expanded = composer.query_one("#console-composer-expanded")
        assert visible_draft.region.height >= 4, visible_draft.region
        assert expanded.region.height >= visible_draft.region.height, (
            expanded.region,
            visible_draft.region,
        )
        draft_last_row_y = visible_draft.region.y + visible_draft.region.height - 1
        assert draft_last_row_y < expanded.region.y + expanded.region.height, (
            "the draft's last row lies outside the parent row's screen region",
            expanded.region,
            visible_draft.region,
        )


@pytest.mark.asyncio
async def test_home_on_a_long_draft_keeps_the_caret_and_leading_text_painted():
    """MEDIUM RED reproduction, mounted: caret at the draft's true start
    (Home) on a >4-row draft must paint both the real leading characters
    and the caret glyph on row 0 -- the reviewer's own repro at paint level
    (pre-fix: row 0 showed no caret at all, in any of the 4 rows, and had
    lost "alp" from "alpha").
    """
    _, host = _ready_host()
    async with host.run_test(size=(120, 30)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        await pilot.pause()

        composer.load_draft(_BOUNDARY_TEXT)
        composer.focus()
        await pilot.pause()
        composer.move_cursor_home()
        await pilot.pause()
        assert composer.cursor_index == 0

        visible_draft = composer.query_one("#console-command-visible-text", Static)
        row0 = visible_draft.render_line(0).text
        assert ConsoleComposerBar.CURSOR_GLYPH in row0, row0
        assert row0.lstrip().startswith(
            ConsoleComposerBar.CURSOR_GLYPH + _BOUNDARY_TEXT[:10]
        ), row0


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
