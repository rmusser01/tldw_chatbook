---
id: TASK-21501
title: >-
  Cursor blink tick arms a full screen layout pass twice a second while the composer is idle
status: Done
assignee: []
created_date: '2026-08-23'
updated_date: '2026-08-23'
labels:
  - performance
  - console
  - composer
priority: medium
dependencies: []
---

## Description

The Console composer's cursor-blink tick repaints the visible-draft `Static` through
`Static.update(...)`, whose `layout` parameter defaults to `True`. Every blink phase
therefore schedules a full screen layout pass — `Screen._refresh_layout` and a whole
`Compositor.reflow` — for as long as the composer merely holds focus, whether or not
the user is typing. The blink interval is 0.53 s, so an idle focused composer costs
roughly two full layout passes per second.

This directly contradicts the contract the method states for itself: its docstring
says it "must stay cheap and must not trigger a layout recompute on every blink
phase". The cost is invisible to every existing test because nothing counts layout
operations.

Found while burning down the 2026-08-22 holistic performance review
(`Docs/Design/2026-08-22-holistic-perf-review.md`), alongside finding 21120's note
that the blink tick already re-scans prompt history per tick.

## Acceptance Criteria

- [x] A blink phase flip performs no more screen layout work than an idle event-loop settle, measured by counting real layout operations rather than by reading the source
- [x] The caret still visibly blinks: the visible phase paints the caret glyph and the hidden phase does not
- [x] Both blink phases occupy identical geometry at every wrap boundary — empty draft, a draft filling the last cell, a draft landing exactly at the wrap width, a multi-row wrapped draft, and a double-width CJK draft
- [x] Draft mutation, focus/blur, resize and collapse still recompute the composer's height (the paths that legitimately need a layout pass are untouched)
- [x] A regression test fails against the un-fixed implementation

## Implementation Plan

1. Instrument Textual's layout path under a real-CSS Console harness and count
   `Screen._refresh_layout` / `Compositor.reflow` / arrangement-cache misses across a
   fixed number of blink ticks, on a spread of draft shapes.
2. Establish that the rendered SIZE genuinely cannot differ between blink phases
   before choosing a blanket `layout=False`; if any phase can change the row count,
   implement a narrower fix instead.
3. Apply the fix, re-measure with the identical probe, and diff the two arms.
4. Add regression tests: one that pins the layout cost against an idle control arm,
   one that pins phase-to-phase geometry identity at the wrap boundaries.
5. Mutation-test both tests against deliberately broken implementations.

## Implementation Notes

`_render_visible_draft_only` (`tldw_chatbook/Widgets/Console/console_composer_bar.py`)
now calls `Static.update(renderable, layout=False)`. That method has exactly one
caller — `_toggle_cursor_blink` — so the change is scoped to the blink path and does
not touch `_refresh_visible_draft`, which still recomputes the row count and calls
`_apply_draft_height` with `layout=True`.

**Measured, under a real-CSS `ConsoleHarness` at 140x42, 6 driven blink ticks per
draft shape (empty / short / wrapped 3-row / exactly-at-wrap-width / width-1 /
CJK double-width):**

| | before | after |
|---|---|---|
| `Screen._refresh_layout` calls | 6 | 0 |
| `Compositor.reflow` calls | 6 | 0 |
| `Widget.arrange` calls | 396 | 0 |
| arrangement-cache misses | 6 | 0 |
| `Static._layout_updates` delta | 6 | 0 |
| time inside `_refresh_layout` | 3.1–6.5 ms/tick | 0 |

Identical in all six shapes. At a 0.53 s interval that is ~1.9 layout passes/second
removed from an idle focused composer.

**Why `layout=False` is safe here** (two independent reasons):

1. The two phases are cell-identical by construction. `_draft_renderable` reserves
   exactly one display cell at the caret position in both phases — the glyph while
   visible, a plain space while hidden — and wraps it in the same pass, a property
   its own comment already documents and which `_visible_draft_row_count` budgets for
   via `reserve_trailing_cell`. Both `CURSOR_GLYPH` (`▌`) and its ASCII fallback
   (`|`) are single-width. The probe confirms the painted row count and per-row cell
   widths are equal in both phases at every boundary tested, including the
   exactly-at-wrap-width case where the caret legitimately spills onto its own row
   (2 rows in *both* phases, because the row count is computed with the reserved cell
   already included).
2. Even if the content did change, the widget's size is not derived from it. The
   visible-draft `Static` gets `width: 1fr`, `text_wrap = "nowrap"` and
   `text_overflow = "clip"` inline at compose time, and explicit
   `height`/`min_height`/`max_height` from `_apply_draft_height`. This was confirmed
   accidentally during mutation testing: breaking the reserved-cell invariant changed
   the painted row count from 1 to 2 between phases while `outer_size` stayed
   `Size(93, 2)` — the inline height absorbed it.

**Regression tests** (`Tests/UI/test_console_composer_cursor.py`):

- `test_console_composer_blink_tick_arms_no_layout_pass` — counts
  `Screen._refresh_layout` over N settles with and without blink ticks and asserts the
  blink arm costs no more than the idle arm (an A/A control rather than a bare `== 0`,
  so an unrelated timer cannot turn it into a flake), then asserts the caret glyph
  still toggles so a caret that stopped blinking cannot pass.
- `test_console_composer_blink_phases_are_geometry_identical` — asserts equal
  `outer_size`, row count and per-row cell widths across both phases for seven draft
  shapes, under a `CSS_PATH`-true harness (a bare `ConsoleHarness` loads none of the
  app stylesheet, so geometry conclusions under it are void).

**Mutation results:** removing `layout=False` fails the cost test with
"6 blink ticks cost 6 extra screen layout passes (idle floor 0)". Removing the
reserved caret cell from the hidden phase fails the geometry test — on painted cell
widths for a short draft (`[5] vs [6]`), and on row count for the exactly-at-width
draft (`1 vs 2`).

**Shutdown/error paths:** unchanged. The timer is still created paused in `on_mount`
and resumed only while `has_focus_within and not self._collapsed`; Textual stops a
widget's timers on close, so no tick can outlive the composer. The `NoMatches` guard
is untouched, and `refresh(repaint=True)` still clears the cached dimensions and the
rich-style cache, so the repaint half is intact.

**Modified files:** `tldw_chatbook/Widgets/Console/console_composer_bar.py`,
`Tests/UI/test_console_composer_cursor.py`.
