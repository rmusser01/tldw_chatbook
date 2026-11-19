---
id: TASK-2762
title: 'Home rail and canvas cannot scroll; overflow work is unreachable and invisible'
status: Done
assignee: ['@t2-home']
created_date: '2026-08-06'
updated_date: '2026-10-11'
labels: [home, bug, ui]
dependencies: []
---

## Description (the why)

`HomeRail` and `HomeCanvas` subclass `Vertical` (Textual default
`overflow: hidden hidden`) and no CSS overrides it — runtime-verified:
`overflow_y=hidden`, `allow_vertical_scroll=False`,
`show_vertical_scrollbar=False` on `#home-rail` and `#home-canvas`
(dev @ 84e4b33f0, 2026-08-06). Contrast `#library-rail`, which sets
`overflow-y: auto` explicitly.

Capacity is real: Needs Attention/Running can hold up to 20 watchlist runs
plus every queued/parsing/writing/failed Library ingest job, and Recent holds
8 — each row 2 cells tall plus a 1-cell margin. Rows past the fold get no
scrollbar, no overflow cue, and — because Home declares no key bindings at
all — no keyboard path either. Work simply becomes invisible.

## Acceptance Criteria (the what)

- [x] The rail (and the canvas, if it can overflow) scrolls when content
      exceeds the viewport, with a visible scrollbar or overflow cue.
- [x] Every rail row is reachable at 24-row terminal height with a full
      attention list.
- [x] A test pins the scrollability (or an explicit overflow indicator).

## Implementation Plan

1. Premise-verify at origin/dev 9f8533686c: `_home.tcss` carries no
   overflow rules for `#home-rail`/`#home-canvas`; both widgets subclass
   Vertical; probe confirms overflow_y hidden at runtime.
2. Red-first: mount a 12-row failed attention list at 120x24 and assert
   `overflow_y == "auto"`, `max_scroll_y > 0`, scrollbar visible, and the
   LAST row becomes visible after scrolling to it (all RED pre-fix:
   'hidden' != 'auto'). Companion canvas test asserts the canvas pane
   scrolls too.
3. Apply the `#library-rail` remedy to both panes in
   `css/features/_home.tcss` (`overflow-y: auto; overflow-x: hidden;`) and
   rebuild the consolidated stylesheet with `css/build_css.py`.
4. Stabilize: the file gained an autouse stub suppressing the rail-state
   DISK write (the toggle test's persisted `[home.rail_state]` leaked into
   later tests under the shared bootstrap profile and re-broke the scroll
   test's geometry); the scroll assertion scrolls TO the row.

## Implementation Notes

- `#home-rail, #home-canvas` now carry `overflow-y: auto; overflow-x:
  hidden;` in `css/features/_home.tcss`, mirroring `#library-rail`; the
  consolidated `tldw_cli_modular.tcss` regenerated via
  `python tldw_chatbook/css/build_css.py`. With Textual's default
  scrollbar behavior the overflowing pane shows its vertical scrollbar
  (asserted `show_vertical_scrollbar is True` at 120x24 with 12 rows,
  `max_scroll_y == 35`).
- Keyboard/mouse: rows are focusable Buttons and the pane is now
  scrollable (wheel, scrollbar, `scroll_to_widget`), so rows past the fold
  are reachable at 24 rows — pinned by scrolling the LAST of 12 attention
  rows into view.
- Tests: `test_home_rail_scrolls_to_rows_past_the_fold`,
  `test_home_canvas_scrolls_when_content_exceeds_the_pane` (both RED
  pre-fix). Stabilizers: autouse `_stub_home_rail_preference_disk_writes`
  fixture (same shared-config leak class as the home_screen file's stub;
  the toggle test's in-memory assertion is untouched) and retry-based
  region settling.
- Files: `tldw_chatbook/css/features/_home.tcss`,
  `tldw_chatbook/css/tldw_cli_modular.tcss` (generated),
  `Tests/UI/test_home_triage_rail.py`.
- Verified: token-governance + full triage file x2 (18 passed each),
  full home suites (98 passed) at the fix commit.

ADR required: no
ADR path: N/A
Reason: two-property CSS change inside an existing feature sheet following
an established in-repo pattern; no new tokens or literals (overflow
keywords are not token-governed values).
