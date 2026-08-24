---
id: TASK-21114
title: >-
  Transcript drag-selection re-wraps and re-renders the message body on every
  mouse-move
status: Done
assignee:
  - '@claude'
created_date: '2026-08-22'
updated_date: '2026-08-23 10:26'
labels:
  - performance
  - console
  - transcript
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21114).

During an active drag (MouseMove at 50-100 Hz), `console_transcript.py` per event: rebuilds the
full message render text for plain rows (uncached `get_display_text()`, :1727-1729), runs
`Content(text).wrap(width)` over the entire body (:2278-2311), calls `set_selection_range` ->
full `body.update()` re-render even when the range did not move (:1740-1787), and sweeps every
mounted row calling `clear_selection()` (:5055-5063; watermarks default 20k/12k lines so
hundreds of rows can be mounted). Tens of ms per event on multi-KB rows - seconds of cumulative
lag per drag on slow hardware. The whole feature is post-baseline.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Plain-row display text is cached and invalidated in sync_message; the wrap table is memoized per (text, width) for the drag's lifetime
- [x] #2 set_selection_range early-returns on an unchanged range; the drag remembers the single row holding a selection instead of sweeping all mounted rows
- [x] #3 A counter/timing probe over a drag across a ~20 KB row shows the per-event cost drop; numbers in the task
- [x] #4 Selection behavior (visuals, copy, menus) unchanged - existing selection tests green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Baseline: run all selection/transcript suites (selection_rows/core/transcript/contract/keyboard/menu/end_to_end/app_smoke/selection_prune_bound, transcript windowing/two_sided_window/window_reconcile/pruning/stream_scrollback, visual transcript) teed to test-logs/.
2. Red-first probe: new test counting Content.wrap calls + body Static.update calls over a synthetic drag (MouseDown + N MouseMove) across a ~20KB plain row, asserting bounded wraps/renders; watch it fail on the current code.
3. Fix (a): cache plain-row display text on ConsoleTranscriptMessage (compute _message_body_render_text once, reuse Content and .plain), invalidated in sync_message exactly where _message/_presentation are reassigned (mirrors markdown _body_text discipline).
4. Fix (b): extract the wrap table (wrapped line -> source offset) into an lru_cache'd pure helper keyed on (text, width); preserve the defensive-fallback semantics (fallback only for cells at/after the first unmodeled wrap edge).
5. Fix (c): early-return in set_selection_range on all three row arms (plain, markdown strip, diff strip) when the effective stored range is unchanged.
6. Fix (d): drop the per-MouseMove all-rows clear_selection sweep; sweep once at drag arm (on_mouse_down) and rely on _selection_origin_row; make SelectionManager.extend_drag report unchanged offsets so on_mouse_move skips the re-render entirely.
7. Timing/counter probe script over a ~20KB row: per-event wrap calls and body re-renders before/after; numbers into task notes.
8. Re-run all suites from step 1 + full --collect-only sweep; A/B any red against base 8e949873e.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
All four per-MouseMove costs removed; behavior-preservation proven by an exhaustive
old-vs-new mapping equivalence check plus the full selection/transcript suites.

**Probe numbers** (M-series mac, 20,048-char plain row, real handlers driven with synthetic
events; scratch probe `drag_probe.py`, logs in `test-logs/t21114-probe-{before,after}.txt`):

| metric (per 150-event drag: 100 sweeping + 50 at-rest moves) | before | after |
|---|---|---|
| `Content.wrap` calls (module mapping path) | 151 (1/event) | 1 (memoized table) |
| body `Static.update` full re-renders | 150 (1/event) | 99 (offset-changing moves only; 0 during the 50 at-rest moves) |
| other-row `clear_selection` sweep calls | 150 (1/event) | 1 (once at drag arm) |
| sweeping-move handler time | 1.702 ms/event | 0.053 ms/event (32x) |
| at-rest (same-cell) handler time | 1.701 ms/event | 0.008 ms/event (213x) |

On slow hardware the wrap alone was the tens-of-ms/event term; it is now O(1) per event.

**Fixes** (all in `tldw_chatbook/Widgets/Console/`):
- (a) `console_transcript.py` `ConsoleTranscriptMessage`: new `_body_render_cache`
  (`_body_render_content()`); `get_display_text()` returns its `.plain` (O(1));
  invalidated at the top of `sync_message` immediately after `_message`/`_presentation`
  are reassigned - the ONLY post-construction writers - mirroring the markdown row's
  `_body_text` discipline (and before the selection clamp reads the text). `compose` and
  the highlight-restore path reuse the same cached Content.
- (b) new module-level `_body_wrap_table(text, width)` with `functools.lru_cache(maxsize=4)`:
  wrapped lines + aligned source offsets in one memoized table; `_body_cell_to_offset`
  is now a table lookup. Text change (streaming) and width change (resize mid-drag) are
  new keys, so invalidation is structural. Defensive-fallback semantics preserved exactly
  (fallback only for cells at/after the first unmodeled wrap edge): verified by an
  exhaustive old-vs-new equivalence sweep - 210,904 (text,width,cell) combinations across
  54 texts (unicode, wide chars, tabs, blank lines, random whitespace soup), 0 mismatches.
- (c) `set_selection_range` early-returns on an unchanged effective range in ALL THREE
  arms - plain body re-render, markdown strip, diff strip (the strips had the same
  ungated shape).
- (d) `console_selection.py` `SelectionManager.extend_drag` now returns bool (False for
  unchanged offset and the pre-existing no-op arms); `on_mouse_move` skips everything on
  False. The all-mounted-rows `clear_selection()` sweep moved from per-move to once per
  drag at arm time (`_clear_other_selection_highlights` in `on_mouse_down`); no other row
  can GAIN a highlight mid-drag (writers are the origin row and keyboard mode, which a
  press exits), so the per-move sweep was redundant after an arm-time sweep.

**Red-first evidence**: new `Tests/UI/test_console_selection_drag_perf.py` (4 tests):
wrap-count bound (was 30 wraps/30 moves -> asserts <=2), unchanged-offset no-re-render
(was 11 updates -> asserts 1), no per-move sweep (was 20 -> asserts 0), and a guard that
the arm-time sweep still clears a prior drag's stale highlight (green before AND after -
it pins the guarantee the per-move sweep used to provide). First three verified red on the
unmodified base, all four green after.

**Test evidence** (teed to `test-logs/`, all vs base `8e949873e`, `-p no:randomly`):
- selection batch A (rows/core/transcript/contract/keyboard + new perf file):
  before 4 failed/91 passed -> after 4 failed/95 passed, IDENTICAL failure node-ids
  (menu_open_row_body_click_dismisses..., citation_row_is_focusable...,
  count_only_change_reconciles_footer..., menu_anchor_derives_from_row_region...);
  all 4 also fail solo on the untouched base tree.
- selection batch B (menu/end_to_end/app_smoke/prune_bound): 117 passed -> 117 passed.
- windowing/prune invariants (windowing, two_sided_window, window_reconcile, pruning,
  stream_scrollback, visual_transcript): baseline 52 passed + 3 pre-existing
  visual_transcript teardown errors -> unchanged after (no new failures in these files;
  hydration/prune machinery untouched).
- transcript-adjacent (annotation_markers, fence_throttle, jump_pill, tail_follow,
  conversation_hydration, rail_state_prune): 4 annotation_markers failures - A/B'd by
  restoring the two touched files to base: identical failure set, pre-existing.
- native_transcript / transcript_region / markdown_widget (10 failed + 1 error): A/B'd
  the same way - identical failure sets at base, all pre-existing.
- Full `--collect-only` sweep (excluding the known-hanging fleet_teardown_notice):
  56,530 tests collected; 5 collection errors, all `ModuleNotFoundError: playwright`
  under Tests/Web_Scraping/Confluence (optional dep absent from this venv, unrelated).
- Ruff on the three touched/added files: clean.

**Deliberately not changed**: `_markdown_cell_to_offset` / `_diff_cell_to_offset` still
split the source per event (O(n) but a plain `str.split`, microseconds vs the wrap's
milliseconds - not part of the AC); `sync_message`'s unconditional
`_refresh_body_highlight` on signature change (streaming needs the re-render; reconcile
only calls it on signature change).
<!-- SECTION:NOTES:END -->
