---
id: TASK-22218
title: >-
  Composer caret blink: no per-tick draft wrap, history scan, or under-modal
  ticking
status: Done
assignee:
  - '@claude'
created_date: '2026-08-24'
updated_date: '2026-08-25 22:02'
labels:
  - performance
  - console
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22218).

Pre-existing (the TASK-21692 layout fix holds — verified live, idle layouts 0 at tip vs 8
at pin — but the per-tick COMPUTE remains). `Widgets/Console/console_composer_bar.py:
2952-3003`: at 1.89 Hz while the composer has focus (the Console steady state), each blink
fires 2 `query_one` + placeholder/draft render; with a non-empty draft it additionally
runs `_ghost_suffix()` — a linear `startswith` scan over up to 1000 history entries
(`Chat/prompt_history.py:261-264`) — and a grapheme-aware `cell_len` wrap of the ENTIRE
draft (window sliced after the full wrap, `:2282-2296`): a pasted 20 KB draft is re-wrapped
1.89x/s forever. The resume gate is `has_focus_within` (`:2993-2996`), which survives
`push_screen` — every modal leaves the blink ticking and repainting underneath.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A blink tick with unchanged draft, width, and history performs no wrap and no history scan (memoized by those inputs; only the caret cell repaints)
- [x] #2 The wrap, when it does run, is bounded to the visible window rather than the whole draft, or the whole-draft cost is measured and accepted
- [x] #3 The blink pauses while the composer's screen is not the active screen (modal on top)
- [x] #4 Tick cost with a 20 KB draft measured before/after
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Red-first probes in Tests/UI/test_console_composer_cursor.py (reusing the 21692 real-CSS harness): (a) count _wrap_draft_line_slices + PromptHistory.complete calls over 6 idle blink ticks with a 20KB draft (nonzero today, 0 after warm-up after the fix); (b) covered-by-modal probe: push a ModalScreen, drive ticks, count renders/phase flips (flips today, frozen-solid + near-zero work after).
2. Add a revision counter to PromptHistory (bumped on every _entries mutation: load, optimistic append, cap trim, rollback) as the cheap ghost-input invalidation source.
3. Memoize _current_visible_draft_renderable by key (display draft, width, focused, segments-initialized, canonical text, canonical+display cursor index, style-range tuple, selection state, history index, history revision) -> per-phase {cursor_visible: Text} dict; a steady-state tick becomes key-compare + dict hit + Static.update(layout=False). The ghost suffix is part of the cached OUTPUT, invalidated via the revision in the key.
4. Modal gate in _toggle_cursor_blink, 22219 shape: keep the timer ticking, early-out on `not self.is_attached or not self.screen.is_active`; on a covered tick force the caret solid once (matching _sync_cursor_blink_state's pause convention) so resume is simply the next tick after the screen is active again -- no pause/resume bookkeeping.
5. AC2: keep the whole-draft wrap but measure it (20KB draft) and accept with numbers -- windowing the wrap needs total row count + caret row by construction; with the memo it no longer runs on ticks at all.
6. Measure per-tick ms before/after (20KB draft). Targeted suites + 21692 guard tests + collect-only sweep, preflight, mutation tests (drop draft from memo key; remove revision bump; remove modal gate), teardown walk.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Memoized the composer's visible-draft renderable and gated the blink tick on screen
activity; added a revision counter to PromptHistory as the cheap ghost-invalidation
source. Base dev `70d28febc`.

**Approach.** `_current_visible_draft_renderable` now keeps a one-entry render memo
`(key, {blink_phase: Text})`. The key (`_visible_render_memo_key`) contains every input
that shapes the output — display draft text, wrap width, focus, segment-model init,
canonical + display caret offsets, canonical text, style-range tuple, selection state,
recall index, and `(id(history), history.revision)` — so a stale hit is impossible by
construction rather than by remembering to clear the cache at each of the composer's
many mutation sites; the two O(draft) joins per tick are the price (µs against the ms
wrap they gate). The two blink phases cache separately: the caret cell (glyph vs space)
participates in the word wrap (space is a chunk boundary, glyph is not), so one phase's
wrapped output is NOT derivable from the other by character substitution. The ghost
suffix is part of the memoized OUTPUT; a history record while the composer idles reaches
the next tick via the new `PromptHistory.revision` (bumped on load, optimistic append,
cap trim, and write-failure rollback — every `_entries` mutation).

**Modal-gate decision (AC3).** Kept the TASK-22219 shape rather than pause/resume
bookkeeping: `_toggle_cursor_blink` early-outs on `not self.is_attached or not
self.screen.is_active` (Textual 8.2.8: `app.screen is self`). Textual posts
ScreenSuspend/ScreenResume to the Screen, not its descendants, so a widget has no clean
cover/uncover event; with the early-out the timer itself IS the resume path — the first
tick after the screen is active again blinks, nothing to un-pause. A covered tick parks
the caret SOLID once (matching `_sync_cursor_blink_state`'s pause convention; the
composer can be partly visible under a dialog), then costs one property check
(measured 0.001 ms).

**Numbers (real-CSS harness, 20 KB draft, width 91, 1000-entry history).**
Tick body before: median 1.579 ms (1 full-draft wrap ≈1.34–1.39 ms + 1 history scan
0.047 ms + renders), i.e. 6 wraps + 6 scans per 6 idle ticks (red-first probe).
After: median 0.109–0.112 ms per tick, 0 wraps + 0 scans per 6 idle ticks after the
two-phase warm-up (the hidden phase's first render after an edit is real work, once —
off the typing path, on the first 0.53 s tick). Covered-by-modal tick: 0.001 ms.
AC2 taken on its "measured and accepted" arm: the whole-draft wrap (≈1.34 ms at 20 KB)
now runs only on memo misses — real edits/resizes/history changes, where a render is
needed anyway; windowing the wrap itself would need total row count + caret row, which
require the full wrap by construction, and is not worth the surgery once ticks no
longer pay it.

**Verification.** Red-first probes committed in `Tests/UI/test_console_composer_cursor.py`:
idle-tick wrap/scan counter (6/6 before → 0/0 after), under-modal freeze+resume (phases
`[F,T,F,T,F,T]` before → `[True]×6` + resume after pop), plus two staleness guards
(typing repaints new draft + caret; `history.append` while idle updates the painted
ghost). Both TASK-21692 guard tests stay green (`layout=False` untouched; geometry
identity untouched). Targeted run: 14 suites, 576 passed / 9 failed — all 9 reds
reproduced identically with the base-SHA production files (send/collapse/undo/command
cluster; pre-existing dev reds, list in the branch tee logs), 0 new. Mutation tests:
M1a memo key made constant → both staleness guards red; M1b history revision dropped
from key → exactly the ghost guard red; M2 modal gate removed → modal probe red; all
restored (working tree byte-identical to commit). Teardown walk: tick after `remove()`
is a no-op (no NoScreen), tick while collapsed/after expand raises nothing. Collect-only
sweep: 59,420 collected, 28 errors — all optional-dependency modules (numpy/audio/TTS/
transcription/Confluence), pre-existing class. `./scripts/preflight.sh` all green.

**Files.** `tldw_chatbook/Widgets/Console/console_composer_bar.py` (memo key + cache,
`_build_visible_draft_renderable` split, tick gate, `set_prompt_history` cache drop),
`tldw_chatbook/Chat/prompt_history.py` (`revision`), `Tests/UI/test_console_composer_cursor.py`
(4 new tests + helpers).
<!-- SECTION:NOTES:END -->
