---
id: TASK-22218
title: >-
  Composer caret blink: no per-tick draft wrap, history scan, or under-modal
  ticking
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-24'
updated_date: '2026-08-25 16:07'
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
- [ ] #1 A blink tick with unchanged draft, width, and history performs no wrap and no history scan (memoized by those inputs; only the caret cell repaints)
- [ ] #2 The wrap, when it does run, is bounded to the visible window rather than the whole draft, or the whole-draft cost is measured and accepted
- [ ] #3 The blink pauses while the composer's screen is not the active screen (modal on top)
- [ ] #4 Tick cost with a 20 KB draft measured before/after
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
