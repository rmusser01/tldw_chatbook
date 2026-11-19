---
id: TASK-21244
title: >-
  Console left rail still repaints its fold copy through a layout-forcing
  update
status: Done
assignee:
  - rmusser01
created_date: '2026-08-23'
labels:
  - console
  - performance
dependencies: []
priority: low
---

## Description

Source: close-out of the 2026-08-22 holistic performance review burn-down; informational
finding from the TASK-21117 review (PR #2016, `7489a0ec8`) promoted to a follow-up.

TASK-21117 split the Inspector right rail's pure-scroll path from its layout reconcile and
documented the underlying Textual trap in a new `backlog/docs/lessons-textual.md` entry:
`Static.update(str)` schedules `refresh(layout=True)`, so a copy-only repaint costs roughly
**two extra whole-screen layout passes**. `UI/Console_Modules/left_rail.py:1086` is now the
**only remaining instance** of that trap in the Console rails — it repaints the fold hint with
`hint.update(text)` on rail transitions.

The same file also keeps a shadow `_outer_hint_text` field (`:323`, `:1083`, `:1085`) — the
shape TASK-21117's AC3 forbade on the right rail. Here it is safe: the skip requires the
shadow **and** the live renderable to match (`text == self._outer_hint_text and
str(hint.renderable) == text`), so the worst case is one redundant write rather than a missed
repaint. The left rail has no stand-down path and the two rails' copy predicates genuinely
differ, so this is not a copy-paste of the right-rail fix. The TASK-21117 reviewer confirmed
none of this is a latent left-rail bug — the cost is real but bounded to transitions, which is
why it is low priority rather than dropped.

## Acceptance Criteria

- [x] A left-rail fold-copy change does not force a whole-screen layout pass when only the text changed
- [x] Left-rail fold and unfold copy is unchanged in every state the existing rail tests cover
- [x] The shadow-field shape is brought in line with the right rail's, or the reason the left rail needs a different one is recorded in the code
- [x] A test fails if a copy-only left-rail update reintroduces a layout pass

## Implementation Plan

1. Mirror the right rail's TASK-21117 fix shape: a `_paint_outer_hint`
   static method that reads the live widget's content (no shadow), skips
   unchanged copy, and writes with `layout=False`.
2. Remove the `_outer_hint_text` shadow field; `_update_outer_hint`
   delegates to the painter.
3. Record the pin justification in code: the slot's one-row height is
   pinned by INLINE styles at compose time (min/max_height = 1), which
   beat every stylesheet rule -- the lessons-textual.md note.
4. Tests: a direct copy-only probe counting `Screen._refresh_layout`
   (mutation-verified), an in-test-baseline scroll integration pin, and
   an unchanged-copy skip + shadow-absence pin.

## Implementation Notes (2026-10-11, dev `9f8533686c`)

- **Fix**: `ConsoleLeftRail._paint_outer_hint` (left_rail.py) mirrors the
  right rail's helper verbatim in shape: read `hint.content`, skip when
  unchanged, `hint.update(text, layout=False)`. `_update_outer_hint` now
  only computes the copy and delegates; the `_outer_hint_text` shadow
  (the TASK-21117 AC3 shape) is deleted from `__init__` and the predicate.
  The pin is INLINE (`min_height = 1` / `max_height = 1` assigned at
  compose, line ~2667), documented in the painter's docstring per the
  lessons-textual.md "the pin may not be in the stylesheet" note.
- **Tests**: `Tests/UI/test_console_left_rail_outer_hint_t21244.py` (3).
  `test_copy_only_outer_hint_update_schedules_no_screen_layout` is the
  AC4 trap: 3 direct copy-only writes over the mounted rail, zero
  `Screen._refresh_layout` passes, hint region pinned across the writes.
  Mutation check: dropping `layout=False` from the painter makes exactly
  that test FAIL (verified by edit-run-restore); restored, all 3 pass.
  The scroll-integration test pins the copy blanking at scroll end with a
  within-test baseline (a copy-preserving scroll) because Textual
  coalesces a `layout=True` copy write into the scroll's own
  `_on_timer_update` pass -- at base the coalesced gesture shows NO delta,
  which is why the direct probe is the load-bearing trap (recorded here so
  the integration pin is not mistaken for one). The third test pins the
  unchanged-copy skip and the shadow field's absence.
- **No regression**: rail batteries
  (`test_console_rail_reconciliation.py`, `test_console_left_rail.py`,
  `test_console_context_rail_fits.py`, `test_console_resize_reflow.py`,
  `test_console_context_rail_header.py`): 62 failures byte-identical to
  base (`diff` of sorted FAILED lists empty -- the pre-existing
  shared-profile `RecoveryRequired` class). The TASK-21595 static-update
  census shows the same 3 pre-existing failures at base and after; my 3
  new tests are the +3 passed delta.
- **Ruff**: no new findings (only line-number shifts of pre-existing
  BLE001s); format hunk count unchanged (0).
- **ADR required: no** -- a one-line repaint discipline fix inside an
  existing widget, mirroring an already-shipped sibling pattern.
