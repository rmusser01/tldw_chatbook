---
id: TASK-22888
title: >-
  Third whole-screen recompose in _sync_library_canvas should be canvas-scoped
status: To Do
assignee: []
created_date: '2026-08-26'
updated_date: '2026-08-29'
labels:
  - performance
  - library
priority: medium
dependencies: []
---

## Description

`_sync_library_canvas()` performs **three** whole-screen `screen.refresh(recompose=True)`
calls (`library_screen.py:1917`, `:1932`, `:1993` at dev `251ea46e11`; these line
numbers drift — locate them by function, not by line). A whole-screen recompose on a
canvas **sync** path rebuilds the entire Library screen rather than the canvas subtree,
which is the regression class `Tests/UI/test_library_recompose_ratchet.py` exists to
catch and the same family as TASK-15457 ("convert per-click whole-screen recomposes to
canvas-scoped sync").

This is a **performance question, not a broken gate.** See the history below: the task
was filed when the ratchet was red, and that redness is gone.

## History — filed as a blocker, since downgraded

**Filed 2026-08-26.** The ratchet failed on pristine dev at **75 sites found, 74
allowed**, blocking the required gate for every Library PR. The third
`screen.refresh(recompose=True)` in `_sync_library_canvas()` arrived with the
adaptive-reader / Prompts-migration series (`3e8b104f6f` and follow-ups `c1a1adbe6b`,
`f1275c8846`, `82b2d626f8`), none of which moved the pin. Attribution at the time:
dev's own `library_screen.py` swapped into an unrelated feature branch failed
identically at 75/74, and an AST census diff against `732105c2d` named exactly one
added entry.

**Re-checked 2026-08-29 against dev `251ea46e11`: the ratchet PASSES** (4 passed, pin
still 74). Dev came back under the pin by removing whole-screen recomposes elsewhere,
so the third call in this function was effectively paid for out of other savings rather
than assessed. The three calls are still there.

**A note on a stale snapshot:** a review of this task observed only *two* counted calls
in the function and concluded the third was fictional. That was accurate for the dev
state it read; the count moved twice while this task sat open. Verified again at
`251ea46e11`: three real calls (a fourth and fifth textual match are prose in a comment
and a docstring, not call sites). Re-run the census before acting rather than trusting
any line number written here.

## Acceptance Criteria

- [ ] The three `screen.refresh(recompose=True)` calls in `_sync_library_canvas()` are
      each assessed: narrowed to a canvas-scoped refresh, or kept with the reason
      recorded next to the call
- [ ] `Tests/UI/test_library_recompose_ratchet.py` still passes afterwards
- [ ] If any call is narrowed, the pin drops to match the new census rather than
      staying at 74 — the headroom is not banked
- [ ] Prompts-in-adaptive-reader behaviour that these recomposes were added to fix still
      works (the migration's own tests stay green)
