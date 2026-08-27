---
id: TASK-22888
title: >-
  Library recompose ratchet is RED on dev - a third whole-screen recompose landed in _sync_library_canvas
status: To Do
assignee: []
created_date: '2026-08-26'
labels:
  - dev-red
  - performance
  - library
priority: high
dependencies: []
---

## Description

`Tests/UI/test_library_recompose_ratchet.py::test_library_screen_whole_screen_recompose_count_is_ratcheted`
fails on pristine `dev` (`4cee590b83`): **75 whole-screen recompose sites found, 74 allowed**.

The 75th site is a third `screen.refresh(recompose=True)` inside `_sync_library_canvas()`,
which previously held two. It arrived with the adaptive-reader / Prompts-migration series
(`3e8b104f6f` "feat(library): migrate Prompts to adaptive reader" and the follow-ups
`c1a1adbe6b`, `f1275c8846`, `82b2d626f8`), none of which moved the ratchet pin.

This is the exact regression class the ratchet exists to catch: a whole-screen recompose on
a canvas **sync** path rebuilds the entire Library screen rather than the canvas subtree.
It is the same family as TASK-15457 ("convert per-click whole-screen recomposes to
canvas-scoped sync"), which should be read alongside this.

The red blocks the required gate for every Library PR, so it needs resolving either way.

**The pin must not simply be bumped to 75.** Bumping absorbs the regression and spends the
ratchet's entire purpose; the decision is whether this recompose is *necessary*, and only if
it is does the pin move — with the justification recorded next to it.

## Attribution evidence

Established by swapping `dev`'s own pristine `library_screen.py` into an otherwise unrelated
feature branch and re-running the single test: it fails identically at 75/74, so the site is
dev's, not the branch's. An AST census diff of dev's file against `732105c2d` (this series'
fork point) reports exactly one added entry:

```
_sync_library_canvas  ->  screen.refresh(recompose=True)
```

Discovered while rebasing PR #2129 (TASK-22500) onto current dev.

## Acceptance Criteria

- [ ] The third `screen.refresh(recompose=True)` in `_sync_library_canvas()` is assessed:
      either narrowed to a canvas-scoped refresh, or kept with a recorded justification
- [ ] `Tests/UI/test_library_recompose_ratchet.py` passes on `dev`
- [ ] If the pin moves to 75, the reason is recorded next to the pin; if the recompose is
      narrowed instead, the pin stays at 74
- [ ] Prompts-in-adaptive-reader behaviour that the recompose was added to fix still works
      (the migration's own tests stay green)
