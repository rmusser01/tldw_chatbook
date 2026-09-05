---
id: TASK-31636
title: >-
  canvas_sync: queue_after_recompose(None) on a failure path clears an owner's
  queued follow-up callback
status: To Do
assignee: []
created_date: '2026-09-05 15:19'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Media wave5-F final review (M-3, 2026-09-05, .superpowers/sdd/2026-09-05-media-ux-wave5-pr-f/final-review.md) found that canvas_sync.py's five failure-path calls to queue_after_recompose(None) (lines ~659, ~674, ~736, ~740, ~747 at branch head 1b1d8b8d84) unconditionally clear whatever recompose follow-up an owning caller had queued, across all four canvas kinds (media/conversations/notes/prompts), not only media. This predates fix/media-wave5-f and was accepted as out of scope for that branch's own focus-restore seam, which is bounded by its own guards (it only moves focus off None or a grip, never a deliberately-set target) so the observed worst case is benign today. It should be looked at on its own so a future caller that relies on queue_after_recompose surviving a failure path is not silently broken.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each of the five queue_after_recompose(None) call sites on a canvas-sync failure path is reviewed and the intended contract (clear vs preserve a queued follow-up on failure) is documented or fixed
- [ ] #2 If fixed, all four canvas kinds (media, conversations, notes, prompts) are covered, not only media
- [ ] #3 No regression in the task-31567 focus-restore seam's own tests
<!-- AC:END -->
