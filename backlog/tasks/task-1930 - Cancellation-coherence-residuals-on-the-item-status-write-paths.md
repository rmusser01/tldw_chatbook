---
id: TASK-1930
title: Cancellation-coherence residuals on the item-status write paths
status: To Do
assignee: []
created_date: '2026-08-02'
labels:
  - watchlists
  - correctness
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from task-1541's fix-wave re-review (report:
`.superpowers/sdd/briefings-residuals/task-1541-fixwave-rereview.md`), which APPROVED the wave
and parked two Minors:

1. **Write-failure + coincident-cancellation race in `_update_item_status`'s CancelledError
   handler** (`UI/Screens/watchlists_collections_screen.py`). The handler assumes the
   `to_thread` write landed (threads don't cancel) and synchronously patches the cached dict
   before re-raising. In the narrow window where the write itself RAISED (DB error) and a
   cancellation arrives in the same sub-millisecond, asyncio's `to_thread`/`wrap_future`
   plumbing discards the real exception unlogged and the handler patches the cache to a status
   the DB never reached. Not practically reachable today; the docstring currently overstates
   the guarantee.

2. **`_mark_item_unread` lacks the sibling cancellation-safety treatment.** The explicit
   "Mark unread" button shares the cross-item supersede worker group with mark-read-on-open but
   did not get task-1541's CancelledError cache patch: a cancelled toggle can leave the cached
   dict stale and produce no toast. Milder than the fixed data-loss case (self-healing on next
   load, no permanent status overwrite), which is why it was parked rather than folded into the
   wave.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The CancelledError handler only patches the cache when the write genuinely landed (or the docstring states the residual window honestly and the discarded-exception path at least logs type-only)
- [ ] #2 `_mark_item_unread` gets the same cancellation-coherence treatment as `_update_item_status` (patch-before-re-raise), with a test mirroring `test_a_cancelled_mark_read_still_leaves_the_cached_dict_coherent`
- [ ] #3 Existing item-status tests pass unchanged
<!-- AC:END -->
