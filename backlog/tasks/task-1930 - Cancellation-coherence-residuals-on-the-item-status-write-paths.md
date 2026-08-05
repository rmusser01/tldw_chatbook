---
id: TASK-1930
title: Cancellation-coherence residuals on the item-status write paths
status: Done
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
**Superseded by task-1541's Qodo redesign (desired-status coalescing) -- see that task's
Implementation Notes, "Qodo redesign" section. Both residuals below are now moot: the mechanism
they described no longer exists in the code.**

Originally filed from task-1541's fix-wave re-review (report:
`.superpowers/sdd/briefings-residuals/task-1541-fixwave-rereview.md`), which APPROVED that wave
and parked two Minors against its `exclusive=True` "supersede" worker-group model:

1. **Write-failure + coincident-cancellation race in `_update_item_status`'s CancelledError
   handler** (`UI/Screens/watchlists_collections_screen.py`). The handler assumed the
   `to_thread` write landed (threads don't cancel) and synchronously patched the cached dict
   before re-raising. In the narrow window where the write itself RAISED (DB error) and a
   cancellation arrived in the same sub-millisecond, asyncio's `to_thread`/`wrap_future`
   plumbing could discard the real exception unlogged and the handler would patch the cache to
   a status the DB never reached.

2. **`_mark_item_unread` lacked the sibling cancellation-safety treatment.** The explicit
   "Mark unread" button shared the cross-item supersede worker group with mark-read-on-open but
   did not get task-1541's CancelledError cache patch: a cancelled toggle could leave the cached
   dict stale and produce no toast.

A follow-up, broader Qodo review of the same fix wave went further than either of the two Minors
above: it found the whole cancellation-based "supersede" model unsound for a durable write (not
just in these two narrow corners) and had it replaced outright with desired-status coalescing
(`_dispatch_item_status`/`_drain_item_status`/`_ItemStatusIntent`) -- see task-1541's notes. That
redesign deletes the `except asyncio.CancelledError` handler in `_update_item_status` entirely
(AC #1's target) and deletes `_mark_item_unread` entirely, folding its gate logic into the shared
`_item_status_write_allowed` helper every gated write now goes through (AC #2's target). Nothing
in the new code is ever cancelled by this screen's own logic, so there is no
"cancellation-coherence" gap left to patch on either path -- the premise both ACs were written
against no longer exists.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Moot: `_update_item_status`'s `except asyncio.CancelledError` handler this AC targeted was deleted outright by task-1541's Qodo redesign, not patched -- nothing calls this method under a cancellation any more, so there is no write-failure/cancellation race left to close
- [x] #2 Moot: `_mark_item_unread` this AC targeted was deleted outright by the same redesign, folded into `_item_status_write_allowed` (shared by both gated writes) and `_drain_item_status` (which never cancels); there is no sibling treatment left to add because there is no cancellation to be coherent against
- [x] #3 Existing item-status tests pass unchanged by this bookkeeping-only task (no production code touched here; task-1541's own redesign commit is verified separately in that task's notes)
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
No code changes in this task. Closed as obsolete-by-supersession (honest bookkeeping, not
"Won't Do" -- the underlying CONCERN this task raised, cache/DB divergence around a cancelled
write, is actually resolved, just not by the incremental patch these ACs originally described).
The redesign that supersedes both residuals is documented in task-1541's Implementation Notes
("Qodo redesign (desired-status coalescing replaces cancellation)" section) and landed on branch
`fix/task-1541-item-status-off-loop`. Left this file in place per project convention (never delete
a filed task) rather than removing it now that its content is stale.
<!-- SECTION:NOTES:END -->
