---
id: TASK-3310
title: >-
  Design decision: ingest queue placement, one consent grammar, and batch retry
status: Done
assignee:
  - '@claude'
created_date: '2026-08-07 19:30'
labels:
  - library
  - ingest
  - ux
  - design
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Three product questions raised by the 2026-08-07 Media Ingestion critique (tracking file `.impeccable/critique/2026-08-07-media-ingest-ux-options-review.md`), needing an owner ruling before implementation:

1. **Queue placement.** The ingest canvas welds a compose form to a monitoring ledger; Console is the live work surface and Home already mirrors ingest jobs. Should the canvas end at Start (one-line live status) with the ledger living in the rail/Home?
2. **One consent grammar.** Guaranteed failures gate inline; missing tooling raises a blocking modal — the worse outcome gets the quieter treatment. Could the commit-summary line carry all consent, retiring the modal (task-3300 fixes its rendering meanwhile)?
3. **Batch retry.** After installing a named missing dependency, the likeliest next action is the same source again, but the form auto-clears; per-row Retry is buried in the queue. Should "Retry this batch" be first-class?
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 Owner ruling recorded per question (keep/change, with rationale)
- [x] #2 Approved changes filed as their own implementation tasks
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Owner rulings taken 2026-08-08 (AskUserQuestion, follow-up batch kickoff):

1. **Queue placement: KEEP on the canvas.** The task-3304 receipt-into-view
   scroll + fold hint made the compose/monitor weld navigable; moving the
   ledger to rail/Home would be a full IA arc for marginal gain now.
2. **Consent grammar: UNIFY INLINE — retire the guardrail modal.** Approved;
   filed as task-3314. Tooling-warning consent folds into the commit/gate
   grammar (the two-press pattern the queue's Clear already uses); the
   task-3300 modal fix keeps it functional until 3314 lands.
3. **Batch retry: SHIP first-class.** Approved; filed as task-3313 —
   re-stage the same source with the same options in one action after a
   finished/failed run.

In the same ruling round the owner also decided sibling reachability tasks:
task-3307 images = SHIP; task-3308 XML = defer with honest unsupported copy.
<!-- SECTION:NOTES:END -->
