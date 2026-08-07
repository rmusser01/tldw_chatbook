---
id: TASK-3310
title: >-
  Design decision: ingest queue placement, one consent grammar, and batch retry
status: To Do
assignee: []
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
- [ ] #1 Owner ruling recorded per question (keep/change, with rationale)
- [ ] #2 Approved changes filed as their own implementation tasks
<!-- AC:END -->
