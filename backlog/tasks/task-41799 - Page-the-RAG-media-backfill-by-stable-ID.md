---
id: TASK-41799
title: Page the RAG media backfill by stable ID
status: In Progress
created_date: 2026-11-18 17:56
assignee:
- '@codex'
labels:
- performance
- rag
references:
- https://github.com/rmusser01/tldw_chatbook/pull/3610
documentation:
- Docs/QA/console-library-responsiveness/README.md
updated_date: 2026-11-18 18:28
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove repeated prefix scans in existing media reconciliation and document enumeration.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both media passes use bounded ascending ID seeks and keep the same active, non-trash membership.
- [x] #2 Backfill remains resumable and reconciliation respects deletion, existing authority and recovery behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Test both native Media passes across gaps and deletion before the cursor advances.
2. Seek after the last visited ID using the existing execute_query owner and page size. Keep both entry and ID results equivalent.
3. Run targeted backfill and recovery tests and compare the large SQLite sweep.
ADR required: no
ADR path: backlog/decisions/030-derived-index-lifecycle-and-atomic-media-migrations.md
Reason: The existing derived index lifecycle and source ownership apply to the changed internal scan.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Both media walks now resume after the latest visited primary key under the original active and trash predicates. A real deletion between pages fails on the frozen offset query and passes with keyset pagination. Six backfill cases and 22 projection recovery controls passed in the existing bootstrap profile. Exact active membership matched the frozen source. The 100,000-row median fell from 6,167.9 to 72.8 ms. ADR-030 applies. Local acceptance is verified. Hosted qualification is pending.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
