---
id: TASK-41800
title: Bound Chatbook summary projection to its requested window
status: In Progress
created_date: 2026-11-18 17:56
assignee:
- '@codex'
labels:
- performance
- library
references:
- https://github.com/rmusser01/tldw_chatbook/pull/3610
documentation:
- Docs/QA/console-library-responsiveness/README.md
updated_date: 2026-11-18 18:28
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make request-local artifact work scale with ranks and selected summaries rather than hashing every record.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Artifact windows keep exact counts, ranks, sort boundaries, identity and revision across the same scopes.
- [x] #2 Only the requested page and exact detail build summaries; browsing keeps the existing request-owned registry snapshot.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Pin identity and revision parity while counting actual summary digest construction.
2. Rank lightweight order keys, then build summaries for selected IDs. Use the same predicate for exact summary.
3. Run registry and composite catalog controls and compare window CPU.
ADR required: no
ADR path: backlog/decisions/172-library-artifacts-browse-and-navigation.md
Reason: The existing artifact read boundary and revision contract stay in place.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The existing request snapshot ranks matching lightweight keys before constructing page summaries. A native exact lookup uses the same filter and revision construction. Sixty Chatbook and Library cases passed. At 10,000 records the first 20 summaries took 3.02 ms CPU from 46.61 ms. The bounded digest test and exact before/after/inclusive comparisons passed. ADR-172 applies. Local acceptance is verified. Hosted qualification is pending.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
