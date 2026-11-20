---
id: TASK-41801
title: Bound Watchlists Reader page selection work
status: In Progress
created_date: 2026-11-18 17:57
assignee:
- '@codex'
labels:
- performance
- watchlists
references:
- https://github.com/rmusser01/tldw_chatbook/pull/3610
documentation:
- Docs/QA/console-library-responsiveness/README.md
updated_date: 2026-11-18 18:28
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reduce disk work before the Reader returns a usable exact page.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 First and continuation pages preserve timestamps, ties, counts, filters, and previews while using a bounded row-read plan.
- [x] #2 Current and existing local databases preserve data, recovery admission, and their registered schema upgrade path.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm the exact chronological and status query shapes, migrations, and read-only compatibility.
2. Keep the exact read snapshot. Let global chronological first pages use the existing date index and build previews only for page IDs.
3. Compare All, Unread and keyword Reader queries and run pagination and recovery controls.
ADR required: no
ADR path: backlog/decisions/042-watchlists-reader-first-ia.md
Reason: The existing index and stable Reader snapshot contract apply. The query changes preserve the stored schema.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The initial count and high-water share a read snapshot. The Reader selects bounded IDs before projecting previews, using the existing chronological index for global All. Explicit and continuation bounds stay in the query. Matched first and continuation pages passed for All, Unread and FTS. At 100,000 rows All improved from 240.0 to 13.9 ms. Watchlists database, Reader and native lifecycle verification passed. The existing ADR-042 and schema apply. Local acceptance is verified. Hosted qualification is pending.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
