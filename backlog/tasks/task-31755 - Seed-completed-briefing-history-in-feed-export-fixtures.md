---
id: TASK-31755
title: Seed completed briefing history in feed export fixtures
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 20:17'
updated_date: '2026-09-05 20:20'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Seven feed-directory export tests create several default generating briefing rows for one watchlist, violating the current single-active-briefing constraint before export is exercised. These fixtures represent finished audio history, not concurrent work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All seven export regressions reach their original filesystem and feed assertions using completed briefing history
- [x] #2 The active-briefing uniqueness and admission guards remain unchanged and their tests pass
- [x] #3 Complete affected feed and DB admission test files plus scoped static checks pass
- [x] #4 Feed query ordering and pagination fixtures also model completed historical briefings, with all four newly exposed query failures resolved
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the seven-failure export baseline and inspect the insert/admission contract.
2. Seed completed briefing status explicitly through the existing DB insertion API in export fixtures. Complete-file verification also exposed the identical default-generating setup in the query helper (four failures); apply the same correction there without changing status-filter assertions.
3. Run complete feed export/query/RSS and DB provenance/admission files, lint and changed-range formatting, then review and record evidence.
ADR required: no
ADR path: N/A
Reason: Test-only correction to completed historical fixture data; no storage, scheduling, or API contract change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Changed only the two feed test helpers to pass status=complete through insert_briefing; exported history no longer occupies the single-generating-briefing claim. Original export baseline7failed/18passed; first broader selection then exposed the same fixture mistake in4query tests. Final four complete files:76passed1.59s, covering export files/RSS/ordering/paging/path safety and the unchanged DB migration/admission/concurrent-claim guards. XML:/private/tmp/tldw-31755-feed-history-fixed.xml. Whole two-file Ruff, changed complete-helper formatting and diff checks pass. No production/schema/constraint change; ADR not required. Files:Tests/Subscriptions/test_briefing_feed_export.py and test_briefing_feed_query.py.
<!-- SECTION:NOTES:END -->
