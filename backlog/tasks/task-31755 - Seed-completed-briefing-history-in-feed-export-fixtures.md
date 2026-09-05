---
id: TASK-31755
title: Seed completed briefing history in feed export fixtures
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 20:17'
updated_date: '2026-09-05 22:58'
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
- [x] #5 Briefing script scope fixtures reach their unchanged isolation assertions using legal completed history
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the original feed-export/query failure evidence and its completed-history repairs.
2. Reproduce the remaining briefing-script scope failure; seed its two historical briefing rows with explicit complete status without altering scope assertions or active-run constraints.
3. Run complete presets, feed export/query/RSS and DB provenance/admission files, lint and formatting checks, then record evidence.
ADR required: no
ADR path: N/A
Reason: Test-only correction of historical fixture data; existing storage and admission contracts remain unchanged.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Changed only the two feed test helpers to pass status=complete through insert_briefing; exported history no longer occupies the single-generating-briefing claim. Original export baseline7failed/18passed; first broader selection then exposed the same fixture mistake in4query tests. Final four complete files:76passed1.59s, covering export files/RSS/ordering/paging/path safety and the unchanged DB migration/admission/concurrent-claim guards. XML:/private/tmp/tldw-31755-feed-history-fixed.xml. Whole two-file Ruff, changed complete-helper formatting and diff checks pass. No production/schema/constraint change; ADR not required. Files:Tests/Subscriptions/test_briefing_feed_export.py and test_briefing_feed_query.py.

Continuation: the whole presets file reproduced 1 failure/24 passes because its script-scope fixture seeded two generating rows. Both are now explicitly complete; the scope assertion is unchanged. Final five complete files passed 101 tests (4.39s), including unchanged DB admission/migration guards. XML: /private/tmp/tldw-31755-script-scope-final.xml. Whole-file Ruff and changed-function formatter check pass; whole-file formatter reports unrelated pre-existing drift. No production change.
<!-- SECTION:NOTES:END -->
