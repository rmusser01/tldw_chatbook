---
id: TASK-31694
title: Align Watchlists refresh and retry tests with committed Reader contracts
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 18:36'
updated_date: '2026-09-05 18:46'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update stale regression expectations to the explicitly shipped retained-open-row and visible retry chrome behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Refresh tests assert the open row is retained while new page contents and selected Reader content remain correct.
- [x] #2 Reader failure coverage asserts visible retry chrome with stale content cleared and recovery still available.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read the merged refresh-retention and visible retry-chrome contracts and preserve the three existing RED cases. 2. Assert retained open rows alongside new first-page data and Reader identity; assert visible retry controls while stale data is cleared. Keep real service requests and bounded settlement. 3. Run both complete affected Watchlists files, scoped checks, parent review, and scoped commit. ADR required: no. ADR path: N/A. Reason: test-only alignment with explicitly merged behavior, no product changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Aligned stale tests with the explicit Reader contracts already merged in TASK31725 / b6740f08912 (retain the open row on same-context refresh) and a62d92f590c (keep retry chrome visible). Refresh still requests a fresh first page and preserves selected content; assertions now also pin retained rows without inflating the service snapshot count. Failure coverage retains all stale-data and retry checks and adds real search/filter/pager visibility and geometry. Both complete Watchlists files passed, 48 pagination plus 57 scoped rebuilds, within the clean 269-test combined gate (/private/tmp/tldw-31693-31695-final.xml). Scoped Ruff and diff checks passed. No product changes or new ADR required.

Parent completed bounded final diff review with no actionable findings.
<!-- SECTION:NOTES:END -->
