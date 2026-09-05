---
id: TASK-31705
title: Seed retry-capable runs in busy-state rendering coverage
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 18:49'
updated_date: '2026-09-05 18:57'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make Watchlists busy-state tests exercise operations that are eligible under the existing recovery policy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Busy-state tests use an actual retry-capable failure and preserve all in-place labels, identity, and enabled versus disabled assertions.
- [x] #2 Unrelated busy operations leave the eligible selection enabled, and the full runs pane file passes with existing ineligible-run coverage intact.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve both busy-state RED cases and compare their incomplete run rows with canonical failure projection and existing retry eligibility tests. 2. Seed failed status plus connection_failure category in only these two fixtures. Preserve every busy label, table identity, and disabled/enabled assertion. 3. Run the complete runs pane file, scoped checks, review, and commit. ADR required: no. ADR path: N/A. Reason: fixture correction to the already shipped fail-closed recovery policy; no product changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The two busy-state fixtures now carry a failed run with canonical connection_failure recovery metadata, so they actually qualify for retry under the existing fail-closed policy. Every original enabled/disabled, label, unrelated-target and in-place table identity assertion remains unchanged. All 39 runs-pane tests passed within the clean 125-test fixture gate (/private/tmp/tldw-31702-31703-31705-first.xml), including existing eligible and ineligible recovery cases. Scoped Ruff/format and diff checks passed. No product changes or new ADR required.

Parent reviewed the final scoped diff with no actionable findings.
<!-- SECTION:NOTES:END -->
