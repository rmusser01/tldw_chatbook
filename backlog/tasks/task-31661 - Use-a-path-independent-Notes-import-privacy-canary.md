---
id: TASK-31661
title: Use a path-independent Notes import privacy canary
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 17:44'
updated_date: '2026-09-05 17:49'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep translated-error privacy checks meaningful when the checkout itself lives below a path containing private.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All seven expected-fault cases still reject leaked exception details and pass under a /private checkout path.
- [x] #2 The full Notes import executor tests and scoped static checks pass without changing runtime exception behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce all seven privacy failures and verify the matched text is only the checkout /private path in traceback frames. 2. Give injected expected faults one unmistakable payload canary and assert that canary is absent from translated str, repr and full traceback; retain type and chain-clearing checks. 3. Run the affected cases and full Notes import executor file, scoped static checks and self-review; document environment-related testing evidence. ADR required: no. ADR path: N/A. Reason: test-only privacy evidence correction, no runtime error translation or safety change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
All seven RED privacy checks matched /private in traceback source paths, not leaked fault payloads. Injected a unique NOTE-IMPORT-EXPECTED-FAULT-CANARY into every expected fault and retained absence checks on str, repr and full formatted traceback plus exact exception types and cleared cause/context. No runtime changes. Full Notes import executor file:137 passed90.37s; full-file Ruff lint and changed-region formatting pass; self-review/diff check clean. Added actual temp-root metadata and privacy-canary incidents to lessons-testing-evidence; the independent Notes conflict-executor qualification passed140 with staff-owned fixtures. ADR not required: test-only evidence correction.
<!-- SECTION:NOTES:END -->
