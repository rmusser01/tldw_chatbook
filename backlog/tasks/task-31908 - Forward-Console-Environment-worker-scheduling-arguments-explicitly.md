---
id: TASK-31908
title: Forward Console Environment worker scheduling arguments explicitly
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 19:55'
updated_date: '2026-09-05 20:02'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the Environment local and network worker scheduling contract explicit at the screen boundary so the worker-group guard can verify the already-distinct groups without altering runtime behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both Environment dispatch tiers retain their distinct existing groups, thread execution, exclusive scheduling and exit_on_error false policy.
- [x] #2 Worker-group Architecture check and complete Environment controller/wiring tests pass without adding an arbitrary group or relaxing the guard.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A; mechanical preservation of existing Environment scheduler interface and DESIGN.md explicit wiring. Caller census: environment.py has exactly two _run_worker calls, both supply thread=True, exclusive=True and the existing LOCAL_WORKER_GROUP or NET_WORKER_GROUP. 1. Record current worker guard RED. 2. Add real screen-wiring assertion for both exact scheduler argument sets. 3. Replace kwargs forwarding with named keyword-only parameters and explicit forwarding, retaining exit_on_error=False. 4. Verify worker guard and full Environment files, lint and root review before separate commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Preserved the existing two-tier Environment scheduler through explicit keyword-only thread/exclusive/group forwarding; exit_on_error remains False and no new group/default was introduced. Extended the real screen-wiring regression to dispatch both tiers and assert the complete exact kwargs sets. Existing worker Architecture guard was RED at the upstream kwargs-only constructor. Full Environment controller/wiring/section/state and worker guard96passed54.17s (XML /private/tmp/console-environment-31745.xml); Ruff lint and changed-region format checks pass. Root reviewed and approved the exact diff. No ADR required: existing scheduler contract preserved. Final screen17266lines/571methods, leaving393lines/12methods above unchanged main ceiling; no overall Architecture completion claimed.
<!-- SECTION:NOTES:END -->

## PR 2427 rebase renumbering provenance

Review-owned TASK-31745 was renumbered to TASK-31908 on 2026-09-06
while rebasing PR 2427 onto dev c4d45c0926. The user approved preserving
upstream task identities and renumbering review-created collisions only.
Original creation dates, task history, and literal verification artifact paths
are retained. See backlog/docs/pr-2427-rebase-reconciliation.md for the mapping.
