---
id: TASK-40727
title: Provision Node for executable CI contracts on disposable runners
status: Done
created_date: 2026-11-13 22:08
assignee:
- '@codex'
updated_date: 2026-11-13 23:23
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The local PR Fast Lane fails executable JavaScript CI contracts because Node is absent from PATH. Declare the job runtime instead of relying on the hosted image inventory.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PR Fast Lane installs a supported Node runtime before running the real JavaScript contract tests.
- [x] #2 Existing JavaScript contract tests remain required and pass; focused routing checks verify setup precedes execution.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/360-isolated-self-hosted-ci-runners.md
Reason: Explicitly provide an existing test dependency within the accepted runner boundary.
1. Read failure logs and runtime consumers; add a focused workflow prerequisite contract and observe it fail.
2. Add official setup-node to the PR Fast Lane, without changing tests or image/controller state.
3. Run targeted CI contracts, independently review, and verify the normal PR run.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
PR Fast Lane now explicitly installs Node24 using actions/setup-node@v4 before its executable CI contracts. No test skips or production-image changes. A workflow prerequisite regression failed before the change;447 focused routing and actual JavaScript cleanup/cost-report cases passed afterward (2.32s). Independent review caught the existing exact step-count contract; it now includes the one Node step. Expanded targeted verification passed570 cases (7.12s), including the entire queue-pressure contract module. Ruff passes; existing queue-pressure formatting drift is retained. Current Node-less CI failures are in run43510161110 job132180868986; all63 failures belong to the two Node consumers. Existing ADR-360 applies; live Linux qualification passed.

Live normal PR qualification: run43519057842 job132211761772 passed4367 tests with3 existing skips in1835.36s on ghq-test-a6ab2c5e; job completed2026-11-13T23:14:15Z. The real Node-dependent contracts remained selected and passed. Evidence: /tmp/tldw-ci-node-live.log. Full gate remains in progress; this is lane qualification, not successful full-run timing. Bookkeeping stays local to avoid restarting healthy CI.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
