---
id: TASK-31666
title: Align cost-cache tests with staged evidence and durable replies
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 17:58'
updated_date: '2026-09-05 18:09'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the cost-cache regression preconditions after staged evidence moved to Next Send and reply persistence became mandatory.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Staged evidence causes no current-spend tokenization on repeated ticks while its Next Send contribution remains covered.
- [x] #2 The warm-reply cache-break projection test reaches a real durable reply and retains its exact one-estimation assertion.
- [x] #3 Full affected cost-cache and current-spend test files pass with scoped static checks.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce both current failures and verify merged Current/On next send contract from TASK-31591 and existing staged-evidence semantic test. 2. Require zero current-spend tokenization of staged evidence across ticks, while independently proving positive Next Send contribution; keep exact unchanged-tick equality. 3. Attach real ChaChaNotesDB using existing helper before sending the warm reply and retain exact one-estimation projection/cache assertions. 4. Run full cost-cache and cost-screen files plus scoped static checks and self-review. ADR required: no. ADR path: backlog/decisions/095-conversation-owned-console-generation-settings.md (existing). Reason: test-only restoration of existing spend and durable-send preconditions, no pricing or runtime behavior changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Updated only the stale cache fixtures: staged evidence stays out of Current spend while increasing Next Send, and the warm-reply case uses the existing real ChaChaNotesDB helper. Exact zero/repeated-tick and one-projection-call assertions remain. Both complete affected files pass: 39 tests in105.94s, report /private/tmp/tldw-review-cost-cache-and-spend-20260905.xml. Full-file Ruff, changed-range format, diff whitespace and self-review pass. ADR not required: test-only existing spend/durability contract.
<!-- SECTION:NOTES:END -->
