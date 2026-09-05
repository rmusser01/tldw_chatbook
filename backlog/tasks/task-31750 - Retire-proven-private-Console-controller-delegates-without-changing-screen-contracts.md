---
id: TASK-31750
title: >-
  Retire proven private Console controller delegates without changing screen
  contracts
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-05 20:06'
updated_date: '2026-09-05 22:37'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove redundant private screen forwarding methods so existing controller ownership is the normal call path, preserving public, framework, and documented screen-name contracts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Exactly the64 approved private delegates are removed and their real callers resolve the existing owners, while public/event/action methods, agent-bridge injection and transcript ephemeral lookup remain intact.
- [x] #2 Callback lookup phases, arguments, await/return behavior and targeted fault-injection semantics are preserved, with explicit late-bound hook regressions.
- [ ] #3 Affected whole-file checks and static ownership tests pass or separately evidenced baseline failures remain tracked, and actual screen size satisfies unchanged ceilings without unrelated compression or boundary moves.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A. Follow root-approved Docs/superpowers/plans/2026-09-05-console-private-delegate-cleanup-proposal.md exactly, after TASK31749 purehelper stage. 1. Read task and retain exact64-method caller/patch/phase inventory. 2. Remove delegates by existing-owner cohorts, retarget only actual screen callers and named late-bound wiring lambda bodies, preserving documented excluded screen methods. 3. Migrate receiver-specific test patches and bare-shell wiring, preserving every behavioral assertion and failure/cancellation phase. 4. Verify argument/return fidelity, delayed/replacement-owner callbacks, whole affected83-file census and architecture guards; measure actual unchanged-ceiling counts. 5. Root reviews each cohort before scoped commits; no shared diagnostic fixture edits or STAY extension.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The 64 private delegates and late-bound callbacks are verified; 65 ownership guards pass after rebasing onto 53194eee67. Whole-file census remains incomplete: pre-rebase 1749 passed / 7 failed, two owner fixtures subsequently repaired. New upstream Console work measures 16899 lines / 508 methods against the retained 16818 / 505 ceiling. Further bounded paydown and remaining UI cases are still required. No new ADR; boundaries unchanged.
<!-- SECTION:NOTES:END -->
