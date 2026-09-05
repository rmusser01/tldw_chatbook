---
id: TASK-31750
title: >-
  Retire proven private Console controller delegates without changing screen
  contracts
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-05 20:06'
updated_date: '2026-09-05 22:15'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove redundant private screen forwarding methods so existing controller ownership is the normal call path, preserving public, framework, and documented screen-name contracts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Exactly the64 approved private delegates are removed and their real callers resolve the existing owners, while public/event/action methods, agent-bridge injection and transcript ephemeral lookup remain intact.
- [ ] #2 Callback lookup phases, arguments, await/return behavior and targeted fault-injection semantics are preserved, with explicit late-bound hook regressions.
- [ ] #3 Affected whole-file checks and static ownership tests pass or separately evidenced baseline failures remain tracked, and actual screen size satisfies unchanged ceilings without unrelated compression or boundary moves.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A. Follow root-approved Docs/superpowers/plans/2026-09-05-console-private-delegate-cleanup-proposal.md exactly, after TASK31749 purehelper stage. 1. Read task and retain exact64-method caller/patch/phase inventory. 2. Remove delegates by existing-owner cohorts, retarget only actual screen callers and named late-bound wiring lambda bodies, preserving documented excluded screen methods. 3. Migrate receiver-specific test patches and bare-shell wiring, preserving every behavioral assertion and failure/cancellation phase. 4. Verify argument/return fidelity, delayed/replacement-owner callbacks, whole affected83-file census and architecture guards; measure actual unchanged-ceiling counts. 5. Root reviews each cohort before scoped commits; no shared diagnostic fixture edits or STAY extension.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Checkpoint, not complete: 64 approved private delegates retired with preserved assertions and delayed callback lookup. 145 architecture/import/inventory tests and 121 question/composer tests pass. The 84-file run stopped for the new dev rebase after 1749 passed and 7 failed; two owner-fixture failures subsequently pass in a 203-test selection. Five other observed cases and the unexecuted remainder still require review. Current measured size and tightened cap are 16818 lines / 505 methods. ADR not required; existing controller boundaries unchanged.
<!-- SECTION:NOTES:END -->
