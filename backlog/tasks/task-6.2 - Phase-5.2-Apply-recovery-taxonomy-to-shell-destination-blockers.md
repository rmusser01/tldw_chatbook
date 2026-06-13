---
id: TASK-6.2
title: Phase 5.2 Apply recovery taxonomy to shell destination blockers
status: Done
assignee:
  - '@codex'
created_date: '2026-05-05 02:55'
labels:
  - unified-shell
  - phase-5
  - recovery
  - destination-blockers
dependencies: []
documentation:
  - >-
    Docs/superpowers/qa/unified-shell/phase-5/2026-05-05-shared-recovery-taxonomy.md
  - >-
    Docs/superpowers/qa/unified-shell/phase-1/2026-05-03-destination-action-audit.md
  - >-
    Docs/superpowers/qa/unified-shell/phase-4/2026-05-05-phase-4-destination-service-adoption-closeout.md
parent_task_id: TASK-6
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Apply the shared Phase 5 recovery taxonomy to the highest-impact shell destination blocked states so users can understand and recover from unavailable ACP, Schedules, Workflows, and Artifacts actions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ACP runtime-unconfigured state exposes what is unavailable, why, next action, recovery target, authority owner, stable selector, and disabled tooltip copy.
- [x] #2 Schedules and Workflows empty active-run states expose taxonomy-aligned recovery copy and disabled tooltips without false Console-launch affordances.
- [x] #3 Artifacts empty Chatbook state exposes taxonomy-aligned recovery copy and disabled tooltip without implying retry can fix missing selection/data.
- [x] #4 Automated UI regressions verify the taxonomy fields or visible copy for representative blocked shell destination controls.
- [x] #5 Durable Phase 5 QA evidence records the running-app or focused widget walkthrough result and residual risks.
<!-- AC:END -->

## Implementation Plan

1. Inspect the current ACP, Schedules, Workflows, and Artifacts destination blocked-state rendering and existing tests.
2. Add failing UI regressions that assert taxonomy-aligned visible copy, selectors, and disabled tooltips for representative destination blockers.
3. Add the smallest shared helper or local state updates needed to make those blocked states expose the taxonomy fields consistently.
4. Add Phase 5 QA evidence and update Phase 5 tracking docs for `TASK-6.2`.
5. Run focused UI tests and documentation contract tests before opening the PR.

## Implementation Notes

Added a shared `DestinationRecoveryState` helper for destination-shell blocked states and applied it to ACP runtime configuration, Schedules empty Console follow, Workflows empty Console launch, and Artifacts empty Chatbook Console launch. Updated existing destination tests plus a new Phase 5 regression to assert visible taxonomy fields and disabled tooltips, and recorded QA evidence under the Phase 5 evidence directory.
