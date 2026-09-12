---
id: TASK-32493
title: Verify agent orchestration invariants in the PR fast lane
status: Done
assignee:
  - '@codex'
created_date: '2026-09-12 03:00'
updated_date: '2026-09-12 03:55'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The orchestration rebase needs repeatable provider-free CI evidence for bounded communication, execution ownership, durable upgrades, and process-isolated task-store callbacks. The local host cannot allocate the semaphores used by six callback checks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The existing PR fast lane exercises bounded messaging and capacity, durable budget and runtime-owner migrations, and all session task-store callbacks.
- [x] #2 The required gate and its dependency boundary remain unchanged; no external provider or full-suite run is added.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR. ADR path: backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md. Reason: extend the explicit bounded target list within the existing core dependency and required-gate contract. 1. Add non-overlapping focused modules to the existing serial fast lane. 2. Validate workflow contract and local feasible cases. 3. Require clean-runner results for semaphore-dependent cases before merge.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added eight bounded provider-free modules to the existing serial fast lane without changing dependencies, event cadence or the required gate (ADR-103). Exact workflow contract: 26 passed. Local fast lane: 1097 passed; twelve cases fail in host SemLock allocation before application behavior and remain enabled. The wake-attempt module also pins every canonical terminal parent status after final review. Clean-runner CI verification is still pending; task remains In Progress.

Clean-runner verification passed on published head f66a87d12f882aec095e457ebe2e4435f553b71c: PR Fast Lane ran all 1125 tests successfully in 574.06 seconds, including every semaphore-dependent case. The required derived-artifact job also passed. Evidence: https://github.com/rmusser01/tldw_chatbook/actions/runs/34671103740 . The exact eight-module target contract passes locally; dependency/event/gate structure is unchanged. ADR-103 applies. Final PR updates will rerun the same required checks before merge.
<!-- SECTION:NOTES:END -->
