---
id: TASK-34411
title: Retire replacements of borrowed connections after finite database callbacks
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-05 22:10'
updated_date: '2026-10-05 22:41'
labels: []
dependencies: []
references:
  - >-
    backlog/tasks/task-31993.5 -
    Retire-finite-background-database-work-on-its-worker-threads.md
  - backlog/decisions/126-complete-local-backup-and-recovery.md
  - 'https://github.com/rmusser01/tldw_chatbook/pull/3024#issuecomment-6004049427'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent a completed finite database callback from retaining a native connection acquired after its original borrowed handle was retired. Preserve unchanged caller-owned connections and their active transactions under the existing finite-operation ownership contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A finite local character metadata callback crossing exact-file quiescence retires its newly acquired replacement after physical callback completion, on cold and warm entry.
- [x] #2 The installed raw core database owners retire replacements acquired inside the finite operation, including on an operation error, while an unchanged borrower and its transaction remain alive.
- [ ] #3 Targeted regressions, resource observations, static checks, and scoped review pass without changing memory/custom routes, cancellation timing, native admission, or general connection lifetime policy.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a real SQLite regression that parks the installed character metadata callback inside the existing finite-operation guard, retires its exact-file borrowed handle, then releases and awaits physical completion. Capture warm RED and cold control. Cover replacement/error and unchanged-transaction controls for the three installed raw core owners.
2. Capture the exact original native connection in both supported guard branches; close only a non-original current handle or a cold operation-owned handle. Keep existing supported/excluded owner routes and caller-thread close behavior.
3. Run targeted finite-reader, borrowed-transaction, cancellation, core-owner, activation/profile and resource checks; retain any failed qualification receipts. Run changed-path lint/format and artifact guards. Include the regression in the existing admission-sensitive PR invocation, preserving its collection-time profile boundary without a new job, dependency or timeout.
4. Obtain scoped independent review and update task, QA receipt, and the incident lesson before publishing to PR3024. Do not claim this proves the uncaptured intermittent profile leak or waives native/Windows/participant/latency gates.

ADR required: no
ADR path: N/A (existing backlog/decisions/126-complete-local-backup-and-recovery.md applies)
Reason: routine correction of existing exact native finite-operation ownership; no new lifetime, admission, schema, service or platform policy.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Earlier implementation checkpoint (the completed affected batch is recorded below):

Implemented exact original native identity capture in both supported operation_owned_connection branches; cleanup retires a different current handle or a cold operation-owned cache, preserving unchanged borrowers and active transactions. Valid repository RED7 failures/7 controls in1.99s; GREEN136 targeted cases in64.73s, no pytest warnings, strict parent resource gate0 with136 zero-database observations. Raw native replacement controls include real SQL-error propagation. Existing owner-thread cancellation and memory/custom controls pass. Scoped independent review found no actionable findings. New test Ruff/format clean, edited helper format-range clean, six inherited whole-file Ruff diagnostics unchanged, existing formatter ratchet/whitespace/all eleven artifact guards pass. Nine-file affected batch including unchanged1000-send control still running; old intermittent profile cause and native/Windows/participant/latency qualification remain unproven. Existing ADR126 applies; no new ownership/admission/lifetime policy. QA receipt and incident lesson updated.

Complete combined affected verification: 363 passed in 574.06s, no pytest warnings, strict process exit 0; all 363 parent post-teardown observations contain zero test database files. The original 1,000-send retention control passed in 387.96s with count/assertions/timeout unchanged. The failure-conditioned profile observer reports null, so the old intermittent failure is still not causally attributed. Receipt /tmp/pr3024-replacement-affected.log SHA256 56d8b421bd063d11f4d9e8b72df233ae1992937f6abcdf0c87d706f17072eb9e. Fresh PR-head CI and actual external review remain required before integration; keep task In Progress and broad qualification tasks unchanged. No full repository sweep or native UI workaround ran.

CI coverage: added the new bootstrap-profile regression to the existing admission-sensitive PR invocation only, with no new job/dependency/timeout. Combined local bodies: 280 passed, 2 existing xfails, 6 warnings in 379.97s; strict parent descriptor observer returns exit 1. First retained databases occur at the existing manually mounted startup-app test, after all 14 new native-retirement controls record no database files; session descriptor growth 598. No explicit basetemp was provided, so this receipt covers named private profiles only. Older pytest garbage cleanup also warned; no unrelated cleanup, warning suppression, threshold change or application-lifetime repair made. Failed receipt cd6dd40fd8e6d421edee7b7223b5128ba1b47c5e5e3e393132aec6ed99f6fa9e retained at /tmp/pr3024-replacement-ci-cohort.log. All eleven artifact guards and incremental independent review pass; keep AC3 unchecked and resource/integration qualification HOLD.
<!-- SECTION:NOTES:END -->
