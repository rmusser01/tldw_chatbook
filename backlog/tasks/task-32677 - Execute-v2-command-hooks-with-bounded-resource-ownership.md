---
id: TASK-32677
title: Execute v2 command hooks with bounded resource ownership
status: Done
assignee:
  - '@codex'
created_date: '2026-09-16 04:23'
updated_date: '2026-09-30 23:47'
labels:
  - plugins
  - implementation
  - hooks
dependencies:
  - TASK-32676
documentation:
  - Docs/superpowers/plans/2026-09-15-expanded-hooks.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Run reviewed hook commands with fair application-wide limits and cancellation that retains ownership until actual cleanup.

Design: Docs/superpowers/specs/2026-09-15-managed-plugins-design.md; Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md. ADR required: yes. ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md. This task implements the accepted contracts without creating a separate permission or runtime owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both synchronous agent-thread and asynchronous Console entries execute argv-only handlers with bounded input/output capture and no implicit shell.
- [x] #2 The exact per-runtime and application execution, reservation and observation limits include provisional, nested, late and cleanup-pending work with fair admission.
- [x] #3 Cancellation closes admission immediately, keeps local children counted until reaped, and separates notification deadlines from the five-second host-reap allowance.
- [x] #4 Real controlled child processes establish successful execution and kill/reap controls; timeout, failed launch, cancelled waiters and surviving-child paths preserve honest outcomes and metadata-only diagnostics.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/163-expanded-console-hook-runtime.md; backlog/decisions/148-console-run-hooks.md; backlog/decisions/197-console-hook-configuration-review.md
Reason: Integrate the reviewed H2 application-owned execution and shutdown contract; retain current legacy consent/recovery/runtime owners. No new architecture or permission owner.
1. Read TASK-32677 and H2 plan/spec, the reviewed implementation at 17ee201cec and current runtime lifecycle callers. Keep original cdeb1687b6 branch intact. Validate exact source/child profile provenance and targeted current shutdown/viewless baseline.
2. Reuse reviewed H2 tests and committed process-isolation bootstrap. Establish a runtime-entry RED for absent immutable v2 session creation, with valid H1 normalization and legacy engine controls before production edits.
3. Reuse hooks_v2 ownership, budgets, command_executor and engine at reviewed H2 checkpoint 17ee201cec. Keep plugin discovery and later event producers out; preserve argv-only bounded capture, current authority, lifetime tickets, fairness, cancellation, shutdown and R47 Windows refusal.
4. Adapt only the current Console runtime lifecycle to shared app-loop budgets, immutable session engines and retained/shielded cleanup. Preserve exact legacy consent epochs, controller ticket validation, voice/worktree/recovery owners and current close fences. Do not reapply already-landed controller fixes or regress newer contracts.
5. Qualify actual controlled commands, mixed runtime budget pressure, publication races, cancellation, incomplete cleanup, agent-thread synchronous facade and own-loop refusal. Resolve any verified bounded-accounting issue at its existing shared owner with a regression test. Update only stale tuple-API test assertions if observed.
6. Carry the reviewed H2 spec/ADR clarifications, then run exact H2 execution/budget/shutdown files and relevant legacy/viewless/consent ownership tests. Targeted runs only; authored syntax/lint/format and shared-file baseline checks, source provenance and whitespace. Record platform/resource limits accurately.
7. Self-review all four ACs, record actual current evidence and any prerequisite harness repairs, close TASK-32677 via CLI and commit independently before proceeding to H3.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Integrated reviewed H2 checkpoint 17ee201cec on current dev. One application loop owns fair bounded argv-only command execution, immutable session engines and retained process cleanup. Current exact-definition legacy consent, RLock, controller close tickets, worktree recovery and voice custody remain intact. Added the reviewed private child bootstrap rather than repointing the shared editable environment.

Confirmed and fixed empty budget records retained for rejected runtime IDs at application capacity. The shared reserve boundary now stores counters only after successful admission; execution and observation pressure regressions failed before the fix and pass afterward.

Verification: current shutdown/viewless baseline 48 passed; missing-runtime-entry RED 1 failed/1 passed with normalization control. Initial integrated H2 execution/budget/shutdown 85 passed. Resource-inventory RED 2 failed/4 passed. Final scoped execution, budgets, shutdown, viewless, legacy hooks, consent and metadata: 262 passed in 55.13s, no skips. Seven new Python files pass full Ruff; eight authored/test files pass formatting. Runtime/shutdown keep exactly their prior 30/2 Ruff diagnostics; changed/new Python parses and git diff --check passes. Evidence and exact command: Docs/superpowers/reviews/2026-09-30-expanded-hooks-integration.md.

ADR required: no new ADR. Existing ADR-163, ADR-148 and ADR-197 govern this integration. H2 spec and ADR163 owner interfaces are documented. Self-review covers all four ACs. Darwin real subprocess launch, timeout, kill/reap, retained custody and cancellation were qualified; Windows v2 command execution remains explicitly refused, Linux not qualified. No full suite, live provider, new permission owner, plugin composition or lifecycle producer activation is claimed; those remain their dependency tasks.
<!-- SECTION:NOTES:END -->
