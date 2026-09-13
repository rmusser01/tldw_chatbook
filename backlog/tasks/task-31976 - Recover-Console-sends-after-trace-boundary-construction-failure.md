---
id: TASK-31976
title: Recover Console sends after trace boundary construction failure
status: Done
assignee:
  - '@codex'
created_date: '2026-09-07 20:53'
updated_date: '2026-09-07 21:08'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A fresh Console send can stop at Trace capture blocked, and Send without capture then hides the card without contacting the provider or displaying its refusal. Restore the explicit recovery path while retaining safeguards against uncertain delivery.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Send without capture dispatches exactly once after a proven pre-adapter trace construction failure and completes the original accepted turn.
- [x] #2 Unknown or already-started dispatch remains protected against automatic replay.
- [x] #3 A refused recovery remains visibly actionable and explains why it did not send; the UI does not silently discard the result.
- [x] #4 Targeted real-database and mounted-Console regressions cover recovery and timer settlement.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing real-database and mounted-Console regressions for a first-call factory failure followed by Send without capture, plus visible refusal and uncertain-delivery controls.
2. Preserve exact live ownership and proof of no provider entry across trace construction failure; admit explicit Capture Off only with that proof. Keep uncertain dispatch guarded.
3. Preserve and show recovery refusal results instead of hiding the trace card on a state transition alone.
4. Run focused trace/controller/UI tests, lint changed code, self-review the diff, and update task notes and relevant lessons.
ADR required: no (new ADR).
ADR path: backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md
Reason: Repair the explicit one-shot Capture Off recovery already required by ADR-097 section 15; no schema, ownership, or dispatch-safety policy change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed the fresh-send recovery dead end under existing ADR-097 section 15. The gateway retains exact live first-call construction-failure proof separately from durable reservation status and consumes it once for explicit Capture Off. Another owner/gateway, a forged failure, Capture On, a later call, rebinding, and uncertain dispatch cannot reuse it. The controller publishes uncertain-delivery handoff state so Retry anyway and Discard render. The trace callout preserves safe refusal copy across transcript refreshes.
Changes: console_provider_gateway.py, console_chat_controller.py, provider_continuation_recovery.py; real-database proof tests, mounted 80x24 send/handoff tests, and persistent-refusal render coverage. Added the observed test-coverage trap to lessons-testing-evidence.md.
Verification: original mounted tests failed with zero dispatch and a hidden refusal; they now pass. The new persistent-refusal test also fails against the untouched dev UI. Focused suites: 75 trace/runtime/UI tests, 73 gateway trace/capture tests, 47 controller recovery and continuation UI tests (195 total) passed. Independent code review found no actionable issues. Changed ranges pass Ruff formatting; new-file lint and baseline-relative lint pass with zero introduced diagnostics (237 pre-existing findings in the touched legacy files). git diff --check passes. Full suite not run per repository policy. Existing requests dependency warning remains.
ADR required: no new ADR; direct repair of backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md. No schema changes. The original reporter-specific trace failure and initial flickering remain unconfirmed; this change fixes the reproduced recovery failure, not an unobserved initial cause.
<!-- SECTION:NOTES:END -->
