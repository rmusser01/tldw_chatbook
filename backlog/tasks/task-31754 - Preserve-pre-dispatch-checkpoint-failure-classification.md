---
id: TASK-31754
title: Preserve pre-dispatch checkpoint failure classification
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 20:16'
updated_date: '2026-09-05 20:25'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Repair a checkpoint-transition failure being misreported as provider-started and routed through terminal assistant persistence before the gateway is called.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A failure before checkpoint dispatch transition retains accepted recovery with provider_started false and makes no provider or terminal assistant write.
- [x] #2 Explicit retry preserves the same durable owners and completes once after the checkpoint failure clears; actual post-dispatch failures retain their existing recovery semantics.
- [x] #3 Complete affected dispatch recovery files and scoped static checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Record the checkpoint CAS RuntimeError, attempted terminal write while gateway count is zero and checkpoint accepted, and cascading settlement exception.
2. Add RED immediate and deferred/sanitized callback-error regressions plus a normal post-dispatch provider-error control; require no premature terminal write and exact-owner retry.
3. Track local dispatch-callback failure, resetting before each callback attempt and after success. Re-raise only when this boundary failed, before generic provider-failure projection; preserve typed catches, cancellation, and finally cleanup.
4. Run full affected recovery/first-send files and static checks, then independent review before commit.
ADR required: no
ADR path: backlog/decisions/079-console-library-conversation-authority.md
Reason: Existing accepted-versus-dispatch-started contract. An identity-only exception guard was rejected after reading the production gateway worker, which sanitizes callback exceptions into distinct ChatProviderError objects; local callback outcome preserves that boundary without a new exception type or interface.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The direct provider loop now tracks whether its local dispatch callback failed and re-raises that error before generic provider failure/terminal projection. The marker resets before each callback attempt and after success. Existing typed exception handlers, cancellation handling, and finally cleanup are unchanged; resume uses its existing completed-checkpoint classification and retains the accepted owner. No new exception type, interface, or persistence policy. ADR required: no new ADR; preserves ADR-079.
Causal probe observed a checkpoint CAS RuntimeError, terminal mark_message_failed attempt with gateway0/checkpoint accepted, and cascading settlement exception reporting provider_started true. An exception-identity-only fix was rejected because the real generic gateway worker sanitizes callback failures into distinct ChatProviderError objects. The local outcome survives that transformation without treating normal provider errors as callback failures. Added an incident follow-up to lessons-testing-evidence.
RED: both immediate and deferred/sanitized callback tests failed premature terminal writes; the normal post-dispatch provider-failure control passed (/private/tmp/tldw-31754-red.xml). GREEN: complete round2, first-send, and trace-first-send files54 passed21.55s (/private/tmp/tldw-31754-boundary-files.xml); remaining seven complete recovery/round1 files108 passed81.42s (/private/tmp/tldw-31754-recovery-files.xml). Total162 passed. Runs emitted existing Requests warnings and aggregate FD-growth warnings234 and378 respectively; no broad cleanup or threshold change was made. New fixture shutdown/quiescence asserts zero exact-DB connections.
Whole test-file Ruff lint/format, changed complete controller-method format, and whitespace checks passed. Whole controller Ruff still reports the same27 baseline findings with identical signatures, not claimed clean. Parent independently reviewed the exact production and regression diff and approved scoped commit after qualified verification.
<!-- SECTION:NOTES:END -->
