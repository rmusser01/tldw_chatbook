---
id: TASK-31799
title: Reconcile reviewed fork transition route inventories
status: Done
assignee:
  - '@codex'
created_date: '2026-09-06'
updated_date: '2026-09-06 01:39'
labels:
  - tests
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The fork census still expects the pre-dev endpoint and controller topology.
Three public endpoint/settings adoption and rollback methods already enter the
canonical transition, while the controller now has three legitimate transient
rollback branches: shutdown during readiness, thinking-persistence preflight,
and queued Capture On without a durable conversation. Keep these inventories
exact without exempting the other unreviewed mutation routes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The direct-route inventory includes only the three reviewed existing guarded owners plus the repaired combined-settings owner from TASK-31798.
- [x] #2 Controller rollback inventory rejects missing, extra and wrong-receiver calls while accepting all three current exact source-owner calls.
- [x] #3 Existing owner-aware guard checks and unclassified-mutation failures remain enforced; no runtime route is changed by this task.
- [x] #4 Targeted verification, mutation checks, scoped static analysis and independent review support the reconciliation.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the 24-pass/3-failure census baseline. Read the complete three store methods and controller rollback branches. Verify each store method with the existing owner-aware `_transitioned` scanner.
2. In `Tests/Chat/test_console_fork_transition_census.py`, add `adopt_session_ephemeral_endpoint`, `rollback_session_ephemeral_endpoint_adoption` and `rollback_session_settings_replacement` to the direct inventory. Replace the rollback count-only assertion with exactly three `self.store.rollback_transient_send(session.id, echoed_user.id, title=pre_send_title, persisted_conversation_id=pre_send_conversation_id)` shapes.
3. Rerun the complete census. Expect the two reviewed inventory checks green and the independent mutation-classification check still red. Exercise process-local missing-call, extra-call and same-count wrong-receiver mutants against the exact controller guard; all must fail.
4. Run the complete endpoint/first-send and provider-apply files, scoped Ruff/format, review the diff and record the precise remaining routes. Save to the existing draft PR without claiming complete fork qualification.

ADR required: no
ADR path: backlog/decisions/092-console-chat-fork-copy-and-authority-boundary.md
Reason: Test-only reconciliation of existing guarded routes; no new ownership or copy policy.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reconciled only the three reviewed existing guarded endpoint/settings rollback methods and the exact three controller optimistic-echo rollback calls. The controller guard now checks receiver, session/message arguments and prior title/conversation keywords, not only count. Missing-call, extra-call, wrong-receiver and wrong-owner process-local mutations all fail; the unmodified controller passes. Complete census, trace-first-send and provider-apply UI files: 70 passed and one independent mutation-classification failure in 114.90s, three existing warnings; /private/tmp/tldw-fork-route-reconciliation.xml. Census improves from 24 passed/3 failed to 26 passed/1 failed; seven unclassified routes remain explicitly open in the checkpoint report. No scanner logic, safe exemption or runtime route changed by this task. Scoped Ruff/format and diff checks pass; independent review clear. ADR check: existing ADR-092, no new ADR required.
<!-- SECTION:NOTES:END -->
