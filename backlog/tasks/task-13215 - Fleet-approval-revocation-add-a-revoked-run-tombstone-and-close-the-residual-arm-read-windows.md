---
id: TASK-13215
title: >-
  Fleet approval revocation: add a revoked-run tombstone and close the residual
  arm/read windows
status: To Do
assignee: []
created_date: '2026-08-10 01:37'
updated_date: '2026-08-10 02:15'
labels: []
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from supervisor-fleet PR 2a Task 7 review. Revocation sweeps rounds that are already armed, which leaves two narrow fail-open windows: (a) revoke-then-arm — an in-flight provider invoke() that reaches its single-call approval fallback AFTER the last revoke pass arms a card nobody will ever revoke (bounded only by the 120s approval timeout); (b) the worker can read was_revoked==False and have revocation land before it returns, and on the MCPToolProvider.invoke/LocalToolProvider fallback paths there is no later cancellation checkpoint. A set of revoked run ids consulted at ARM time (return all-deny immediately when the owner is already revoked) closes (a) outright and narrows (b). Also from the same review: the sibling retained-payload rule is correct but untested — replacing its guard with an unconditional _parked_approval_payloads.pop leaves all 235 tests green, and regressing it reproduces TASK-1050 Defect B (a live sibling child's card unrecoverable on switch-away/back, badge lit until timeout). And a round armed with an empty run-id owner (lost ContextVar binding) is silently unrevocable — worth a warning log.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A revoked-run registry is consulted at arm time so a card armed after revocation resolves all-deny immediately
- [ ] #2 The sibling retained-payload rule has a regression test (unconditional pop must fail it)
- [ ] #3 Arming a round with an empty run-id owner logs a warning
- [ ] #4 _revoke_skill_script_rounds' cross-lock window is closed: it snapshots still_armed under _pending_skill_script_lock, releases it, then pops _parked_skill_script_payloads under _approval_state_lock — a sibling confirm armed in that window has its payload popped (TASK-1050 Defect B: card unrecoverable on switch-away/back, badge lit until timeout). Fix with an identity-guarded pop (only when the stored payload's request_id is among the revoked ids), matching _clear_pending_skill_script_if_round_is_current. Note the same window pre-exists in request_skill_script_confirm's own teardown.
<!-- AC:END -->
