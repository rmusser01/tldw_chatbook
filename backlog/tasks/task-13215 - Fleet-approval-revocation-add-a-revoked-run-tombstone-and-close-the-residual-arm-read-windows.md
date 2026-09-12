---
id: TASK-13215
title: >-
  Fleet approval revocation: add a revoked-run tombstone and close the residual
  arm/read windows
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-10 01:37'
updated_date: '2026-09-12 06:47'
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
- [ ] #4 Revoking or tearing down one skill-script round preserves the exact retained payload and remountability of a live sibling round.
- [ ] #5 The final tool-approval decision snapshot is atomic with revocation, so revocation before snapshot completion cannot return a partially approved batch.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR; clarify the existing cancellation contract
ADR path: backlog/decisions/067-indefinite-human-approval-waits.md
Reason: close late-arm and mixed-snapshot gaps in existing per-run cancellation authority; preserve primary-only unswept kinds and sibling ownership.
1. Use current InterruptRoundHost probe evidence: late arming after revoke can approve and execute fs_write; no empty-owner warning; payload guards lack revoke-specific mutation coverage. Old cross-lock premise is stale after extraction.
2. Retain revocation tombstones per run and swept kind for the host lifetime, under the same lock as registration. No TTL, capacity eviction, or session-close clearing can reactivate an abandoned daemon thread.
3. Fail revoked arms before publishing cards, discard preregistered state by identity, return unresolved deny, and warn without payload content when revocable rounds have no run owner.
4. Make MCP final decision snapshots atomic with revocation under the approval lock, keeping UI and audit side effects outside that lock. Preserve exact round-keyed sibling payload cleanup.
5. Add deterministic regressions for post-revoke MCP/local/skill prompts, sibling payload remountability, empty-owner warning, and snapshot/revoke ordering. Run targeted approval/host/provider tests and changed-line static checks; independent review before Done.
<!-- SECTION:PLAN:END -->
