---
id: TASK-13215
title: >-
  Fleet approval revocation: add a revoked-run tombstone and close the residual
  arm/read windows
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-10 01:37'
updated_date: '2026-09-12 07:02'
labels: []
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Cancellation must fence delayed approval fallbacks even when no round existed at the revoke pass. The current interrupt host already owns one shared lock and exact round-keyed payloads; the remaining gaps are revoke-before-arm admission, mixed final MCP verdict snapshots, and missing empty-owner diagnostics. Preserve sibling payloads and remountability with mutation-sensitive regressions. Indefinite default waits make timeout an unsuitable fallback; a fully committed approval cannot be retroactively retracted by later cancellation.
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented host-lifetime per-kind revoked-run tombstones with atomic shared admission at early controller registration and host entry. Late MCP/local/script fallbacks fail closed before configuration, publication, badge, or payload retention; unowned legacy arms warn once without content and remain answerable. MCP verdicts commit as a whole under the sweep lock, with revoked unresolved provenance and audit callbacks outside. Exact-round payload cleanup is unchanged.

ADR required: no new ADR. Clarified backlog/decisions/067-indefinite-human-approval-waits.md: memory grows with distinct revoked runs for the host lifetime; eviction requires proof every physical invocation drained. Completed approvals and side effects are not retroactively retractable.

Regression evidence: unchanged production gave 12 intended failures and 4 passing controls; repaired admission/snapshot/sibling selection gave 20 passes. A destructive same-session unpark mutation fails all four approval/script revoke/teardown remount cases. The real local fs_write fallback creates no file after revoke. Local approval selection: 16 passed. Scoped same-session script harness now waits for actual badge and retained payload and joins workers through begin_shutdown; its old preregistration-only wait raced publication and leaked a waiter on failure.

Affected-file initial run: 201 passed, 9 failed. Exact original-source rerun of those nodes: 8 reproduced and the script readiness race passed once; human-wait passes alone (7), and corrected script plus human-wait passes together (12). Baseline UI failures and their exact node IDs are recorded in the task report; they remain outside this bounded fix. No full suite, dependency, guard/cap, provider authority, or cancellation-policy changes. Final combined host/wiring/script/human-wait/MCP run: 197 passed, 5 baseline mounted-UI cases deselected; parked-payload/approval targeted selection: 30 passed. Changed-line Ruff lint/format checks are clean; whole-file inherited debt is compared in task-2-static.log. Self-review preserved per-kind ownership, check_revoked=False, callback lock boundaries, sibling cleanup, and terminal snapshot semantics. Status remains In Progress for independent root review.
<!-- SECTION:NOTES:END -->
