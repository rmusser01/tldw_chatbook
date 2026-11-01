---
id: TASK-36760
title: Keep served control and projection draining independent of phone clients
status: Done
created_date: 2026-10-25 17:39
dependencies:
- TASK-36729
labels:
- console
- web-server
- mobile
- adr-244
priority: high
documentation:
- Docs/superpowers/plans/2026-10-25-shared-console-mobile-supervision-implementation.md
- Docs/superpowers/specs/2026-10-25-shared-console-mobile-supervision-design.md
- backlog/decisions/244-shared-console-ownership-and-mobile-supervision.md
assignee:
- '@codex'
updated_date: 2026-10-25 20:17
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the authenticated process control channel available independently of Canvas and mobile visits, and prevent projection congestion from becoming execution-owner loss.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every admitted child has authenticated runtime control and lifecycle-owned event draining before publication, even with Canvas disabled, the mobile surface disabled or no phone subscriber.
- [x] #2 Projection overflow uses bounded coalescing or reset state without closing the host control channel, blocking agent execution, dropping command outcomes or resolving native decisions.
- [x] #3 Snapshots and events expose a detectable sequence gap and recover current state; malformed protocols, revoked child credentials and genuine control loss remain distinct failures.
- [x] #4 Existing Canvas authentication, feature revocation and browser capability isolation remain enforced under the revised control protocol.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/244-shared-console-ownership-and-mobile-supervision.md
Reason: Direct implementation of independent authenticated Console control and bounded projection recovery.

1. Read Task 2 requirements and current control/server/publisher wiring; add RED tests for bounded sticky-reset events and overflow before mobile use.
2. Add strict host/target identity models and bounded Console event buffer with one wake signal; bump both ends of the private protocol to version 3.
3. Keep correlated command responses separate from replaceable projection traffic; preserve authenticated connection on projection overflow and require snapshot reset.
4. Move draining to child registration before publication, with zero phone clients/mobile disabled; keep Console control independent of Canvas revocation.
5. Run targeted broker/publisher/event/auth/mobile checks, scoped static analysis and docs updates; commit explicit scope and submit independent review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented ADR-244 Task 2: authenticated private control version 3 now starts independently of Canvas/mobile exposure; child registration installs bounded lifecycle-owned projection history before publication. Projection overflow retains sticky reset and wakes observers without disconnecting control or touching native outcomes/pending response correlation. Current snapshot acknowledgement retires history while retaining a gap floor for stale readers. Added frozen opaque host/target identities, closed bounded Console method envelopes (unsupported native handlers refuse), bounded/coalesced child publication and browser queues, generation-fenced legacy relay observation, and a minimal existing phone status-line reset/unavailable state. Canvas revocation keeps the restart latch and revokes gateway capabilities while retaining runtime control. Required/connected transport/auth/mobile API/page selection: 195 passed with one inherited aiohttp deprecation warning; additional existing relay selection: 29 passed. Exact inherited auth/Canvas config-selection failures reproduced at Task2 BASE before using documented per-node/module bootstrap_profile fixtures; security assertions unchanged and stores stay temporary. Default scoped Ruff compared to exact BASE has no new per-file diagnostics, new files pass format/lint, and existing modules use format_ratchet. Task remains In Progress and ACs remain unchecked for independent review. Final commands/evidence and any remaining qualification limits are in .superpowers/sdd/2026-10-25-shared-console-mobile-supervision-implementation/task-2-report.md. Existing ADR-244 applies; no new ADR, schema, provider, migration, dependency manifest, push or merge.
Final qualification: the seven-file transport/auth/mobile API/page selection passes 195 tests (one inherited aiohttp warning), existing targeted mobile/control selection passes 29 tests, and connected Canvas gateway/spawn/isolation selection passes 87 tests. Three real served-child browser cases (snapshot mismatch, old version, absent broker) reach the same unrelated ChatScreen resize failure at UI/Screens/chat_screen.py:21567 (NoMatches '#console-left-rail') on both Task2 BASE and current source with the same installed Chromium and private-profile fixture; full logs and zero-leftover-child process evidence are retained in the Task 2 report/evidence. These are qualification blockers for later retained-terminal work, not repaired or hidden here. Final scoped static checks cover 15 Python files: AST passes, full default Ruff has 160 inherited diagnostics and zero new per-file diagnostic multiplicities, new modules pass Ruff/format, existing-module format_ratchet passes, node --check and git diff --check pass. No new ADR is required (direct implementation of ADR-244). Status/ACs remain pending independent review.
Task2 review fix1: verified the Important late-subscription reset visibility gap. New subscriptions now seed one bounded reset frame from outstanding current-child projection loss, including when the observer already processed overflow and the child is idle. Refresh/new-page cannot erase that unavailable state; existing status text says current Console approvals cannot be verified. Subscriber-only queue overflow still recovers via a new subscription when native/parent projections are current. No native recovery handler or second decision owner added. RED: late-subscription timeout and incorrect refresh recovery instruction; GREEN amended module16 passed, including actual Chromium new-page/reset display and subscriber-only resubscription. Connected legacy relay selection12 passed/25 deselected. Scoped default Ruff/AST on2 changed Python files:10 inherited diagnostics, zero new versus reviewed HEAD7a25a1085e; format ratchet, node --check and diff check pass. Full commands, output and fix self-review appended to task-2-report.md. Task remains In Progress/AC unchecked pending scoped re-review.
Independent task review found a late-subscription reset visibility gap. Commit1c6e05588c seeds outstanding child loss for new subscriptions and keeps recovery copy truthful; scoped re-review cleared the finding with no new breakage. Main selection195 passed; focused mobile29 passed; connected Canvas87 passed plus3 exact BASE-reproduced ChatScreen resize failures. Fix selection16 reset/browser+12 relay tests passed. Default Ruff has zero introduced per-file findings and committed format ratchets pass. Existing aiohttp warning and scoped inherited lint debt remain documented. Full-App resize qualification is carried into Tasks3/11; no full sweep/providers/user-data migration. ADR244 applies.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
