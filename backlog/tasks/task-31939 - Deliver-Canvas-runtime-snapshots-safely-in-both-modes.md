---
id: TASK-31939
title: Deliver Canvas runtime snapshots safely in both modes
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-06 22:14'
updated_date: '2026-09-07 06:44'
labels:
  - canvas
  - v2
dependencies:
  - TASK-31938
documentation:
  - >-
    backlog/decisions/124-canvas-mermaid-subset-and-immutable-runtime-profiles.md
  - Docs/superpowers/specs/2026-09-06-chatbook-canvas-v2-mermaid-design.md
  - >-
    Docs/superpowers/plans/2026-09-06-chatbook-canvas-v2-mermaid-implementation.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep native and served browser delivery consistent with verified restart-bound runtime policy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Native host and served parent-child delivery use matching immutable build, catalog and policy snapshots and verified cached bytes.
- [x] #2 Restart into a revoked policy, mixed process identities and stale loads fail closed; explicit Canvas disable still stops live execution.
- [x] #3 Two-browser capability, selection freshness, source-only recovery and confirmed bridge isolation pass without adding a public port or weakening authentication.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing closed protocol-v2 snapshot identity and mismatch-handshake tests; share captured snapshot through actual parent/child/native ownership.
2. Deliver exact retained per-profile assets and private V2 runtime data through existing capability/load-scoped routes and shell; retain CSP, V1 bytes, revocation and source-only rules.
3. Verify real native and served browsers, two-session capability isolation, stale delivery, disable/restart and mutable-file refusal with targeted protocol/gateway/browser tests; document and self-review.
ADR required: yes
ADR path: backlog/decisions/124-canvas-mermaid-subset-and-immutable-runtime-profiles.md (existing, extends ADR-121)
Reason: Direct implementation of approved process-lifetime snapshot authentication and immutable asset delivery; no new listener or guest permissions.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented protocol-v2 snapshot authentication before child scope attachment, exact shared owner snapshots across native and real served app initialization, retained runtime/shell bytes, and capability-scoped V2 data separated from the closed plan. Refused served launches cannot fall back to native delivery; unavailable previews preserve artifact acceptance. Source-only recovery and delayed plan failures preserve selection/load fences. ADR-124/ADR-121 applied; no new ADR or production V2 admission. Evidence: 155 targeted protocol/gateway/native/kill-switch tests and 72 native/served Chromium tests pass, including actual separate TldwCli V1/V2 create/update/save/reopen and strict bootstrap/confinement checks. Follow-up refused-child coverage and detailed RED/GREEN receipts are in .superpowers/sdd/2026-09-06-chatbook-canvas-v2-mermaid-implementation/task-6-report.md. Restart policy testing uses new-owner simulation plus existing old-load/receipt invalidation, not a deployed package replacement. Existing screen-size ratchet remains failing at baseline; chat_screen is reduced by two lines, library untouched, ceilings unchanged. Existing app/Console/screen lint baselines retained; modified Canvas/serve/tests clean. README and testing lesson updated. Leave In Progress for independent review.
Review fix round 1: unconditional non-secret served-child identity now survives an absent broker after parent disable, so an enabled fresh child cannot create a native gateway. Independent load generations fence overlapping reloads of the same revision, including delayed source recovery. RED tests reproduced both failures; scoped owner and actual-child/browser GREEN evidence is appended to task-6-report.md. Actual parent/all-child process replacement qualification remains Task8 work, not satisfied by Task6 new-owner simulations. Status remains In Progress.
<!-- SECTION:NOTES:END -->
