---
id: TASK-31939
title: Deliver Canvas runtime snapshots safely in both modes
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-06 22:14'
updated_date: '2026-09-07 06:10'
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
- [ ] #1 Native host and served parent-child delivery use matching immutable build, catalog and policy snapshots and verified cached bytes.
- [ ] #2 Restart into a revoked policy, mixed process identities and stale loads fail closed; explicit Canvas disable still stops live execution.
- [ ] #3 Two-browser capability, selection freshness, source-only recovery and confirmed bridge isolation pass without adding a public port or weakening authentication.
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
