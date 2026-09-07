---
id: TASK-31938
title: Preserve Canvas V2 profiles across revision lifecycles
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-06 22:13'
updated_date: '2026-09-07 03:34'
labels:
  - canvas
  - v2
dependencies:
  - TASK-31937
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
Keep exact runtime semantics across updates, branches, temporary promotion and archive portability.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All production creation, update, rename, historical and replay paths use one profile resolver and preserve optimistic parent checks.
- [ ] #2 Temporary and durable histories retain exact profiles through atomic commit, rollback, cancellation and promotion without new storage or sync.
- [ ] #3 Real conversation and Chatbook export-import preserve source and graphs; unknown and revoked profiles remain inert and archives cannot install runtime bytes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing real-controller profile transition tests and integrate one captured snapshot and prepared-plan identity admission.
2. Preserve exact profiles through durable/staged creation, update, rename, branch/replay and post-compilation ownership fencing.
3. Verify temporary atomic promotion/close and real conversation/Chatbook export-import with inert unavailable profiles and rollback; self-review and targeted static checks.
ADR required: yes
ADR path: backlog/decisions/124-canvas-mermaid-subset-and-immutable-runtime-profiles.md (existing, extends ADR-121)
Reason: Direct implementation of approved profile ownership and archive contracts; schema68/archive3.0 and sync exclusion remain unchanged.
<!-- SECTION:PLAN:END -->
