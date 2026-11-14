---
id: TASK-34884
title: Design combined Persona and Tool Profile pack portability
status: To Do
assignee: []
created_date: '2026-09-27 15:11'
labels:
  - packs
  - personas
  - design
dependencies: []
references:
  - >-
    backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md
  - backlog/decisions/107-portable-tool-use-packs.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define how a user can transfer a Persona, its expressions and visuals, and an associated Tool policy profile as one reviewed package, preserving the existing actor and permission authorities.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An approved specification defines portable associations, identity mapping, and handling of missing or changed tools without transferring credentials or local identifiers.
- [ ] #2 The import design preserves separate actor activation, policy installation, and explicit binding review, with defined cancellation and partial-failure recovery.
- [ ] #3 A canonical ADR records the format and cross-service authority decisions, with a verification matrix and atomic implementation task breakdown.
<!-- AC:END -->

## ID allocation

Created through the Backlog CLI, then moved from its uncommitted offer of
TASK-32953 to TASK-33010. Refreshed remote refs, reachable object paths, and local
worktree task paths showed a maximum of TASK-33009 before filing.

## Design boundary

ADR required: yes, before implementation.
Existing ADRs: ADR-074 and ADR-107. The new decision must define composition and
recovery across actor and permission authorities without transferring grants.
