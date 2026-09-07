---
id: TASK-31941
title: Qualify and admit Canvas V2 Mermaid
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-06 22:15'
updated_date: '2026-09-07 08:10'
labels:
  - canvas
  - v2
dependencies:
  - TASK-31940
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
Enable the first immutable diagram profile only after end-to-end security and usability qualification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Mandatory real Chromium zero-egress and containment gates include diagram attacks, combined budgets, useful mixed documents and positive controls.
- [ ] #2 Native and same-origin served workflows, two-browser isolation, archive recovery, restart revocation, packaging and reproducibility have fresh recorded evidence.
- [ ] #3 Only a fully qualified immutable profile is admitted; failures leave V2 unavailable without changing V1 limits, and user and operator documentation report actual scope and coverage.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (existing ADR applies)
ADR path: backlog/decisions/124-canvas-mermaid-subset-and-immutable-runtime-profiles.md
Reason: Qualification and admission of the exact approved immutable profile under ADR124/ADR121, no new authority.
1. Follow Task8 of Docs/superpowers/plans/2026-09-06-chatbook-canvas-v2-mermaid-implementation.md and plan-scoped task8 integration notes.
2. Add missing release gates for adversarial/quota behavior, real process restart and recovery, exact examples, packaging/reproducibility and actual CI browser collection.
3. Run targeted Canvas native/served/browser/lifecycle/archive/static checks, inspect required visual fixtures, record exact scope and limits in Docs/Canvas/V2_VERIFICATION.md.
4. Only after required gates pass, observe production-admission RED then freeze/admit exact candidate and rerun final targeted checks; otherwise keep candidate disabled and report failing design gate.
5. Update docs and backlog evidence, commit and obtain independent task review; whole-branch review follows.
<!-- SECTION:PLAN:END -->
