---
id: TASK-32521
title: Design native reasoning prefill for Console
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-13 02:53'
updated_date: '2026-09-13 02:59'
labels: []
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-09-12-console-native-reasoning-prefill-design.md
  - backlog/decisions/159-console-native-reasoning-prefill.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define a reviewable Console design for native reasoning continuation across qualified providers, with next-send and pinned lifetimes, tool-round replay, precise recovery, and conversation-owned persistence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The design specifies native-only support and both prefill lifetimes, including disable and precedence behavior.
- [x] #2 The design resolves retry, queue, cancellation, tool replay, storage, and provider compatibility risks from review.
- [x] #3 A canonical ADR records provider and data-ownership decisions and is linked from the design and task.
- [x] #4 The written design passes self-review and is presented to the user before implementation planning.
- [ ] #5 The user approves the written spec before implementation planning begins.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/159-console-native-reasoning-prefill.md
Reason: Native provider continuation contracts, per-turn ownership, and conversation persistence.

1. Record the approved scope and review refinements in the design spec.
2. Record architectural decisions and alternatives in ADR-159 and update the ADR index.
3. Self-review lifecycle, provider qualification, persistence, and UI contracts; verify local links and document formatting.
4. Present the written spec for user review before implementation planning.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Wrote the native-only Console reasoning-prefill design and proposed ADR-159, and added the ADR index entry. The design includes both lifetimes, enable/disable, exclusive queued-turn reservations, revision-safe failure/retry handling, native qualification, tool replay, precise whitespace, and versioned pin portability. Used-seed retention follows existing protected continuation/capture rules; next-send configuration remains ephemeral.

Self-review checked lifecycle consistency, ownership, replay attribution, recovery, and scope. Three document checks passed, including 16 relative links and placeholder/conflict-marker/whitespace checks. The repository-wide Backlog ID guard reports existing duplicate tasks unrelated to TASK-32521; the new ID was swept against 210 refs and 41 worktrees and is absent from the guard failures. The CLI initially offered 32516, below the swept maximum 32520, so this newly created task was immediately renumbered to 32521 before linking it.

No runtime code, full test suite, or live provider calls were part of this documentation task. The written spec and ADR remain proposed pending user review; implementation planning has not started.
<!-- SECTION:NOTES:END -->
