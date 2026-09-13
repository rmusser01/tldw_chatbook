---
id: TASK-32521
title: Design native reasoning prefill for Console
status: Done
assignee:
  - '@codex'
created_date: '2026-09-13 02:53'
updated_date: '2026-09-13 03:34'
labels: []
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-09-12-console-native-reasoning-prefill-design.md
  - backlog/decisions/159-console-native-reasoning-prefill.md
  - Docs/superpowers/plans/2026-09-12-console-native-reasoning-prefill.md
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
- [x] #5 The user approves the written spec before implementation planning begins.
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
Completed the native-only Console reasoning-prefill design and ADR-159. The user approved the written spec; ADR-159 is now Accepted. The design covers both lifetimes, enable/disable, exclusive queued-turn reservations, revision-safe retry/recovery, native qualification, tool replay, exact whitespace, and versioned pin portability. Updated the ADR index and linked the implementation plan.

Wrote Docs/superpowers/plans/2026-09-12-console-native-reasoning-prefill.md with eight atomic tasks, explicit interfaces, red/green examples, exact integration paths, targeted checks, and live qualification gates. Created TASK-32523 through TASK-32530 as To Do using Backlog CLI; each links the spec, plan, and ADR and depends only on earlier tasks. Plan review accounted for the message-only Sync handler with a scoped encrypted pin record and a content-free versioned thinking replay-eligibility field.

Verification: 16 embedded Python examples parse; 34 local links resolved before adding the two final cross-links; all eight task files pass the scoped Backlog guard and have earlier-only dependencies. The repository-wide Backlog guard has existing unrelated duplicate IDs and the CLI emits hydration warnings for malformed task YAML on another branch; these do not involve the new task records. Task/ADR IDs were scanned against all local refs and worktrees. No application implementation, full test suite, or live provider qualification was performed. Execution remains a separate step from this completed design and planning work.
<!-- SECTION:NOTES:END -->
