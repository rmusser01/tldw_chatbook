---
id: TASK-21511
title: Add Research Copy sharing and Deep Research return
status: To Do
assignee: []
created_date: '2026-08-24 05:54'
updated_date: '2026-08-24 05:54'
labels:
  - research
  - workspace
  - copy
  - sharing
dependencies:
  - TASK-21507
  - TASK-21508
  - TASK-21509
  - TASK-21510
references:
  - Docs/superpowers/specs/2026-08-23-research-workspace-design.md
  - Docs/superpowers/plans/2026-08-23-research-workspace-copy-sharing-deep-research.md
  - backlog/decisions/078-research-workspace-authority-and-screen-boundaries.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add explicit, receipted cross-authority Copy, real server sharing, and a durable launch/preview-return bridge between Research Workspace and separately owned Deep Research Runs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Cross-authority transfer starts only from explicit Copy, freezes one source/destination manifest, displays per-item treatment/conflicts/redaction/size/unsupported content, and requires confirmation; toggling authority never transfers data.
- [ ] #2 Copy uses stable transfer and item idempotency keys, persists per-stage receipts, resumes partial work without duplicating acknowledged items, and exposes Completed, Partially completed, Rolled back, retryable, and terminal outcomes.
- [ ] #3 Copy conflict choices are capability-gated and destructive replacement requires a second confirmation plus destination version checks; v1 provides neither Move, automatic merge, continuous sync, nor silent fallback.
- [ ] #4 Local mode exposes Export bundle, Copy to Server, and Copy to Server and Share; Server mode exposes supported workspace shares, permissions, clone policy, private-link limits, active shares, revoke, shared-with-me, verification, import, and clone through existing sharing services.
- [ ] #5 `Copy to Server and Share` cannot open Share until Copy completes, and all sharing/copy preflights explicitly exclude device-only overlays.
- [ ] #6 Deep Research launch stores the qualified origin workspace, selected source identities/versions, initiating output/message, normalized query, authority-specific chat identity, return route, and timestamp, while the run remains owned by Research Interop/server Research.
- [ ] #7 Runs can return a bundle only to its matching origin context; Workspace validates run/origin identity, previews the import, and creates a draft Report only after confirmation, with idempotent re-import and explicit new-version choice.
- [ ] #8 Restart reconstructs Copy, sharing, and Deep Research launch/return state from durable owners/receipts rather than UI guesses, and stale context results cannot repaint another workspace.
- [ ] #9 Targeted manifest/receipt migration, conflict, idempotent retry, partial recovery, sharing policy, launch identity, bundle mismatch, return preview, mounted navigation, and live server-contract tests pass without a full-suite claim.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR

ADR path: `backlog/decisions/078-research-workspace-authority-and-screen-boundaries.md`

Reason: ADR-078 already defines Copy-only transfer, receipt ownership, server-only sharing, overlay exclusion, and Deep Research ownership/return. The durable tables and UI implement that accepted boundary.

Follow `Docs/superpowers/plans/2026-08-23-research-workspace-copy-sharing-deep-research.md` task-by-task with test-first checkpoints and one scoped commit per completed plan task.
<!-- SECTION:PLAN:END -->
