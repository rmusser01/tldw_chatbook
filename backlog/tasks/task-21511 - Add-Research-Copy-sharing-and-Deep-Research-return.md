---
id: TASK-21511
title: Add Research Copy sharing and Deep Research return
status: In Progress
assignee:
- '@codex'
created_date: 2026-08-24 05:54
updated_date: 2026-11-24 20:47
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
- backlog/decisions/322-research-workspace-authority-and-screen-boundaries.md
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

ADR path: backlog/decisions/322-research-workspace-authority-and-screen-boundaries.md

Reason: ADR-322 defines Copy-only transfer, receipt ownership, server-only sharing, overlay exclusion, and Deep Research return. The client implements those existing boundaries.

Follow the approved Copy, Sharing, and Deep Research plan in order with targeted tests and a scoped commit for each completed step. TASK-21510 has a canonical Server Study save contract. Generic Copy still requires both owners to expose a safe transfer contract.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The workspace Actions menu now opens explicit Copy, Local bundle export, and Server sharing. Copy shows the frozen manifest, device-only exclusions, and durable receipts. A repeat Server-to-Local Copy finds its previous owner from the exact qualified receipt. Replacement requires the captured owner revision and a second confirmation. Server sharing supports access edits, guarded links, recipient verification and import, and clone through the sharing service. The compound action opens Share after Copy and its Server migration both complete. Local export uses the same transfer policy and an atomic private file write. A detached Sources pane discards late refresh results.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
