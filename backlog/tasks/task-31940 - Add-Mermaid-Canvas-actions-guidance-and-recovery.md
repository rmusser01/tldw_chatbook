---
id: TASK-31940
title: Add Mermaid Canvas actions guidance and recovery
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-06 22:14'
updated_date: '2026-09-07 07:02'
labels:
  - canvas
  - v2
dependencies:
  - TASK-31939
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
Expose the approved diagram experience with accurate assistant guidance and honest preview status.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Mermaid fences open through existing replay and ownership checks with exact escaped text and no script interpolation.
- [x] #2 Bounded profile-aware guidance and executable examples retain unchanged tool parameters and source-free logs and cards.
- [ ] #3 Saved, pending, ready, failed and unavailable states are distinguished per load; source, View previous and confirmed unsent repair remain usable in both modes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (existing ADR applies)
ADR path: backlog/decisions/124-canvas-mermaid-subset-and-immutable-runtime-profiles.md
Reason: Direct implementation of approved Mermaid authoring and recovery contracts, retaining ADR121 authority boundaries.
1. Follow Task7 of Docs/superpowers/plans/2026-09-06-chatbook-canvas-v2-mermaid-implementation.md with RED escaped-wrapper and stable fence identity tests.
2. Implement bounded exact-profile authoring guidance without changing tool parameters or persistent context.
3. Add load-fenced honest preview states, explicit source/previous and confirmed unsent repair in incumbent native/served shells.
4. Verify focused provider, action, card, native/served browser flows and static checks; update user guide, commit, and obtain independent task review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

Implemented bounded escaped Mermaid fence authoring with independent Mermaid identities and unchanged HTML replay semantics; exact captured-profile guidance and executable examples; current-context guide dedup with unchanged tool parameters; per-load honest preview status, bounded V2 diagnostics and explicit confirmed unsent repair. Existing native authority and model-context consumers were extended instead of adding parallel state. Candidate remains disabled/default null; V1 runtime bytes, storage, archive and authority boundaries remain unchanged (ADR124/ADR121).

Verification: final nonbrowser affected run 278 passed; broad affected browser/runtime/UI run 430 passed, 2 explicit asset-cache skips, 1 failed. The failure is intermittent actual Chatbook subprocess V1 update-preview absence; diagnostic reproduction also observed shell api/state 503 gateway_unavailable. Passing isolated reruns are not remediation. AC3 remains unchecked pending independent investigation/review; do not use this task to admit the candidate release. Native actual browser failure/stale status and served mounted production-authority confirmed repair tests passed, separately from actual subprocess normal lifecycle evidence.

One bounded impeccable hardening batch and desktop/narrow confirmation were inspected by the parent. Manual detector was degraded by missing optional parsers; no clean automated accessibility claim. Proven fixture defects fixed: atomic owner-receipt publication and stale fake-compiler signature, with deterministic receipt RED/GREEN. Full commands, evidence distinctions, screenshots, warnings, limitations and modified file list: `.superpowers/sdd/2026-09-06-chatbook-canvas-v2-mermaid-implementation/task-7-report.md`. Status intentionally remains In Progress for independent review.
