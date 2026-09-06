---
id: TASK-31825
title: Correct four post-merge Canvas V1 review defects
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-06 13:36'
updated_date: '2026-09-06 14:02'
labels:
  - canvas
  - review
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Correct four independently reproduced defects in the merged Canvas V1 implementation before adding Mermaid support. Restore the accepted execution-profile, historical-selection, archive identity, and virtual form-state contracts without broadening runtime authority.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Unknown or unsupported imported runtime profiles remain source-inspectable and exportable but cannot compile or execute in native or served Canvas.
- [ ] #2 The next assistant Canvas read or update honors the live session historical selection and preserves branch reachability and stale-authority fences.
- [ ] #3 Same-identity Canvas archive restore rejects any divergent canonical conversation or message graph atomically while exact restores remain idempotent.
- [ ] #4 Untouched textarea and select defaults have matching rendered and virtual values in real Chromium; edits and reconstruction preserve supported form behavior.
- [ ] #5 All four findings have permanent regressions with observed RED then GREEN, focused preservation tests and independent scoped review, without weakened sandbox or performance limits.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR
ADR path: backlog/decisions/121-local-versioned-canvas-artifacts-and-browser-sandbox.md
Reason: direct corrections to accepted behavior; no new authority, schema, or runtime capability.

### Task 1: Correct the four verified V1 findings

Scope: all acceptance criteria of TASK-31825. Base: 017cf826c, repair branch codex/canvas-v1-review-fixes in .worktrees/canvas-v1. Spec: Docs/superpowers/specs/2026-09-03-chatbook-canvas-design.md (runtime profiles, sections 8.2, 9 and 14); existing ADR-121 applies.

Global Constraints: strict zero egress; no native global, parent DOM, network, filesystem, cookies or new runtime powers. No profile guessing/conversion. Preserve source inspection/export, exact owner/incarnation/branch/expected-parent fences, atomic archives, existing quotas and scheduling boundaries. No V2, unrelated refactoring, schema or dependency changes. Preserve all existing worktrees, refs and ignored evidence. Root alone executes Git, tests, interpreters, browser and formatters. Workers use static reads and apply_patch only, no subagents. Targeted tests only, no full suite or user database.

1. Add permanent regression tests first, report exact selectors to root, and wait for root-observed RED before implementation. Repeat per finding if practical. Use existing real SQLite and real Chromium harnesses; mocks only at external agent seams.
2. P1: native_authority.resolve_render_plan and Web_Server/serve.py compile source while discarding the stored runtime_profile. Carry and validate profile at native and private served read boundaries before compilation; imported canvas-v99 must stay inert and readable/exportable. Inspect gateway read representation and preserve stale-plan fences.
3. P2: console_chat_controller registers CanvasScope with selected IDs None. Hand the authoritative live historical selection to the next assistant run, preserving reachability and incarnation fences. Reproduce three real submit_draft turns (create r1, update r2, pin r1 through NativeConsoleCanvasAuthority, then actual provider read/update expected r1); expect first HTML and a staged branch, not second HTML/conflict.
4. P2: chatbook_importer same-identity preflight compares only Canvas rows and then skips the whole conversation. Compare canonical conversation and message graph plus Canvas under the existing transaction; reject any divergence atomically while exact restore is idempotent. Seed/export the real Canvas graph fixture, update an originating message with db.update_message and preserve_descendants=True, then restore: it must fail without mutation. Preserve import-as-new behavior.
5. P2: canvas_runtime_worker.js installNode initializes values from attributes only. Untouched textarea text and select selected-option defaults render correctly but canvas.submit reports empty values. Correct the supported virtual form-state initialization and maintenance; real Chromium must submit hello/b for textarea hello and second selected option b, preserve edits/reconstruction, and pass zero-egress observations. Use virtual DOM only and retain bounded operations.
6. Root runs focused RED/GREEN plus preservation tests, package integrity regeneration/checks as required, lint and formatting checks limited to touched code. Worker self-reviews and records evidence. Independent scoped review verifies all four corrections and new breakage; root performs final task hygiene and reports any remaining limitations.

Report concerns before expanding architectural authority. Do not commit/push/merge from a worker.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the four post-merge Canvas V1 corrections under ADR-121: fail-closed stored-profile compilation in native/served paths; authoritative historical selection at assistant dispatch with exact temporary-promotion proof; canonical conversation/message/Canvas equality for atomic same-identity restore; and synchronized virtual textarea/select state. Added permanent RED-to-GREEN regressions and form-subset documentation. Focused verification: 403 Python tests passed; 26 Chromium zero-egress tests passed (Firefox/WebKit unavailable); five post-format selection controls passed. Offline pinned runtime assets regenerated and verified. Existing Requests warning and inherited lint debt remain; no new lint diagnostics at checkpoint. Independent scoped review pending; no PR publication or merge performed.
<!-- SECTION:NOTES:END -->
