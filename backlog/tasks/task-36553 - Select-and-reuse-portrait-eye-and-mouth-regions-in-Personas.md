---
id: TASK-36553
title: Select and reuse portrait eye and mouth regions in Personas
status: Done
created_date: 2026-10-24 14:24
dependencies:
- TASK-36552
labels:
- console
- characters
- portrait-compositing
references:
- backlog/decisions/242-console-radio-portrait-performance.md
documentation:
- Docs/superpowers/specs/2026-10-23-console-radio-portraits-design.md
- Docs/superpowers/plans/2026-10-24-portrait-compositing-implementation.md
- backlog/decisions/243-source-bound-portrait-compositing.md
assignee:
- '@codex'
updated_date: 2026-10-24 16:34
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users mark expression regions once on their prepared portrait and inspect them precisely before spending generation requests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The existing Living portrait editor offers three labeled source-space regions, contain-mapped two-corner mouse selection, keyboard-accessible numeric adjustment and detailed mask preview; letterboxing and terminal-cell aspect do not shift saved coordinates.
- [x] #2 Users explicitly confirm an open-eyes/closed-mouth prepared base before saving a recipe; unsupported bases retain the existing ready-made six-frame import workflow.
- [x] #3 Save regions reuses unchanged canonical pixel identity/settings after reopen without provider calls or revision churn from equivalent PNG encoding; exact source/profile/editor/revision checks preserve active artwork and Cancel discards only unsaved edits.
- [x] #4 Saving changed regions retires the exact draft sample and atomically installs a rest-only/zero-component snapshot for the new recipe, without relabeling old frames as a complete manual set; the applied DB performance remains unchanged.
- [x] #5 Focused controls, narrow layouts, Static/Reduce motion, shared image budgets and token-backed styling work with production CSS; mounted coordinate/keyboard/stale-save/reopen tests reject late preview/save work.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/243-source-bound-portrait-compositing.md
Reason: Implement the approved one-time source-space region editor and independent settings Save using existing ownership and token contracts.

1. Read Unit 3, task criteria, incumbent editor/controller, image preparation, production CSS and design guidance.
2. Add red mounted tests for contain mapping, keyboard coordinates, confirmation, Save/reopen/idempotence, draft reset and stale authority.
3. Implement bounded static source/mask preview and a focused keyboard-accessible three-region modal, reusing source pixels and shared budgets.
4. Integrate region load/edit/Save into existing controller, retire exact speech preview on changed settings, install an authoritative rest-only candidate, and preserve active performance and manual imports.
5. Verify mounted normal/narrow/Static and stale preview/save controls, token/component/CSS governance, actual captures and targeted authoring callers. Self-review, document and commit before TASK-36554.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented source-space region modal with labeled two-corner and numeric selection, contain/aspect/letterbox mapping, source/compositor-mask crop preview and explicit narrow-readable neutral-base confirmation. New portrait_regions helper reuses exact prepared pixels/settings CAS and existing native/editor/profile ownership; changed Save stops the exact draft sample and installs a rest-only snapshot, unchanged Save retains candidate/revision/sample, and active artwork stays unchanged. Reopen validates settings and recovers applied components only through full pixel invariants; unsupported recipes permit confirmed reselection. Token-backed production CSS rebuilt, expansion measurement refreshed, and actual wide/narrow/source/mask/keyboard captures reviewed. Expanded authoring callers: 38 passed; final editor/CSS sync: 19 passed; focused final keyboard/budget controls: 2 passed. Ruff/format/diff pass; mechanical detector []. Three unrelated governance failures have seven offender sources byte-identical to pre-task db7ffce35f; none weakened. See Docs/superpowers/reviews/2026-10-24-task-36553-review.md, updated Architecture/user guide, and QA/portrait-compositing captures. ADR-243 applies. No full suite or live-provider claim.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
