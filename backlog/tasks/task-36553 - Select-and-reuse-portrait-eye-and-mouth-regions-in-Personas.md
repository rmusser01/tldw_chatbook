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
updated_date: 2026-10-24 18:23
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
- [x] #6 A valid saved source that cannot produce a bounded generation reference still permits manual frame Prepare/import/Apply/reopen/Remove; compositing alone is unavailable, limits and original source bytes remain intact.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/243-source-bound-portrait-compositing.md
Reason: Existing ADR-243 requires ordinary manual import/removal compatibility; this correction adds no architectural decision and keeps memory/validation limits intact.

The original implementation and verification remain recorded in the linked plan/review. Review correction:
1. Add a failing mounted test with a valid 4096x4096 saved source and already-applied manual frames; verify Prepare, manual replacement/Apply/reopen/Remove and source preservation.
2. Make optional reference/region preparation fail independently of ordinary draft creation; disable/refuse unavailable compositing actions with a useful explanation.
3. Run the regression and affected authoring/region/generation controls, self-review all callers and document the result before moving to the pair-viewer finding.
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

Full-review correction: optional generation-reference failure no longer prevents
ordinary manual authoring/removal. A valid 4096×4096 source regression was red at
Prepare, then green through import/replacement, Apply, reopen and Remove without
changing source bytes or memory limits. Compositing controls alone are disabled.
Authoring/region/generation checks: 62 passed in 165.67s; Ruff/format/diff pass.
Self-reviewed all reference callers, control reset and unchanged source/CAS boundaries.
Existing ADR-243 applies; see the appended TASK-36553 review.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
