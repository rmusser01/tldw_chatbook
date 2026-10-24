---
id: TASK-36553
title: Select and reuse portrait eye and mouth regions in Personas
status: To Do
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
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users mark expression regions once on their prepared portrait and inspect them precisely before spending generation requests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The existing Living portrait editor offers three labeled image-space regions, a contain-mapped two-corner mouse selection, keyboard-accessible numeric adjustment and a detailed mask preview; letterboxing and terminal-cell aspect do not shift saved coordinates.
- [ ] #2 Users explicitly confirm the prepared base has open eyes and closed mouth before saving a composition recipe; unsupported base artwork can still use the existing ready-made six-frame import workflow.
- [ ] #3 Save regions reuses unchanged saved settings after reopen without provider calls, uses exact source/profile/editor/revision checks, and leaves the active portrait unchanged; Cancel discards only unsaved region edits.
- [ ] #4 Saving changed regions invalidates disposable generated components, requires a new complete composition candidate and retains the currently applied performance; source/profile changes reject late preview/save work.
- [ ] #5 Focused controls, narrow layouts, Static/Reduce motion, shared image budgets and token-backed styling work with production CSS; geometry/coordinate mapping, keyboard, stale-save and reopen tests verify the actual mounted editor.
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
