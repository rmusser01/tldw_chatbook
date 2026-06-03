---
id: TASK-75
title: Add optional Console background effects
status: Done
labels:
- console
- ui
- settings
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add optional ambient background effects for the Console transcript/event stream so users can enable snow, rain, or matrix-style visuals without affecting Console controls, transcript content, or readability by default.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Console background effects default to off and load safely from config.
- [x] #2 Users can configure effect, scope, intensity, and frame rate from Settings and config.
- [x] #3 Transcript-scoped effects render behind the main transcript/event stream without changing transcript focus, selectors, row reconciliation, or exports.
- [x] #4 Controls, rails, composer, provider recovery, and inspector remain unaffected in transcript scope.
- [x] #5 Workbench scope is either implemented safely or visibly gated/falls back without pretending it is active.
- [x] #6 Focused tests cover config normalization, Settings persistence, renderer behavior, Console selector stability, and transcript keyboard behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Follow `Docs/superpowers/plans/2026-06-02-console-background-effects-implementation.md` task-by-task.
2. Complete Task 1: config model and defaults with failing-first tests.
3. Complete Task 2: Settings ownership, controls, and nested persistence.
4. Complete Task 3: background effect renderer.
5. Complete Task 4: Console transcript wiring and CSS regeneration.
6. Complete Task 5: workbench scope gate or safe implementation.
7. Complete Task 6: focused verification, self-review, and task hygiene.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added normalized Console background effect configuration defaults and validation for disabled-by-default snow, rain, and matrix effects.
- Added Settings ownership, controls, nested persistence, and visible Workbench-scope fallback for unavailable wider-scope effects.
- Added a non-focusable Textual renderer and transcript surface that layers effects behind `#console-native-transcript` without changing transcript widget identity.
- Wired transcript-scoped settings from `app_config` into the live Console session surface and regenerated modular CSS after TCSS changes.
- Preserved Workbench scope as a config value but gated it in Settings and transcript runtime so it cannot silently activate behind controls.
- Verified with focused config, Settings, renderer, Console selector, and transcript keyboard tests.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented optional Console background effects behind the transcript/event stream, default off. Workbench scope is visibly gated to Transcript in this build. Final focused verification passed: `30 passed, 8 warnings`; CSS build passed with the existing missing `_evaluation_v2.tcss` warning; `git diff --check` passed.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] Acceptance criteria checked.
- [x] Implementation plan followed with review gates per task.
- [x] Focused automated tests run.
- [x] CSS build run after TCSS change.
- [x] Diff hygiene checked.
- [x] Implementation notes added.
- [x] Status set to Done.
<!-- DOD:END -->
