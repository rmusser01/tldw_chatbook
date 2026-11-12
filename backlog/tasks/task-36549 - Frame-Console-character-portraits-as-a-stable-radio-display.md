---
id: TASK-36549
title: Frame Console character portraits as a stable radio display
status: Done
created_date: 2026-10-23 22:56
dependencies:
- TASK-36545
- TASK-36546
labels:
- console
- ui
- accessibility
documentation:
- Docs/superpowers/specs/2026-10-23-console-radio-portraits-design.md
- Docs/superpowers/plans/2026-10-23-console-radio-portraits-implementation.md
- backlog/decisions/242-console-radio-portrait-performance.md
- Docs/superpowers/reviews/2026-10-23-console-radio-portraits-implementation-review.md
- Docs/superpowers/reviews/2026-10-23-task-36549-review.md
updated_date: 2026-10-24 03:40
assignee:
- '@codex'
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add restrained radio-screen identification, truthful voice status and an audio-level meter around the living portrait without shifting the Console sidebar.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The existing Character section presents a recessed fixed-geometry portrait frame, character identification, compact voice status and a measured speech-level meter; ordinary-layout rows reserve space through rest, speech and timing loss/recovery using visibility hiding, with row removal restricted to compact layout.
- [x] #2 Ready, Preparing voice, Transmitting, Paused and voice-error feedback reflect actual owned activity; silence/underrun empties the meter and unqualified timing shows truthful status without invented movement.
- [x] #3 Theme colors and authored portrait colors are preserved; any brief transmission effect uses central tokens, adds no sound/continuous static and respects motion preferences.
- [x] #4 Narrow layouts drop the meter before sacrificing controls, preserve contain fit/rail scroll and retain full-size portrait viewing and keyboard/focus contracts.
- [x] #5 Only source TCSS is edited and rebuilt; all new visual/motion values use defined design tokens and canonical component patterns.
- [x] #6 Production-CSS mounted checks prove stable meter-row height, actual portrait/control bounds and widget identity through timing loss/recovery and rest/transmission; reduced/static motion and narrow/tall resize remain correct, compactness follows layout budget rather than speech state, and token/generated-CSS guards pass.
- [x] #7 Character names and voice/status labels render as plain text; markup-shaped names cannot crash Console synchronization or alter label styling.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/242-console-radio-portrait-performance.md
Reason: Implement approved stable radio presentation around the existing portrait and owned speech channel without a new runtime/audio boundary.

1. Read the actual left-rail compose/allocation path and Console snapshot ownership; add failing pure status and production-CSS geometry tests.
2. Add a small status/meter widget and recessed token-backed portrait frame, preserving existing identification, controls, viewer, authored color and plain-text labels.
3. Reuse the avatar presentation timer for living-portrait meter updates and existing Console sync for status/fallback. Add no independent presentation timer, image decode, per-frame SQLite read or remount.
4. Reserve identification/status/meter rows before image fitting, account for fixed frame border geometry, and derive compactness only from measured layout budget. Ordinary suppression uses visibility hiding; only compact layout removes the meter row.
5. Test actual owned Preparing/Transmitting/Paused/Ready/failure, timing loss/recovery, silence, reduced/static motion, hostile names, widget/portrait/control geometry and narrow/tall resize with production CSS.
6. Rebuild governed CSS, run relevant guards and targeted tests, bounded visual inspection, update guidance/plan, self-review and commit before integrated qualification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented stable token-backed radio framing, literal identity labels and truthful owned voice status/level projection with existing clocks. Ordinary row suppression preserves geometry; measured compact layout drops the meter and border-aware contain fitting retains the existing scroll/viewer/controls. Fixed shared Rich label interpretation for markup-shaped character names. ADR-242 applies; no new boundary/dependency/token. Modified focused radio widget, character/rail/wiring/renderer, control/workbench labels, source/rebuilt CSS and guides. Self-review: Docs/superpowers/reviews/2026-10-23-task-36549-review.md. Evidence: 66 targeted UI checks passed; 17 final UI/visual/coalescing checks passed; 86 architecture/style/timer guards passed, one skipped, three pre-existing failures reproduced with original source; two current screen budgets passed. New code lint/format and diff check pass, no added legacy lint diagnostics. Two bounded visual rounds and detector [] recorded under Docs/superpowers/qa/console-radio-portraits/2026-10-23/radio-ui.md. Full suite and live provider/device qualification were not run in this unit; TASK-36550 owns live evidence.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
