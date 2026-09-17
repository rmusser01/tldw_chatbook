---
id: TASK-32752
title: Keep empty Notes folder actions fully visible
status: Done
assignee:
  - '@codex'
created_date: '2026-09-17 19:35'
updated_date: '2026-09-17 20:05'
labels:
  - design-system
  - ui
  - library
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The integrated native app clips the disabled Remove placement label in empty Library Notes at 170x48 in both themes. Preserve complete visible action labels and truthful disabled-state guidance at actual pane widths.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Empty Notes folder actions paint their complete labels within the canvas at wide and compact supported sizes in both themes.
- [x] #2 Selected-note and selected-placement action packing and disabled-state explanation remain correct.
- [x] #3 Targeted layout, token and generated-style checks plus inspected private native evidence qualify the repair.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: existing backlog/decisions/150-design-token-system-and-design-language.md and backlog/decisions/161-component-pattern-library.md
Reason: routine packing correction within the existing Notes toolbar contract, without new architecture or visual token values.

1. Reproduce empty Notes action clipping with production CSS at actual wide/compact pane geometry; measure button chrome, row budget and canvas content.
2. Correct the proven budget or layout defect using the existing wrap-to-next-row policy; preserve compact omissions and disabled reasons.
3. Verify targeted empty/selected/protected action paint, resize stability, token floors and generated styles.
4. Review independently and inspect private native dark/light wide/compact captures with lifecycle/default isolation evidence.
5. Update task/report/ledger and save the verified repair to draft PR #2704.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed empty Notes folder-action clipping using the stable Items width minus host/canvas chrome and the measured tree button/row costs. Overflowing composed rows repack on shrink, including the initial unmeasured frame; harmless extra rows remain on growth. Clearing derived tree caches when composing a legacy list preserves the filter widget through that state transition.

Added 13 production-CSS cases covering empty/selected/protected/folder actions, themes, actual pane bounds and paint, resize/growth identity, measured fallback and the tree-to-legacy transition. Five affected existing layout tests now use the existing private-profile helper. Verification: 18 affected plus 44 governance/build/budget cases pass (62 distinct); all derived-artifact checks pass; scoped static checks add no debt. Independent review finding reproduced with cache clearing removed, then resolved. Four final real terminal captures show complete wide labels and preserved compact policy; ten private DBs healthy, clean process/terminal shutdown, no app errors and unchanged default files.

ADR required: no; existing ADR-150/161 apply. Updated Notes guide, Library workflow audit, completion ledger and QA evidence at Docs/superpowers/qa/2026-09-17-notes-empty-toolbar/README.md. Initial harness errors and pre-import native refusal remain recorded as unqualified attempts. No full suite or external service/generation qualification. Remaining Settings/destination reviews stay open.
<!-- SECTION:NOTES:END -->
