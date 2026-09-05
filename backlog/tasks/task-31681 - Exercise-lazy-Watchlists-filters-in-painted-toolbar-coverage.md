---
id: TASK-31681
title: Exercise lazy Watchlists filters in painted toolbar coverage
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 18:20'
updated_date: '2026-09-05 18:37'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the toolbar visibility regression to the shipped progressive-disclosure interaction while preserving painted controls and containment checks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The visibility test opens Filters through the real control and waits for its mounted laid-out editor.
- [x] #2 Every original toolbar control stays within its pane and its expected labels paint on the correct current row at both terminal sizes.
- [x] #3 Affected Watchlists visual cases pass without weakening viewport or paint assertions or changing product layout.
- [x] #4 Painted reverse-video checks are bounded to their table region and reject duplicate labels in adjacent panes.
- [x] #5 Tree highlight coverage uses a real paged Reader service so successful navigation can commit before the original style assertions.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve RED2toolbar sizes plus treehighlight and source reversevideo; diagnostic showed Filters is lazy, Readerfake lacks paging, and realsource row isreverse but helper returnsFalse at duplicated Inspectorlabel. 2. Click realFilters and await mounted editor; keep exactcontrols/containment and assert labels at actual per-control paintedrows. 3. Add synthetic duplicate-label-outside-region regression, verifyRED, then crop each compositor strip to the givenregion beforematching. Preserve missinglabel/unstyled semantics. 4. Remove obsolete emptyScopeService fake onlyfromtreehighlighttest so existing realpagedReader cancommit realclick; wait committedscope beforeoriginalclass/color/paint assertions. 5. Run allWatchlists and reverse-video parametrizations in affected visualfile, all helper regressions, scoped lint/format,parentreview, evidence,done,scopedcommit. ADR required:no. ADR path:N/A. Reason: faithful visualtestfixtures and oraclebounds only; no productbehavior orlayoutchanges.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Restored real lazy Filters interaction and per-control painted-row checks at both sizes. Cropped reverse-video evidence to the actual table region, with two RED/GREEN duplicate-label regressions. The tree highlight now uses the existing real paged Reader service and asserts successful scope commit before unchanged visual checks. All 38 Watchlists and reverse-video visual cases passed in 38.31s; report /private/tmp/tldw-31681-watchlists-visual-final.xml. Scoped Ruff and diff checks passed. Parent reviewed all changes with no actionable findings. Test-only changes; no product layout changes or new ADR required.
<!-- SECTION:NOTES:END -->
