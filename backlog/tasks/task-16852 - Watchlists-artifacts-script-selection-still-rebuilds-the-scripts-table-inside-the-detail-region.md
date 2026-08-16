---
id: TASK-16852
title: 'Watchlists artifacts: script selection still rebuilds the scripts table inside the detail region'
status: To Do
assignee: []
created_date: '2026-08-16'
labels:
  - ui
  - watchlists
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-15779 (PR #1732) fixed briefing selection destroying the briefings table by moving
the detail chrome into a `BriefingDetailRegion` that recomposes alone. Its
Implementation Notes disclose the deliberately unexpanded scope — the same defect one
level down — and it still holds at dev `ee741cf10`:

Selecting a **script** rebuilds the WHOLE detail region, including the scripts table the
user is interacting with. `UI/Watchlists_Modules/artifacts_pane.py:1840-1842` —
`watch_selected_script` calls `_refresh_detail_region()` (`:1789`), which does
`region.refresh(recompose=True)`, and `#artifacts-scripts-table` is composed inside
`compose_briefing_detail()`, i.e. inside the region being torn down. So the 15779 bug's
symptom set (focused table destroyed under the user, scroll position lost, the
immediately-following arrow key press dead because the focused widget was unmounted)
recurs at the scripts level. The briefings table is unaffected either way — 15779's fix
stands.

Fix direction is the 15779 recipe applied one level down: split the script-detail chrome
into its own sub-region (or patch script-selection-dependent bits in place) so a script
selection updates the script detail without recomposing `#artifacts-scripts-table`.
The 15779 pin suite (`Tests/Watchlists/test_watchlists_artifacts_selection_in_place.py`)
is the template for the born-red evidence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Selecting a script updates the script detail without unmounting `#artifacts-scripts-table` (same widget instance, focus and scroll preserved; born-red test)
- [ ] #2 An arrow-key press immediately after a script selection moves the selection on (the 15779 AC symptom, at the scripts level)
- [ ] #3 The 15779 selection-in-place suite and the artifacts pane suites stay green
<!-- AC:END -->
