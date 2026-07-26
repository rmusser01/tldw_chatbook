---
id: TASK-670
title: Extend RecomposeCaptureGuard to remaining recompose widgets
status: To Do
assignee: []
created_date: '2026-07-26 12:00'
labels:
  - followup
  - ui
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-637 guarded the 5 originally-named non-screen recompose sites with the RecomposeCaptureGuard mixin, but its repo sweep found ~22 more same-bug-class sites left out of scope to bound the task: 7 UI/Watchlists_Modules/*_pane.py files plus ~15 across Widgets/Home, Widgets/Chat_Widgets, Widgets/Evals, Widgets/Library, Widgets/Console (7 files), Widgets/status_widget.py, Widgets/file_list_item_enhanced.py, Widgets/TTS/*. Full enumerated list in task-637's report. A capture held by a descendant of any of these at recompose time still leaks app-wide.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All enumerated remaining recompose sites carry the mixin (or a documented exemption)
- [ ] #2 At least two newly-guarded sites have regression tests (one simple, one teardown-drain)
- [ ] #3 Existing capture/navigation tests stay green
<!-- AC:END -->
