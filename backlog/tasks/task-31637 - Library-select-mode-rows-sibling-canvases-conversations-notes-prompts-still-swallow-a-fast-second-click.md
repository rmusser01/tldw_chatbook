---
id: TASK-31637
title: >-
  Library select-mode rows: sibling canvases (conversations/notes/prompts) still
  swallow a fast second click
status: To Do
assignee: []
created_date: '2026-09-05 15:20'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Task 2 of the media wave5-F branch (task-31631 AC#2, commit d7d8c687df, see .superpowers/sdd/2026-09-05-media-ux-wave5-pr-f/task-2-report.md) fixed a real bug on Library Media's select-mode rows: each row is one full-width Button, and Textual's Button._on_click drops any click landing inside the previous press's 0.2s -active flash, so a fast second click on the same row (e.g. marker then title) was silently swallowed. The fix was one line, active_effect_duration = 0, at the media row's construction site. library_conversations_canvas.py, library_notes_canvas.py and library_prompts_canvas.py build their own select-mode rows the same way and never zero active_effect_duration, so the identical swallow still reproduces on those three canvases' rows. There is no shared row-factory to fix this once; each canvas needs its own one-line change (plus, ideally, the same whole-row-click test shape media now has).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 conversations, notes and prompts select-mode rows toggle correctly on a fast second click landing within 0.2s of the first (marker then title, or two title clicks)
- [ ] #2 Each canvas's browse-mode row press behavior (opening the item) is unchanged
- [ ] #3 A real-mouse-event pinning test exists per canvas, following the shape of test_every_click_on_a_media_row_toggles_it_in_select_mode in Tests/UI/test_library_multiselect_media.py
<!-- AC:END -->
