---
id: TASK-38757
title: "File Notes narrow navigator gives the work pane a sliver and its controls take focus off screen"
status: To Do
assignee: []
created_date: 2026-11-05 07:11
labels:
- follow-up
- library
- file-notes
- ux
- accessibility
priority: medium
references:
- tldw_chatbook/Widgets/Library/library_file_notes_workspace.py
- tldw_chatbook/Utils/adaptive_reader_state.py
- Tests/UI/test_library_file_notes_workspace.py
- Docs/User_Guide/library/file-notes.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 7 of the 2026-10-23 follow-up burn-down. Checked on dev bb80905eb3, 2026-11-04.

Below 80 columns, the File Notes workspace switches to narrow navigation (`_apply_responsive_layout`, `Widgets/Library/library_file_notes_workspace.py:3536`). In the navigator view it resolves the shared adaptive reader layout with `priority="items"` (:3561-3571). The items pane and two 5-cell grips are seated first, and the work pane gets what is left.

A probe on this base called `resolve_adaptive_reader_layout(width, AdaptiveReaderLayoutPreferences(), AdaptiveReaderLayoutProfile(), priority="items")`. With items at 32 cells, the work pane got 37 cells at 79 columns, 18 at 60, 8 at 50 and 2 at 44, and none at 42 and below. At 40 the items pane itself dropped to 30.

That leftover pane is still laid out, and its controls stay in the focus chain. During the File Notes narrow-width work in PR #3403 (not yet on dev), at 40 columns, "More file actions" (`#file-notes-maintenance-toggle`) took focus at x=41, off screen, and the editor pane painted 1 cell wide at x=40. #3403 fixes the width-0 crash in that layout (its one-row buttons now truncate instead of wrapping). It left hiding the sliver alone, because that is a change to the shared resolver.

So between 40 and 79 columns the navigator shows a work pane too thin to use, and Tab can land on a control the user cannot see.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 In the narrow navigator view (40-79 columns), the work pane is either hidden or wide enough to use; no sliver of a few cells is shown
- [ ] #2 At every width from 40 to 79, Tab from the navigator never focuses a widget outside the visible region
- [ ] #3 In the narrow editor view (after opening a file, and after Back), the work pane and its controls still show and take focus as before
- [ ] #4 A width-sweep test over 40-79, in the navigator and editor views, pins #1-#3 and is red before the change
- [ ] #5 If the narrow layout changes, the File Notes User Guide page describes it
<!-- AC:END -->
