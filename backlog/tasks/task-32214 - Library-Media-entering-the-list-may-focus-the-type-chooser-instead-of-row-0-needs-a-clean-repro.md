---
id: TASK-32214
title: >-
  Library Media: entering the list may focus the 'type:' chooser instead of row
  0 (needs a clean repro)
status: Done
assignee: []
created_date: '2026-09-10 14:53'
updated_date: '2026-09-10 17:32'
labels:
  - library
  - media
  - keyboard
  - critique-9
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Assessor B observed: click the rail Media row, `Down` moved nothing, the `type: All types` button carried the heavy focus border and `Enter` opened the type strip; the repro had earlier keypresses in the session and `test_library_shell.py::test_library_media_list_focuses_first_row_and_arrow_keys_move_it` asserts the opposite. Reproduce cleanly (fresh launch, rail click only) before fixing. Evidence: critique #9 snapshot (.impeccable/critique/2026-09-10T*__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 02374bf66a on fresh + seeded profiles, 235x52/100x30/60x24). Register row 11.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A clean live repro either confirms the entry focus lands on the type chooser (then fix: focus row 0 on entry in every path, incl. re-entry after the chooser was used) or closes this task with the capture
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Spike: clean live repro on the seeded power profile (rail click only, no prior keys)
2. Decision rule from the plan: fix the entry-focus arm if it reproduces, otherwise close by evidence
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Not reproducible from a clean entry at dev 3315241674 (the plan's brief said e6cb464239; this worktree is off origin/dev at 3315241674). Live capture on the seeded power profile (tmux socket crit9-media-list, 235x52): command palette to Library, then the rail's Media row clicked with the mouse escape sequence only and NO other key. Focus landed on #library-media-row-0 -- 'Old draft — to discard' painted with the █ left-edge bar -- and a single Down moved the bar to row 1 ('Meeting recording 2026-09-05'). The 'type: All types' opener carried no focus frame at any point. Assessor B's repro carried earlier keypresses, which disarm the entry-focus arm (see Tests/UI/test_library_crit8_keyboard.py:210). The pin test_library_shell.py::test_library_media_list_focuses_first_row_and_arrow_keys_move_it holds (re-run green on this branch). No code change. Captures: /private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/92b26568-46c4-4abe-9da5-9676d5282276/scratchpad/crit9/wave/media-list/caps/32214-a-after-rail-click.txt and 32214-b-after-down.txt
<!-- SECTION:NOTES:END -->
