---
id: TASK-32214
title: >-
  Library Media: entering the list may focus the 'type:' chooser instead of row
  0 (needs a clean repro)
status: To Do
assignee: []
created_date: '2026-09-10 14:53'
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
- [ ] #1 A clean live repro either confirms the entry focus lands on the type chooser (then fix: focus row 0 on entry in every path, incl. re-entry after the chooser was used) or closes this task with the capture
<!-- AC:END -->
