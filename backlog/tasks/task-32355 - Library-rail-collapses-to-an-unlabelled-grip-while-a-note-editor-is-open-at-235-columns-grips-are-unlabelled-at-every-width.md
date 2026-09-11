---
id: TASK-32355
title: >-
  Library rail: collapses to an unlabelled '--->' grip while a note editor is
  open at 235 columns; grips are unlabelled at every width
status: To Do
assignee: []
created_date: '2026-09-11 06:17'
labels:
  - library
  - layout
  - critique-10
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
With a note open at 235 columns the rail is replaced by '--->' with no label (A cap 11; the rail returns on Escape, cap 13); three literal '<---'/'--->' runs sit in the gutters at 235x52 (B D10, cap 17). PROVEN source: library_adaptive_reader_shell.py:151-156. library.md promises a slim 'Nav' handle and the rail beside Notes at 120 columns and wider. Evidence: critique #10 snapshot (.impeccable/critique/2026-09-11T06-13-11Z__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 1f3184655b, captures under the session scratchpad crit10/A and crit10/B).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 At 120 columns and wider the rail stays beside the note editor, or the collapse is documented as intended and the handle is labelled 'Nav'
- [ ] #2 Every pane grip carries a label or a footer hint
- [ ] #3 Pinned
<!-- AC:END -->
