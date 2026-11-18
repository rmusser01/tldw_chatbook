---
id: TASK-41915
title: "Collections at 80 columns lands focus on nothing"
status: To Do
assignee: []
created_date: 2026-11-18 20:31
updated_date: 2026-11-18 20:31
labels:
- follow-up
- ui
- library
- a11y
priority: low
references:
- tldw_chatbook/UI/Screens/library_screen.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41504's review (round 16), not a regression (base ends on nav-home). At 80x24 a code-driven switch into Collections leaves focus on nothing, because its rows, filter and Quick Capture sit in the collapsed Items pane. Media has the same compact limit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 At compact widths the list entry lands on a control in the visible pane, for Collections and Media
- [ ] #2 An 80x24 pilot pins the focused widget id
<!-- AC:END -->
