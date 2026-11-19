---
id: TASK-42001
title: "A cold landing on Library puts keyboard focus on the rail's first row"
status: To Do
assignee: []
created_date: 2026-11-19 19:39
updated_date: 2026-11-19 19:39
labels:
- follow-up
- ui
- library
priority: medium
references:
- tldw_chatbook/UI/Screens/library_screen.py
- Docs/User_Guide/library.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41527 (round 17). Booting straight to Library, or arriving from another screen (Console then Ctrl+3), leaves focus on the nav bar's Home button: a blind Enter leaves Library, and Tab walks all 14 nav buttons before the rail. Owner decision, 2026-11-19: "Content (rail list)", so a cold landing focuses the Library rail's first row, the way Console lands in its composer.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Booting to Library and arriving from another screen both land focus on the rail's first row (or the rail search box when the rail is empty)
- [ ] #2 A rail click or deep link that already sets focus keeps its own landing
- [ ] #3 Tests cover boot and screen-switch arrival, red before the fix
- [ ] #4 Docs/User_Guide/library.md describes the new landing
<!-- AC:END -->
