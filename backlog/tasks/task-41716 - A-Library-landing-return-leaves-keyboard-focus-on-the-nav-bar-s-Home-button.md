---
id: TASK-41716
title: "A Library landing return leaves keyboard focus on the nav bar's Home button"
status: To Do
assignee: []
created_date: 2026-11-18 06:57
updated_date: 2026-11-18 06:57
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
Found by TASK-41500's review (round 16). A fresh Library landing and a reused bare return both leave focus on `NavigationButton#nav-home`; Esc-to-hub instead focuses the hub's first action (`_focus_library_hub_entry`). The landing's accelerators still work, so this is a consistency question, not a defect.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A fresh landing and a reused bare return focus the same landing control, chosen and recorded in the notes
- [ ] #2 A pilot test pins the focused widget id on both paths
<!-- AC:END -->
