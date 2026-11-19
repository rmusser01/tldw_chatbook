---
id: TASK-41716
title: "A Library landing return leaves keyboard focus on the nav bar's Home button"
status: To Do
assignee: []
created_date: 2026-11-18 06:57
updated_date: 2026-11-19 19:58
labels:
- follow-up
- ui
- library
- a11y
priority: low
references:
- Docs/User_Guide/library.md
- tldw_chatbook/UI/Screens/library_screen.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41500's review (round 16). A fresh Library landing and a reused bare return both leave focus on `NavigationButton#nav-home`; Esc-to-hub instead focuses the hub's first action (`_focus_library_hub_entry`). The landing's accelerators still work, so this is a consistency question, not a defect.

Owner decision, 2026-11-19: "Content (rail list)". A cold landing (booting straight to Library, or arriving from another screen) lands on the rail's content, the way Console lands in its composer, not on the nav bar's Home button. Found again by TASK-41527 (round 17): from a cold landing, Tab walks the other 14 nav buttons before the rail, and a blind Enter leaves Library.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A fresh landing (boot to Library, or arrival from another screen) and a reused bare return focus the rail's first Browse row (Media); on the Get started compact rail of a new profile they focus Import…, as F6 does today; with the rail collapsed for the session, they focus the collapsed rail's **Nav** handle
- [ ] #2 A pilot test pins the focused widget id on each landing path in AC#1 (boot, arrival from another screen, reused return, Get started, collapsed rail), each red before the fix
- [ ] #3 A landing that already sets its own focus, such as a deep link into a destination or a click on a rail row, keeps it
- [ ] #4 Docs/User_Guide/library.md describes the landing
<!-- AC:END -->
