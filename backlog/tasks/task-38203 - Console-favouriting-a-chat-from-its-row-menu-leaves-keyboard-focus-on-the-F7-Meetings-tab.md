---
id: TASK-38203
title: >-
  Console: favouriting a chat from its row menu leaves keyboard focus on the F7
  Meetings tab
status: To Do
assignee: []
created_date: '2026-10-30 06:21'
labels:
  - console
  - left-rail
  - focus
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**Impact.** A keyboard user who opens a Conversations row's menu with 'm' and chooses Favourite (or Remove favourite) loses their place: focus lands on the navigation bar's last tab, F7 Meetings, instead of returning to the row. The next 'm' does nothing, and Enter would open Meetings. This is the same class of defect TASK-33621.12 fixed for the Save .md… prompt.

**Evidence (live, real app, real model, isolated profile, 160x45).** Two chats were created with real replies. The first row was focused from the filter with Tab x3; 'm' opened its menu; Enter on Favourite starred the chat; a capture taken 3 s later shows F7 Meetings highlighted, and a following 'm' opened no menu. The result is the same on dev ff815f8559, on the TASK-33622.6 branch before its review round (639f103d06), and on the branch after it. Captures: dv-fav-*, pf-fav-*, r2-05/06/07 under the Console UX review session's evidence/bd5/U12/. Closing the menu's 'Icon and colour…' prompt with Escape sent focus to the composer instead (r2-08/09), so the restore path differs per menu action. Root cause not investigated.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After Favourite or Remove favourite is chosen from a row's menu by keyboard, focus returns to that chat's row (wherever the star moved it), and a following 'm' opens that chat's menu
- [ ] #2 Focus never lands on a navigation-bar tab or another section's control as a side effect of a row-menu action
- [ ] #3 A Pilot test on the real ChatScreen opens a row menu with 'm', chooses Favourite with Enter, and asserts the focused control is that chat's row; it fails on the pre-fix code
<!-- AC:END -->
