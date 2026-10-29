---
id: TASK-37620
title: Closed Console row menus keep claiming Esc
status: To Do
assignee: []
created_date: '2026-10-29 12:18'
labels:
  - console
  - keyboard-focus
  - ux-review-2026-09-29
dependencies:
  - TASK-33622.4
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The conversation- and workspace-row action menus defined a private method named _detach, which shadows the Textual hook that clears a removed widget's parent. A closed row menu therefore kept its parent, the menu registry kept reporting it as open, and the Console's Esc dismissed that ghost instead of returning to the composer. The ghost lasts for as long as anything still references the menu, which for a long-lived object means until a full garbage collection. Found while testing TASK-33622.3's menu fold; it breaks TASK-33622.4's Esc-to-composer rule after any row menu has been used.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After a conversation-row or workspace-row menu closes, the menu registry no longer reports it, even while the menu object is still referenced
- [ ] #2 After a row menu closes, Esc from the transcript returns focus to the composer
- [ ] #3 A Pilot test on the real ChatScreen opens and closes each row menu, then presses Esc from the transcript and asserts the composer has focus; it fails on the pre-fix code
<!-- AC:END -->
