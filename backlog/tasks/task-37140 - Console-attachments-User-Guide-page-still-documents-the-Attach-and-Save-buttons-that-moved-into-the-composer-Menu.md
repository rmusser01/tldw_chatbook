---
id: TASK-37140
title: >-
  Console attachments User Guide page still documents the Attach and Save
  buttons that moved into the composer Menu
status: To Do
assignee: []
created_date: '2026-10-27 18:30'
labels:
  - console
  - docs
  - p3
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the review of TASK-33620.15.22 (2026-10-27). Docs/User_Guide/console/attachments-images-voice.md still describes a composer action row with **Attach** and **Save** buttons. Attach and Save as Chatbook moved into the composer Menu (console_composer_menu_modal.py: the row keeps only Send, Mic and the conditional Stop and ✕; the Menu entry is "Attach file"). The page is stale in four places: "Getting there" lists Send, Mic, Attach and Save to the right of the draft; the Layout tour says the Attach button relabels to **Attach +** when a file is staged, and its screenshot (images/console/attachment-staged.svg) shows that old row; "Three ways to attach a file" item 1 says to press the **Attach** button; "Managing staged attachments" says **Attach +** stages another file. The page also calls the dictation button **Mic** (seven times); the composer labels it Dictate (console_composer_bar.py, idle label "Dictate"; seen live in the TASK-33620.15.22 run). A reader following the page looks for buttons that are no longer there.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every place the page names a composer control matches the current composer row and Menu, checked against the running app
- [ ] #2 The layout screenshot shows the current composer with a staged attachment
<!-- AC:END -->
