---
id: TASK-31957
title: >-
  Library media - the preview pane does not show the analysis marker the row
  does
status: To Do
assignee: []
created_date: '2026-09-07 08:26'
labels:
  - library
  - media-ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
I final review M6: task-28008's description asked for the analysis marker in the row secondary line and in the preview pane; only the row line shipped, a deliberate v1 omission because the acceptance criteria did not require the pane. The preview pane (library_media_state.py ~1090) still shows Title / Type / Updated, so the two surfaces disagree about the same item.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The preview pane reports the same analysis state as the row's secondary line
- [ ] #2 A painted pin covers an analysed and an un-analysed item
<!-- AC:END -->
