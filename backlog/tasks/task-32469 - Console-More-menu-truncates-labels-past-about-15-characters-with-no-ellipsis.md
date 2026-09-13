---
id: TASK-32469
title: Console More menu truncates labels past about 15 characters with no ellipsis
status: To Do
assignee: []
created_date: '2026-09-11 23:27'
labels:
  - console
  - ux
  - critique-notes-2026-09
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Widgets/Console/console_message_more_menu.py fixes the per-message More menu at 24 cells (width: 24 / MENU_WIDTH) and Textual cuts any button label past about fifteen characters silently: no ellipsis, no tooltip. Evidence from task-32146's live captures (wave3-caps/capture-console/01-more-menu-235x52.txt and 04-more-menu-100x30.txt): TASK-31759's Summarize up to here as note renders as Summarize up to and Save transcript up to here as note as Save transcript, so neither names a destination; task-32146 shipped a 15-character label (Capture as note) to fit. The action-row contract tests assert label strings, never rendered width, so this cannot fail a test today, and Docs/User_Guide/console/chat-basics.md currently documents the cut labels as-is. The fix is the menu, not the labels.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every More-menu label renders whole at 235x52 and 100x30, or is shortened with a visible ellipsis and a tooltip carrying the full label
- [ ] #2 A test pins the rendered More-menu label width so a label that would be cut fails before it ships
- [ ] #3 Docs/User_Guide/console/chat-basics.md no longer describes the truncated labels
<!-- AC:END -->
