---
id: TASK-3749
title: Composer DraftChanged message would unblock six on_key branches
status: To Do
assignee: []
created_date: '2026-08-08 21:06'
labels:
  - refactor
  - console
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave 5 task 1 moved 7 of on_key's composer branches into ConsoleComposerBar. Eleven more could not move because they call screen methods after editing the draft: six of those call _sync_console_workbench_actions_from_draft (workbench state + slash-command popup) and _dismiss_console_guidance (repaints the transcript). If the composer posted a DraftChanged message that the screen subscribed to, those six branches would become composer-only and could move, taking the keymap with them. This is a DESIGN change, not an extraction, which is why wave 5 did not do it: an extraction that also changes how components communicate gives a regression two candidate causes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The composer notifies the screen of draft changes rather than the screen calling back after each edit
- [ ] #2 The six blocked branches move to ConsoleComposerBar.handle_console_key
- [ ] #3 No behaviour change: the workbench actions, slash-command popup and guidance repaint still update on every draft edit
<!-- AC:END -->
