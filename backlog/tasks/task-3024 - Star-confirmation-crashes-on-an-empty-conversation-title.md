---
id: TASK-3024
title: Star confirmation crashes on an empty conversation title
status: To Do
assignee: []
created_date: '2026-08-07 16:19'
labels:
  - bug
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
ConsoleWorkspaceController._console_star_conversation formats its confirmation toast with title.splitlines()[0], which raises IndexError when the title is empty. console_workspace_context.py sets star_button.conversation_title from row.title raw, so an untitled conversation reaches it. Pre-existing (identical code at chat_screen.py:18058 before wave 4 moved it verbatim). The durable star/unstar write completes first, so no data is lost -- the user loses the confirmation toast and the workspace-context re-sync that follows it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Starring a conversation whose title is empty shows a confirmation and re-syncs the workspace context
- [ ] #2 A regression test covers the empty-title path
<!-- AC:END -->
