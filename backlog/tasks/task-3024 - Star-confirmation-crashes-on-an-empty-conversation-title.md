---
id: TASK-3024
title: Star confirmation crashes on an empty conversation title
status: Done
assignee: []
created_date: '2026-08-07 16:19'
updated_date: '2026-08-07 19:45'
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
- [x] #1 Starring a conversation whose title is empty shows a confirmation and re-syncs the workspace context
- [x] #2 A regression test covers the empty-title path
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`"".splitlines()` is `[]`, so the first-line read raised `IndexError` on an
untitled conversation -- *after* the durable star write, so the toggle landed
while the user saw no confirmation and the context rail never re-synced. The
empty case was always intended: the `title_suffix` logic below already drops
the quoted name when the title is falsy; only the read was unguarded.

Fixed with `next(iter(...), "")`. The regression test is mutation-verified: it
reproduces the `IndexError` against the original expression, and asserts the
durable write landed as well as the toast, so it pins both halves of the defect.
<!-- SECTION:NOTES:END -->
