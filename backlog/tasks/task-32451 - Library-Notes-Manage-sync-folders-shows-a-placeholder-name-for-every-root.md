---
id: TASK-32451
title: Library Notes Manage sync folders shows a placeholder name for every root
status: To Do
assignee: []
created_date: '2026-09-11 16:10'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - sync
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while walking the lasting-sync chapter for task-32269, once task-32243 made a root reachable at all.

Every row in Manage sync folders reads "Sync folder (name unavailable before cutover)", including a root the user just created and typed a display name for. `library_notes_sync_controller.py:709` hard-codes that literal for every row because the path-free `NotesSyncRootRuntimeSnapshot` carries only root_id/status/next_action -- the display name the user typed at setup has no route to the row, and neither does the folder. With one root the row is merely unhelpful; with two it is unusable, because nothing on screen distinguishes them.

The name is not private the way the path is: the user typed it, it is already shown in the notes folder tree ("Vault sync"), and it is validated as a bounded non-path label by `NotesSyncRootSetup`. The fix is a route for it, not a new path field.

Evidence: captures cap-06 and cap-07 under the wave-3 sync scratchpad (live walk at 235x52 against a 179-file vault, 2026-09-11).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each row in Manage sync folders shows the display name the user gave that root
- [ ] #2 Two roots are distinguishable from the list alone
- [ ] #3 No absolute path reaches the row or any log record through this change
- [ ] #4 A migrated legacy candidate, which has no user-typed name, still shows something honest rather than a placeholder that claims a name is unavailable
<!-- AC:END -->
