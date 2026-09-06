---
id: TASK-31796
title: Library notes list keeps stale note title after rename until a filter re-query
status: To Do
assignee: []
created_date: '2026-09-05 19:15'
labels:
  - bug
  - ui
  - library
  - notes
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the 2026-09-05 pre-release live UAT sweep (fresh scratch profile, dev tip 8e9d1128d4, real tmux-driven app). Create a blank note (autosaves as 'Untitled'), set its title in the editor, wait for 'Saved', Esc back to the list: the Unfiled entry still reads 'Untitled', and stays stale even after reopening the note. Only typing a filter query and pressing Enter re-queries and shows the real title. DB row confirmed correct throughout, so this is purely the list widget not refreshing on save. With several new notes, every one shows as 'Untitled' in the primary nav surface.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Renaming a note updates its row in the notes list on save (or at latest on returning to the list), without requiring a filter re-query.
- [ ] #2 A regression test covers title propagation from editor save to the list row.
<!-- AC:END -->
