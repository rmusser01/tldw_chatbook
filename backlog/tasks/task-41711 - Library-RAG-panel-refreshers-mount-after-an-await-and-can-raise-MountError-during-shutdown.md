---
id: TASK-41711
title: "Library RAG panel refreshers mount after an await and can raise MountError during shutdown"
status: To Do
assignee: []
created_date: 2026-11-18 06:57
updated_date: 2026-11-18 06:57
labels:
- follow-up
- ui
- library
priority: medium
references:
- tldw_chatbook/UI/Library_Modules/library_rag_search_controller.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41500 (round 16). TASK-41500 guarded the RAG scope recovery mirror against mounting after the app's exit fence (`app._exit` makes every node read as detached, so Textual raises "Can't mount ... before X is mounted"). `_refresh_search_rag_panel_state_widgets` carries on after the scope block's early return into the answer, results and history remove/mount calls, and the query-status, answer, history and results refreshers have the same window between an await and a mount.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No RAG panel refresher mounts once the app's exit fence is set, and none swallows a mount error while the app runs
- [ ] #2 A deterministic test sets the fence between the await and the mount and is red before the fix
<!-- AC:END -->
