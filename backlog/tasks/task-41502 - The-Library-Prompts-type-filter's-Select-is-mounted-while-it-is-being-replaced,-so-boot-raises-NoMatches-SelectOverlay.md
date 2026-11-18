---
id: TASK-41502
title: "The Library Prompts type filter's Select is mounted while it is being replaced, so boot raises NoMatches SelectOverlay"
status: To Do
assignee: []
created_date: 2026-11-17 05:27
labels:
- follow-up
- library
- prompts
- ui
priority: medium
references:
- tldw_chatbook/UI/Screens/library_screen.py
dependencies:
- TASK-41127
- TASK-41141
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41127 and re-measured in its fix round (round 15). `#library-prompts-type-filter` mounts a `Select` while the previous one is being replaced; `NoMatches 'SelectOverlay'` / `'#label'` is raised. Boot with the Prompts default tab fails 3 of 3 and the Prompts route 2 of 3. TASK-41141 fixed the exit-time mount fence for stock Select widgets; this is a different, mount-time overlap.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Booting with the Prompts default tab and opening the Prompts route raise no `NoMatches` for the type filter, 5 of 5 runs
- [ ] #2 The two failing nodes are named in the Implementation Notes with red-before and green-after evidence
<!-- AC:END -->
