---
id: TASK-412
title: Skills editor - guard dirty edits on Library rail switch (silent data loss)
status: To Do
assignee: []
created_date: '2026-07-21 15:18'
labels:
  - skills
  - ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
P0 from the 2026-07-21 Skills UX/NNG review (verified live). _select_library_rail_row guards the notes and prompts dirty flushes but omits the skill flush: switching to another Library rail row while a skill edit is dirty silently discards the edit with no warning and no undo. Notes/prompts editors on the same screen are protected; skills are not. NNG heuristics 5 (error prevention) and 3 (user control).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Switching Library rail rows while the skill editor has unsaved changes no longer silently discards them (same veto/flush contract as notes and prompts),Screen-level flush behavior for skills is covered by a regression test that fails on the pre-fix code,Existing notes/prompts rail-switch guards keep working
<!-- AC:END -->
