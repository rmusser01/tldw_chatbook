---
id: TASK-448
title: Skills editor - guard dirty edits on Library rail switch (silent data loss)
status: Done
assignee:
  - '@claude'
created_date: '2026-07-21 15:18'
updated_date: '2026-07-21 15:36'
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
- [x] #1 Switching Library rail rows while the skill editor has unsaved changes no longer silently discards them (same veto/flush contract as notes and prompts),Screen-level flush behavior for skills is covered by a regression test that fails on the pre-fix code,Existing notes/prompts rail-switch guards keep working
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce in a unit test: drive _select_library_rail_row with a dirty skill editor and assert the switch is vetoed (test fails on current code)\n2. Add the skill flush call to _select_library_rail_row mirroring the existing notes/prompts guards\n3. Confirm flush contract matches _flush_library_skill_save veto semantics (explicit-save-only, returns False while dirty)\n4. Run skills + library test suites
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added the missing skill flush guard to _select_library_rail_row, mirroring the prompt guard exactly: 'if not await self._flush_library_skill_save(): return' after the notes/prompt checks. _flush_library_skill_save is veto-only (explicit-Save-only editor), so the rail switch now aborts while _library_skill_dirty is set instead of silently resetting the editor. Regression test test_library_shell_rail_switch_vetoed_while_skill_editor_dirty (Tests/UI/test_library_skills_canvas.py) drives the real harness: create editor, real Input edit marks dirty, browse-media rail press is vetoed with the edit intact - watched fail on pre-fix code (switch went through to browse-media). Suites: skills canvas 41 passed; library shell 253 passed, 4 failures reproduced identically on clean origin/dev (pre-existing ingest/config baseline, unrelated). Visible-feedback follow-up is task-449 by design.
<!-- SECTION:NOTES:END -->
