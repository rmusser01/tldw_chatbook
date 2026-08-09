---
id: TASK-3800
title: Skills-canvas dirty-guard test hits the fixture-bypass AttributeError
status: To Do
assignee: []
created_date: '2026-08-09 15:50'
labels:
  - tests
  - library
  - tech-debt
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
test_action_library_skill_back_honors_dirty_guard (Tests/UI/test_library_skills_canvas.py) fails on dev: it drives LibraryScreen.action_library_skill_back / _arm_library_list_entry_focus against a SimpleNamespace fake that bypasses LibraryScreen.__init__ (via a local _bind_editor_active helper), so the fake never gets _library_list_entry_focus_timer, an __init__-only attribute the guarded-exit path reads. This is the exact fixture-bypass shape task-3022 diagnosed and fixed for a different cluster of tests in this same file family (construct the fixture properly -- object.__new__(LibraryScreen)-style or SimpleNamespace bypass without the __init__-only attributes present -- rather than patching production code); it escaped task-3022's own exit-bar suite sweep because this particular test was not in the suites that task swept.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The test's fixture sets or provides every __init__-only attribute action_library_skill_back's guarded-exit path (_arm_library_list_entry_focus) reads, so it no longer raises AttributeError
- [ ] #2 test_action_library_skill_back_honors_dirty_guard passes on dev
- [ ] #3 No production code in library_screen.py is changed to accommodate the fixture
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Filed, not fixed, per the fix/library-polish-batch final review's Important #2: out of that wave's scope to fix, filing the debt task was the ask.
<!-- SECTION:NOTES:END -->
