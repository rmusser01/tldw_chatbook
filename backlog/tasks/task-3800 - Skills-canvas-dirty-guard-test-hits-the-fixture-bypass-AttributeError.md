---
id: TASK-3800
title: Skills-canvas dirty-guard test hits the fixture-bypass AttributeError
status: Done
assignee:
  - '@claude'
created_date: '2026-08-09 15:50'
updated_date: '2026-08-09 18:18'
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
- [x] #1 The test's fixture sets or provides every __init__-only attribute action_library_skill_back's guarded-exit path (_arm_library_list_entry_focus) reads, so it no longer raises AttributeError
- [x] #2 test_action_library_skill_back_honors_dirty_guard passes on dev
- [x] #3 No production code in library_screen.py is changed to accommodate the fixture
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the AttributeError in isolation to confirm the exact missing attribute and the call site that reads it.
2. Read _arm_library_list_entry_focus (library_screen.py) to enumerate every __init__-only attribute its guarded-exit path touches.
3. Compare against the existing SimpleNamespace fixture (the 'clean' fake in test_action_library_skill_back_honors_dirty_guard) to find which of those attributes are missing.
4. Add the missing attribute(s) to the SimpleNamespace fixture at the value __init__ would set (None), matching task-3022's established fixture-repair shape used elsewhere in this file family -- no production code changes.
5. Run the test 5x; run the full file to confirm no regression.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Filed, not fixed, per the fix/library-polish-batch final review's Important #2: out of that wave's scope to fix, filing the debt task was the ask.

**Fix (2026-08-09).** Confirmed the exact site: `_arm_library_list_entry_focus` (library_screen.py) reads `self._library_list_entry_focus_timer` (an `is not None` check, to stop() any prior timer before scheduling a new one -- PR #1410's stored-timer-handle guard) before it ever assigns to it. The `clean` `SimpleNamespace` fake in `test_action_library_skill_back_honors_dirty_guard` set `set_timer` and `_disarm_library_list_entry_focus` but never the `_library_list_entry_focus_timer` attribute itself, so the read raised `AttributeError` on a bare `SimpleNamespace` (which has no attribute defaults). Added `_library_list_entry_focus_timer=None` to that `SimpleNamespace(...)` call, matching the value `LibraryScreen.__init__` sets, with a comment naming task-3022's fixture-bypass shape and why this specific attribute was missing. No other attribute was missing -- the rest of `_arm_library_list_entry_focus`'s reads (`call_after_refresh`, `set_timer`) were already provided.

No production code touched (`git diff --stat tldw_chatbook/` empty for this task). Verified the whole file (108 tests) still passes after the change, and 5/5 consecutive runs of the target test.

**Files changed:** `Tests/UI/test_library_skills_canvas.py` (`test_action_library_skill_back_honors_dirty_guard`'s `clean` fixture: added the missing `_library_list_entry_focus_timer=None`).
<!-- SECTION:NOTES:END -->
