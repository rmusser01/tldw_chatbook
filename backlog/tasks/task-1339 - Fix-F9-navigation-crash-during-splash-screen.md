---
id: TASK-1339
title: Fix F9 navigation crash during splash screen
status: Done
assignee: []
created_date: '2026-08-04 23:47'
updated_date: '2026-08-05 00:31'
labels:
  - navigation
  - crash
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Pressing F9 while the splash screen is active raises IndexError: pop from empty list in switch_screen via app.py:5735 handle_screen_navigation (headless repro 2/2). A fast typist hitting F9 during the ~1.5s splash crashes navigation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 F9 (and other destination keys) during splash is queued or safely ignored
- [x] #2 No exception reaches the user
- [x] #3 Pilot regression test covers navigation keypress during splash
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add Pilot regression test in Tests/UI/test_screen_navigation.py: press F9 ~0.3s into splash while splash_screen_active, assert no exception and app lands on initial screen (nav swallowed).
2. Guard handle_screen_navigation in tldw_chatbook/app.py: ignore NavigateToScreen until the initial screen has been pushed (covers splash-active and the post-splash startup window) — choose 'safely ignore' over queueing (simpler, satisfies AC1).
3. Run new test plus Tests/UI/test_screen_navigation.py and splash-related suites.
ADR required: no — routine bug fix.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Chose "safely ignore" over queueing (simpler; satisfies AC1): navigation requests that arrive before the initial screen exists are swallowed with an info log.

Root cause: all shell navigation (F7/F8/F9, Ctrl+digit layer, command palette, nav bar) funnels through `NavigateToScreen` → `handle_screen_navigation` → `switch_screen`. While the splash screen is up (or the post-splash startup push is still in flight) the screen stack has no result callback to pop, so `switch_screen` raised `IndexError` in Textual's `_pop_result_callback`.

Fix: a single guard at the top of `handle_screen_navigation` in `tldw_chatbook/app.py` returns early until `_initial_screen_pushed` is set (in `_push_initial_screen`). Guarding on the flag rather than `splash_screen_active` also covers the interleaving window after the splash `Closed` message clears that flag but before the initial screen push completes. This is the only `switch_screen` call site, so the one guard covers every destination key, not just F9.

Regression test: `test_navigation_keypress_during_splash_is_safely_ignored` in `Tests/UI/test_screen_navigation.py` presses F9 via Pilot 0.3s into a forced 5s splash (config patched so it never reads the real user config), asserts no exception, startup completes, and the app settles on its initial screen rather than Settings. Verified failing pre-fix with the exact reported `IndexError: pop from empty list`, passing post-fix.

Four existing navigation tests that call `handle_screen_navigation` directly in a simulated post-startup state needed `app._initial_screen_pushed = True` added to match the handler's new precondition; no assertions changed.

Modified files: `tldw_chatbook/app.py` (11-line guard), `Tests/UI/test_screen_navigation.py` (new regression test + 4 test precondition updates).

Verification: `pytest Tests/UI/test_screen_navigation.py Tests/UI/test_settings_category_sweep.py Tests/Utils/test_startup_polish_regressions.py Tests/UI/test_settings_splash_screen_viewer.py` → 158 passed.
<!-- SECTION:NOTES:END -->
