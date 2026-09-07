---
id: TASK-2720
title: >-
  Escaped screen-navigation exception silently kills all tab navigation until
  restart
status: Done
assignee: []
created_date: '2026-08-06 17:00'
labels:
  - navigation
  - bug
  - uat
  - robustness
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Full-app UAT walkthrough on `origin/dev` `b0185749c` (fresh profile seeded with a copy of real DBs, first-run wizard completed, 235x52 tmux): clicking the Console nav tab raised `worker_failed exception_type=PermissionError operation=handle_screen_navigation` (16:10:16), and a Library click two minutes later failed identically (16:12:17). Both were single sanitized lines in the persistent log; the raising site is unknown (ADR-029 keeps tracebacks off disk, and the escaping exception bypasses the `logger.opt(exception=True)` guards, so nothing reached the in-app Logs buffer either).

The trigger did not reproduce: three faithful replications (same wizard walk, same clicks; one with a fully reset profile, pending v30→v31 migrations, and purged worktree `__pycache__`) all navigated cleanly, and sandbox/write probes at the failing paths passed. Transient cross-process contention is the leading suspect (POSIX `fcntl` lock conflicts surface as `EACCES` → Python `PermissionError`; other app instances run concurrently on this machine), but that is unproven.

What IS proven is how badly the app degrades when any exception escapes the navigation worker, and every piece of it is app-side:

1. **Silent failure.** `_notify_navigation_failure` only covers screen construction and `switch_screen`; an exception at any of the unguarded steps in `_handle_screen_navigation_locked` / `_complete_screen_navigation` (`_resolve_screen_navigation_target`, `_current_runtime_identity`, `screen_state_store.restore`, `acquire_navigation_transition()`) escapes to the worker (`exit_on_error=False`), and the user sees nothing at all.
2. **Tab-bar desync.** The nav-bar highlight had already moved to the destination, so the bar said Console while the body remained Home.
3. **No retry.** Because the nav state believed Console was active, re-clicking Console became a no-op — the destination was unreachable for the rest of the session (observed live: second Console click produced no worker attempt at all).

One PermissionError from any transient cause therefore converts to "this tab is dead until app restart", with no message. The nav worker body should be guarded so ANY escaping exception rolls back the tab highlight, surfaces `_notify_navigation_failure`, and leaves the destination re-clickable.

Evidence: scratch-profile persistent log lines at 16:10:16 / 16:12:17 2026-08-06; pane captures showing Console highlighted over Home content.
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: three tests — escaped exception (injected at `_current_runtime_identity`, an actually-unguarded step) must notify + roll the nav bar back; a dispatched failure must still record `worker_failed` (guard re-raises); widget-level `restore_active` must undo the optimistic highlight and make the destination re-clickable.
2. GREEN: `MainNavigationBar.restore_active(route)`; `_notify_navigation_failure` additionally rolls the current screen's bar back to the on-stack route; `handle_screen_navigation` wraps the locked body — recover, then re-raise for ADR-029 diagnostics.
<!-- SECTION:PLAN:END -->

## Acceptance Criteria

<!-- SECTION:ACCEPTANCE_CRITERIA:BEGIN -->
- [x] An exception raised at any point inside the navigation worker (including target resolution, runtime-identity read, snapshot restore, and transition admission) results in a user-visible failure notification.
- [x] After a failed navigation, the nav-bar highlight reflects the screen actually on the stack (no split-brain).
- [x] After a failed navigation, clicking the same destination again issues a fresh navigation attempt (no dead tab).
- [x] A regression test injects an exception into an unguarded step and verifies notification + highlight rollback + retryability.
- [x] The `worker_failed` diagnostics line for navigation failures is preserved (ADR-029 sanitization unchanged).
<!-- SECTION:ACCEPTANCE_CRITERIA:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root mechanism (from the live incident): the nav bar activates a destination optimistically at click time, and its already-active check then swallows every retry once the navigation worker dies — one transient exception made a tab unreachable until restart, silently. Fix in three parts: (1) `MainNavigationBar.restore_active(route)` re-points highlight + active ids at the route actually on the stack; (2) `_notify_navigation_failure` now performs that rollback too, which also repairs the pre-existing desync on the already-guarded construction/mount failure paths; (3) `handle_screen_navigation` catches anything escaping the locked body, recovers the user-facing state, and re-raises so the `worker_failed operation=handle_screen_navigation` diagnostics line keeps being written (pinned by test). The transient PermissionError trigger itself remains unreproduced (3 faithful replications); this hardens the degradation path per the ACs. An always-green app-level retry test was deleted during RED verification — app-level retry was never broken; retryability lives in the bar and is pinned at the widget level. Files: tldw_chatbook/app.py, tldw_chatbook/UI/Navigation/main_navigation.py, Tests/UI/test_screen_navigation_failure_recovery.py.
<!-- SECTION:NOTES:END -->
