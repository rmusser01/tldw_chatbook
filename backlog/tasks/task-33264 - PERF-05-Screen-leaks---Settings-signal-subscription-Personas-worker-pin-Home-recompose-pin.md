---
id: TASK-33264
title: 'PERF-05: Screen leaks - Settings signal subscription, Personas worker pin,
  Home recompose pin'
status: Done
created_date: 2026-09-28 18:02
labels:
- performance
- memory
- ui
- perf-audit-2026-09
priority: high
references:
- qa/perf-structural-audit-2026-09-27/report.md
- qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md
assignee:
- '@claude'
updated_date: 2026-09-29 00:36
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Every Settings visit retains the whole SettingsScreen, about +71k objects and +10.7 MB per visit. theme_changed_signal is subscribed in on_mount and never dropped, and Textual's Signal keeps subscribers in a WeakKeyDictionary whose bound-method values pin their keys. Departed Personas screens (6 of 10) are pinned by the thread-worker active_worker ContextVar. The reused Home screen whole-screen recomposes on every visit and its query_one cache pins the discarded trees. These leaks inflate every gen-2 GC pause. Source: the 2026-09-27 structural efficiency audit at dev 840ed2ca58 (qa/perf-structural-audit-2026-09-27/report.md, section 4, PERF-05; every issue with file:line is listed under PERF-05 in qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After 10 Settings visits and 10 Personas visits, no departed screen instance remains reachable
- [x] #2 Home visits no longer whole-screen recompose when the targeted triage sync suffices
- [x] #3 A regression test fails if a screen subscribes to an app-level Signal without unsubscribing on unmount
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Test-first: add a full-app private-profile leak test (Tests/Performance) that visits Settings and Personas 10x each (Home in between), then asserts no departed screen is reachable via weakref + gc.collect; watch it fail.
2. Add a Home test: a completed chatbook-snapshot refresh on a reused Home does not recompose the screen and syncs the triage in place; watch it fail.
3. Add a static regression guard: any tldw_chatbook class that subscribes to an app-level Textual Signal (theme_changed/screen_change/app_suspend/app_resume/mode_change) must also unsubscribe in the same class.
4. Investigate retention paths with gc.get_referrers (Settings signal, Personas active_worker ContextVar) before fixing.
5. Fix: Settings + ThemePicker unsubscribe theme_changed_signal on unmount; stop thread workers pinning their node via the pool thread's context (fix where all thread workers route through, not only ccp_character_handler); Home replaces the post-snapshot recompose with _sync_home_triage().
6. Verify: new tests, every test file touching settings_screen/personas/home_screen/ccp_character_handler, preflight; compare any failure against base c174e30f6b.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Three leak mechanisms fixed, plus the regression guards.

Settings (P0):
- on_unmount now unsubscribes theme_changed_signal. Textual's Signal keys subscribers in a WeakKeyDictionary whose bound-method values pin the key, so every visit retained the whole screen (audit: 10/10, about +10.7 MB per visit).
- The two app-lifetime writer states (_ConsoleToggleWrite, _PermissionSummaryWrite) also held the screen. They now drop it on unmount; a late result falls back to the screen that started it.
- ThemePicker gets the same unsubscribe.

Personas (P1):
- Textual 8.2.8's Worker._run_threaded sets the active_worker ContextVar in the pool thread's own context and never resets it. Each pool thread therefore kept its last Worker, and through it the screen that started it (audit: 6/10 retained, about 14 MB each).
- Fixed where every thread worker routes through: ThreadWorkerContextGuard (Utils/text_selection_crash_guard.py, composed into TldwCli as TextualAppGuards) sets the loop's default executor at Load to FreshContextExecutor, which runs each job in a fresh contextvars.Context.
- asyncio.to_thread is unaffected: it already submits a Context.run partial carrying the caller's context.

Home (P1):
- The chatbook-snapshot completion now calls _sync_home_triage() instead of refresh(recompose=True), which recomposed once per visit (about 68 ms vs 3 ms) and pinned discarded trees in the query_one cache.

Tests:
- Tests/Performance/test_screen_leaks.py boots the real app on a private profile, visits Settings and Personas 10x each, and asserts no departed instance is reachable. It also carries a static guard that fails if a screen subscribes to an app-level Signal without unsubscribing, plus that guard's own negative control.
- Tests/UI/test_home_screen.py adds the in-place sync test.

Verification:
- Leak, Home, Settings, theme-picker, text-selection-guard and boot-worker-census suites: all new tests pass.
- The 62 failures are a subset of base c174e30f6b's 63 (RecoveryRequired harness class, TASK-33370, plus diagnostic-inventory drift in untouched files).
- Architecture/perf ratchets: the identical 22 failures on branch and base.
- Preflight passes.

Built by a background agent that stopped at the usage limit before finishing notes; completed and verified by the controller.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
