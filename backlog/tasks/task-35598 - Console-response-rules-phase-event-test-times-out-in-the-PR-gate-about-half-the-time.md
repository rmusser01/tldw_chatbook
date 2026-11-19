---
id: TASK-35598
title: Console response-rules phase-event test times out in the PR gate about half
  the time
status: Done
created_date: 2026-10-19 18:04
updated_date: 2026-10-21 07:19
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
test_console_response_rules.py::test_rule_phase_events_project_only_the_mounted_current_session has timed out on 4 of ~7 PR-gate runs observed across three different PR heads on 2026-10-18/19 (variants [rebuild] twice and [maintenance] once; each time green on rerun). Seen on PR #3140's runs (39152595713, 39160862810, 39275849597, 39287157642) — never related to that PR's diff (zero overlap). It was added to the PR-gate census recently (TASK-35444). Under runner congestion it hits the private-profile timeout machinery; suspect a hang in phase-event projection during maintenance/rebuild phases that only manifests under load.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Repro or hang-site identified under load,Test passes reliably in the PR gate or is quarantined with a tracked reason
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Pull the failing PR-gate jobs' logs and read the private-profile child's blocked-stack dump
2. Identify the hang site and mechanism; check whether a fix already landed (TASK-35645/35836 merged 2026-10-20, after the last failure)
3. Verify the fix covers the exact stack path; check post-fix gate runs for this test
4. Loop-run the full parametrize >=5x locally under CPU load
5. Close as fixed-elsewhere with evidence, or fix/quarantine if still flaky
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- **Diagnosis (from the CI child's own blocked-stack dump, jobs 118189348984 [maintenance] and 117749113055 [rebuild]):** the test's watchdog printed the UI loop blocked at `tldw_chatbook/Backup_Recovery/config_participants.py:427` (`while not lock.acquire(timeout=0.05)`) inside `load_settings()` -> `operation()`'s REBUILD-then-FILE two-lock acquire, reached from the UI loop via `_sync_native_console_chat_ui` -> `_current_console_rail_state` -> `_build_console_inspector_state` -> `_active_console_provider_model_display` -> `_build_console_provider_selection` -> `_provider_readiness_app_config` -> `load_settings()` (config.py:2194 on those heads).
- **Mechanism:** the test holds `_settings_rebuild_lock()` in a helper thread on purpose (to prove phase-event projection never waits for a config rebuild) and releases it via a 2s watchdog. On the pre-fix heads, Stop's UI sync rebuilt the provider selection with a live `load_settings()` ON the UI loop, which spun in the lock-acquire loop until the watchdog fired; the test's own 1s poll after `stop.press()` (`async with asyncio.timeout(1)`) expired because the loop never got a turn -> TimeoutError. Not a projection hang; a config-read-on-UI-loop hang exposed by this test's deliberate lock hold.
- **Resolution: fixed elsewhere, after the last failure.** TASK-35645 (PR #3180, merged 2026-10-20 15:57 PT) + TASK-35836 made every Console refresh read defer/own config instead of waiting out a rebuild (`_console_readiness_projection_active` serves the projection's owned config during sync passes; mount refreshes retry). All 4 failing runs are 2026-10-18/19, i.e. strictly pre-fix. A covering pin, `test_console_syncs_never_wait_for_a_config_rebuild[full-ui-held]`, drives the exact failing path (`_sync_native_console_chat_ui` under a held rebuild lock) and is itself in the census.
- **Evidence test passes reliably now:** post-fix gate run 39542773944 all 10 UI Fast Lane lanes green (census includes this test); post-fix run 39539214227's only lane failure was a different test (`test_console_rename_consistency`), our test passed in its lane. Locally on this worktree (dev @ 906497d9b4): full parametrize 8/8 PASS under 16 busy-loop threads on 18 CPUs (plus 4/4 baseline), file suite 17/18 -- the one failure (`test_workspace_command_opens_same_manager_in_registered_scope`) passes in isolation, is NOT in the census, and is untouched by this task (pre-existing, zero code changes made here).
- **No code changes; no census quarantine needed.** ADR required: no (investigation-only; the architectural call was made under TASK-35645/35836).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
