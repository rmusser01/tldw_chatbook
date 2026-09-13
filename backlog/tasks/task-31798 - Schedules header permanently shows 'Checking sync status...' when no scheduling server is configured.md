---
id: TASK-31798
title: Schedules header permanently shows 'Checking sync status...' when no scheduling server is configured
status: Done
assignee: []
created_date: '2026-09-05 19:15'
labels:
  - bug
  - ui
  - schedules
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the 2026-09-05 pre-release live UAT sweep (fresh scratch profile, dev tip 8e9d1128d4, real tmux-driven app). With no scheduling server, the Schedules DestinationHeader keeps its compose-time seed 'Checking sync status...' indefinitely (verified 20+ minutes and across navigations) while the same screen's footer correctly says 'Local schedules - no scheduling server connected; sync is off.' Source lead: UI/Screens/scheduling/schedules_workbench.py:586 seeds the label; the refresh at ~line 4315 that would set 'Local only - no server connection' never runs on this path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The header resolves to the local-only status (matching the footer) shortly after mount when no server is configured.
- [x] #2 Test covering the no-server header path.
<!-- AC:END -->

## Implementation Plan

1. Reproduce on current dev; identify why the header stays on the compose-time "Checking sync status…" seed.
2. Trace `_refresh_owner_select`: `server_reachable is None` is the "still checking" signal, but the mount-time reachability probe can leave it `None` forever.
3. Add a RED test that pins the settled stuck state, then fix, then confirm GREEN + live.

## Implementation Notes

**Root cause.** A fresh LOCAL profile ships the placeholder `[tldw_api] base_url = http://127.0.0.1:8000`, so `derive_configured_server_binding` makes `runtime_state.active_server_id` truthy even though the app is in local mode. `SchedulesWorkbench.on_mount` runs `_refresh_owner_select()`, then kicks `_refresh_server_reachability()`. The probe (`SchedulingService.refresh_server_reachability` → `get_capabilities`) is `_enforce`-gated on a server permission; in local mode it raises `ServerClientPolicyError`, whose handler deliberately leaves `_server_reachable` **unchanged** (`None`) and only records `_server_permission_denied = True`. `_refresh_owner_select` treated `server_reachable is None` as "a probe is still in flight" → the header sat on the transient "Checking sync status…" copy permanently, while the footer correctly read local-only (it only gates on `not server_available`).

**Fix.** In `_refresh_owner_select`, inside the `not server_available` / `server_reachable is None` branch, distinguish "probe still pending" from "probe completed and could not establish a usable connection" using the existing honest signal `service.server_permission_denied`. `server_reachable` can only stay `None` after a completed probe via the `ServerClientPolicyError` path, which always sets `server_permission_denied` — so it is the exact settled-state signal. When set, resolve the header to `"Local only — no server connection"` (the same copy the `not active_server_id` branch and the footer already use). The genuinely-pending window (permission-denied still False) keeps painting "Checking sync status…", so the existing `test_header_paints_checking_not_a_false_unreachable_during_mount_probe` is unaffected.

**Tests.** Added `test_header_resolves_local_only_when_probe_is_policy_refused` (RED→GREEN) modelling the exact settled state (truthy `active_server_id`, `server_reachable=None`, `server_permission_denied=True`, no sync errors).

**Live verification** (isolated scratch profile, tmux, current dev): with the placeholder `[tldw_api]` URL configured (truthy `active_server_id`), the header reads "Local only — no server connection" and the footer reads "Local schedules — no scheduling server connected; sync is off." — they match, and no "Checking sync status…" persists.

**Files:** `tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py` (fix), `Tests/UI/test_schedules_workbench.py` (regression test).
