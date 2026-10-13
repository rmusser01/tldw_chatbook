---
id: TASK-34912
title: Stop-or-dirty-check-the-app-level-Buddy-scope-reconcile-timer
status: Done
assignee: []
created_date: 2026-10-13 14:44
labels:
- performance
- ui-perf
- wave-1-timers
dependencies: []
priority: high
updated_date: 2026-10-13 15:41
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
BuddyManagement.start_scope_tracking arms an App-level set_interval(0.5s, reconcile_scope) but nothing ever stops it: _scope_timer has no stop call anywhere in the repo. Once armed (any Buddy view presentation or settings save), the app-wide timer sweeps runtime sessions and rewrites sink scope 2x/second for the rest of the session with no change detection, burning idle CPU on every screen. UI/Navigation/buddy_management.py:1010-1014.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Scope timer stops or disarms when no Buddy view or scope consumer remains
- [x] Each reconcile_scope tick cheaply no-ops when nothing changed
- [x] No app-level 0.5s timer remains armed after the Buddy overlay closes (test)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read buddy_management.py scope tracking + persona_buddy_overlay arming/unmount sites
2. Stop the app-level timer when no Buddy view remains; add cheap dirty-check in reconcile_scope
3. Extend Tests/UI buddy tests for stop + no-op paths
4. Targeted test run + task notes
<!-- SECTION:PLAN:END -->

## Implementation Notes

- Added `BuddyManagementCoordinator.stop_scope_tracking()` (buddy_management.py): clears the timer, runs one final `reconcile_scope()` to settle speech/sink state for the closed presentation, then stops the timer; idempotent.
- `reconcile_scope()` now builds a cheap fingerprint (runtime object identity, binding, speech/presentation flags, per-session (id, persisted_conversation_id) tuple) and returns early when it is unchanged. Runtime identity preserves the rebind-replay contract; persisted ids catch session promotion. Promotion retries are deadline-driven and explicitly bypass the gate so the timer keeps its retry semantics.
- Stop sites, both under the overlay lock so no concurrent reconcile can re-arm afterwards: `PersonaBuddyOverlay.reconcile()` not-desired branch (view retired) and `PersonaBuddyOverlay.shutdown()` (permanent close). Re-arming is unchanged (`start_scope_tracking` on presentation + settings save). The stop call uses `getattr(app, "_buddy_management")` so it never materializes the coordinator lazily.
- Tests: new `Tests/UI/test_buddy_scope_tracking.py` (7 tests: stop/idempotency, unchanged no-op gate, rebind replay, session-change follow, promotion bypass, overlay retire stop, overlay shutdown stop) — 7/7 pass. Regression: `Tests/UI/test_buddy_speech.py` 14/14, `Tests/UI/test_persona_buddy_app_mount.py` 29/29 (full-app overlay lifecycle). `test_persona_buddy_host.py` (stale import of a deleted module) and `test_persona_buddy_widget.py` (config-load setup error) fail identically with the change reverted — pre-existing tree breakage, unrelated.
- ADR required: no — routine perf fix preserving existing boundaries and contracts.
