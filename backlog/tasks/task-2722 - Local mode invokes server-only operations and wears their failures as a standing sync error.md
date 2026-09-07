---
id: TASK-2722
title: >-
  Local mode invokes server-only operations and wears their failures as a
  standing sync error
status: Done
assignee: []
created_date: '2026-08-06 17:00'
labels:
  - schedules
  - home
  - bug
  - uat
  - local-server-split
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Full-app UAT on `origin/dev` `b0185749c`, local-only profile (no server configured, backend "Local"):

- The Schedules screen header wears a persistent **"1 sync error"** badge, and its sync strip shows `notifications.reminders.list.server requires server mode.` — i.e. the screen itself called a `*.server` operation while in local mode and then reports the predictable refusal as a sync error to the user.
- Home's active-work adapter does the same: `Home.active_work_adapter WARNING Failed to fetch server event feed for Home: notifications.feed.list.server requires server mode.` (twice per visit in the session log buffer).

A local-mode user who has never configured a server sees a standing error badge they cannot clear and did nothing to cause. Server-only feeds should be gated on the active runtime source rather than called-and-caught, and a "requires server mode" refusal in local mode should not be classified as a sync error.

Evidence: Schedules pane captures + Logs-screen warning entries, 2026-08-06 UAT session.
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. `SyncEngine.pull`/`sync_now`: return early unless the target owner is a server owner (`server:` prefix) — local owners never legitimately sync, and today's attempt is what generates the policy refusal that gets persisted as a sync error.
2. Schedules workbench `_refresh_owner_select`: for a local owner, server-sync health is not applicable — header shows "Local schedules" and the strip gets no sync errors (also hides refusals persisted before the gate existed).
3. Home `_server_event_status_fields`: catch `PolicyDeniedError` specifically — return the quiet unavailable state without a warning log.
4. TDD each layer RED→GREEN with the existing fixtures (Tests/Scheduling/test_sync_engine.py, Tests/UI/test_schedules_workbench.py, Home adapter tests).
<!-- SECTION:PLAN:END -->

## Acceptance Criteria

<!-- SECTION:ACCEPTANCE_CRITERIA:BEGIN -->
- [x] With runtime source = local and no server configured, the Schedules screen shows no sync-error badge from server-only operations.
- [x] Server-only feed/list calls are not issued while in local mode (or their local-mode refusal is classified as "not applicable", never as an error surfaced to the user).
- [x] Home renders without logging server-feed failure warnings in local mode.
- [x] Switching to server mode restores the current behavior (real failures still surface).
<!-- SECTION:ACCEPTANCE_CRITERIA:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The first fix attempt (gate SyncEngine on `server:` owners) was WRONG and reverted: five existing tests proved local-owner sync is the designed local→server push flow (pending mutations). The real defect is classification: `SchedulingServerClient` translated `PolicyDeniedError` into plain `ServerClientValidationError`, erasing the "mode refusal, not failure" signal before the engine recorded it. Fix: (1) new `ServerClientPolicyError(ServerClientValidationError)` + client translation keeps refusals typed without breaking existing catches; (2) `SyncEngine.pull`/`sync_now` treat it as "not applicable" (info log, nothing persisted); (3) the Schedules workbench filters refusal-shaped entries persisted by older builds out of the error surface (badge + strip) — real errors still surface (guard test); (4) Home's `_server_event_status_fields` catches `PolicyDeniedError` before the generic handler: quiet unavailable state, "Server events apply in server mode.", no per-visit warning. Deviation from plan step 1 documented above; plan steps 2-3 shipped as written. Tests: Tests/Scheduling/test_scheduling_server_client_policy.py (new), test_sync_engine.py (+2), test_schedules_workbench.py (+2), test_active_work_adapter.py (+1) — each watched RED first. Note: `test_conflicts_tab_renders_rows_and_resolves` fails on pristine origin/dev (pre-existing copy drift, verified in a throwaway worktree; not addressed here). Files: Scheduling/services/server_client.py, Scheduling/services/sync_engine.py, UI/Screens/scheduling/schedules_workbench.py, Home/active_work_adapter.py.
<!-- SECTION:NOTES:END -->
