---
id: TASK-70.3
title: Add manual Sync v2 preview and execution plan
status: Done
assignee: []
created_date: ''
updated_date: 2026-06-02 00:25
labels:
- sync
- sync-v2
- manual-sync
- ux
dependencies: []
documentation:
- Docs/superpowers/specs/2026-05-26-chatbook-sync-v2-completion-roadmap-design.md
parent_task_id: TASK-70
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a manual Sync v2 workflow that lets users preview pending Notes and Chat work, explicitly start sync, and understand success, partial success, blocked, failed, or conflicted outcomes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Users can preview outgoing Notes and Chat work before any mutation request is sent to the server.
- [x] #2 Users must explicitly start manual sync; no app-launch, scheduled, or background mutation loop is introduced.
- [x] #3 Push, pull, partial failure, conflict, blocked, and success outcomes are visible in user-facing copy and profile status.
- [x] #4 Actual app QA verifies the visible manual sync workflow in every touched surface.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a Sync_Interop manual sync controller that builds a preview from the existing profile state and pending Notes/Chat outbox entries without calling server mutation APIs.
2. Add explicit run orchestration that delegates to LocalFirstSyncService.sync_once only after a user-triggered call and maps success, partial failure, conflict, blocked, and failed outcomes into user-facing copy.
3. Wire the controller into app startup alongside ServerSyncService/SyncScopeService without introducing app-launch, scheduled, or background sync.
4. Surface the preview/run/result state in a minimal Settings sync section entry point, preserving Settings as a config/status hub rather than hiding sync execution in generic copy.
5. Add focused unit and mounted UI regressions, then capture actual rendered QA evidence for any touched UI.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `ManualSyncControlService` to build a local-only Notes/Chat outbox preview and run exactly one Sync v2 cycle only after explicit user action.
- Wired local-first sync and manual sync control services during app startup without adding app-launch, scheduled, or hidden background mutation loops.
- Added a Settings Overview `Manual Sync v2` block with status, preview/result rows, pending outgoing counts, and explicit preview/run actions.
- Added focused unit and mounted UI regressions for preview counting, explicit run mapping, blocked preflight states, app service wiring, and Settings row rendering.
- Verification: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Sync_Interop/test_manual_sync_control.py Tests/UI/test_settings_configuration_hub.py Tests/UI/test_screen_navigation.py::test_app_initializes_watchlists_and_notifications_services --tb=short` passed with `142 passed, 8 warnings`.
- Verification: `git diff --check` passed.
- Visual QA: user inspected the live app in the Settings screen and approved the visible `Manual Sync v2` workflow. Automated local screenshot capture was attempted, but macOS focus repeatedly captured Codex instead of the Terminal TUI, so the stored screenshot artifact is not included as evidence.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
- Manual Sync v2 now has a Settings Overview entry point for read-only preview and explicit one-shot run control.
- The workflow blocks unsafe runs when profile, dataset identity/key, or local apply-store prerequisites are missing, and maps sync results to user-facing `success`, `partial-failure`, `conflict`, `blocked`, and `failed` copy.
- Focused regressions cover the controller, app wiring, and Settings UI state.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
