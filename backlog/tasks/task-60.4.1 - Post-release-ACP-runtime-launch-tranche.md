---
id: TASK-60.4.1
title: Post-release ACP runtime launch tranche
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-22 12:56'
labels:
  - post-release
  - acp
  - runtime
  - ux
dependencies: []
parent_task_id: TASK-60.4
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Plan and implement the ACP runtime launch path only after the post-release actual-use audit evidence shows ACP remains recoverably blocked rather than broken by UI affordance.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ACP launch requirements are derived from TASK-60.3 actual-use audit evidence.
- [x] #2 ACP unavailable states remain source-honest until a real runtime can be launched.
- [x] #3 Console, ACP, Home, and relevant handoff surfaces expose consistent runtime readiness and recovery copy.
- [x] #4 QA verifies the ACP task/run package path with actual app use before completion.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add TDD coverage for ACP process states: missing config, configured launchable runtime, successful launch payload, failed launch, Console readiness, and Home readiness.
2. Implement an ACP-owned runtime process manager with shell-free subprocess launch, status transitions, stop/restart hooks, timeout/error reporting, and session payload generation.
3. Wire `TldwCli` to own one ACP process manager instance and expose runtime/session state helpers without moving ACP setup into Settings.
4. Update `ACPScreen` controls and copy so Launch, Stop/Restart, and Follow in Console reflect real process state.
5. Update Console and Home readiness to consume the same ACP runtime state.
6. Update QA evidence and task notes, run focused tests plus diff checks, then capture actual textual-web/CDP screenshots for approval.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added an ACP-owned runtime process manager that launches configured runtimes with `subprocess.Popen(..., shell=False)`, records a Console-followable session payload, and reports configured, running, failed, stopped, and not-configured states.
- Wired `TldwCli`, ACP, Console, and Home to the same ACP runtime/session state so readiness and recovery copy do not contradict across screens.
- Reworked ACP into the approved destination-native three-column layout with Schedules-style framed columns, visible session selection, actionable runtime controls, and hidden launch/restart actions while a runtime is already running.
- Added focused regressions for runtime launch/failure/stop state, ACP screen hierarchy, Console source readiness, ACP primary action routing, and Home ACP readiness.
- Captured and received user approval for the actual textual-web/CDP running-state screenshot at `Docs/superpowers/qa/product-maturity/post-release-ux-hci/acp-schedules-style-polish-final-2026-05-22.png`.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
ACP now has a real local runtime launch path behind ACP-owned configuration. Missing runtime remains honestly blocked, configured runtime can launch into a running session payload, Console can follow that ACP session, and Home/Console/ACP all reflect the same runtime readiness.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] Acceptance criteria checked.
- [x] Implementation plan followed.
- [x] Focused runtime, ACP, Console, and Home regressions pass.
- [x] `git diff --check` passes.
- [x] Actual textual-web/CDP screenshot captured and approved.
- [x] QA evidence added for the ACP runtime-launch tranche.
<!-- DOD:END -->
