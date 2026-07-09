---
id: TASK-132
title: Console UAT final integration replay
status: Done
assignee:
- '@codex'
created_date: 2026-06-21 21:36
updated_date: 2026-06-21 21:36
labels:
- console
- uat
- qa
priority: high
dependencies:
- TASK-128
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replay the completed Console UAT workstream matrix after all parallel streams merge so remaining gaps, screenshots, regression evidence, and residual risks are tracked in one final closeout.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All Console UAT workstreams are listed with merged branch or PR evidence.
- [x] #2 Focused regression commands are rerun or explicitly linked from the merged stream evidence.
- [x] #3 Rendered CDP/Textual-web screenshot evidence is checked for each screen state required by the harness protocol, or the gap is documented as a residual risk.
- [x] #4 A concise final integration summary identifies remaining Console gaps before additional feature work continues.
- [x] #5 The final integration task does not introduce new Console feature behavior beyond verification, documentation, or small regression harness fixes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reconfirm the merged Console UAT matrix against current origin/dev and record PR/branch evidence for each completed stream.
2. Run the focused regression commands from the reusable command list using the project virtualenv interpreter where the shell lacks a python shim.
3. Launch the real app through Textual-web/CDP with isolated HOME/XDG paths and live local llama.cpp at 127.0.0.1:9099 where provider-response evidence is required.
4. Replay the minimum Console states from the CDP evidence protocol, capture actual rendered screenshots under Docs/superpowers/qa/console-uat-parallelization/, and inspect them before presenting for approval.
5. Document residual gaps and risks without implementing new Console feature behavior; only apply small regression harness fixes if verification exposes one.
6. Update TASK-132 acceptance criteria, implementation notes, and the acceptance matrix after verification and approval.

ADR required: no.
ADR path: N/A.
Reason: This is a QA replay and evidence closeout. It does not change storage/schema, sync policy, provider/runtime boundaries, service contracts, security policy, or long-lived product architecture.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Replayed the merged Console UAT matrix against `origin/dev` after PR #549 (`15259871`) and updated the matrix to show merged stream evidence plus final replay status.
- Reran focused verification for Console native chat flow, session settings, generic provider fallback, Backlog ID hygiene, and `git diff --check`.
- Added a test harness helper that opens the right inspector rail before activating the Console settings summary button, because the approved default rail state keeps that control hidden until the inspector rail is visible.
- Updated stale Console settings layout assertions so they match the approved layout: settings summary belongs to the right Run Inspector rail, while staged context and workspace context remain in the left rail body.
- Captured a fresh Textual-web/CDP wide screenshot from an isolated app runtime on `127.0.0.1:18937`; visual approval was received for the rendered wide Console capture.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Final replay did not identify a new Console feature blocker. The rendered wide Console capture `Docs/superpowers/qa/console-uat-parallelization/task-132-final-replay-console-wide-cdp-2026-06-21.jpg` was approved. Residual risk: default in-app browser viewport captured only the left rail, so the final evidence uses an explicit `2048x1220` CDP viewport to show the complete Console shell.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
