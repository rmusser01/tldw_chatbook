---
id: TASK-24200
title: Isolate Console raw CLI worker group
status: Done
assignee:
  - '@codex'
created_date: '2026-08-29 13:08'
updated_date: '2026-08-29 13:12'
labels:
  - console
  - concurrency
  - tests
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent the Console raw-CLI worker adapter from forwarding potentially exclusive work into Textual’s shared default worker group, where unrelated workers on the Chat screen can cancel one another.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Console raw-CLI work is always forwarded with an explicit work-specific group
- [x] #2 The repository exclusive-worker group inventory passes without allowlisting the adapter
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A. Reason: localized correction to an existing Console worker-admission adapter; no new runtime boundary or scheduling policy. Use the failing architecture inventory as the red gate, add a wiring regression that observes the exact run_worker arguments, explicitly bind the adapter to the console-raw-cli work group, run the exact regression plus containing wiring/raw-CLI and architecture tests, then static checks and close.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Bound the Console raw-CLI worker adapter to the explicit console-raw-cli group before forwarding caller options to ChatScreen.run_worker. This keeps current non-exclusive raw-command workers behaviorally unchanged while ensuring any future exclusive use cannot enter Textual's shared default group and cancel unrelated Chat work. Added a wiring regression that observes the exact forwarded arguments. ADR required: no; ADR path: N/A. Verification: raw-CLI wiring regression plus full repository exclusive-worker inventory, Console-specific worker-group guard, and raw-CLI send suite: 40 passed in 10.82s; Ruff check passed; compileall passed; git diff --check passed. Ruff format baseline ratchet is unchanged: wiring.py is format-red at HEAD and current, while the touched test file is formatted. The full wiring file also exposed two independent stale characterizations, recorded as TASK-24201 for immediate follow-up.
<!-- SECTION:NOTES:END -->
