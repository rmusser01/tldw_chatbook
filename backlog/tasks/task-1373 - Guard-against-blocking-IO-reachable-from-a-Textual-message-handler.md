---
id: TASK-1373
title: Guard against blocking I/O reachable from a Textual message handler
status: In Progress
assignee: []
labels:
  - performance
  - testing
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Stop the freeze bug class fixed by TASK-1320 from reappearing, and fix the one
live instance a call-graph sweep found outside the mount path.

Textual runs message handlers on a serialized pump. Anything blocking that a
handler reaches — directly or through the functions it calls — stops the app
from processing clicks, keys and navigation for the duration. TASK-1320 fixed
four mount-path instances (measured stalls of 1030ms, 1140ms, and up to 300s
against an unreachable server), but nothing prevents a new one being written,
and the class was never limited to `on_mount`: a button handler that blocks
freezes the app identically.

A sweep for this found a live instance: `ChatbookCreationWizard.on_button_pressed`
calls `subprocess.run(["open"/"xdg-open"/"explorer", folder])` with no timeout,
on the pump. On macOS `open` returns promptly, but `xdg-open` is a shell script
that in several desktop environments does not return until the launched handler
exits — an unbounded block triggered by a button press.

The guard has to walk the call graph, not just each handler's body. The blocking
call is almost always a level or more down (`on_mount` -> `_refresh_chatbooks` ->
`.glob(`), and a body-only scan reports a clean result against code known to be
broken — verified: an early draft of this scan returned zero against the
pre-TASK-1320 chatbooks file.

Scope is deliberately limited to calls whose cost is user-visible. Small local
filesystem operations (`mkdir(exist_ok=True)`, `read_text` of a config file)
measure 0.049ms and 0.014ms respectively; flagging them would produce a large
noisy baseline for no benefit and invite churn with real risk, since deferring
work into a worker introduces its own failure modes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Opening the export folder from the Chatbook wizard cannot block the app, on any platform
- [ ] #2 A repo-wide test fails when a new blocking call becomes reachable from a message handler
- [ ] #3 The guard is proven to catch a known instance, not just to report a clean result
- [ ] #4 Pre-existing accepted paths are baselined individually with a stated reason, not silenced wholesale
- [ ] #5 The guard cannot be fooled by the same call name appearing in a comment or string
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Fix the wizard's `subprocess.run` so opening a folder cannot block the pump,
   with a test that fails against the current call.
2. Build the guard as an AST call-graph walk from handler entry points
   (`on_*`/`watch_*`/`action_*`) to blocking leaves, terminating at any hop that
   hands work off (`to_thread`, `run_worker`, `call_from_thread`, `set_timer`,
   `call_after_refresh`).
3. Prove the guard fails against the pre-TASK-1320 chatbooks source before
   trusting any clean result from it.
4. Baseline the remaining accepted paths one by one with reasons.

ADR required: no
Reason: adds a test-suite guard and one localized fix; no policy, framework or
storage contract changes.
<!-- SECTION:PLAN:END -->
