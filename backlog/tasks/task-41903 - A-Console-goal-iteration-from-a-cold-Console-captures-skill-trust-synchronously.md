---
id: TASK-41903
title: "A Console goal iteration from a cold Console captures skill trust synchronously"
status: Done
assignee:
- '@claude'
created_date: 2026-11-18 20:31
updated_date: 2026-11-19 16:13
labels:
- follow-up
- console
priority: medium
references:
- tldw_chatbook/Chat/console_runtime.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41507 (round 16). `ConsoleRuntime._submit_goal_iteration` captures its configuration synchronously, as the launch wake did, so a goal iteration from a cold Console would hit the `console_capture_source_not_ready` refusal TASK-41507 fixed for wakes. Untested.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A goal iteration from a cold Console reaches the provider or shows a correct refusal, never `console_capture_source_not_ready`
- [x] #2 A test starts cold and is red before the fix
<!-- AC:END -->

## Implementation Plan

1. Reproduce cold: the goal dispatch rig (`Tests/Chat/test_console_goal_dispatch.py::build_goal_rig`) runs on a stand-in app with no MCP service or skills stack, so its capture never reads the trust slot. Give it the app's own bound MCP service and skills stack with the trust service unbuilt, and record what the runtime submitter raises.
2. Fix as TASK-41507 did for the launch wake: `ConsoleRuntime._submit_goal_iteration` awaits `publish_cold_skill_trust(controller.app)` before its synchronous capture.
3. Red at base, green after; covering goal suites base vs head; ratchets and preflight.

## Implementation Notes

**Where it refused.** Goal iterations start from the mounted Console's goal UI (launch or resume), so "cold" means a Console whose skill discovery has not built the app's skill trust service yet. `ConsoleRuntime._submit_goal_iteration` captures its configuration synchronously (`resolve_runtime_turn_configuration_snapshot`). With a stock MCP service the capture is pending, so `remember_pending_mcp_capture` reads the sources, and `_ready_attribute` refuses an empty trust slot with `RecoveryRequired('console_capture_source_not_ready')`. `_dispatch_owned` keeps only `UNKNOWN_EFFECT`; nothing reached the provider.

**Fix (`Chat/console_runtime.py`, +4).** `_submit_goal_iteration` awaits `publish_cold_skill_trust(controller.app)` before its capture, as `ConsoleFleetWakeCoordinator._deliver` does for the launch wake (TASK-41507). The helper builds only when the capture would refuse and returns at once for any other app. A failed build ends the iteration as any other pre-acceptance exception does (no change there).

**Test.** `Tests/Chat/test_console_goal_dispatch.py::test_a_goal_iteration_from_a_cold_console_builds_the_skill_trust_it_captures` (the file runs in `derived-artifacts.yml`'s Console lane). `build_goal_rig` gained an optional `app`; the test passes a stand-in carrying the app's own bound MCP service and skills scope from `_build_test_app(user_data_dir=config.get_user_data_dir())`, with the trust service unbuilt (asserted). It records what the runtime submitter raises and requires none, one provider call, a done outcome and a built trust slot.
- Red at base: `RecoveryRequired('console_capture_source_not_ready')`.
- The first version, on the rig's plain stand-in, passed at base: that stand-in has no MCP service, so its capture never reads the trust slot (lesson added). The real `TldwCli` as backend hung in the coordinator's recovery wait, so the stand-in keeps only the two services it needs.

**Runs.** The covering run is the one in TASK-41900's notes (goal dispatch, goal authority, light-surface goal runs, goal controls, send start window, at base and head): the only difference in these files is this test, red at base and green at head; the goal reds there (authority 6, controls 2) fail the same at base. `Tests/Architecture/test_module_size_ratchet.py`: the `console_runtime.py` row rises 7078 -> 7082, the change's own lines, with a dated note (owner rule). Preflight passed.

Files: `tldw_chatbook/Chat/console_runtime.py`, `Tests/Chat/test_console_goal_dispatch.py`, `Tests/Architecture/test_module_size_ratchet.py`.
