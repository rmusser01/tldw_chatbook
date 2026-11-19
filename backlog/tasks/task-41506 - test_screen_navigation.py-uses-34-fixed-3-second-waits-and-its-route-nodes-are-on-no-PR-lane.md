---
id: TASK-41506
title: "test_screen_navigation.py uses 34 fixed 3-second waits and its route nodes are on no PR lane"
status: Done
assignee:
- '@claude'
created_date: 2026-11-17 05:27
updated_date: 2026-11-18 08:21
labels:
- follow-up
- tests
- navigation
priority: medium
references:
- Tests/UI/test_screen_navigation.py
dependencies:
- TASK-41127
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41127 (libtests, round 15). The file holds 34 fixed 3 s waits plus `_wait_for_initial_screen`; at load 85 its 12-node slice took 96.1 s serial, and its route nodes were left off lane.

2026-11-18, from the TASK-36952.3 media series' wide A/B (chain over dev 5e3c8b2bd4, 2026-11-17, before TASK-41127's wall-clock waits): five real-app route nodes failed on base and head alike against the fixed 3 s Library wait: `test_media_route_round_trips_to_the_library_media_row`, `test_search_route_round_trips_to_the_library_rag_row`, `test_boot_with_prompts_default_tab_lands_on_library_with_prompts_row_selected`, `test_prompt_receipt_owner_vetoes_real_app_navigation_until_settlement` and `test_research_workspace_runs_round_trip_restores_independent_context`. Over six interleaved serial rounds at load 70-160, the search route failed 2 of 6 on head and 1 of 6 on base, and the media route 2 of 6 on head and 3 of 6 on base. TASK-36952.3.7's reruns also saw `test_search_all_palette_command_lands_on_library_with_honest_toast` stay on ChatScreen with base code, landing 0 of 4 times even with the wait raised to 30 s, so that node has a navigation race beyond the fixed wait.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Fixed waits are replaced by condition waits with a bounded deadline
- [x] #2 The route nodes run on a PR lane, with their measured serial time recorded
- [x] #3 The file's wall-clock time at load under 40 falls measurably, with before and after in the notes
<!-- AC:END -->

## Implementation Plan

1. Confirm at base: the 34 `range(150)` loops were already converted to `_settle` by TASK-41127 review round 1 (942ee7fe52); list the fixed waits that remain and the route nodes still off lane.
2. Replace the remaining fixed waits that gate an async outcome (fixed-count pause loops, `asyncio.sleep`, `pause(0.5)`) with `_settle` condition waits.
3. Cut the wall clock: real-app nodes that do not test the splash skip the 7 s boot splash (`_cli_no_splash`, the `--no-splash` seam other test files already use); both boot paths run `_push_initial_screen`.
4. Lane the off-lane route nodes that pass (not TASK-41502's Prompts nodes or TASK-23149's two), with serial lane-flag times.
5. Measure the whole file's wall clock before and after at `-n 4`, load recorded.

## Implementation Notes

**AC#1: fixed waits.** At this base, TASK-41127 review round 1 (942ee7fe52) had already replaced the 34 `range(150)` × `pause(0.02)` loops and `_wait_for_initial_screen` with `_settle` (a 15 s wall-clock condition wait). The fixed waits left after it are now condition waits:
- the storm's 200-pause loop and the research workspace's 100-pause loop;
- the shielded-save test's `asyncio.sleep(1.0)`;
- the nav-overflow tests' four `pause(0.5)`;
- the splash test's pauses and its 60-pause loop, and the mcp boot's single 20 s deadline (both in TASK-41505's line).

Storm and FIFO now wait for `_initial_screen_pushed`, not just for the screen type, because a `NavigateToScreen` posted before it is dropped. Left as they are: single `pilot.pause()` layout waits; the `pause(0.1)`/`pause(0.05)` in harness tests with a stubbed `switch_screen` or a compose-only nav bar, where there is nothing async to wait for; and the deep-link nodes' hand-rolled loops, which are already bounded condition waits.

**Wall clock.** After the conversion, most of each real-app node's time was the 7 s boot splash. `_build_unsplashed_test_app` sets `_cli_no_splash` (the `--no-splash` seam three other test files use). Both boot paths run `_push_initial_screen`. Every real-app node that boots and waits uses it, except three: the splash test, the two Prompts nodes (TASK-41502 owns their boot race), and the stubbed-switch nodes.

**AC#2: lane.** Two route nodes were still off lane:
- `test_search_route_lands_on_library_rag_canvas` (green since TASK-41500);
- `test_legacy_alias_explicit_context_merges_default_under_caller_keys`.

They go at the census tail, with TASK-41501's four nodes. Each passed 3 serial lane-flag rounds at load 14-21, call times 3.1-3.3 s and 3.7-4.2 s; the search route also passed 10 of 10 at `-n 4`. TASK-41502 lanes the two Prompts route nodes. The ten other route nodes from TASK-41127's list were laned in its review round 1.

Not laned:
- `test_mcp_deep_link_with_explicit_context_lands_on_tools_hub` and `test_tools_settings_deep_link_with_profile_payload_lands`. Over 44 head runs at `-n 4` (load 11-15), one failed at teardown on `WorkerFailed: NoMatches 'MCPServersMode'`. `MCPWorkbench._disarm_canvas_delete` runs as a worker after a mode change, and it queries the canvas after awaiting `_sync_children_lock`. Base passed 20 of 20, which proves nothing either way. Proposed as a follow-up.
- TASK-23149's two `_FakeMediaRowButton` nodes, which are red.

**AC#3: whole-file wall clock**, `-n 4`, base and head run alternately in one session:

| Run | Wall | Result | Load |
|---|---|---|---|
| base 1 | 109 s | 4 failed, 144 passed | 13 -> 12 |
| head 1 | 64 s | 2 failed, 148 passed | 12 -> 14 |
| base 2 | 105 s | 3 failed, 145 passed | 14 -> 11 |
| head 2 | 59 s | 2 failed, 148 passed | 11 -> 12 |

The head failures are TASK-23149's two nodes, red at base too. Base also fails the storm both times, and the Prompts route node once. Route nodes that took 9-11 s each now take 3-5 s.

**Files.** `Tests/UI/test_screen_navigation.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`.
