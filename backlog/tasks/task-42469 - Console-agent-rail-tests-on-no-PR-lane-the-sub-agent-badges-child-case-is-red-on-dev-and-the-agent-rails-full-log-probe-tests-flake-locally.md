---
id: TASK-42469
title: "Console agent-rail tests on no PR lane: the sub-agent badge's child case is red on dev, and the agent rail's full-log probe tests flake locally"
status: To Do
assignee: []
created_date: 2026-11-23 06:38
updated_date: 2026-11-23 07:09
labels:
- follow-up
- console
- agents
- tests
- dev-red
priority: medium
references:
- Tests/UI/test_subagent_badge_primary_token.py
- Tests/UI/test_console_agent_rail.py
- Tests/UI/test_console_agent_controller.py
- scripts/ui_pr_gate_census.txt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 22 of the follow-up burn-down (2026-11-23), in TASK-42426's covering runs. Checked on the round-22 stack 336d15338b (dev 2f7790873a plus round 22's stacked branches); paths start at the repository root.

**Why.** Neither file is in `scripts/ui_pr_gate_census.txt` or a `derived-artifacts.yml` step, so their reds and flakes report nowhere.

**Sub-agent badge (red every time).** `Tests/UI/test_subagent_badge_primary_token.py::test_real_badge_count_work_tracks_subagents_not_primary_publication[child]` fails at base and on TASK-42426's branch, and again on the round-22 stack before its rebase (142e821116, `-n 4`): `TypeError: a coroutine was expected, got <function _controller.<locals>.repaint ...>`. It is raised by the test's `schedule` double (`asyncio.create_task(awaitable)`, line 184), reached from `_settle` (line 213). The `_controller` double (line 178) defines `async def repaint()`, but `schedule` is now handed the function itself rather than a call of it. Whether the product's scheduling call or the double changed is not known. The other three cases (`primary_only`, `ttl` and `database_owner`) pass. The file last changed in 3e618d7119 (2026-10-15).

**Agent rail full-log probes (flaky locally).** `Tests/UI/test_console_agent_rail.py` (46 tests) fails locally on the known `RecoveryRequired: raw_source_selection_changed` trap, which is out of scope here. Under a temporary `bootstrap_profile` mark, TASK-42426 ran it with `test_console_agent_controller.py` and `test_console_fleet_historical_detail.py`: 12 failures on its branch and 13 at base, 10 of them the same. The rest were full-log probe tests in `test_console_agent_rail.py` (the "View full log" availability tests from line 846, such as `test_full_log_probe_is_cached_per_run_id_while_open` and `test_full_log_slow_probe_is_off_thread_and_rejects_stale_target`) whose failures changed from run to run: two more branch runs failed 11 and 9. The node ids of the 10 consistent failures were not recorded.

**Earlier record.** TASK-32470 (open, medium, 2026-09-11, "Fix dev-tip test harness drift in console suites") already lists `test_console_agent_controller.py`'s `test_agent_bridge_is_*` tests as harness drift (its SimpleNamespace lacks `get_message_exchanges` and `capture_console_settings_origin`). Some of the 10 consistent failures may be those; AC#3 names them either way.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The badge `[child]` red has a recorded cause (test-double drift after a named commit, or a product defect) and is fixed: a defect in product code with a test shown red first, drift in the test with its assertions kept
- [ ] #2 The full-log probe tests that flake are named by node id, each with its race named and fixed where it lives (product race in product code, test synchronisation in the test), and they pass in repeated runs of the file under parallel load (at least 20 runs, or a soak recorded in the notes)
- [ ] #3 The three files' consistent failures under the bootstrap mark are named by node id in the notes, each with a cause or with the reason it is an artifact of the mark
- [ ] #4 The badge file and the agent-rail file run in a PR lane (their green nodes at least), with their serial time under the lane's flags recorded in the notes
<!-- AC:END -->
