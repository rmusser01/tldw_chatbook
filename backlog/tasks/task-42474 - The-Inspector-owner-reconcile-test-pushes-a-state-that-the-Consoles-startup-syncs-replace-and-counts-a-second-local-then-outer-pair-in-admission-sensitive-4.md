---
id: TASK-42474
title: "The Inspector owner reconcile test pushes a state that the Console's startup syncs replace, and counts a second local-then-outer pair in admission-sensitive 4"
status: Done
assignee:
- '@claude'
created_date: 2026-11-23 08:52
updated_date: 2026-11-23 09:27
labels:
- follow-up
- ci
- console
- tests
- flaky
priority: high
references:
- Tests/UI/test_console_rail_reconciliation.py
- tldw_chatbook/UI/Screens/chat_screen.py
- tldw_chatbook/UI/Console_Modules/right_rail.py
- tldw_chatbook/Widgets/Console/console_settings_summary.py
- .github/workflows/derived-artifacts.yml
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 22 of the follow-up burn-down (2026-11-23), in the round's final gate. Checked on the round-22 stack a999508ae7 (dev fe8c87e3e2 plus round 22's stacked branches); paths are under `tldw_chatbook/` unless they start with `Tests/` or `.github/`.

**Why.** `Tests/UI/test_console_rail_reconciliation.py::test_inspector_descendant_owners_reconcile_local_then_outer[settings]` runs in the required "Run admission-sensitive suites 4" step (`.github/workflows/derived-artifacts.yml:962`). In the final gate's local run of that step (2026-11-22 23:50Z, 598 passed, 1 failed) it failed at line 2029:

```
assert events.count("outer") == 1
assert 2 == 1   where events == ['local', 'outer', 'local', 'outer']
```

The test opens the Inspect rail, waits five bare `pilot.pause()` calls, stops the transcript poll, starts recording local (`session-settings` section) and outer reconcile requests, calls `ConsoleSettingsSummary.sync_state` with a probe state, waits ten more pauses, and asserts exactly one local-then-outer pair. A second pair means something else synced the settings owner, or reconciled its section and the rail, inside that window.

A serial A/B at load 7-10 (each run covers the three parameters) failed 4 of 28 runs at dev and 9 of 28 at the stack top, nearly always in `[settings]`. That difference is not significant, but round 22 changes Console code (`UI/Console_Modules/agent.py`, `Notes/agent_lessons.py`), so whether it contributes has to be settled.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The origin of the second local-then-outer pair is named with probe evidence (call stacks of every owner sync and reconcile request in the window), and the notes say whether round-22 code adds or delays it
- [x] #2 Delaying that origin into the observation window makes the unfixed test fail every time, and the fixed test passes under the same delay
- [x] #3 The test starts observing only after the open Inspect rail and the Console's own startup syncs have been quiet on consecutive checks, a later Console sync cannot undo the state it pushes, and its count assertions are unchanged and still fail when the owner adds a redundant late pair
- [x] #4 The file's other parametrized cases and the sibling test with the same open-then-settle shape are checked under the same load and fixed the same way where they are exposed
- [x] #5 The file passes 20 runs at -n 4 under load and 5 serial runs with the lane's flags, and the admission-sensitive 4 step passes once
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Probe (scratch pytest plugin, no worktree file): log the stack and queue-time origin of every `ConsoleSettingsSummary.sync_state`, `session-settings` reconcile request and rail outer request, the credential poll's gates, the Console native syncs and console-sync workers, and count calls into round 22's AgentRuns guard and agent-lessons seed. Run it at dev and at the stack top, under normal and background QoS, until failures are captured on both or one side.
2. Forced delay: hold the named origin and release it as the window opens; show the base test red every time, then the fixed test green under the same holds.
3. Fix where the race lives. If the second pair is a product sync that correctly re-applies the screen's state, it is a test synchronisation gap: wait for the rail and the Console's startup syncs to be quiet on two consecutive checks (TASK-42415, TASK-42148), and hand the probe state to the screen's builder so a later sync republishes it (TASK-42149). Keep every assertion; show a redundant late pair still fails.
4. Check `[sources]`, `[run]` and `test_chat_screen_inspector_mutation_paths_delegate_one_owner_request` under the same load; fix them the same way where exposed.
5. Soak: file 20x at -n 4 and 5x serial with the lane flags; the admission-sensitive 4 step once; preflight and `ruff check --select F`.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Cause: the Console's own startup syncs, landing in the observation window; a test synchronisation gap, not a product defect.** After the five-pause settle the Console was often still converging: syncs deferred on a cold config come back 0.2 s later, coalesced control-bar replays back off, and the 0.25 s credential poll makes its first full pass only once attach has reconciled and config is warm. Each of these re-syncs the Inspector owners from the screen's own builders (`_context_spend._build_console_settings_summary_state`, `_build_console_inspector_state`). The test had pushed a probe state straight into the owner, so the late sync legitimately put the screen's state back, which queues the owner's own local-then-outer request a second time. Stopping `_console_transcript_sync_timer` stops none of these.

**Probe evidence.** A temporary pytest plugin (scratch only) logged the stack and queue-time origin of every `ConsoleSettingsSummary`/`ConsoleRunInspector`/`ConsoleStagedContextTray.sync_state`, every `session-settings` local and rail outer request, every native Console sync, console-sync worker and credential-poll tick, and counted calls into round 22's code.
- Normal QoS at load 7-10: 0 of 12 runs failed at each side with the probe on.
- Background QoS (`taskpolicy -b`), load 8-12, serial: dev fe8c87e3e2 7 of 18 runs failed, stack top a999508ae7 8 of 18. 11 failures were `[settings]`, with CI's exact `['local', 'outer', 'local', 'outer']`, and 4 were `[run]`, 3 of them with the list CI's `[run]` failure printed.
- The second pair's origin, 40-230 ms into the window, the same mix at both sides: a console-sync worker's native sync (`_sync_native_console_chat_ui > project_console_tick > _sync_console_rail_and_controls`, 9: base 5, top 4), the credential poll's first full pass (`_poll_console_credential_readiness`, pre-gate `None` then committed, 4: base 2, top 2), and a config-rebuild retry (`_sync_console_identity_surfaces`, 2, top). Each re-sync carried `changed=True` (the screen's real state replacing the probe).
- GitHub CI: of 159 completed admission-4 jobs (every attempt) from 2026-11-19 08:05Z to 2026-11-23 06:05Z, this node failed twice, both `[run]` with `assert 2 == 1` at line 2029: run 44948612804 (job 136982919179, 2026-11-21 22:09Z, branch fix/task-33628.16-command-rows-not-loss) and run 45080756791 (job 137395306063, 2026-11-23 01:01Z, branch fu2/linux-uat, events `['local', 'local', 'outer', 'local', 'local', 'outer', ...]`). The sibling `test_chat_screen_inspector_mutation_paths_delegate_one_owner_request[sources]` failed twice with `assert 3 == (1 + 1)`: run 44668715397 (job 136069675211, 2026-11-20 06:29Z, fu/r17-readiness) and run 44825484327 (job 136702097445, 2026-11-21 04:21Z, fu/r18-consoletests). None of these branches carries round 22.

**Round 22 does not contribute.** At the stack top the probe counted 0 calls to `ConsoleAgentController._agent_runs_reads_closed` and 0 to `initialize_agent_lessons_folder` in all 24 bg2 test runs; Library and Trace screens are never mounted in this harness. The origins are dev's code, at the same rates. The coordinator's 4/28 vs 9/28 is within noise (here 7/18 vs 8/18).

**Forced delay (temporary edits, reverted).**
- Base test, credential poll held from app start and released, with one immediate tick, at the transcript stop: `[settings]` red 4 of 4 with `['local', 'outer', 'local', 'outer']`.
- Base test, Console native syncs held from the rail press and released as a console-sync worker at the transcript stop: `[settings]` and `[run]` red 4 of 4 each; `[sources]` green (no Console sync touches the staged tray; in 16 probed `[sources]` runs its only sync was the test's).
- Fixed test under the same holds: poll released at the transcript stop 9 of 9 green; syncs held through the quiet wait and released into the window 9 of 9 green; a credential transition forced into the window (gates reset, one tick) 9 of 9 green.
- Negative control: a temporary second `_request_section_reconcile` 0.15 s after `ConsoleSettingsSummary.sync_state` keeps the fixed test red 3 of 3 with `['local', 'outer', 'local', 'outer']`.

**Change (test only, `Tests/UI/test_console_rail_reconciliation.py`).**
- New `_wait_for_inspector_quiet(pilot, screen, rail, owner_passes_after=...)`. It returns after two consecutive polls 0.05 s apart (15 s deadline, failure names what is pending) on which: the rail is open; no outer or section reconcile is scheduled; the owner-pass count is unchanged and past `owner_passes_after`; attach is reconciled; no Console sync is running, requested or scheduled; the credential poll's first pass is done (`_console_credential_pre_gate` set); no config-rebuild retry is pending; no screen worker is left. That is TASK-42148's startup signal plus the two sources it missed.
- `test_inspector_descendant_owners_reconcile_local_then_outer` stops the transcript poll right after pressing the open control, then waits for quiet with the rail-open's own owner pass counted (TASK-42415's rule). `[settings]` and `[run]` hand their probe state to the screen's builder (TASK-42149), as the sibling test already did, so any later Console sync republishes it. The ten-pause settle after the sync is now the same wait, past the baseline pass. Every assertion is unchanged.
- `test_chat_screen_inspector_mutation_paths_delegate_one_owner_request` has the same shape and was exposed: besides its two CI failures, `[sources]` failed 1 of 6 whole-file runs at -n 4 (load 11) with the first version of this fix. A probe under the sync hold showed why: the test stages a launch and syncs only the tray, and a Console sync landing in the window re-synced the settings summary and run inspector for that launch, a second owner pass (`assert 3 == (1 + 1)`), red 3 of 3. It now uses the same two waits; under the same hold (syncs held across the old five-pause settle, then released) it is 9 of 9 green. Its `[settings]` and `[run]` cases already used the builder.

**Soak and lane.**
- Whole file at -n 4, 20 runs at load 6.3-10.9 (alongside the serial soak and the step): 20 of 20 green, 54 passed each.
- Whole file serial with the lane flags (`-p no:xdist --timeout=300`), 5 runs at load 6.3-10.4: 5 of 5 green, 54 passed each.
- Both fixed tests under background QoS (`taskpolicy -b`, load 8-12), the condition that failed 7 and 8 of 18 runs before: 10 of 10 runs green, 60 of 60 cases.
- "Run admission-sensitive suites 4" step (`r19-step.sh`, `-p no:xdist --timeout=300`), started at load 6.3: 599 passed in 20:01 (the final gate had 598 passed, 1 failed).
- Preflight passed; `ruff check --select F` on the file is clean; `Tests/lint/ruff_gate.py` exits 0. No product module, census entry or lane changed, so no size ratchet or census floor applies.

**Siblings and TASK-42463.** TASK-42463's three reconciliation sites are untouched; after this change they are at lines 1494-1524 (`test_inspector_geometry_only_generation_reconciles_without_owner_pass`), 3256-3286 (`test_inspector_section_collapse_still_runs_the_full_reconcile`) and 3332 (`test_scroll_cost_probe_still_detects_pre_split_routing`), and `_wait_for_inspector_quiet` is there for them. `_open_production_inspector` (press, eight pauses) feeds those and the owner-demand tests; none failed in this task's runs.

**Files.** `Tests/UI/test_console_rail_reconciliation.py`; `backlog/docs/lessons-testing-evidence.md` (a recurrence note on the TASK-42149 entry: stopping the transcript poll quiets nothing, and TASK-42148's signal misses the credential poll's first pass and config retries).
<!-- SECTION:NOTES:END -->
