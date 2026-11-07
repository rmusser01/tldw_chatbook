---
id: TASK-39145
title: "With config unparseable and the machine heavily loaded, the Console's core push can lose every non-blocking config-lock probe for 10 s or more and never run"
status: Done
assignee: ['@claude']
created_date: 2026-11-07 02:28
updated_date: '2026-11-07 22:08'
labels:
- follow-up
- console
- config
priority: low
references:
- tldw_chatbook/UI/Console_Modules/config_sync.py
- tldw_chatbook/UI/Console_Modules/console_spend_projection.py
- Tests/UI/test_console_projection_reads_off_loop.py
dependencies:
- TASK-38967
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11 of the 2026-10-23 burn-down; checked on dev 2684e5453f, 2026-11-07, and in TASK-38967's notes (round-11 branch fu/r11-console). Line numbers are dev's.

**How the push waits.**
- `sync_console_core_state` (`UI/Console_Modules/config_sync.py:264`) probes both config locks without blocking (`:283-287`).
- If either is busy, it replays the whole FULL sync (`:292`) instead of pushing.

**Why it can starve.**
- With config unparseable, every off-loop config read holds those same locks through a rebuild that fails.
- So under heavy load the push can lose every probe.
- TASK-38967's notes record two failures of `Tests/UI/test_console_projection_reads_off_loop.py::test_an_unparseable_config_leaves_an_idle_console_idle` (`:602`) at load 48-122 in this mode: "the full sync never pushed", 4 FULL syncs and 0 core pushes in the 10 s deadline.

**What is already fixed.** Round-11 branch fu/r11-console fixes the separate replay loop (`rewarm_console_presentation`). This mode remains, and TASK-38967 leaves it as a proposed follow-up because making it deterministic needs a design change.

**Why it matters.** While config cannot be parsed, the Console never pushes its provider selection, and each attempt costs a whole FULL sync.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 With config unparseable and the config locks held continuously by failing off-loop reads (held deterministically in a test), the core push still completes, or reaches a defined state the user can see, within one bounded period; the test is red on dev
- [x] #2 Lock contention no longer costs a whole FULL sync replay per attempt; replays are counted in the test, before and after
- [x] #3 `test_an_unparseable_config_leaves_an_idle_console_idle` and the TASK-38967 replay pins stay green, and the unparseable-config test passes 10 of 10 at `-n 4` at load 40 or more, with the load recorded
<!-- AC:END -->

Possible directions (not required): a cold read that does not hold the config locks, or not replaying the whole sync on lock contention.

## Implementation Plan

Owner decision (2026-11-07): "Bounded stop + notice: after a bounded wait (a few seconds), stop retrying and show a visible notice in the Console's status area, e.g. 'Config file can't be read — showing the last loaded settings.' It clears on the next good load. Honest, simple, keeps the lock guarantee."

1. Keep TASK-36511's guarantee: the push still runs only while it holds both config locks.
2. In `config_sync.sync_console_core_state`, a lost probe no longer replays the whole FULL. It schedules one push-only retry (`CORE_PUSH_RETRY_DELAY`, 0.5 s), at most `CORE_PUSH_RETRIES` (6) per episode, so it stops about 3 s after the first refusal. A won probe ends the episode. The cold-read deferral (one whole replay per cold episode) is unchanged.
3. At the stop, if the last config bootstrap failed to parse (`config.get_config_load_failure()`), record `_CONFIG_CACHE_INSTALLS` and show one alert segment on the status strip ("Config file can't be read: showing the last loaded settings"), once per episode. Once the install counter moves (a good load), the next FULL, control-bar sync or successful readiness read clears it and starts a new episode.
4. Tests, red on base: a real failing read parked inside `_preserve_corrupt_config_aside` holds both locks; the push stops within the bound without any whole replay, and the notice appears once. After a good load the notice clears and the next push runs. Update the TASK-36511 pin's contention leg to the new rule.
5. Update the Console User Guide's status-strip copy.

## Implementation Notes

Owner decision (2026-11-07): "Bounded stop + notice: after a bounded wait (a few seconds), stop retrying and show a visible notice in the Console's status area, e.g. 'Config file can't be read — showing the last loaded settings.' It clears on the next good load. Honest, simple, keeps the lock guarantee."

**Approach** (`UI/Console_Modules/config_sync.py`):
- **Lock guarantee kept.** The push still runs only while it holds both config locks, probed without blocking (TASK-36511).
- **No whole replay on contention.** `sync_console_core_state` used to call `replay_whole_console_sync` when it lost the probe. It now owes one push-only retry (`retry_console_core_push`, `CORE_PUSH_RETRY_DELAY = 0.5` s). The cold-read deferral (one whole replay per cold episode) is unchanged.
- **The bound.** At most `CORE_PUSH_RETRIES = 6` retries per episode (about 3 s); the 7th lost probe stops the retries. A won probe ends the episode and stops any owed retry.
- **The notice.** At the stop, if the last config bootstrap failed to parse (`config.get_config_load_failure()`), the push records `_CONFIG_CACHE_INSTALLS` and shows one alert segment on the status strip: "Config file can't be read: showing the last loaded settings". It matches the strip's "Emergency stop: runs and schedules held" grammar, and segment labels are markup-escaped on render. Later lost probes in the same episode neither retry nor re-show it.
- **Clearing.** `sync_config_unreadable_notice` is the one rule: once the install count moves (any good load), it clears the segment and starts a new episode. FULL syncs (`sync_console_core_state`), control-bar syncs (`run_console_presentation_sync`) and a successful readiness read (`_refresh`) all call it. It queries the DOM only while a notice is shown or being cleared.
- **The segment** lives on `ConsoleStatusChips` (`sync_config_unreadable`, `CONSOLE_CONFIG_UNREADABLE_STRIP_LABEL`). `chat_screen.py` is untouched.

**Tests.**
- New: `Tests/UI/test_console_projection_reads_off_loop.py::test_a_core_push_held_off_by_failing_reads_stops_and_says_so`.
  - A real failing `load_settings(True)` is parked inside `_preserve_corrupt_config_aside`, holding both config locks. Every retry is fired by hand (no clock).
  - It checks that 1 + 6 lost probes stop with no push and no whole-sync replay, that the notice shows once, and that a later FULL probe changes nothing.
  - Then a good load: the next push runs and the notice clears.
  - Red with this task's product changes reverted: `probe 1 lost the config locks and scheduled no push retry (whole-sync replays requested: 1)`. Before, every lost probe requested a whole-sync replay; after, 0 over the episode's 7 probes (AC#2).
- Changed: `test_the_core_state_push_never_waits_for_or_rebuilds_config` (TASK-36511). Its contention leg now expects one push-only retry timer and no whole replay, which is AC#2. Its lock assertions are unchanged.

**AC#3.**
- The whole file passed at `-n 4` (22 passed, load 28-34), including `test_an_unparseable_config_leaves_an_idle_console_idle` and the TASK-38967 replay pins.
- The unparseable-config test passed 10 of 10 at `-n 4` under a temporary `parametrize('_rep', range(10))`, removed afterwards. The 1-minute load average, sampled every 10 s, stayed at 42-47 (background `Tests/CI` runs held it there). An earlier 10-of-10 run saw load 31-46.

**Docs.** `Docs/User_Guide/console.md`, Quirks & troubleshooting: what the notice means and when it clears.

**Covering runs (`-n 4`, through run.sh).**
- Head: 557 passed, 30 failed. Base 8b762b52fa, same files: 567 passed, 19 failed. The files: the off-loop file, status chips, emergency stop, chip overflow, status row collapse, workbench contract, cost chip screen, mode strip, config-sync cadence, readiness projection, pending-facts cold readiness, cost config retry, checked display scope, temporary status copy, first-persistence attribution, Backup_Recovery config-sync lifetime, status chips cost, display contention, timer-path inventory, module size ratchet, layer direction.
- Only on head: 9 `test_console_cost_chip_screen.py` nodes and 2 `test_console_cost_config_source_retry.py` nodes timed out waiting for 'warm reply' at load 60+. On a rerun of both files: 39 passed, 1 failed (`test_idle_draft_edit_burst_coalesces_one_cost_refresh`). Alone, that node failed once on base and passed twice on head. The timer-path inventory and workbench-contract failures fail on base too.
- `Tests/CI`: 1220 passed, 1 skipped. Preflight passed. `bootbudget.sh`: 32 passed.
- The new test takes about 21 s alone. Its file is a whole-file UI census entry.

Files: `tldw_chatbook/UI/Console_Modules/config_sync.py`, `tldw_chatbook/UI/Console_Modules/console_spend_projection.py`, `tldw_chatbook/Widgets/Console/console_status_chips.py`, `Tests/UI/test_console_projection_reads_off_loop.py`, `Docs/User_Guide/console.md`.
