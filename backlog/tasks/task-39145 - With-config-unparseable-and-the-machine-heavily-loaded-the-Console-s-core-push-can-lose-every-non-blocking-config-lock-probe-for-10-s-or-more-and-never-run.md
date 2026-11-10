---
id: TASK-39145
title: "With config unparseable and the machine heavily loaded, the Console's core push can lose every non-blocking config-lock probe for 10 s or more and never run"
status: Done
assignee: ['@claude']
created_date: 2026-11-07 02:28
updated_date: '2026-11-10 00:39'
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
- [x] #2 While config.toml cannot be parsed, refused core pushes stop owing whole FULL sync replays once the refusal episode passes about 3 s (until then each refused attempt still costs one, as on dev), and ordinary contention with a parseable file never stops replaying; replays are counted in the test, before and after the stop (amended 2026-11-09, owner decision)
- [x] #3 `test_an_unparseable_config_leaves_an_idle_console_idle` and the TASK-38967 replay pins stay green, and the unparseable-config test passes 10 of 10 at `-n 4` at load 40 or more, with the load recorded
<!-- AC:END -->

Possible directions (not required): a cold read that does not hold the config locks, or not replaying the whole sync on lock contention.

## Implementation Plan

Owner decision (2026-11-07): "Bounded stop + notice: after a bounded wait (a few seconds), stop retrying and show a visible notice in the Console's status area, e.g. 'Config file can't be read — showing the last loaded settings.' It clears on the next good load. Honest, simple, keeps the lock guarantee."

Owner decision (2026-11-09), on the contradiction with dev 0aa96f8d1b (ADR-121 amendment): "Dev's rule, keep stop+notice". The chosen option read: "Keep dev's whole-screen refresh path and owed handoff; round 12 only counts the refused refreshes, stops after ~3 s and shows the notice, cleared on the next good load. No retry timer. Each refused attempt still costs a whole refresh until the stop. TASK-39145's 'no whole refresh' criterion is amended."

1. A refused core push behaves as on dev: it owes the CORE handoff, retires it with `owed=True` and calls `replay_whole_console_sync`. The coalesced callback makes the one guarded attempt, and nothing pushes before warmTrue. Remove the push-only retry timer, `_retry_core_push_later`, `retry_console_core_push` and the `retry` keyword.
2. Bound only an unreadable file. While `config.get_config_load_failure()` is set, the first refusal opens an episode on `config_sync._clock`. A refusal `CORE_REFUSAL_EPISODE_SECONDS` (3.0) or more later stops owing the replay and shows the status-strip notice, once. A pushed core state ends the episode. A good load (the install count moves) clears the notice and starts a new episode. With a parseable file a refusal always replays. The install count is read before the parse failure (M3).
3. Tests: dev's two contract tests pass unchanged. New tests drive every refusal by hand on a replaced clock, each red under a temporary edit.

## Implementation Notes

Owner decision (2026-11-07): "Bounded stop + notice: after a bounded wait (a few seconds), stop retrying and show a visible notice in the Console's status area, e.g. 'Config file can't be read — showing the last loaded settings.' It clears on the next good load. Honest, simple, keeps the lock guarantee."

Owner decision (2026-11-09), resolving the contradiction with dev 0aa96f8d1b (the ADR-121 amendment, whose tests pin a whole replay and an owed CORE handoff for a refused push): "Dev's rule, keep stop+notice". The chosen option read: "Keep dev's whole-screen refresh path and owed handoff; round 12 only counts the refused refreshes, stops after ~3 s and shows the notice, cleared on the next good load. No retry timer. Each refused attempt still costs a whole refresh until the stop. TASK-39145's 'no whole refresh' criterion is amended." AC#2 is amended to match.

**Approach** (`UI/Console_Modules/config_sync.py`):
- **Dev's rule.** A refused core push behaves exactly as on dev: it owes the CORE handoff, retires it with `owed=True` and calls `replay_whole_console_sync`. The coalesced callback makes the one guarded attempt, and nothing pushes before warmTrue. The push still runs only while it holds both config locks (TASK-36511).
- **Removed.** Round 12's push-only retry timer is gone: `CORE_PUSH_RETRY_DELAY`, `CORE_PUSH_RETRIES`, `_end_core_push_contention`, `_retry_core_push_later`, `retry_console_core_push` and the `retry` keyword. The keystroke census wrapper is back to dev's version.
- **The bound.** `_refusal_owes_replay` decides whether a refusal still owes its replay.
  - Only while `config.get_config_load_failure()` is set: the first refusal opens an episode on `config_sync._clock`.
  - A refusal `CORE_REFUSAL_EPISODE_SECONDS` (3.0) or more after it stops owing the replay and records `_CONFIG_CACHE_INSTALLS`, read before the parse failure (M3). That shows the status-strip notice "Config file can't be read: showing the last loaded settings" once; later refusals in the episode neither replay nor re-show it.
  - A core push that runs ends the episode.
  - With a parseable file a refusal always replays and never shows the notice.
- **Clearing.** `sync_config_unreadable_notice` is the one rule: once the install count moves (a good load), it clears the notice and starts a new episode. FULL and control-bar syncs and a successful readiness read call it.
- **Notice segment.** Unchanged: it lives on `ConsoleStatusChips` (`sync_config_unreadable`).

**Tests** (`Tests/UI/test_console_projection_reads_off_loop.py`). Dev's `test_the_core_state_push_never_waits_for_or_rebuilds_config`, `test_a_finite_checked_reader_leaves_a_core_push_release_window` and `Tests/Backup_Recovery/test_canvas_policy_worker.py::test_stock_canvas_poll_leaves_the_retired_read_callback_handoff` pass unchanged; the first is restored to dev's text. Each new test drives every refusal by hand on a replaced `config_sync._clock`:
- `test_refused_core_pushes_stop_replaying_an_unreadable_config_and_say_so` (mounted). A real failing read is parked inside `_preserve_corrupt_config_aside`, holding both locks. 15 refusals 0.2 s apart each owe a replay; the refusal at 3.0 s does not and shows the notice; a later refusal changes nothing. Then a good load: the notice clears and the next push runs.
  - Red on dev's `config_sync.py`: "refused pushes kept owing whole replays past the episode".
  - Red with the good-load clear disabled: "the notice outlived a good config load".
- `test_parseable_contention_never_stops_the_whole_replay`: 40 refusals over 8 s all replay, with no notice. Red with the parse-failure check removed: "parseable contention stopped replaying after 15".
- `test_a_won_push_starts_a_new_refusal_episode`: a push between refusals restarts the episode. Red with the reset removed: "a refusal after a won push was charged to the old episode".
- `test_an_install_landing_at_the_stop_never_leaves_the_notice_up` (M3). Red with the read order swapped: "the notice outlived the good install that landed at the stop".

**AC#3.** The unparseable-config test and the new mounted test, each under a temporary `parametrize('_rep', range(10))` (removed afterwards), ran together at `-n 4`: 20 passed. The 1-minute load average, sampled every 10 s, was 40.4-94.6 throughout (background `Tests/CI` runs held it up).

**Superseded.** Round 12's earlier push-only retry design (reviews r12p-rev1 I2/M1/M2/M3 and r12p-rev2 I2/M2) is replaced by the above. Its pause-flag and FULL-probe counting no longer exist, and its census-wrapper change was reverted.

**Docs.** `Docs/User_Guide/console.md`, Quirks & troubleshooting: "it keeps trying for about three seconds, then stops trying and shows this notice once".

Files: `tldw_chatbook/UI/Console_Modules/config_sync.py`, `tldw_chatbook/UI/Console_Modules/console_spend_projection.py`, `tldw_chatbook/Widgets/Console/console_status_chips.py`, `Tests/UI/test_console_projection_reads_off_loop.py`, `Tests/Performance/test_console_keystroke_work_census.py` (reverted to dev), `Docs/User_Guide/console.md`.
