---
id: TASK-38967
title: "Console readiness warm-up hands back a display proof about to expire, so the FULL sync it prepared refuses and replays 0.2 s later"
status: Done
assignee:
  - '@claude'
created_date: 2026-11-06 01:21
updated_date: 2026-11-07 01:58
labels:
- follow-up
- console
- performance
- tests
priority: low
references:
- tldw_chatbook/UI/Console_Modules/console_spend_projection.py
- tldw_chatbook/UI/Console_Modules/config_sync.py
- tldw_chatbook/UI/Screens/chat_screen.py
- Tests/UI/test_console_pending_facts_cold_readiness.py
- Tests/UI/test_console_pending_facts_direct_control_cold_readiness.py
- Tests/UI/test_first_persistence_getter_attribution.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b, 2026-11-05. No branch due to merge before this one changes the code below.

**Mechanism** (files under `tldw_chatbook/UI/Console_Modules/`):
- `ConsoleReadinessConfigProjection.warm()` (`console_spend_projection.py:496-511`) returns True whenever the proof is younger than `max_age`, which is 1.0 s (`:510`).
- `run()` re-reads only when the proof is at least `life` old (`:425-435`). `life` is 1.0 s, or 0.5 s while the run-poll timer exists (`REFRESH_AHEAD_SHARE`, `:200`). So with no run poll, a 0.99 s proof is handed back as warm.
- `_checked_display_status` then refuses the proof once it is `min(1.0, max_age)` old or more (`:865`). `_run_checked_display_sync` (`:1109-1123`) calls `defer`, which is `_request_console_control_bar_sync(delayed=True)` (`config_sync.py:236`) and waits 0.2 s (`config_sync.py:218`).
- In production:
  - a FULL sync awaits `prepare_console_presentation_sync` (`chat_screen.py:17853`), which calls `warm()` (`config_sync.py:141`);
  - then `project_console_tick` returns False at `chat_screen.py:17869` ("Replay owns it");
  - the whole sync replays 0.2 s later.

**Measured.** `warm` and `_checked_display_status` were wrapped in the three files below and each file was run 5 times, at load 9-11:

| What | Count |
|---|---|
| `warm()` calls, all returning True (mostly production FULL preparation) | 191 |
| … handing back a proof at least 0.5 s old | 72 |
| … handing back a proof at least 0.9 s old | 15 |
| … of those 15, followed straight away by an expiry refusal | 11 |
| Refusals in total | 140 |
| … of which were expiry refusals (the rest are the tests' deliberate cold reads) | 27 |

The 11 proofs were 0.928-0.999 s old when warmed and 1.006-1.311 s old when refused. TASK-38745 (Done) recorded the same thing independently and fixed only its own test, with a bounded retry; the product is unchanged. TASK-38696 (Done) is the refusal backoff, which is a different issue.

**Tests carrying the race.** Each warms, runs one `_sync_console_rail_and_controls()`, waits one `pilot.pause()`, and never checks that the sync rendered:
- `Tests/UI/test_console_pending_facts_cold_readiness.py:564-566`
- `Tests/UI/test_console_pending_facts_direct_control_cold_readiness.py:565-567`
- `Tests/UI/test_first_persistence_getter_attribution.py:93-95`

None of them is on a PR lane. Each passed 6 of 6 here.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Measured first: over a stated workload, the number of FULL syncs whose warm-up returned True but whose render refused on proof age is counted before and after the change; if a refresh-ahead margin does not reduce it, the notes record the numbers and the product stays unchanged
- [x] #2 If changed, warm-up re-reads a proof within a margin of max_age before returning True, so the render that follows it does not refuse on age; a deterministic test with the proof aged to max_age minus a small epsilon is red on dev
- [x] #3 The margin adds no config read to an idle Console, with re-reads per idle minute measured before and after
- [x] #4 The three tests assert that their sync rendered, or retry as TASK-38745's _prepared does, and pass 10 of 10 -n 4 runs with the load recorded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Measure on the base first: run the three named files 5 times at `-n 4` under a call-through probe that counts FULL syncs whose warm-up returned True and whose render (`project_console_tick`) then refused on proof age, plus the display re-reads. Measure an idle Console's re-reads over a minute.
2. If a refresh-ahead margin can help, have `ConsoleReadinessConfigProjection.warm()` re-read a proof past `REFRESH_AHEAD_SHARE` of its life (the share the run poll's projections already use) before it returns, so the FULL render that follows gets a fresh proof. `run()` takes the shorter life as a keyword; its other callers keep theirs.
3. Add a deterministic test with the checked display's clock replaced: age the proof to `max_age` minus a small epsilon, warm it, let the render take a little time, and project. Show it red on the base.
4. Re-measure the same workload and the idle minute on the branch.
5. Make the three tests retry until their sync renders, as TASK-38745's `_prepared` does, and run them 10 times at `-n 4`, recording the load.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Approach.** `ConsoleReadinessConfigProjection.warm()` now re-reads a display that is past `REFRESH_AHEAD_SHARE` (half) of its 1 s life before it returns. That is the share the run poll's projections already refresh ahead at, so the FULL render that follows starts with a fresh proof. `run()` takes the shorter life as a keyword (`life=`), and every other caller keeps the life it had. An idle Console calls `warm()` only from a FULL sync's preparation, so idle cost is unchanged.

**Measured (AC#1).** Workload: the three named files, `-n 4`, through run.sh. A call-through probe (`r11c_warm_probe`, loaded with `PYTEST_PLUGINS` so that xdist workers and private-profile children load it too; not committed) wraps `prepare_console_presentation_sync`, `project_console_tick` and `_checked_display_status`. It counts each FULL sync whose warm-up returned True, and whether that sync's own render then refused on proof age or for another reason.

| Base and branch, interleaved, 6 rounds each, load 44-54 | base | branch |
|---|---|---|
| FULL syncs whose warm-up returned True | 127 | 125 |
| ... warm-up handed back a proof at least 0.5 s old | 39 | 0 |
| ... at least 0.9 s old | 8 | 0 |
| ... render refused on proof age | 1 | 0 |
| ... render refused for another reason (all on `storage._lock.acquire(blocking=False)`) | 11 | 11 |
| ... render returned False, so the whole sync replayed | 11 | 9 |
| Checked-display re-reads | 112 | 138 |

- Earlier sequential sets, not interleaved, agree on the age figures. Base, 10 runs: 209 FULL syncs, 9 warm-ups at least 0.9 s old, 2 age refusals. Branch, 10 runs: 223 FULL syncs, 0 and 0.
- **What the margin does not fix.** The remaining replays are storage-admission lock contention, not age. In the interleaved base arm, 8 of the 35 FULL warm-ups that re-read were refused on that lock, against 3 of the 92 that did not re-read. The branch showed 7 of 61 against 4 of 64.
  - So a render straight after a re-read meets a busy storage lock more often. These files also write config, which both invalidates the display and takes storage locks, so that correlation may be the tests' own.
  - The margin roughly doubles the warm-ups that re-read (35 -> 61), and the lock refusals stayed level (11 and 11). Listed as a proposed follow-up, not changed here.
- **Cost.** About 23% more checked-display re-reads while FULL syncs run (112 -> 138). They are off-loop reads.

**AC#2.** `Tests/UI/test_console_projection_reads_off_loop.py::test_a_warm_up_rereads_a_display_near_expiry_so_its_render_accepts_it` replaces the checked display's clock and stops the run poll. It ages the proof to `max_age - 0.01`, awaits `warm()`, advances 0.05 s for the render, and projects.
- Base: red, `the render refused the display its warm-up had just handed back` (`False is True`).
- Fix: green.

**AC#3.** An idle Console minute (send harness at 160x45, after a 3 s settle; scratch probe `test_r11c_idle_rereads.py`) made 0 checked-display re-reads, 0 warm-ups and 0 FULL preparations. That held on both base and branch, in 2 interleaved runs each, at load 20-53.

**AC#4.** The three files now call `sync_rendered` (in `test_console_pending_facts_cold_readiness.py`), which retries warm-up plus sync until the sync renders, as TASK-38745's `_prepared` does.
- 10 of 10 runs at `-n 4` passed, 3 tests each, at load 26-62.
- Before this change, each file had passed 6 of 6 for the filer.

**Covering.** 18 files that use the readiness projection, `prepare_console_presentation_sync` or the checked display, at `-n 4` with `-p no:randomly`, run on head and on base with the same command. Head: 2 failed, 244 passed. Base: 2 failed, 242 passed. The same 2 nodes failed on both (`test_console_readiness_config_projection.py::test_actual_mode_bar_defers_cold_confi…` and `test_console_shared_context_policy.py::test_actual_settings_open_waits_for_shared_c…`). Head's 2 extra passes are the new tests.

**Review round (r11c-rev1).** While checking the review fixes, `test_an_unparseable_config_leaves_an_idle_console_idle` failed on this branch 3 times in a burst at load 100-170. Each time the idle Console replayed 2-3 full syncs in the 2 s window, against a bound of 1.
- A probe that recorded `_CONFIG_CACHE_INSTALLS` across the cold window showed no install there (2 -> 2), so TASK-38949's episode key was not the cause.
- One path this task added did amplify replays. With config unparseable, every re-read fails and arms its own retry sync (TASK-38649), and the warm-up's read-ahead added a failing re-read to every FULL sync that found a display at least 0.5 s old.
  - The warm-up now reads ahead only while the last read succeeded (`_retry_delay` is 0). While reads fail, it keeps the base rule: re-read only an expired display.
  - The test above gained a step: with a failed read recorded, a warm-up at 0.6 of the display's life must not re-read. It was red without the guard (`the warm-up read ahead while the display's reads were failing`) and is green with it.
- The test is load-flaky on the base too, with the same message.
  - Interleaved runs at load 68-99 after the guard: base failed 3 of 6, head failed 1 of 6.
  - An earlier interleaved set at load 60-135: base 4 of 4 passed, head 4 of 4 passed.
  - It is a whole-file census entry, so this is listed as a proposed follow-up.

**Review round (r11c-rev2): the self-replaying FULL sync.** Instrumenting that flake found a product loop in which a FULL sync replays itself.
- **The loop.** A FULL warms the checked display before its projection, then awaits the transcript and the archive view before its rail-visibility step. Under load those awaits outlived the display's 1 s life, so the rail step refused the expired display and the whole sync replayed. The replay did the same thing.
  - Measured: in a failing run, all 4 refusals in a 4 s window were on proof age, at 1.20-1.33 s. Every run, passing or failing, showed FULL syncs every 1-2.5 s for as long as the load lasted.
  - Base has the same path.
- **The fix.** Before its rail step, a FULL now re-reads the display if, and only if, it has expired: `config_sync.rewarm_console_presentation`, which calls `run(life=max_age)` and awaits the read. One read replaces a whole replay, and a live display reads nothing. With config unparseable, every read is a rebuild under the config locks, so the rail step must not add reads.
  - A first version called `warm(ahead=...)`. That broke `_refused_warm_up`'s call-through wrapper, which takes no keyword arguments, in 3 tests, so `warm()` keeps its signature.
- **Test.** `test_a_full_sync_slower_than_the_display_life_does_not_replay_itself` uses the clock harness. The transcript step advances the clock by 1.2x the display life.
  - Red before the fix: `a FULL sync whose awaits outlived the display replayed itself`.
  - A second FULL that stretches by only 0.6x must not re-read at the rail step. Under a variant that reads ahead there, it fails with `the rail step re-read a live display`.
- **Review r11c-rev3.** While reads failed, the rail step's re-read could still start a second failing read in the same sync. The reviewer's clock-harness probe measured 4 reads against 3 with the re-warm off, and the retry backoff doubled from 0.8 s to 1.6 s. `rewarm_console_presentation` now reads nothing while `_retry_delay` is non-zero, the same guard as `warm()`. A third step of the test above records a failed read, makes reads fail, and stretches a FULL past the display life. It fails without the guard (`the rail step started a read while the display's reads were failing`, 3 == 2) and passes with it.
- **Stress.** The unparseable-config test with three neighbours, `-n 4`:
  - final version, 10 of 10 runs passed at load 40-53;
  - then 5 interleaved rounds each at load 21-40, base 5 of 5 and head 5 of 5;
  - earlier, at load 40-87, base failed 2 of 10 (`kept replaying`, 2 full syncs in 2 s).
- **Not shown deterministic.** At load 48-122, before the final version, two failures had a second mode: `the full sync never pushed`, 4 FULL syncs and 0 core pushes in the 10 s deadline. The core push takes the config locks without blocking and replays the sync when they are busy. With config unparseable, every off-loop config read holds those locks for a failed rebuild, so under extreme load the push can lose every probe for 10 s. Making that deterministic needs a design change, for example a cold read that does not hold the locks, or not replaying the whole sync on lock contention. It stays a proposed follow-up.

**Files:** `tldw_chatbook/UI/Console_Modules/console_spend_projection.py`, `tldw_chatbook/UI/Console_Modules/config_sync.py`, `tldw_chatbook/UI/Screens/chat_screen.py`, `Tests/UI/test_console_projection_reads_off_loop.py`, `Tests/UI/test_console_pending_facts_cold_readiness.py`, `Tests/UI/test_console_pending_facts_direct_control_cold_readiness.py`, `Tests/UI/test_first_persistence_getter_attribution.py`.
<!-- SECTION:NOTES:END -->
