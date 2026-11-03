---
id: TASK-36511
title: Console attach holds the UI thread 180-270 ms in synchronous reconcile work
status: In Progress
assignee:
- '@claude'
created_date: 2026-10-23 22:44
updated_date: 2026-11-03 16:11
labels:
- follow-up
- performance
- console
priority: medium
references:
- Tests/Performance/_compact_original_config_gate.py
- Tests/Performance/test_compact_setup_ownership.py
- Tests/Performance/test_compact_model_original_checked_read.py
dependencies:
- TASK-36390
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-36390 (2026-10-23). The compact-setup gates have 14 nodes in `test_compact_setup_ownership.py` and `test_compact_model_original_checked_read.py`. While a compact read is held, they require the shared loop to answer a probe within 100 ms.

Once TASK-36390 fixed the gate's install step, 3-5 of the 14 nodes failed per run at load average 9-18. The probe delays were bimodal: 0 ms, or 179-271 ms. Stacks (`TLDW_LOCAL_COMPACT_STACKS=1`) put the main thread in synchronous Console attach work at the probe:
- `_reconcile_console_after_attach`
- `_prepare_allocation_reconcile`
- `_sync_native_console_chat_ui` -> `_build_console_provider_selection`

Owner ruling (2026-10-23): keep the 100 ms bound and treat this attach work as a real UI-thread stall. The compact gates stay off a PR lane until they pass.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No synchronous segment of a Console attach blocks the UI thread for more than 100 ms on the compact gates' fixtures
- [ ] #2 All 14 compact-setup gate nodes pass 20 of 20 runs under parallel load
- [ ] #3 The compact-setup gate files run on a PR lane
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Measure on the gates' own fixtures (dev 9bbd5c2515): time every UI-loop step with a local-only sitecustomize sampler, census the UI thread's config reads around the probe, and keep the failing receipts' `TLDW_LOCAL_COMPACT_STACKS` leads. Name the segments over 100 ms and what they wait on.
2. Fix the largest root cause first, in the shared function its callers route through, reusing the existing never-wait config pattern (TASK-35645/35836) rather than a new mechanism.
3. Re-measure after each change; fix the next root cause only if the gate still fails on it and the fix stays small.
4. Pin each fix with a unit test in a file a PR lane already runs; show it red before the fix.
5. AC#2: 20 rounds of the 17-node gate files at `-n 4` under the shared machine's load, recording the load.
6. AC#3: put the gate files on a PR lane (census route if the checker allows it; otherwise report the blocker).
7. Regression check: Console attach / reconcile / first-mount test files at `-n 4`, compared with dev on the same command; size ratchets; preflight.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Status: partly done; all three criteria still open (reasons below).** Base is dev 9bbd5c2515. The gate files now hold 17 nodes: 14 run the loop probe, and `custom_reader`, `custom_body` and `malformed` take the direct route.

**What stalled the loop.** Every probe failure whose stack I captured was the UI loop doing config work inside the gate's window. The gate holds a compact read inside the guarded config scope and edits `config.toml` while holding it. Any UI-loop `load_settings()` in that window either waits for that read (lock wait at `config_participants.operation` line 427) or rebuilds the now-cold settings on the UI loop. The triage stack `_sync_native_console_chat_ui -> _build_console_provider_selection` is this case. The leads show it bottoming out in `load_settings()`, called from `sync_console_core_state`. That function probed `config_read_would_wait()` and then read without holding the locks, so a worker read could start in between (a TOCTOU race), and it never checked whether the read would be cold.

**What changed** (shared mechanism: hold both config locks without blocking, or defer):
- `config_sync.sync_console_core_state`: the push runs only while it holds both config locks and the live read is warm. Otherwise it replays the whole sync, and that pass's own preparation warms config off-loop.
- `console_spend_projection.retry_after_config_rebuild`: the identity and width refreshes also defer while a live read is cold, and kick the readiness projection's off-loop read. The shared check is `live_config_read_is_cold`: only a disk-loaded snapshot reads live, so other snapshots never loop.
- Review round 1, bounded deferral (`defer_cold_live_read`). Config can stay cold for good: an unparseable `config.toml`, a strict decryption failure or an unexpected load error installs neither cache, and the app keeps running in that state. Deferring every cold read replayed an idle Console forever. Each owner (the core-state push, and each `retry_after_config_rebuild` refresh by name) now defers once per cold episode. Its next call reads live, as base did. Any warm read ends the episode for every owner (round 2: ending it only for the owner that saw the warm read left a stale entry for rarely called owners such as the Resize width refresh). Outside triggers therefore cannot start new deferrals while config stays cold. The 0.2 s retry stays a single pending timer per refresh.
- `app.py` subscriptions backfill starter: the pre-check uses the wired `SubscriptionsDB.db_path`, as the scheduler's pre-check does, instead of a checked `get_subscriptions_db_path()` on the UI loop. Size row 5687 -> 5689 with a dated waiver.
- `Tests/Backup_Recovery/test_console_config_sync_lifetime.py`: the `Surface` proxy re-bound classmethods. The new disk-loaded check reached that in `native-worker-teardown`; the proxy is fixed.

**Measurement.** A local-only sitecustomize timed every UI-loop step (`Handle._run`), sampled the main stack every 2 ms, and counted UI-thread config calls with `sys.monitoring`. Failing receipts were kept with `TLDW_LOCAL_COMPACT_STACKS=1`. CPU is `thread_time`. Wall time includes machine load.

UI-loop steps of 30 ms or more:

| Segment | Before: n, wall med/max, CPU med/max (ms) | After: n, wall med/max, CPU med/max (ms) |
|---|---|---|
| core-state push (`sync_console_core_state`) | 5, 46/237, 7/78 | 0 |
| UI-loop config operation, attach (lock wait or cold rebuild) | 8, 48/274, 40/78 (3 over 100 wall) | 0 in attach (2, at shutdown) |
| boot starter (`_admit_staggered_boot_workers`) | 2, 230/274, 60/67 | 0 |
| identity refresh rebuilding cold config | seen in stress leads | 0 after the wrapper fix (79.5 CPU / 139.6 wall before it, A1) |
| `_reconcile_console_after_attach` | 11, 43/135, 8/26 | 3, 35/43, 8/9 |
| full sync (`_sync_native_console_chat_ui`) | 205, 44/237, 23/78 | 113, 45/142, 23/36 |
| control-bar sync | 43, 45/178, 27/61 | 19, 46/86, 22/36 |
| left-rail `_prepare_allocation_reconcile` | 52, 53/248, 38/53 | 31, 80/219, 39/74 |
| Textual `Screen._refresh_layout` | 219, 45/219, 28/54 | 145, 45/201, 28/54 |

- Before: 7 instrumented rounds at load 15-28. After: 4 rounds (A2) at load 24-28.
- Before, the probe lag maxed at 219 ms, with 1 probe over 100 ms. After (A2, load 24-28), it maxed at 106 ms, with 1 probe over 100 ms: that step used 13.9 ms of CPU and 142 ms of wall, so it was the scheduler.

**Evidence**
- Dev, 6 plain rounds: 1 failure in 84 probes. The stall was 238 ms, main thread in `operation` under `sync_console_core_state`.
- Dev, 4 instrumented rounds: 1 failure, 219 ms, from the boot starter's `get_subscriptions_db_path`.
- Interleaved A/B at load 12-15, 16 `queued_body` nodes per round, 4 rounds per arm: dev failed 2 of 64, both the cold push rebuilding on the UI loop (`_load_settings_uncached`). The branch failed 1 of 64, a control-bar render taking 109 ms.
- AC#2, first fixes (lock hold and starter): 18 of 20 rounds green, 280 of 280 probes under 100 ms. Both failures were `queued_body` running the swapped body through the cold push, which led to the cold-replay change.
- AC#2, final code: 19 of 20 rounds green at load 9-28; my own instrumented run was adding load during rounds 1-10. 279 of 280 probes were under 100 ms. The failure was round 7, `closing`, at load 27.4: 135 ms in Textual `Widget.watch_has_focus -> update_node_styles`, a focus restyle with no config work.
- Red/green:
  - `test_the_core_state_push_never_waits_for_or_rebuilds_config` is red on dev (`[True] == [False]`, a worker read started under the push), on the lock-hold-only version (`2 == 1`, a cold push ran), and at 7dd3f03343 on its episode case ("a still-cold replay deferred again").
  - `test_a_mount_refresh_defers_a_cold_live_read_and_warms_it_off_the_loop` is red before the wrapper change (`[True] == []`) and at 7dd3f03343 ("a still-cold retry deferred again").
  - `test_an_unparseable_config_leaves_an_idle_console_idle` (mounted, `@private_profile_test`, breaks only the child's own `config.toml` and restores it):
    - It polls up to 10 s for the first core-state push and identity run, then compares a 2 s quiet window with the idle baseline. Round 2 replaced a fixed 1 s window, which failed 2 of 2 at load 15-21 because the first identity run can take seconds under load.
    - At 7dd3f03343 it is red on the steady state: at load 32, idle 0 full syncs against 7 in the quiet window.
    - At HEAD it passed 10 of 10 runs of the whole file at `-n 4`, load 30-41.
  - `test_subscriptions_starter_reads_no_config_on_the_ui_loop` is red against base `app.py` ("the starter read config on the UI loop").
  - All four are green on the branch.
- Reviewer probes, 7dd3f03343 -> branch, load 7-10:
  - Mounted idle Console with an unparseable config: 7dd3f03343 ran 5 full syncs and 26 core-state pushes a second, with the replay flag stuck on. The branch does one burst (3 full syncs and 18 pushes in second 0), then 0 a second, as base did (4 and 22, then 0).
  - Unit probe, 5 rounds of off-loop warm read then push and refresh: 7dd3f03343 made 1 push and 5 replays and never ran the refresh, with a timer always armed. The branch makes 1 replay, pushes every round after the first, runs the refresh from round 1, and leaves no timer.
- Regression check: 20 attach, reconcile, config-sync and boot files at `-n 4`, run on the branch and on dev sources with the same command. The failure sets are identical apart from the new tests (red on dev) and `native-worker-teardown` (the proxy, fixed). After review round 1 it was re-run at load 7-14: 21 failed on the branch and 23 on base sources. All 21 branch failures also fail on base; the base-only 2 are the new unit tests. After review round 2, at load 31-40, both arms failed 26.
  - Branch-only: `test_cost_settings_only_native_postbody_refusal_and_fresh_retry[runtime|source]` and `test_workspace_command_opens_same_manager_in_registered_scope`. All three passed 3 of 3 reruns on the branch at load 38, and in both earlier rounds.
  - Base-only: the two new unit tests, plus a load flake, `test_fixture_failure_retires_held_transport[primary]`. The size ratchets pass except the pre-existing `library_skills_controller` (3171 > 3142, untouched). Preflight passes.

**Criteria left open**
- AC#1 is not met. Lock waits and cold rebuilds are gone from the attach path. The remaining attach steps are CPU-bound and stay under 100 ms of CPU (left-rail allocation peaks at 74 ms), but their wall time passes 100 ms at load 20 and above.
- AC#2 is not met: 19 of 20 rounds (above).
- AC#3 is blocked. `scripts/check_ui_pr_gate_census.py` rejects any entry outside `Tests/UI/`. The only PR lane for `Tests/Performance` is `perf-guard.yml`, which open PR #3380 also edits, so this change leaves it alone. Follow-up after #3380 merges: add both gate files to perf-guard's "Run Console tab-publication and voice gates" step and remove its "Not here" comment (lines 126-128).

**Test placement.** The starter test lives in `Tests/UI/test_app_instance_warning.py`, a whole-file census entry that already pins `TldwCli` boot wiring unbound against a stub. The boot fleet's own suite, `Tests/App/test_boot_worker_stagger_policy.py`, runs in no PR lane.

**Files:** `tldw_chatbook/UI/Console_Modules/config_sync.py`, `tldw_chatbook/UI/Console_Modules/console_spend_projection.py`, `tldw_chatbook/app.py`, `Tests/UI/test_console_projection_reads_off_loop.py`, `Tests/UI/test_app_instance_warning.py`, `Tests/Backup_Recovery/test_console_config_sync_lifetime.py`, `Tests/Architecture/test_module_size_ratchet.py`, and `backlog/docs/lessons-testing-evidence.md` (new entry).
<!-- SECTION:NOTES:END -->
