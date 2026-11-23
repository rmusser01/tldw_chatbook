---
id: TASK-42425
title: "Three Console projection-cadence tests fail UI Fast Lane 36 under CI load"
status: Done
assignee:
- '@claude'
created_date: 2026-11-22 22:10
updated_date: 2026-11-23 03:31
labels:
- follow-up
- ci
- console
- tests
- flaky
priority: high
references:
- Tests/UI/test_console_projection_reads_off_loop.py
- tldw_chatbook/UI/Console_Modules/config_sync.py
- scripts/ui_pr_gate_census.txt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 21 of the follow-up burn-down (2026-11-22), in the CI runs the round's filing read. Checked on the round-21 stack 77f124a740 (dev ddd3c02052 plus seven ports); paths are under `tldw_chatbook/` unless they start with `Tests/` or `scripts/`.

**Why.** In a scan of the 140 Derived Artifacts PR runs created from 2026-11-20 00:00Z to about 21:50Z on 2026-11-22 (every attempt; times are job start times), the required "UI Fast Lane (36)" failed in 11 jobs, from 2026-11-20 09:54Z to 2026-11-22 20:26Z, each on one node of `Tests/UI/test_console_projection_reads_off_loop.py`, a whole-file census entry:
- `test_explicit_syncs_of_a_run_start_share_the_next_ticks_projection` (line 410), 7 jobs (including runs 44954347282, 44974749607, 45031398878, 45048643145 attempt 2, 45050972963): `assert calls[-1] is True` at line 491 (`False is True`), or `assert late_requests and ...` at line 522 (`assert ([])`).
- `test_a_full_sync_slower_than_the_display_life_does_not_replay_itself` (line 227), 1 job (45050945775): `a FULL sync whose awaits outlived the display replayed itself` at line 266.
- `test_a_full_sync_asked_for_during_a_pause_runs_once_after_resume` (line 1827, `@private_profile_test`), 3 jobs (including 44976516789): `assert started_at < armed_at + delay` at line 1914 fails in the child.

The file passes locally at the base (`-n 4`, load 6-8).

All three pin the Console's sync and projection cadence (coalescing a run start's syncs into the next tick, not replaying a slow FULL sync, running a paused request once after resume). They drive a fake clock but still depend on real task ordering, so a loaded runner can reorder the awaits they count. Whether any failure is a real cadence bug (a replayed FULL sync, a request owed to the wrong tick) is not known.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 For each of the three nodes, the notes record whether the CI failure is test ordering or a product cadence defect, with the evidence
- [x] #2 A defect is fixed in product code with a test shown red first; an ordering dependence is removed from the test, shown needed by a temporary delay (reverted) that fails the old test
- [x] #3 The file passes in repeated runs under parallel load (at least 20 runs, or a soak recorded in the notes)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Collect every failed UI Fast Lane (36) job since 2026-11-20 and read each failure (read-only `gh run view --job <id> --log-failed`).
2. For each node, trace which product return path produces the failing value (a projection that declined, a FULL that returned before its transcript, a FULL that returned False before its rail step) and name the race: coordinator or config-lock contention from unrelated work, an expired display, or a product cadence defect.
3. Reproduce each filed failure locally with a temporary forced overlap (a thread holding the contended lock, or a delay) that turns the old test red on the filed line; revert it and record the red.
4. Fix where it lives: product code (with a red-then-green test) for a cadence defect; otherwise isolate the test from the unrelated contention as TASK-40968 does, keeping every behaviour assertion and timing bound that is the product contract; deadline waits, not counted pauses.
5. Check sibling cadence/timing asserts in the same file for the same exposure and fix those that share it.
6. Soak: the file 20 times at -n 4 under extra load, and 5 times serially with the UI lane's flags; then preflight, ruff F and size ratchets.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Verdict (AC#1): test ordering, not a product cadence defect, for all three nodes and one more.** Every failure was the test's own sync meeting the storage coordinator (`storage_admission._lock`) held by another thread at a checked display render. `_checked_display_status` takes that lock with `acquire(blocking=False)` before and after each render, and a busy coordinator declines the render. The product then leaves it to the whole-sync replay, which renders in another task, as designed (TASK-40968). The tests read their own task's render, so they counted that deferral as a cadence failure. The cadence decisions themselves (left to the poll, owed, served) are made before the render and were right in every logged case. Every refused proof was fresh (0.0-0.6 s old against a 1 s life).

**CI evidence.** Of the 228 Derived Artifacts runs created since 2026-11-20, I scanned every attempt of the 213 that did not end in success, and found 12 failed UI Fast Lane (36) jobs (the filing counted 11):
- `test_explicit_syncs_of_a_run_start_share_the_next_ticks_projection`, 7 jobs. Line 491 `assert False is True` (the tick's projection returned False) in 137289973758, 137002801775 and 136379457523. Line 522 `assert ([])` (the FULL returned at its declined projection, before the transcript) in 137297601294, 137237410875, 137073602499 and 136377707759.
- `test_a_full_sync_slower_than_the_display_life_does_not_replay_itself`, 1 job: 137297667635, line 266.
- `test_a_full_sync_asked_for_during_a_pause_runs_once_after_resume`, 3 jobs: 137102797398, 136260569312 and 136064036207. In all three the child reported `assert <Task ...> in set()`: on those trees (checked at their head shas through the contents API) line 1914 was `assert worker._task in deferred_rail_reads`; round 20's port (#3703) later moved it eight lines down on dev. The filing's `started_at < armed_at + delay` is line 1914 on the base, not in the failing trees, and that wall-clock bound never failed.
- Sibling, not in the filing: `test_a_context_snapshot_published_during_a_run_projects_on_the_next_tick`, 1 job: 137004881929, line 403 `assert (0 + 1) in []` (the next tick's projection never rendered).

**Local reproduction at base, with the holder named.** A temporary probe (deleted) ran the unchanged tests from a frozen copy of the base file. It logged every non-accepted `_checked_display_status` with the coordinator's owner thread and stack, every whole-sync replay's caller, and every FULL that began. Load was 35-59, at -n 4 to -n 6. Counts below are test executions. It caught natural failures for all three nodes and one sibling:
- Node 1, 2 failures in 48 executions, both at line 491. The poll tick's `project_console_tick` declined with the coordinator busy (proof age 0.0). The holders were a ChaChaNotes transaction admitting its connection (`_enter_transaction -> get_connection -> participants.core_access -> storage_admission.check`) and a repository operation (`participants.core_operation -> _repository_operation`).
- Node 2, 3 failures in 48 executions. Two were the FULL's projection and one was its rail step (proof age 0.6, after the rewarm). The holders were a ChaChaNotes connection admission, `private_sqlite.admitted -> acquire_storage`, and a repository operation.
- Node 3, 1 failure in 66 executions. The resume's FULL declined at its projection (busy, age 0.011), so it returned False before its rail step and `deferred_rail_reads` stayed empty. This is the CI symptom.
- `test_a_control_bar_sync_builds_the_workspace_context_once`, 2 failures in 24 executions: `_sync_console_control_bar deferred`, busy, age 0.016-0.033. This node also failed once in my first plain base run (-n 4, load about 20).
The snapshot sibling did not fail in 24 local executions. Its CI symptom matches the forced overlap below.

**Forced overlap (AC#2): red on the old tests, green on the new ones.** I generated temporary copies of the base file and of the fixed file (both deleted). In each copy a thread took `storage._lock` and held it for 0.3 s just before the n-th checked render of the measured sync. Old tests, 7 of 7 red with the filed symptoms:
- node 1: `assert False is True` with the tick forced, and `assert ([])` with the late-request FULL forced;
- node 2: "a FULL sync whose awaits outlived the display replayed itself", with the projection forced and again with the rail step forced;
- node 3: the child reported `assert <Task ...> in set()`;
- snapshot: `assert (0 + 1) in []`;
- workspace: `_sync_console_control_bar deferred`.
Fixed copies: 10 of 10 passed, 7 of them under the same forced holds (a log confirmed each hold fired) and 3 unforced.

**Fix (test only; no product change).**
- New helper `_isolate_checked_renders(monkeypatch)` (TASK-40968's pattern, from one choke point). It wraps `console_spend_projection._run_checked_display_sync` so each checked render in the calling task waits for the coordinator (`acquire(timeout=30)`) and holds it, with no await under it. Renders in other tasks are unchanged. An expired or moved display still declines.
- Applied to node 1 (the owner task that counts projections), node 2 (the FULL awaited in the test task) and five siblings with the same exposure. Two siblings have CI or local reds: the context-snapshot tick and the workspace-context bare syncs. Three have no recorded red: the `_project` calls in `test_run_start_rereads_an_expired_display_...` and `test_a_projection_rereads_an_aging_display_...`, and node 2's second FULL, which the same install covers.
- Node 3 already allowed the resume's FULL to defer its rail read "while storage is busy". It assumed the projection before it could not defer. The test now records a deferral from `project_console_tick` as well as from `sync_console_rail_visibility` (`deferred_renders`), and then waits, as before, for a FULL to render. I kept its "launched once", "before the backoff" and "stopped timer delivers nothing" assertions unchanged.

**Negative controls on the fixed tests** (temporary source edits, run, then reverted):
- `rewarm_console_presentation` returns at once: node 2 fails with "replayed itself". The isolation does not hide an expired display.
- `_left_to_the_poll` returns False: node 1 fails with "3 explicit sync(s) projected inside the run start's poll interval".
- Resume asks for a delayed sync: node 3's child times out waiting for the resume's launch.

**Siblings checked and left alone.** The 40968 and round-20 sites were already isolated. The backoff tests' wall-clock bounds (`3 <= len(window) <= 6`, `rendered < armed_at + delay`, the survivor tick's `loop.time() < armed_at + delay`, node 3's `started_at < armed_at + delay`) never failed in the 12 CI jobs, the local probes or the soak below. One exposure remains, seen locally only in a background task: node 2's FULL runs `sync_console_core_state`, whose non-blocking config-lock probe can set the replay flag if another thread holds a config lock. It was never seen in node 2, so I did not guard it; it is listed as a follow-up.

**Soak (AC#3).** On the fixed file, alongside other agents' load (1-minute load average 23-46 across the runs):
- 20 of 20 whole-file runs at -n 4 passed (29 of 29 each). These were two concurrent loops of 10, run beside the serial loop.
- 6 of 6 more at -n 6 passed (load 31-37).
- 5 of 5 serial runs with the UI lane's flags (`-p no:xdist --timeout=180`) passed, 284-370 s each (load 23-42).
For contrast, at base the same load gave the probe failures above. One base run under `taskpolicy -b` failed only on 30 s setup and `private_sqlite_helper_timeout` app-boot timeouts, which is the starvation artefact the `taskpolicy -b` lesson describes, so I did not use `-b` for the soak.

**Other checks.** `ruff check` (all rules, and `--select F`) passes and `ruff format` reports the file already formatted. `Tests/CI/test_console_approval_critical_gate.py`, which names the file, passed 6 of 6. `./scripts/preflight.sh` passed through the wrapper. No size ratchet lists the file. The census entry is unchanged (no new test).

**Files.** `Tests/UI/test_console_projection_reads_off_loop.py`, and a `backlog/docs/lessons-testing-evidence.md` entry: map a CI failure by its assertion text, not its line number, and isolate every own-task render in the file.
<!-- SECTION:NOTES:END -->
