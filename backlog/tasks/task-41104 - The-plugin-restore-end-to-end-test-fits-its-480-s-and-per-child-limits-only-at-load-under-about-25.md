---
id: TASK-41104
title: "The plugin restore end-to-end test fits its 480 s and per-child limits only at load under about 25"
status: In Progress
assignee:
- '@claude'
updated_date: '2026-11-17 00:18'
created_date: 2026-11-15 02:20
labels:
- follow-up
- plugins
- tests
- flaky
priority: low
references:
- Tests/Plugins/test_recovery.py
- .github/workflows/plugin-qualification.yml
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** `test_plugin_backup_restores_in_place_and_waits_for_authority_recovery` is in the Managed plugin qualification workflow (`plugin-qualification.yml:131`, a PR path filter on `tldw_chatbook/Plugins/**` and the backup files). A slow runner fails it on time, not on behaviour.

**Evidence.** `test_recovery.py:1843` sets `@pytest.mark.timeout(480)`; the restore children get 360 s (`:1872`) and 300 s (`:1879`). The comment above it (`:1840-1842`) records 4.0 min at load 15-20 and 5.4-6.9 min at load 25-40. TASK-40308's notes: at load 30-50 its restore children exceed the limits; with relaxed limits it passed in 777 s at load about 40 (children 401 s and 348 s); with the committed limits it passed in 326 s at load 19-41.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Either the test's time limits come from measured CI times of its children (recorded with run ids), with at least 2x headroom, or the slow part (the two Tools-group restores, mostly journal revalidation per the comment) is made faster, with before/after times recorded
- [x] #2 The test passes 3 of 3 locally at load 40 or more
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Measure the test's children locally with the load average (PHASE lines, child seconds) and collect the macOS CI times of the whole test with run ids.
2. Replace the fixed per-child wall budgets with the TASK-40306 hang detector (`_child(..., idle=...)`: fail once the child's process tree stops using CPU), keeping wall caps only as derived backstops, so a loaded child that is only slow is not killed. Restore and authority product code stay untouched (another stream owns them).
3. Run the test 3 of 3 at load 40 or more and record the times and loads.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Summary.** The test failed on load, not on behaviour. Its children got fixed wall budgets (60/360/60/300/60 s) inside a 480 s test limit. A loaded host stretches their wall time five-fold or more while their CPU barely changes. I changed the waiting:
- Each child fails once its process tree uses no CPU for 60 s (TASK-40306), which catches a blocked hang at any load.
- A hang that keeps polling still uses CPU. Every restore lock wait polls (`admission.py` `flock(LOCK_NB)` every 0.01 s, `lock.acquire(timeout=0.05)` loops), so for those only each child's own wall cap stops the hang. The caps are set per child from measurements.

No product code changed: restore and authority logic belong to the concurrent fu-r15-plga stream, so the "make the restore faster" alternative was not taken.

**Criteria (coordinator, r15pb-rev1).** The old criterion #2 began "Or" and was an alternative to #1. The coordinator authorized merging them into one either/or criterion (#1), with #1's original wording; the old #3 is now #2 (met).
- **#1 stays open, and so does the task.** No CI run has per-child times yet. It closes once this PR's qualification run records them (the `plugin_restore_child_*` JUnit properties now do) and the caps are tightened from them, with at least 2x headroom.
- Until then the caps below come from loaded local runs. The coordinator ruled that the test limit may come from loaded local runs, because criterion #2 requires the test to hold there.

**Changes.**
- `Tests/Plugins/test_recovery.py::test_plugin_backup_restores_in_place_and_waits_for_authority_recovery`:
  - Every child runs through `_child(..., idle=60)`, with its own cap in `_RESTORE_CHILD_CAPS`: plugins 160, first 720, unchanged 160, again 710, changed 160 s. The short children's caps are about 4x the slowest short child (seed, 41 s); the restores' are 2x their slowest (359 s and 355 s).
  - The test limit is the caps' sum plus a minute, 1970 s, so a stuck child always fails on its own cap, by name. That is 2.4x the slowest whole run (831 s).
  - Each child's wall and CPU seconds are printed and recorded with `record_testsuite_property("plugin_restore_child_<name>", ...)`, so the qualification lane's `plugin-qualification.xml` keeps per-child CI times. The reviewer's local JUnit showed all five, e.g. `first` "101 s wall, 43 s cpu".
  - The comments say plainly that a polling hang is stopped only by the cap.
- `Tests/Backup_Recovery/test_config_sibling_capture.py::_wait_while_busy` (review r15pb-rev1):
  - It kills the child's descendants before the child, so a hung restore leaves nothing running.
  - It sums CPU per process and skips one that cannot be read. Before, an unreaped zombie descendant threw the whole sample away, and a busy child was killed as idle.
  - The `_child` and `_wait_while_busy` docstrings name the polling-hang limit.
  - Two negative controls: `test_child_wait_kills_a_hung_childs_descendants` and `test_child_wait_counts_a_busy_child_beside_a_zombie_descendant`. Red before the fix ("the hung child's own child is still running"; the busy child was killed "after 2 seconds"); 4 of 4 helper tests green after.
  - The reviewer's probes after the fix: the orphan is gone, and the busy child beside a zombie exits rc 0 after 15.4 s. With `timeout=20`, the 0.01 s and 0.05 s poll hangs are stopped only by the cap (20.8 s, 21.3 s), as the comments now say.

**CI times.** No passing run keeps per-child times: captured output is shown only on failure, and the JUnit has no `system-out` (lesson added). The whole test's `--durations` line in 24 macOS qualification runs carrying 314eb9c4d2 (runs 43698619087 to 43737692261) read 66-310 s; the slowest was 310 s, in run 43698653112.

**Local measurements** (shared 18-core host; 1-minute load):
- Old limits:
  - Load 39-56: passed in 284 s.
  - Load 74-99: **failed** on the 480 s test timeout during `again` (first took 289 s). This is the red.
- Fixed idle, single cap (round 1):
  - Two runs at load 77-98: passed in 552 s and 557 s.
  - Three runs at load 85-127: **3 of 3 passed** (criterion #2) in 633, 646 and 642 s. Children (wall, CPU): plugins (24-33, 4), first (259-301, 48-49), unchanged (16-19, 3), again (206-270, 36-37), changed (15-18, 3).
- Review round, with caps 130/600/130/540/130:
  - One run at load 81-102, beside a heavy Backup_Recovery run: passed in 831 s. Children: plugins (41, 5), first (359, 50), unchanged (28, 4), again (355, 38), changed (20, 3). All were under those caps.
  - These were new slowest times, so the caps were re-derived from them with the same rule (160/720/160/710/160).
- Children's CPU stays about 95 s per run at load 13-127; the rest is contention.
- Negative control (round 1): a `time.sleep(3600)` at the top of the seed child failed the test in 83 s (`timed out after 64 seconds`, 0 s CPU).

**Trade-off.** The macOS qualification job already runs 20.0-28.6 of its 30 minutes. A polling hang in a restore child can reach that limit before the child's 710-720 s cap does, and lose the JUnit; a short child's hang now stops by 160 s. Follow-ups: tighten each cap from the per-child CI times the next qualification runs record, and look at the macOS job's headroom.

**Files.** `Tests/Plugins/test_recovery.py`, `Tests/Backup_Recovery/test_config_sibling_capture.py`, `backlog/docs/lessons-testing-evidence.md`.
<!-- SECTION:NOTES:END -->
