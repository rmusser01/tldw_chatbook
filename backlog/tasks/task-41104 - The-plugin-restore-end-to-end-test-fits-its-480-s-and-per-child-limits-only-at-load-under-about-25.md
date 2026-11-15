---
id: TASK-41104
title: "The plugin restore end-to-end test fits its 480 s and per-child limits only at load under about 25"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-15 19:27'
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
- [x] #1 The test's time limits come from measured CI times of its children (recorded with run ids), with at least 2x headroom
- [ ] #2 Or the slow part (the two Tools-group restores, mostly journal revalidation per the comment) is made faster, with before/after times recorded
- [x] #3 The test passes 3 of 3 locally at load 40 or more
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Measure the test's children locally with the load average (PHASE lines, child seconds) and collect the macOS CI times of the whole test with run ids.
2. Replace the fixed per-child wall budgets with the TASK-40306 hang detector (`_child(..., idle=...)`: fail once the child's process tree stops using CPU), keeping wall caps only as derived backstops, so a loaded child that is only slow is not killed. Restore and authority product code stay untouched (another stream owns them).
3. Run the test 3 of 3 at load 40 or more and record the times and loads.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Summary.** The test failed on load, not on behaviour. Its children got fixed wall budgets (60/360/60/300/60 s) inside a 480 s test limit, and a loaded host stretches wall time five-fold or more while the children's CPU barely changes. I changed the waiting, following TASK-40306: each child now fails once its process tree stops using CPU for 60 s, which is a hang at any load. The wall limits are only backstops, derived from measurements. No product code changed; restore and authority logic belong to the concurrent fu-r15-plga stream, so criterion #2 (making the restore faster) was not taken. It is the "Or" alternative to #1 and stays unticked.

**Changes** (`Tests/Plugins/test_recovery.py::test_plugin_backup_restores_in_place_and_waits_for_authority_recovery`):
- Every child runs through `_child(..., idle=60)` (`_RESTORE_CHILD_IDLE`).
- Each child's wall cap is `_RESTORE_CHILD_CAP = 620` s, twice the whole test's slowest macOS qualification run. A child's own CI time is under the whole test's, so each child has at least 2x headroom.
- The test limit is 1300 s, twice the slowest loaded local run (646 s).
- Each child's wall and CPU seconds (`resource.getrusage(RUSAGE_CHILDREN)` deltas) are printed and recorded with `record_testsuite_property("plugin_restore_child_<name>", ...)`. The qualification lane's uploaded `plugin-qualification.xml` now carries per-child CI times. A local serial `--junitxml` run showed `<property name="plugin_restore_child_plugins" value="63 s wall, 0 s cpu"/>`.
- The old M12 comment ("must fail within 8 minutes") is replaced. A hung child now fails in about 60 s, sooner than the old 300-360 s child limits. A busy hang runs to the backstops.

**CI times** (criterion #1). No passing run keeps per-child times: captured output is shown only on failure, and the JUnit has no `system-out` (lesson added). The whole test's `--durations` line in 24 macOS qualification runs carrying 314eb9c4d2 (runs 43698619087 to 43737692261) read 66-310 s; the slowest was 310 s, in run 43698653112. Run 43738208614 (unknown sha) read 226 s.

**Local measurements** (shared 18-core host; 1-minute load sampled every 20 s):
- Old limits:
  - Load 39-56: passed in 284 s. Children, wall only: plugins 20, first 118, unchanged 11, again 118, changed 7 s.
  - Load 74-99: **failed** on the 480 s test timeout during `again` (first 289 s). This is the red.
- New waiting, two runs in parallel at load 77-98: passed in 557 s and 552 s. Children (wall, CPU): plugins (32, 4), first (262, 49), unchanged (17, 3), again (208, 37), changed (18, 3).
- Criterion #3, three runs in parallel at load 85-127: **3 of 3 passed** in 633, 646 and 642 s. Children CPU: first 48-49 s, again 36-37 s, about 95 s per run, flat across loads. These ran with a 1200 s test limit, which was then set to 1300 s (2x 646); nothing else changed.
- Negative control: a `time.sleep(3600)` added at the top of the seed child, then removed. The test failed in 83 s at load 90-106 (`TimeoutExpired ... 'plugins'] timed out after 64 seconds`, child (64 s wall, 0 s CPU)). It did not wait out a fixed budget.

**Trade-off.** The macOS qualification job already runs 20.0-28.6 of its 30 minutes (12 recent runs). A *busy* hang here can still reach the job limit before the test's own limit, as it could with the old 480 s in the slower runs. A hang that stops using CPU fails in about a minute. Proposed follow-ups: tighten each child's cap from the per-child CI times this PR's qualification run records, and look at the macOS job's headroom.

**Files.** `Tests/Plugins/test_recovery.py`, `backlog/docs/lessons-testing-evidence.md`.
<!-- SECTION:NOTES:END -->
