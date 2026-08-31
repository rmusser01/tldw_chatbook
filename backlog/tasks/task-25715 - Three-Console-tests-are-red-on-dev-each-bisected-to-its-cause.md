---
id: TASK-25715
title: 'Three Console tests are red on dev: two bisected to #2220, one a flake'
status: To Do
assignee: []
created_date: '2026-08-31 14:27'
labels:
  - console
  - testing
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two Console rail tests pass at 4da99a884 and fail on origin/dev at 46c2b0e5f0fb. Both were found while baselining the Context rail UX work (PRs #2233, #2242, #2260) against dev, and neither is caused by it -- each is bisected below to the commit that introduced it. Filed so they are not silently re-attributed to whatever change happens to run next to them, and so the two owning changes get the decision they each imply.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 test_context_section_headers_match_inspector_title_band passes, or the contract it pins is deliberately retired with the Inspector/Context divergence recorded
- [ ] #2 test_active_reveal_queue_retains_only_identity_across_target_and_rail_removal is made deterministic -- it fails intermittently (~1 in 12) at every commit measured, so it is a flake, not a regression
- [ ] #3 test_console_workbench_standard_width_inspector_snapshot passes, or its "Blocked impact" assertion is updated to the Inspector's current copy
- [ ] #4 No test in this set is left red on dev without an owner
<!-- AC:END -->

## Evidence

Measured with `PYTHONPATH=<worktree> <main>/.venv/bin/pytest -p no:randomly`
in detached worktrees, so each commit's own source is what ran.

Both PASS at `4da99a884` (the commit the Context rail work branched from) and
FAIL at `origin/dev` `46c2b0e5f0fb`. A manual first-parent binary search over
the 60 merges between them isolated one cause each.

### 1. Header padding — first bad `c2f64f690` (#2220, Inspect rail UX burn-down)

```
Tests/UI/test_console_left_rail.py::test_context_section_headers_match_inspector_title_band
E  assert Spacing(top=0, right=1, bottom=0, left=1)
       == Spacing(top=0, right=0, bottom=0, left=0)
```

The test ties Context section headers to the Inspector's title band. #2220
changed the **Inspector** side to zero padding; the Context side still carries
`1`. The test is asserting a shared visual band that one of the two sides has
since left, so it fails on the Context header while naming neither change.

Worth noting for whoever picks this up: the same contract test is why
TASK-23193's header-chrome reduction was reverted (a 140x40 fit was given up to
keep the band). If the band is no longer real, that trade should be revisited
rather than inherited.

### 2. Reveal queue — first bad `0ef6f3fd4` (#2252, console-rail-adaptive-row-cap)

```
Tests/UI/test_console_rail_reconciliation.py::test_active_reveal_queue_retains_only_identity_across_target_and_rail_removal
E  textual.css.query.NoMatches: No nodes match '#console-left-rail-body'
     on ConsoleLeftRail(id='console-left-rail', ...)
```

The test removes the rail and then fires the queued reveal callback, pinning
that a deferred reveal holds only identities and degrades quietly when its
target is gone. After #2252 the callback resolves `#console-left-rail-body`
unguarded, so a reveal that outlives its rail raises instead. The queue still
carries no widget references -- the two `_contains_widget_reference` assertions
before it still pass -- so this is the lookup at the end, not the queue design.

Bisect also cleared the neighbouring Console merges: `3c081c79e` (#2233),
`4ae04314c` (#2242), `51d3fbdbf` (#2249) and `41176579f` (#2250) are all GOOD
for the test each is adjacent to.

### 3. Inspector snapshot — first bad `c2f64f690` (#2220), the same commit as (1)

Found on a post-merge sweep of `Tests/UI/`, which is how it stayed hidden: this
file was not in the per-batch sweeps run during the rail work.

```
Tests/UI/test_workbench_visual_snapshots.py::test_console_workbench_standard_width_inspector_snapshot
E  assert 'Blocked impact' in '<svg class="rich-terminal" ...>'
```

Same bisect method, same first-bad commit as the header-padding failure. #2220
renamed or removed the Inspector's "Blocked impact" copy without updating the
snapshot asserting it. So #2220 owns **two** of the three findings here, which is
worth knowing before anyone picks at them one at a time.

**Not in scope, already owned.** The other two failures in that same file --
`test_console_workbench_normal_and_compact_snapshots[normal]` and `[compact]`,
both asserting `'Library search:'` -- fail at `4da99a884` as well, so they
predate this work entirely. TASK-23147 already owns that label drift, and
TASK-23148 owns the rail-handle arithmetic in the same file. Recorded here only
so the next person to run this file sees five failures and knows which is which.

## Correction (2026-08-31): finding 2 is a FLAKE, and its bisect was invalid

Posted before this correction, in PR #2260's body and PR #2266's table:
`test_active_reveal_queue_retains_only_identity_across_target_and_rail_removal`
was attributed to `0ef6f3fd4` (#2252). **That attribution is withdrawn.**

A post-merge stability check caught it. Running the test alone, same command,
`-p no:randomly`, three times in a row: fail, pass, pass. Measured properly:

| Commit | Result |
|---|---|
| `46c2b0e5f0fb` -- the commit the bisect called FIRST BAD | **11 passed / 1 failed of 12** |
| `d81bd7a23` (dev after this work) | 12 passed / 0 failed of 12 |
| `d81bd7a23`, under 8-way CPU saturation | 10 passed / 0 failed of 10 |

It fails intermittently at roughly 1 in 12 to 1 in 24 **at both commits**, so it
is not a regression from #2252 or from anything else. My bisect ran the test
once per step; over a flaky test that search does not converge on a cause, it
converges on wherever the coin happened to land. It produced a specific,
plausible, wrong commit -- and the two initial failures that made me start the
search at all were themselves the flake, hit twice while the machine was busy
with concurrent test runs.

CPU saturation does not reproduce it, so the trigger is not simple load.

**Mechanism: unidentified. My first guess was wrong.** I wrote here that "the
reveal callback should guard the lookup and no-op when its rail is gone."
Reading the code, **it already does** -- at both layers.
`_active_reveal_is_current` returns False on `not self.is_attached`, and
`_reveal_active_section` wraps all three `query_one` calls in
`except (NoMatches, QueryError): return`. So the proposed fix was for a defect
that is not there, and the escape path is still unexplained.

Ruled out, so nobody repeats them (all at `d81bd7a23`, quiet machine):

| Hypothesis | Test | Result |
|---|---|---|
| Plain rarity | 30 consecutive runs | 30 passed |
| CPU contention | 10 runs under 8-way saturation | 10 passed |
| Cold bytecode cache | 6 runs, `__pycache__` purged before each | 6 passed |

Observed rate is roughly **1 in 60** at the tip and **1 in 12** at
`46c2b0e5f0fb`, on samples too small to separate those two numbers. Every
failure so far has appeared while other pytest processes were running, but
deliberate CPU saturation does not reproduce it, so "load" is not the mechanism
either -- possibly disk or scheduler contention rather than CPU.

The `await rail.remove()` / deferred-prune interaction remains the most
plausible area to look, since `remove()` only schedules the prune and
`is_attached` flips only when it runs -- but that is a place to start, **not a
diagnosis**, and it should not be written up as one until someone has a
traceback with `--tb=long` from an actual failure.

**Findings 1 and 3 are unaffected** -- both re-measured 0 passed / 5 failed of
5, deterministic, and their bisects to `c2f64f690` (#2220) stand.

## Notes

Filed in the same spirit as TASK-15512. `origin/dev` had by this point absorbed
the Context rail PRs, so "red on dev" alone no longer told me the failures were
not mine -- both had to be re-run at the pre-branch commit and then bisected
before either could honestly be called someone else's.

## Renumbering provenance

This task previously held id TASK-25713, colliding with the older
"Census-warm-boot-flakes-on-sys.modules-mutation-during-iteration" task that
arrived on dev first (created 14:12; this one 14:27 the same day).
Per the owner rule decided 2026-08-21 in TASK-19601 (**older id keeps it;
the younger task renumbers with a provenance note, regardless of Done
status**), it renumbered to TASK-25715. Citations to TASK-25713 in
this branch's own commit messages and in PR #2260's body refer to THIS
task; the other TASK-25713 holder is the older arrival and keeps the id.
