# ADR-103: Fast PR lane and required gate aggregation

Status: Accepted (amended 2026-08-30 for admin/current-base enforcement); amended 2026-09-27 (nightly disabled, guards removed, GGUF evidence narrowed); amended 2026-10-28 for bounded UI capacity
Date: 2026-08-29
Related Tasks: [TASK-24403](../tasks/task-24403%20-%20Fast-PR-lane-preserves-required-gate-and-full-coverage-cadence.md), [TASK-25705](../tasks/task-25705%20-%20Reconcile-diagnostic-inventory-and-enforce-the-dev-required-gate.md)
Supersedes: N/A

## Decision

Pull requests into `dev` use one serial, minimal-dependency fast-test job whose
result is aggregated by the existing required `Derived artifacts reproduce from
their sources` context; comprehensive test coverage runs on `main`, manual
events, and a dedicated default-branch-owned nightly workflow instead of every
pull-request update.

   (Clarified 2026-09-27: since TASK-32908 the fast test runs as two parallel jobs,
   PR Fast Lane and UI Fast Lane; see Consequences.)

## Amendment (2026-08-30, TASK-25705) — adopted

The `dev` protection rule applies the existing required context to
administrators and requires the result against the latest base revision:

- `enforce_admins` is enabled; and
- required status checks use strict/latest-base enforcement.

The required context remains `Derived artifacts reproduce from their sources`.
No workflow name or prerequisite relationship changes in this amendment.

This closes a reproduced enforcement gap. PR #2228 merged into `dev` at
15:40:01 UTC while its required workflow was still queued: the fast lane did
not start until after the merge and the derived-artifact job did not start
until approximately 15:48 UTC. The merge introduced persistent-diagnostic
owners without regenerating their canonical inventory, leaving the checker
and its dependent summarization privacy boundary red. The required workflow
correctly detected the drift after merge, but branch protection had
`enforce_admins=false` and `strict=false`, so an administrator could merge
before that evidence existed and a result from a stale base could satisfy the
rule.

Repository-wide generated artifacts compose across pull requests. A required
artifact check therefore protects the branch only when every merger is bound
by it and its result describes the current base. The additional queue/rebase
cost is accepted: administrators wait for the same gate as other contributors,
and a base update may require a fresh result before merge. Force-push policy is
unchanged by this amendment.

**Note (2026-09-27):** the owner relaxed strict to `false` on 2026-09-12, recorded
in `backlog/docs/branch-protection-baseline.md` but not here. The owner restored
strict to `true` on 2026-09-27 after stale-base merges caused breakage, so this
amendment's rule is in force again. See the baseline doc for the reasons and
the runner-starvation root cause found the same day.

## Context

The account-wide CI investigation in TASK-22250 found that `tldw_chatbook`
generated approximately 93% of the account's runner use. One ordinary pull
request can request about thirteen concurrent runners after its core/UI shard
limits and focused guards are counted. Two active pull requests can therefore
exceed the former twenty-runner account ceiling, and several active pull
requests can still create avoidable queueing after the account's GitHub Pro
upgrade raised that ceiling to forty.

Trigger deduplication removed redundant `dev` push and permanent promotion-PR
runs, but each real pull-request update still launches the complete six-shard
core and twelve-shard UI matrices plus specialized jobs. Raising account
capacity does not bound that demand.

The `dev` protection rule currently requires only the stable job context
`Derived artifacts reproduce from their sources`. Adding a new required context
would be operationally unsafe: existing pull-request head commits may never
have reported it and can remain waiting for an expected check. Folding package
installation and tests directly into the existing artifact job would avoid
that rollout problem but would destroy the job's intentionally install-free,
roughly ninety-second diagnostic contract.

Repository paths cannot safely select which PRs need tests. Tests consume
files under `Docs/` and `backlog/`, including directory globs, so documentation-
only heuristics can silently skip load-bearing inputs.

The repository's default branch is `main`, but the schedule added to
`test.yml` during TASK-22250 exists only on `dev`; the live `main` version has no
schedule. GitHub evaluates schedules only from the default branch, and the API
shows that this configuration has created no scheduled `Tests` runs. Nightly
coverage therefore needs its own workflow installed on `main`, not merely a
cron entry on `dev`.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Add `PR Fast Lane` as a new required branch-protection context immediately | Existing PR heads may not report the new context, leaving them permanently waiting unless branches are rewritten or PRs are reopened. |
| Install dependencies and run tests directly inside the existing derived-artifacts job | It mixes unrelated ownership, removes the install-free diagnostic path, and makes artifact drift harder to distinguish from environment/test failures. |
| Keep full matrices on every PR and rely on the forty-runner Pro ceiling | The workload remains unbounded across active PRs and can still consume the whole account pool during ordinary bursts. |
| Select tests from changed paths | Documentation and Backlog files are test inputs; path heuristics have already been reviewed and rejected as unsound in this repository. |
| Reduce only matrix shard counts | Fewer shards reduce instantaneous fan-out but retain hours of setup and execution per PR, so several PRs still monopolize the shared pool. |
| Leave the nightly schedule inside `dev`'s `test.yml` | GitHub does not schedule a workflow version that exists only on a non-default branch, so this preserves an intention rather than a running control. |
| Merge all of `dev` to `main` only to activate nightly | It couples CI activation to every unreleased product change; a dedicated workflow can be promoted independently as one reviewed file. |

## Consequences

- The existing required job name remains unchanged, so branch protection needs
  no migration and existing PRs are not stranded.
- The fast-test jobs and derived-artifact job remain separate. The required job
  declares both fast lanes as prerequisites, runs with `always()`, and fails
  explicitly on pull requests unless both prerequisite results are `success`.
- The required workflow runs its test lanes in parallel, then the install-free
  derived-artifact checks. As amended on 2026-09-27 that was two fast lanes
  (at most two runners at a time), three runners with the one routine
  path-scoped guard (perf-guard), instead of approximately thirteen. It has
  grown since: the UI lane currently runs 28 shards, with at most 17 UI jobs running at once (TASK-37236 and the
  dated capacity amendments below), the Console P0 gate is a third lane (TASK-33621.27), and the PR
  Fast Lane's admission-sensitive half (two jobs since TASK-35575, three since Roleplay frame B2b), the Dreams
  tracking tests and the Dreams and Guardian boundary contracts are jobs of
  their own (TASK-34916, TASK-34929, TASK-34986), so an ordinary unlabeled,
  non-GGUF PR now requests 24 test runners in parallel, 25 with perf-guard
  (see the dated notes below).
- That figure is not a global maximum. The two path-scoped GGUF
  evidence matrices can add six jobs, and a synchronize event on a PR carrying
  the opt-in TASK-19637 label can add three more. Those exceptional evidence
  suites retain their explicit contracts.
- The original rollout allowed unchanged PR heads to retain a previously
  reported required result until their next pull-request event. The 2026-08-30
  amendment ends that grandfathering for merge eligibility: the required
  result must now describe the latest `dev` base and applies to administrators.
- The fast lane runs at the supported Python and Textual floor with only core
  application dependencies and explicit test utilities. Optional ML, document,
  and browser stacks remain outside the PR gate.
- Full-tree coverage remains mandatory, but moves to events with bounded
  cadence: `main` pushes, explicit manual dispatch, and a dedicated
  `nightly-deep.yml` installed on default-branch `main` that checks out `dev`.
- The nightly workflow resolves `dev` once and passes that immutable SHA to all
  five matrix legs. Each leg records the SHA, so one cross-platform verdict
  cannot silently combine commits when runners start at different times.
- The change is activated through two atomic PRs: the first changes the `dev`
  PR policy and prepares the reviewed nightly workflow; existing TASK-19600
  promotes only that identical workflow file to `main`, where GitHub can
  actually schedule it.
- The candidate target list must remain non-overlapping. Pytest has silently
  collapsed a directory argument when a file inside that directory was also
  listed, so collection count is part of verification evidence.
- A future change to the required context name, prerequisite relationship,
  event cadence, or fast-lane dependency boundary must update ADR-103 or
  supersede it.

## Links

- [TASK-24403](../tasks/task-24403%20-%20Fast-PR-lane-preserves-required-gate-and-full-coverage-cadence.md)
- [TASK-25705](../tasks/task-25705%20-%20Reconcile-diagnostic-inventory-and-enforce-the-dev-required-gate.md)
- [TASK-19600](../tasks/task-19600%20-%20Nightly-deep-test-tier-has-never-fired-cron-registers-only-from-the-default-branch.md)
- [Fast PR lane design](../../Docs/superpowers/specs/2026-08-29-fast-pr-lane-design.md)
- [TASK-22250](../tasks/task-22250%20-%20CI%20runs%20are%20swept%20by%20simultaneous%20burst%20cancellations.md)

## Amendment (2026-09-27): nightly disabled; duplicated guards removed

- **Full-tree cadence change.** `nightly-deep.yml` was disabled with
  `gh workflow disable` on 2026-09-27 (owner decision). It had produced 0 complete
  runs in 8 nights while using about 22% of the account's runner-minutes and 64% of
  its macOS minutes. Until CI throughput sub-project 3 restores it (`gh workflow
  enable`, once a run can finish and report), full-tree coverage comes only from
  `main` pushes and manual dispatch. The last `main` push was 2026-09-14. This
  records the loss rather than hiding it; the nightly had not produced a complete
  verdict before the change either.
- **Guards.** `css-bundle-guard.yml` and `backlog-guard.yml` were deleted. Their
  checks already ran inside the required job on every pull request and on every
  `dev`/`main` push. The Consequences runner count above is updated to match.
- **GGUF evidence narrowed.** `task-2062-1`/`-2` evidence workflows no longer fire on
  `app.py`, `config.py` or `css/**` edits (about 244 of the PR merges that triggered
  them in 30 days). They still fire on GGUF code, their own tests, `pyproject.toml`
  (the only native macOS/Windows install signal a PR gets) and the shared
  test-harness files they import. Their UI test files are not in the PR gate's UI
  census because neither is green on the fast lane's minimal dependency set (all 11
  failures are `RecoveryRequired: raw_source_selection_changed`). So until that
  harness gate is fixed (CI throughput sub-project 3), an edit confined to `app.py`,
  `config.py` or `css/**` does not exercise the GGUF screens on any PR.
- **GGUF source evidence narrowed again (2026-10-11, owner decision).** `task-2062-2` no longer
  fires on `Tests/UI/app_factory.py` or `Tests/private_profile.py`. In the 13 days after the
  first narrowing, 110 of its 230 runs (1,706 of 3,903 runner-minutes) were fired by those two
  files and no other trigger path. The two-week review (TASK-33160) measured the changed
  workflows at 70% below baseline against an 80% target; replaying those 13 days without the two
  paths gives 82%. `pyproject.toml` and both `conftest.py` files still trigger both workflows.
  The gap this adds: an edit confined to `app_factory.py` or `private_profile.py` does not
  exercise the GGUF source screens on any PR.
- **Strict.** Restored on 2026-09-27 (see the note under the 2026-08-30 amendment).
- Spec: `Docs/superpowers/specs/2026-09-27-ci-conflicts-and-waste-design.md`.

2026-10-03: the required check is now also started by the merge queue's `workflow_dispatch` (input `pr`); see ADR-218.

2026-10-10 (TASK-33621.27): the UI census (`scripts/ui_pr_gate_census.txt`) may list
one test's pytest node id as well as a whole file, so a P0 regression test whose
file is too slow for the lane can be gated on its own. The census checker
verifies a node id's file still defines the test, rejects an entry with
whitespace, rejects a node id beside its own whole file (the overlap rule
above), and resolves a bracketed parametrize id statically, refusing one it
cannot resolve. The same overlap helper pins every PR-lane pytest step.

The Console review's P0 regression tests run in the census (mounted
private-profile node ids) and in a third parallel lane, `console-p0-gate`
("Console P0 regression gate"), with the PR Fast Lane's plugin set and its
sandboxed / admission-sensitive split. They first went into the PR Fast Lane,
which then took 33m41s of its 35-min cap (run 38064483856). The required check
now `needs` all three lanes and fails when any of them is not `success`. This is
one more runner per PR: the PR Fast Lane, four UI shards and the P0 gate make
six concurrent test runners, then the aggregate.
`Tests/CI/test_console_p0_regression_gate.py` pins where each P0 test runs. The
required context name, event cadence and dependency boundary are unchanged.

2026-10-14 (TASK-34916): the PR Fast Lane's admission-sensitive pytest invocation
runs in its own job, `pr-fast-lane-admission` ("PR Fast Lane (admission-sensitive)"),
with the same install line, the same targets in the same order, and the same 35-min
cap; the required check `needs` it and fails when it is not `success`. Over 90
pull-request runs to 2026-10-14 the two invocations in one job took median 31.5 /
max 39.5 min and four runs were cancelled at the cap; each half replays at max
20.8 and 20.1 min. One more runner per PR: seven concurrent test runners, then
the aggregate.

2026-10-14 (TASK-34917): the UI Fast Lane runs six round-robin shards (was four);
the census and its order are unchanged. Four shards measured shard 3 at max 19.7
min against the 20-min cap; a replay of 26 PR runs' shard logs put the heaviest
shard at 17.4 min with five and 14.1 with six. Two more runners per PR: nine
concurrent test runners, then the aggregate.

2026-10-14 (TASK-34929): `Tests/Dreams/test_track_service.py`, in no pull-request
lane before, runs whole on every pull request in its own job, `dreams-tracking`
("Dreams tracking tests", 20-min cap, the PR Fast Lane's plugin set); the required
check `needs` it and fails when it is not `success`. It is not path-filtered: the
file also imports Notifications, Scheduling and the egress guard, and a skipped
job would fail the aggregate's verdict. Its own job, so no existing lane moves
toward its cap; its first run (38588725796) took 6m50s. One more runner per PR:
ten concurrent test runners, then the aggregate.

2026-10-14 (TASK-34986): PR2921's two Dreams/Guardian steps ("Run Dreams and
Guardian boundary contracts", "Run Dreams admission-sensitive contracts") run in
their own job, `dreams-boundary` ("Dreams and Guardian boundary contracts"),
moved verbatim from `console-p0-gate` with the same install line and 20-min cap;
the required check `needs` it and fails when it is not `success`. With them the
Console P0 regression gate measured median 16.1 / max 18.7 min over 76 PR runs
and was cancelled at the cap twice; replaying the 76 runs, the gate keeps median
11.6 / max 13.9 min and the new job takes median 5.2 / max 6.7. In the two cancelled
runs the gate's own steps took ~17.5 (a failing run) and ~16.1 min with setup, so
its worst case keeps about 4 min of headroom. One more runner
per PR: eleven concurrent test runners, then the aggregate.

2026-10-14 (TASK-33007, PR #3057): Phase 7 of the model-configuration redesign
took the UI lane to ten shards (was six). Its 92 census entries sit at the census
tail, ordered by `scripts/ui_lane_shard_replay.py`, and are 66-72 of the
lane's 109-119 pytest minutes in the PR's three CI runs. Replayed over eight
runs' shard logs (those three and five six-shard runs after TASK-34917), each
file at its slowest, the heaviest shard carries 15.2 pytest minutes with nine
shards, 13.7 with ten and 12.4 with eleven, plus ~1.1-1.7 min of setup. Ten is
the fewest that leaves a hung private-profile test's ~3.3-minute report room
under the 20-min cap, as TASK-34917 requires (~15.4 + 3.3 of 20). Four more
runners per PR: that made fifteen concurrent test runners per PR, then the
aggregate, and the matrix sets no `max-parallel`. Those runners bear on the
required check's wall-clock tail that TASK-34917 and TASK-34916 left open for an
owner decision; the owner chose to land the ten shards without `max-parallel`
(2026-10-15).

2026-10-19 (TASK-35575, owner decision): the PR Fast Lane's admission-sensitive
suites run in two jobs. `pr-fast-lane-admission` keeps the 41 targets `dev` had;
Phase 9 of the model-configuration redesign put its 54 (the 52 moved, plus
TASK-33666's two Stop-claim nodes) in a second job,
`pr-fast-lane-admission-2` ("PR Fast Lane (admission-sensitive 2)"), with the same
install line, 35-min cap and `--timeout=300`; the required check `needs` it and
fails when it is not `success`. With all 93 in one job, PR #3169's two runs took
29.8 and 27.4 min of the 35, so one hung test's ~5.3-minute report would have
passed the cap. Over 20 PR runs (2026-10-18/19) `dev`'s 41 alone took median
17.5 / max 20.0 min; the 52 alone ran in 266.9 s serial in the job's install set.
One more runner per PR: that makes sixteen concurrent test runners per PR, then
the aggregate.


2026-10-20 (TASK-35834): thirteen UI shards replace ten, retaining the serial
round-robin checker, all gated tests, dependencies, 20-minute job cap,
180-second test timeout and fail-closed aggregation. TASK-35834: thirteen UI
shards. At that point the total was nineteen concurrent test runners per PR,
twenty including perf-guard. Historical counts retain their dates.

The model-defaults whole-file selector becomes all 16 function selectors,
including every parameter case (415 -> 430 census entries). Only the existing
92-entry Phase 7 block is rebalanced; its first entry stays fixed. Two green
runs' worst file/node costs project 995s pytest at ten shards, 829s at twelve,
774s at thirteen and 797s at fourteen. Adding 110s setup and 15% runner variance
makes thirteen the smallest evaluated layout below the 1020s acceptance cap
(1000s). More shards are not monotonically better. All duration phases are
logged to correct node attribution; unknown costs in older top-25 logs share
only the unmeasured file remainder. Hosted qualification, including three
post-change PR attempts, is recorded in the TASK-35834 QA report.


2026-10-23 (TASK-36454..36458 CI qualification): repeated 20-minute UI job
cancellations require the existing replay/reorder/add-capacity policy. Current
UI lane at that checkpoint: 17 shards, with 23 eligible test jobs per PR;
Perf Guard can add one more. This is a scheduling count, not a concurrency or
queue-latency guarantee. Every selected test, the serial minimal-dependency
checker, 20-minute job cap, 180-second test timeout and fail-closed aggregate
remain unchanged. Timing qualification and the bounded inherited reordering
are recorded in the pending-draft QA report; no hosted pass is implied here.

Worst-observed file/node timing replay over the two recorded layouts requires
at least 17 shards even with ideal balance under the existing 1020s target
(110s setup reserve plus 15% variance). The selected bounded reordering projects
999.83s at seventeen; this is an estimate requiring current-head hosted CI.
The first 181 and last 104 census entries stay fixed, with all 720 unique
selectors preserved. See [qualification evidence](../../Docs/superpowers/qa/2026-10-23-textual-pending-draft-followups.md).

The same follow-up later appends two completion-safety regressions after the
original 720 positions (722 selectors total). The seventeen-shard matrix,
existing caps and aggregation policy stay unchanged; the QA report separates
the earlier passing component timings from final-head qualification.


## Amendment (2026-10-28, TASK-37236): smaller slices with bounded concurrency

Current UI lane: 28 shards. The current total is 35 eligible test jobs per PR.
The UI matrix sets `max-parallel: 17`; the six other required application lanes
therefore retain a peak of 24 test runners, or 25 with Perf Guard (23 and 24 before Roleplay frame B2b's third admission lane).
Eligible job count and peak concurrency are distinct. The existing required
aggregate waits for all 28 shards; every shard stays serial with the same
minimal dependencies, 20-minute job cap and 180-second test timeout. No selected
target, gate or failure condition is removed. The 1,140 census entries are
conserved exactly once, including complete parameter selections.

Three observed 17-shard layouts failed through timeouts despite limited-prefix
reordering; separate assertion failures remain failures to repair, not timing
passes. The last observed checkout consumed 289 seconds in shard 1 and 131 in
shard 13 before pytest could use the remaining job budget. Call-only tail models
also omitted roughly 50–145 seconds of per-case teardown in many jobs. Keeping
most positions fixed and treating checkout variance as a fixed per-shard cost
did not provide reliable capacity.

Use the existing `scripts/ui_lane_shard_replay.py` progress and all-phase costs
at their worst observation across the retained runs, then deterministic
longest-processing-time placement over the complete census. The measured
workload is 16,850.38 seconds including a 120-second allowance for TASK-37235's two
new native controls (116.644 seconds in actual loaded local Windows XML).
That local measurement is an allowance input, not hosted capacity proof. Comparing
17, 18, 20, 22, 24, 26, 28 and 30 shards makes 28 the smallest tested layout
meeting the existing 1,020-second target with a 310-second setup/checkout reserve
and 15 percent work variance: `602.73 * 1.15 + 310 = 1003.14` seconds. This remains
a forecast requiring actual final-head hosted CI. Update the projection if the
new control measurement exceeds its allowance; no hosted pass is inferred.

The additional eligible jobs can run in a later wave. This adds checkout/setup
work and can lengthen the tail of a workflow, but it preserves instantaneous
runner demand and avoids repeated whole-matrix retries or timeout increases.
Keeping 17 shards with the measured LPT ordering forecasts 1,450.75
seconds under the same reserve/variance basis; increasing the job timeout would
hide the capacity problem. Raising simultaneous UI runners would increase
account-wide queue pressure, so the concurrency ceiling stays 17.

GitHub's native [matrix concurrency control](https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/run-job-variations#defining-the-maximum-number-of-concurrent-jobs)
bounds simultaneous matrix jobs independently of the number of variations.
CI contracts pin the ceiling, matrix membership, eligible-versus-parallel
counts, unchanged limits and the fail-closed aggregation. Task and QA records
retain the failed runs, exact selected-target conservation, measured replay
inputs and scoped Superpowers review. This amendment implements the existing
replay/reorder/add-capacity policy rather than creating a second required gate.

Even ideal balance at 27 shards has a lower-bound forecast of 1027.70 seconds under the same assumptions, above the target; 28 is therefore the smallest feasible count in this model, not merely the first sampled success.


2026-10-31 (Roleplay frame B2b, TASK-33910.4, owner decision): a third
admission-sensitive job, `pr-fast-lane-admission-3` ("PR Fast Lane
(admission-sensitive 3)"), runs B2b's mounted Roleplay suites
(`test_roleplay_import_app.py`, `test_roleplay_item_list_app.py` and the painted
search-placeholder contrast test), moved out of lanes 1 and 2, with the same
install line, 35-min cap and `--timeout=300`; the required check `needs` it and
fails when it is not `success`. On PR #3334's run 41169081459 lane 2 with these
suites hit its 35-min cap at 99% (about 17 CI-min for 211 mounted tests, 485 s
serial locally). One more runner per PR: 24 concurrent test runners, 25 with
perf-guard, 35 eligible test jobs.
