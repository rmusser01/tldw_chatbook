---
id: TASK-22250
title: CI runs are swept by simultaneous burst cancellations
status: In Progress
labels:
  - ci
  - infrastructure
priority: high
---

## Description

Whole sets of CI checks are cancelled at the same instant, unrelated to the code
under test. This is the reason `test.yml` has produced no verdict since
2026-06-26, and it survives the sharding work: the shards themselves are fine,
they simply never get to finish.

Two observations from PR #2061 on 2026-08-24:

**Before re-running** — two runs on head `4204db845` were both killed at
20:33:2x despite wildly different durations (Derived Artifacts 4.7 min; GGUF
source ubuntu 20.1 min). Identical kill time across unrelated workflows with
unrelated runtimes is not a timeout and not a code failure.

**After re-running** — head `de126e038` reached `Tests` executing for the first
time under the new sharding (6 core shards + 12 UI shards), and **20 checks were
cancelled simultaneously**: all 6 core shards, all 12 UI shards, MCP min-Textual,
and one artifact-lease leg. The required check
(`Derived artifacts reproduce from their sources`) survived and passed.

Note this is NOT the `cancel-in-progress` concurrency rule: a push-triggered run
on a stable ref was swept too, so a `cancel-in-progress` change alone would not
fix it (this is why the proposed workflow edit was put on hold).

## Acceptance Criteria

- [x] The agent performing the cancellations is identified (account concurrency
      ceiling, manual queue clearing, or workflow configuration)
- [ ] A PR's test workflows can run to completion without being swept
- [ ] `Tests` produces a verdict on at least one PR

## Implementation Plan

1. Pin the approved trigger contract with focused RED tests: full ordinary
   coverage on PRs into `dev`, a release run on pushes to `main`, and only
   `nightly-deep` consuming scheduled runners.
2. Filter synchronization-capable workflows before GitHub creates duplicate
   runs for permanent promotion PR #602, while preserving focused push guards
   on `dev` and `main`.
3. Remove the obsolete PR #602 job-level exceptions and retain the existing
   core/UI `max-parallel: 3` bounds and push-only cancellation behavior.
4. Run the targeted CI contract suite, Ruff, YAML parsing, and diff checks.
5. Rebase on latest `dev`, open the recovery PR, and cancel only obsolete idle
   runs using current PR heads plus `dev`/`main` SHAs as the authority boundary.
6. Verify a fresh post-change `Tests` run starts and reaches a verdict, address
   every Qodo/review finding, rebase and reverify, then merge the exact verified
   head.

Design: `Docs/superpowers/specs/2026-08-29-account-ci-trigger-recovery-design.md`

Plan: `Docs/superpowers/plans/2026-08-29-account-ci-trigger-recovery.md`

ADR required: no

ADR path: N/A

Reason: this is an operational GitHub Actions scheduling and trigger-policy
change; it does not alter application architecture, storage, security, or a
cross-module runtime contract.

## Implementation Notes

Needs owner input: the cancellations originate outside the workflow files, so
they cannot be diagnosed from the repository alone. Candidates worth checking in
order — the account's concurrent-job ceiling (macOS runners are already scarce
and one UI shard has been observed queueing ~90 min), someone clearing the
Actions queue by hand, and org-level runner policy.

Until this is resolved, "re-run the cancelled checks" is the only available
mitigation, and it is not reliable: the second attempt on #2061 was swept more
broadly than the first.


## Update 2026-08-26 — cause identified and half fixed

**The cancellations were `cancel-in-progress`, not an external agent.** My
earlier note here argued the opposite, on the grounds that a push-triggered run
on a stable ref was swept too. That reasoning was wrong: a later push to the same
ref supersedes the earlier push run, same concurrency group. The Actions API
shows each batch's cancellation timestamp matching the NEXT batch's creation
timestamp.

Fixed in PR #2102 by scoping `test.yml`'s cancellation to `push` events only,
matching `derived-artifacts.yml` — which already did this and was, not
coincidentally, the one required check that reliably survived to report.

Confirmed empirically on #2102's own CI. Pushing `40c7f2d9` superseded
`2f10322d`; on the superseded head, same push and same instant:

| workflow | rule | outcome |
|---|---|---|
| Tests (fixed) | push-only | queued — survived |
| Derived Artifacts | push-only | queued — survived |
| Perf Guard | blanket | cancelled |
| CSS Bundle Guard | blanket | cancelled |
| Backlog Guard | blanket | cancelled |

Scope, precisely: this protects a RUNNING batch. GitHub concurrency keeps one
running plus one PENDING member per group, and a third arrival still replaces the
pending one — observed on this PR. Only a per-run group would prevent that, at
the cost of never superseding obsolete commits.

## The other half: runner starvation — still open, owner input needed

Cancellation was one cause of the missing verdicts. It is not the only one.

Measured 2026-08-26 while #2102 waited: **8 runs queued repo-wide for 23–40
minutes with ZERO in progress.** Nothing was cancelled and nothing was running.

The repo is public and user-owned. GitHub Actions concurrency is billed and
limited **per account, not per repository**, so runs here can be starved by other
repositories or branches under the same account. That cannot be diagnosed from
inside this repo.

What the owner would need to check:
- the account's concurrent-job ceiling and what is consuming it
- whether other repositories under `rmusser01` are running Actions concurrently
- any spending limit or billing state affecting Actions

Note the interaction: with cancellation now scoped to push, a superseded PR run
keeps its queue position and the newer run waits behind it, so under starvation
both eventually run rather than one replacing the other. That is correct for
getting verdicts but doubles runner demand for a branch under active work —
which matters only while starvation persists.


## Update 2026-08-26 (second) — there IS a second cancellation source

The supersession finding above is correct but does not explain everything. A
second, distinct mechanism is now documented, and it is the one the original
title described.

Measured on `fix/task-22251-findings` at head `d86e82800`, with **no newer
commit** (local and remote both at that SHA):

| workflow | timeout-minutes | job started | job cancelled | ran for |
|---|---|---|---|---|
| Perf Guard | 15 | 02:02:52 | 02:07:37 | 4m45s |
| CSS Bundle Guard | none (default 360) | 02:02:51 | 02:07:38 | ~4m47s |
| Backlog Guard | none (default 360) | 02:02:51 | 02:07:40 | ~4m49s |
| TASK-2062.1 / .2 GGUF | — | 02:02:51 | 02:07:39-40 | ~4m48s |

These were **running** jobs, killed within ~3 seconds of each other, roughly 5
minutes in. Not supersession — nothing superseded them. Not `timeout-minutes` —
one had 15 minutes, the others had the 6-hour default.

The same shape appeared on an unrelated branch (`fix/setup-wizard-uat`, batch
created 01:38:43): guards and GGUF cancelled, `Derived Artifacts` in progress,
`Tests` queued. So it is systematic, not branch-specific.

Also observed in the same window: 8 runs queued repo-wide for 23-40 minutes with
**zero** in progress, then capacity returning.

### What this needs from the owner

The repo is public and user-owned; Actions concurrency and billing are
**per-account**, so nothing inside this repository can identify the source.
Worth checking, in order:

- the account's concurrent-job ceiling, and what else under `rmusser01` consumes it
- Actions spending limit / billing state (a hit limit can stop and cancel jobs)
- whether any automation, bot, or another session calls the cancel-workflow API
- GitHub's own status for the relevant runner pools during these windows

### What is NOT the cause (ruled out with evidence)

- `timeout-minutes` — one victim had 15 min, others the 6h default, all died at ~5 min
- supersession by a later push — local and remote were the same SHA
- the `cancel-in-progress` rule — `Derived Artifacts` and `Tests`, which are
  push-scoped, were not hit in this burst while the blanket-rule ones were;
  but note that is correlation observed once, not a mechanism

## Update 2026-08-28 — the ceiling MEASURED, and it is not macOS

The remaining open AC ("the agent performing the cancellations is identified —
account concurrency ceiling, manual queue clearing, or workflow configuration")
is answered: **the account's concurrent-job ceiling, saturated by this
workflow's own fan-out.** Measured live via `gh api .../actions/runs/<id>/jobs`
while PR #2129 waited ~4 hours for a 4.5-minute required check.

**The correction that matters: macOS is NOT the bottleneck.** The workflow's own
concurrency comment and this task's Implementation Notes both blame "scarce
macOS runners". At the moment of measurement:

| label | running | queued |
|---|---|---|
| ubuntu-latest | 12 | 87 |
| windows-latest | 1 | 1 |
| macos-latest | **0** | **1** |

Ubuntu is the constrained resource, at roughly 12-13 concurrent slots.

**The mechanism.** One `Tests` run fans out to **25 jobs**, nearly all ubuntu
(12 UI shards + 6 core shards + artifact-lease and MCP legs). Six live runs were
therefore asking for ~100 ubuntu jobs against ~12 slots — about eight waves of
~35-minute shards.

**Why it blocks merges specifically.** `Tests` is not a required check, but its
~100 queued ubuntu jobs sit in the *same FIFO pool* as the short required ones.
`Derived artifacts reproduce from their sources` takes ~4.5 minutes and was the
last check holding PR #2129; it spent hours queued behind UI shards belonging to
other branches. A non-required workflow starves the required gate.

**A run showing `queued` is not idle.** `gh run list` reports the RUN as queued
while its jobs execute — the run inspected had 2 jobs in progress and 4
completed while listed as `queued`. Diagnosing from run status alone produces
the false conclusion "nothing is running at all"; always go to the jobs API.

Nothing was cancellable at the time: all four competing branches had OPEN PRs
(#2155, #2158, #2160, #2161) and both `dev` runs were on the current SHA, so
the queue was legitimate demand, not stale work.

### Options for the owner (each is a trade, none applied here)

1. **Cut the UI shard count.** 12 shards is the single largest contributor. At a
   ~12-slot ceiling, more shards than slots adds queueing without adding
   parallelism — the shards serialize anyway, and each carries its own setup.
2. **Narrow the trigger.** Full 25-job fan-out on every PR push is what
   multiplies across branches; a reduced set per push with the full matrix on
   merge or nightly would leave the required gate unobstructed.
3. **Raise the ceiling** (paid runners / larger plan) — the only option that
   keeps current coverage and cadence unchanged.

Not applied unilaterally: every option either reduces coverage or costs money,
and it affects every contributor's CI, so it is an owner decision.

## Acceptance Criteria (updated)

- [x] The agent performing the cancellations is identified (account concurrency
      ceiling, saturated by a 25-job fan-out; measured 2026-08-28)
- [ ] A PR's test workflows can run to completion without being swept
- [ ] `Tests` produces a verdict on at least one PR
- [x] Owner picks among the three fan-out/ceiling options above (cap concurrent
      shard fan-out and narrow task-specific evidence triggers)

### Why the docs-only skip was withdrawn (Qodo review, 2026-08-28)

The obvious saving — skip the suite when a PR touches only `backlog/` and
`Docs/` — was implemented, reviewed, and **reverted**, because the premise is
false in this repository. Qodo flagged that `Docs/fixtures/console-block-prompts`
holds JSON consumed by core tests. Checking the rest of the tree made it worse:

- **126 test files reference `Docs/`**, including
  `Tests/MCP/test_mcp_documentation_contract.py`, which asserts on specific
  markdown files (`Docs/User_Guide/mcp.md` and others).
- **78 references to `backlog/tasks`**, and tests read named task files.
- Decisively, tests **glob** the directory:
  `test_post_release_ux_hci_validation_plan.py:150` and
  `test_product_maturity_phase1_harness.py:81` both walk
  `backlog/tasks/*.md`. **Adding a task file is itself an input to those
  tests** — which is exactly what a "docs-only" PR does.

So no path heuristic can decide safely here: docs and backlog are load-bearing
test inputs, and a glob means even a brand-new file participates. A skip would
have been silent, and the failure mode is tests that never run.

**What remains, then:** the only safe levers are the ones that do not guess.
The OS-matrix narrowing landed (PRs get ubuntu; merge and nightly get all
three). Beyond that, the dominant cost is unchanged and unavoidable without a
deliberate coverage decision: the UI suite is **41.5 min x 12 shards ~= 8.3
job-hours**, and per-job setup is only ~8% (3.6 min of 45), so re-sharding
moves the number very little. Reducing what runs per PR reverses task-1465;
raising the ceiling costs money. Both are owner calls, unchanged by this work.

## Update 2026-08-28 — selected mitigation implemented, live proof pending

The owner selected the repository-side capacity controls. The implementation is
on `codex/task-22250-ci-queue-pressure` and intentionally leaves the two live
verdict criteria open until GitHub Actions demonstrates them on a PR.

- `Tests` now admits at most three core shards and three UI shards at once. The
  test coverage and shard count are unchanged; the matrix runs in waves so one
  PR cannot request the whole measured ubuntu pool.
- The permanent draft promotion PR (`dev` to `main`, currently #602) reports
  skipped runner jobs while it is a draft. A `ready_for_review` activity trigger
  guarantees that marking it ready starts fresh checks rather than relying on a
  later commit.
- The two TASK-2062 GGUF evidence workflows retain manual dispatch and their
  exact three-OS evidence, but PR runs are limited to their owned production and
  test paths.
- `Tests`, Derived Artifacts, CSS Bundle Guard, Perf Guard, and Backlog Guard now
  cancel only superseded push runs. Pull-request runs are never blanket-cancelled.
- The stale derived-artifacts contract was repaired to include the already
  existing `scripts/check_index_plan_pins.py` sixth checker.

### Review follow-up

- The draft promotion exemption is bound to exact PR #602 as well as its draft
  `dev`-to-`main` shape, so a fork cannot reuse those branch names to bypass CI.
- The two GGUF evidence path filters include the application, configuration,
  CSS, and pytest harness files imported by their selected evidence nodes.
- Rebasing onto `dev` exposed stale diagnostic-inventory drift from PR #2199.
  The two added statements log only stable operation names and exception types;
  the removed statement was the superseded Library Collections warning. The
  reviewed inventory was regenerated so the required check reflects the new
  base truthfully.

Queue cleanup used open-PR head SHAs plus current `dev`/`main` SHAs as the
authority boundary:

- `tldw_chatbook`: 228 queued records audited; 2 current `dev` runs preserved,
  8 obsolete `Tests` runs preserved because they already had executing jobs, and
  204 idle obsolete runs cancelled. This included PR #602's duplicate Perf Guard
  run. Fourteen three-day-old records remain as GitHub ghost entries: they have
  zero jobs, report `status=queued`, and both normal and force-cancel endpoints
  return HTTP 409 (`not been queued yet`). They consume no runner slots.
- `tldw_server`: 62 queued records audited; 54 current open-PR runs preserved.
  Eight three-day-old zero-job ghost entries returned the same HTTP 409 from
  both cancellation endpoints and were left in place rather than deleting run
  history.

The preserved current Derived Artifacts ubuntu job (run `33234414802`) remained
queued after the drain because the eight obsolete `Tests` runs still had jobs in
progress. No live completion or `Tests` verdict is claimed from that probe.

Verification:

- 38 focused CI workflow-contract tests passed.
- Ruff check and format check passed for the five targeted CI test modules.
- All seven changed workflow files parsed as YAML, and `git diff --check` passed.
- The full suite was not run; repository policy calls for targeted verification
  unless the owner explicitly requests a full sweep.

ADR required: no

ADR path: N/A

Reason: the change is operational CI scheduling and trigger policy, not a
long-lived application architecture boundary.

## Update 2026-08-29 — account workload source removed, live proof pending

Authenticated account inspection closed the ambiguity left by the repository-
only probes:

- All 442 active repositories owned by `rmusser01` were scanned. Only
  `tldw_chatbook` and `tldw_server` had live Actions work, and neither had an
  in-progress run at the audit instant.
- The account is on GitHub Free. August public-runner usage was fully discounted
  to a net $0, ruling out private-minute exhaustion as the cause.
- `tldw_chatbook` consumed 510,153 runner-minutes in August, approximately 93%
  of the account total. `tldw_server` consumed 33,634 runner-minutes.
- The live snapshot contained 96 queued plus 28 pending `tldw_chatbook` runs
  and 20 queued `tldw_server` runs. GitHub documents that Actions may be rate-
  limited as usage scales; the API does not expose a personal-account throttle
  flag, so the observed effective serialization is evidence of restriction,
  not a claim about an undisclosed GitHub enforcement reason.

The dominant duplicate source was the trigger contract. Every merge to `dev`
created the 23-job `Tests` suite once for the `dev` push and again when permanent
promotion PR #602 (`dev` to `main`) synchronized. The daily schedule also ran
the ordinary suite before its five-leg `nightly-deep` matrix.

The recovery branch now prevents that work before GitHub queues it:

- `Tests` runs the ordinary suite for PRs into `dev`, runs a release suite on
  pushes to `main`, preserves manual dispatch, and excludes ordinary jobs from
  the scheduled event so only `nightly-deep` consumes scheduled runners.
- Derived Artifacts, CSS Bundle Guard, Perf Guard, and Backlog Guard accept PRs
  into `dev` while retaining their existing `dev`/`main` push checks.
- The obsolete PR #602 job-level conditions were removed. The label-only
  TASK-598/601/602/603 evidence workflows were deliberately left unchanged
  because normal PR synchronization never triggers them.
- The existing core/UI `max-parallel: 3` bounds and push-only cancellation
  policy remain intact.

Local verification after the RED-to-GREEN contract change:

- 40 targeted CI workflow-contract tests passed.
- Ruff check and format check passed for both changed Python contract files.
- All five changed workflows parsed as YAML, and `git diff --check` passed.
- The full suite was not run; repository policy calls for targeted verification
  unless the owner explicitly requests a full sweep.

The two live acceptance criteria remain open. This branch is not the fix until
a fresh PR `Tests` run starts, survives without a simultaneous sweep, and
reports a completed verdict.

ADR required: no

ADR path: N/A

Reason: this remains an operational GitHub Actions trigger and scheduling
policy change, not an application architecture boundary.
