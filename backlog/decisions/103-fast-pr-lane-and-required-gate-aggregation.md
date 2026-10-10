# ADR-103: Fast PR lane and required gate aggregation

Status: Accepted (amended 2026-08-30 for admin/current-base enforcement); amended 2026-09-27 (nightly disabled, guards removed, GGUF evidence narrowed)
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
- The required workflow runs its two fast lanes in parallel (at most two
  runners at a time), then the install-free derived-artifact checks. With
  one routine path-scoped guard (perf-guard), an ordinary unlabeled, non-GGUF PR has a
  peak of at most three runners instead of approximately thirteen.
- That three-runner figure is not a global maximum. The two path-scoped GGUF
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
