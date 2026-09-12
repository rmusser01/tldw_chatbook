# Fast PR Lane Design

## Problem

The trigger-recovery change in TASK-22250 stopped duplicate workflow creation,
but the ordinary pull-request test contract still requests roughly thirteen
concurrent runners per PR:

- three core shards at a time;
- three UI shards at a time;
- one artifact-lease spike;
- one artifact-workflow shape job;
- one minimum-Textual job;
- four focused guard workflows.

Two PRs can request approximately twenty-six runners, and a routine burst of
active PRs can still saturate the account after its GitHub Pro upgrade increased
the standard hosted-runner ceiling from twenty to forty. The fix must bound PR
demand rather than treating added capacity as permission to preserve unbounded
fan-out.

The rollout must also preserve the existing required check. `dev` protection
requires the job context `Derived artifacts reproduce from their sources`.
Creating a new required context can strand current PR heads that never reported
it, while workflow- or path-level skips can leave required checks pending.

## Goals

1. Give every PR into `dev` a fast, truthful test verdict on one runner.
2. Preserve the existing required status context without a branch-protection
   migration.
3. Keep derived-artifact checks install-free and independently diagnosable.
4. Preserve comprehensive test coverage on `main` and manual runs, and make
   the intended nightly coverage actually schedulable from the default branch.
5. Bound an ordinary, unlabeled, non-GGUF PR's peak runner demand to the fast
   lane plus existing focused guards.
6. Avoid path heuristics, optional dependency stacks, and redundant test-path
   selection.
7. Prove the contract locally and on the implementation PR before merging.

## Non-goals

- Making the fast lane representative of every product test.
- Removing, weakening, or re-sharding the comprehensive suites.
- Changing application behavior or dependencies.
- Moving CI to self-hosted or larger runners.
- Automatically rewriting, closing, or reopening existing PRs.
- Treating the GitHub Pro upgrade as the repository-side solution.

## Selected Architecture

The existing `.github/workflows/derived-artifacts.yml` workflow becomes the
owner of two jobs:

1. `pr-fast-lane`, displayed as `PR Fast Lane`, runs only for
   `pull_request` events.
2. `derived-artifacts`, still displayed as
   `Derived artifacts reproduce from their sources`, remains the required
   context and final aggregator.

The required job declares `needs: [pr-fast-lane]` and uses `if: always()` so it
still runs when the prerequisite fails or is skipped. Its first PR-only verdict
step fails unless `needs.pr-fast-lane.result == 'success'`. Every existing
derived checker retains `if: !cancelled()`, so a failed fast lane does not hide
artifact drift and an artifact failure does not hide later artifact failures.

For `push` events, `pr-fast-lane` is skipped, `derived-artifacts` still runs due
to `always()`, and the PR-only prerequisite assertion is skipped. This preserves
the current install-free `dev`/`main` push guard.

The stable job display name is not renamed. Branch protection continues to
require exactly the context it already knows, while the fast lane becomes a
transitive requirement.

### Result contract

| Event | Fast lane | Required derived job | Required outcome |
| --- | --- | --- | --- |
| PR, fast lane succeeds | Success | Runs all derived checks | Passes only if derived checks pass |
| PR, fast lane fails | Failure | Runs, records prerequisite failure, then runs derived checks | Fails |
| PR, fast lane is skipped or cancelled | Skipped/cancelled | Runs when GitHub permits post-cancellation work and rejects non-success; otherwise the required check is cancelled | Cannot satisfy protection |
| Push to `dev` or `main` | Skipped | Runs all derived checks | Reports artifact verdict |

No workflow-level or path filter is added to the required workflow. A required
workflow skipped before jobs are created can remain permanently expected.

## Fast-Lane Test Contract

The lane uses Ubuntu, Python 3.11, and the project's exact Textual 8.2.8 pin. It
runs serially with a twenty-minute job timeout.

The pytest selection is:

- `Tests/CI`
- `Tests/test_smoke.py`
- `Tests/Model_Artifacts/test_operation_leases.py`
- `Tests/Model_Artifacts/test_operation_leases_process.py`
- `Tests/UI/test_mcp_workbench.py`
- `Tests/UI/test_mcp_tools_mode.py`

These paths do not overlap. In particular,
`Tests/CI/test_textual_runtime_contract.py` is covered by `Tests/CI` and must not
also appear as an explicit argument. This repository has previously observed
pytest silently replacing a directory selection with one nested file when both
were listed, producing a misleading green run with most tests absent.

At `origin/dev` commit `0d83188e14`, this exact selection collected and passed
661 tests in 3 minutes 41 seconds under the repository's Python 3.12 virtual
environment. That run establishes the baseline target health, not the final
environment contract. Before the PR is opened, the same selection must collect
the expected nodes and pass in a clean Python 3.11 environment containing only
the dependencies below.

### Dependency boundary

The lane installs:

- the project with `pip install -e .`;
- `pytest`;
- `pytest-asyncio`;
- `pytest-timeout`;
- `packaging`, because a CI contract imports it directly.

The lane does not install `requirements-test.txt`. That file includes Torch,
Transformers, ChromaDB, Playwright, Docling, and other optional stacks whose
download and import costs are incompatible with a fast required gate. PyYAML,
Textual, portalocker, loguru, and the other runtime imports used by these tests
arrive through the core project dependency set.

The lane does not use xdist or pytest-shard. One process avoids additional
dependencies and makes runner consumption exactly one. If the clean GitHub run
approaches the timeout, target ownership must be reviewed before parallelism or
optional dependencies are added.

## Comprehensive Workflow Contract

`.github/workflows/test.yml` no longer listens to `pull_request`. Removing the
event is preferable to creating a workflow full of skipped matrix jobs and
eliminates the current `Artifact Lease Gate`/`Test Summary` failure mode in
which skipped prerequisites are treated as failures.

It also no longer owns `schedule` or the `nightly-deep` job. GitHub evaluates
scheduled workflows only from the repository's default branch, which is
`main`. The live `main` version of `test.yml` has no schedule, so the schedule
currently present only on `dev` has never created a run. A dedicated
`.github/workflows/nightly-deep.yml` owns `schedule` and `workflow_dispatch`,
lives identically on `dev` and `main`, and explicitly checks out `dev` before
running the existing five-environment full-tree matrix. A short prerequisite
job resolves `dev` to one immutable commit SHA; every matrix leg checks out
that same SHA and records it in the job summary. Staggered runner starts must
not turn one cross-platform run into verdicts for different commits.

Event ownership becomes:

| Event | Coverage |
| --- | --- |
| Push to `main` | Sharded core and UI suites, artifact-lease matrix and shape gate, minimum-Textual job, and test summary |
| Schedule | Dedicated default-branch-owned five-environment `Nightly Deep` workflow, explicitly checking out `dev` and running the full `Tests/` tree |
| Manual `Tests` dispatch | Existing sharded/specialized jobs and full manual suite |
| Manual `Nightly Deep` dispatch | The same five-environment full-tree matrix used by the schedule |
| Pull request into `dev` | No `Tests` workflow run; the required workflow owns the fast lane |

The nightly event exercises the full `Tests/` tree once; it does not also launch
the core/UI shard matrices. “Full coverage nightly” means the complete tree,
not duplicate executions through two job topologies. Keeping the deep run in a
dedicated workflow also lets operators dispatch it without launching the
manual sharded suite at the same time.

Because `test.yml` no longer comments on PRs, its `issues: write` and
`pull-requests: write` permissions and PR-comment step are removed. The workflow
retains only `contents: read`.

## Capacity Contract

The required workflow's two jobs are sequential, so it consumes at most one
runner at any instant. The other routine PR workflows have three runnable jobs:
CSS bundle, performance, and Backlog guards. An ordinary, unlabeled,
non-GGUF PR therefore has a peak of at most four runners, and usually fewer
because those three guards are path-scoped.

That is a routine-lane bound, not a repo-wide upper bound. A PR touching both
TASK-2062 GGUF evidence scopes can add six matrix jobs, for a peak of ten. If it
also carries the opt-in `task-19637-platform-evidence` label, a synchronize
event can add three platform jobs, for a peak of thirteen. The other historical
platform-evidence workflows run only on their matching label event and remain
explicit, exceptional evidence suites. This change deliberately preserves
those purpose-specific contracts rather than hiding them inside an inaccurate
global concurrency claim.

The existing PR concurrency policy remains: pull-request runs are not cancelled
in progress, while superseded non-main push runs may be cancelled. A fast lane
finishing in minutes makes bounded completion preferable to restoring the
cancellation loop that previously prevented verdicts.

## Rollout

No branch-protection API change is made.

The rollout uses two atomic PRs because the PR policy belongs on `dev`, while a
scheduled workflow must exist on default-branch `main`:

1. The fast-lane task targets `dev`. It adds the fast lane and aggregation, removes PR
   and schedule admission from `test.yml`, removes that file's embedded nightly
   job, and adds the dedicated `nightly-deep.yml` source.
2. Existing TASK-19600 owns activation on `main` after the first PR merges. It
   adds only the exact reviewed `nightly-deep.yml` file from `dev`; it does not
   promote unrelated `dev` changes. The file is then registered from the
   default branch, manually dispatched once, and observed on its next real
   scheduled event before TASK-19600 is closed.

The dev implementation PR exercises the changed required workflow from its own
merge ref and must report the existing required context. Once merged, every new
or updated PR uses the transitive fast-lane gate. An unchanged existing PR may
retain an older passing instance of the same context until its next PR event
because branch protection is not strict. This bounded grandfathering is
accepted. The rollout does not close/reopen PRs, force-push branches, synthesize
statuses, merge all of `dev` merely to activate a schedule, or claim that a base
update will reliably backfill the lane.

## Security and Failure Isolation

- Workflows retain `contents: read`; the fast lane receives no secrets and does
  not use `pull_request_target`.
- Untrusted PR code executes only on an ephemeral GitHub-hosted runner.
- The install-free derived job remains separate, so dependency/setup failures
  are visible as `PR Fast Lane` failures instead of masquerading as artifact
  drift.
- The required aggregator fails closed for every non-success fast-lane result.
- A twenty-minute timeout prevents dependency resolution or a hung focused test
  from occupying a runner indefinitely.

## Verification

Workflow-contract tests are written before workflow edits and must prove:

- `test.yml` has no `pull_request` trigger;
- `test.yml` has no schedule or embedded nightly job;
- the dedicated nightly workflow owns schedule and manual dispatch, checks out
  one resolved `dev` SHA in every leg, records it, and retains the existing
  five-environment full-tree matrix;
- main and manual coverage ownership remains intact;
- the fast lane uses one non-matrix Ubuntu/Python 3.11 job with a twenty-minute
  timeout;
- the exact target list is present once and contains no directory/file overlap;
- the fast lane does not install `requirements-test.txt` or optional extras;
- the stable required job name is unchanged;
- the required job needs the fast lane, runs with `always()`, and rejects every
  PR prerequisite result other than `success`;
- neither fast-lane nor required-gate jobs or steps may use
  `continue-on-error`, and the fast-lane pytest command is pinned exactly so
  selection-suppressing flags cannot turn collection or a subset into a pass;
- push events still run the derived checks despite the skipped fast lane;
- artifact checker steps remain install-free and continue after earlier
  failures;
- pull-request workflow runs are not cancelled in progress.

Verification then requires:

1. `--collect-only` on the exact target list with the expected nonzero count.
2. The exact target list passing in a clean Python 3.11 minimal-dependency
   environment.
3. The focused CI workflow-contract suite passing.
4. Mutation evidence: remove the aggregator prerequisite assertion and prove a
   contract test fails; change a fast-lane dependency result to skipped/failure
   and prove the required job cannot pass.
5. Ruff on changed Python tests, YAML parsing for changed workflows, and
   `git diff --check`.
6. A live dev implementation-PR run showing no heavyweight `Tests` workflow, one
   `PR Fast Lane`, and a completed truthful
   `Derived artifacts reproduce from their sources` required verdict.
7. A main activation PR containing only the identical dedicated nightly
   workflow; after merge, a successful manual dispatch and a real `schedule`
   event whose jobs reach terminal, truthful verdicts against `dev`.

The repository's full local test suite is not required for this workflow-only
change. The comprehensive GitHub suites and the focused contract tests are the
relevant evidence.

## ADR Check

ADR required: yes

ADR path: `backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md`

Reason: this establishes the repository's long-lived required CI aggregation,
dependency boundary, and coverage cadence across pull-request, main, nightly,
and manual events.
