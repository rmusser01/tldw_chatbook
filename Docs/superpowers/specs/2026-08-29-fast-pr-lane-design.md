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
4. Preserve comprehensive test coverage on `main`, nightly, and manual runs.
5. Bound one PR's peak runner demand to the fast lane plus existing focused
   guards.
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

## Heavy Test Workflow Contract

`.github/workflows/test.yml` no longer listens to `pull_request`. Removing the
event is preferable to creating a workflow full of skipped matrix jobs and
eliminates the current `Artifact Lease Gate`/`Test Summary` failure mode in
which skipped prerequisites are treated as failures.

Event ownership becomes:

| Event | Coverage |
| --- | --- |
| Push to `main` | Sharded core and UI suites, artifact-lease matrix and shape gate, minimum-Textual job, and test summary |
| Schedule | Existing five-environment `nightly-deep` full-tree run, explicitly checking out `dev` |
| Manual dispatch | Existing sharded/specialized jobs, full manual suite, and nightly-deep matrix |
| Pull request into `dev` | No `Tests` workflow run; the required workflow owns the fast lane |

The nightly event already exercises the full `Tests/` tree through
`nightly-deep`; it does not also launch the core/UI shard matrices. “Full
coverage nightly” means the complete tree, not duplicate executions through two
job topologies.

Because `test.yml` no longer comments on PRs, its `issues: write` and
`pull-requests: write` permissions and PR-comment step are removed. The workflow
retains only `contents: read`.

## Capacity Contract

The required workflow's two jobs are sequential, so it consumes at most one
runner at any instant. With the four existing guard workflows, one PR's peak is
five runners. Under the forty-runner account ceiling, eight such worst-case PR
sets can execute concurrently before account-level queueing; typical demand is
lower because path-scoped guards often do not create runnable jobs.

The existing PR concurrency policy remains: pull-request runs are not cancelled
in progress, while superseded non-main push runs may be cancelled. A fast lane
finishing in minutes makes bounded completion preferable to restoring the
cancellation loop that previously prevented verdicts.

## Rollout

No branch-protection API change is made.

The implementation PR exercises the changed workflow from its own merge ref and
must report the existing required context. Once merged, every new or updated PR
uses the transitive fast-lane gate. An unchanged existing PR may retain an older
passing instance of the same context until its next PR event because branch
protection is not strict. This bounded grandfathering is accepted. The task does
not close/reopen PRs, force-push branches, synthesize statuses, or claim that a
base update will reliably backfill the lane.

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
- main, nightly, and manual coverage ownership remains intact;
- the fast lane uses one non-matrix Ubuntu/Python 3.11 job with a twenty-minute
  timeout;
- the exact target list is present once and contains no directory/file overlap;
- the fast lane does not install `requirements-test.txt` or optional extras;
- the stable required job name is unchanged;
- the required job needs the fast lane, runs with `always()`, and rejects every
  PR prerequisite result other than `success`;
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
6. A live implementation-PR run showing no heavyweight `Tests` workflow, one
   `PR Fast Lane`, and a completed truthful
   `Derived artifacts reproduce from their sources` required verdict.

The repository's full local test suite is not required for this workflow-only
change. The comprehensive GitHub suites and the focused contract tests are the
relevant evidence.

## ADR Check

ADR required: yes

ADR path: `backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md`

Reason: this establishes the repository's long-lived required CI aggregation,
dependency boundary, and coverage cadence across pull-request, main, nightly,
and manual events.
