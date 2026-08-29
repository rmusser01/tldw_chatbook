# Account CI Trigger Recovery Design

## Problem

GitHub Actions is effectively serialized across the `rmusser01` account while
current workflows continue to create work faster than that restricted capacity
can drain it. Authenticated account evidence collected on 2026-08-29 showed:

- 442 active owned repositories audited;
- only `tldw_chatbook` and `tldw_server` had queued or pending Actions runs;
- no owned repository had an in-progress run at the audit instant;
- `tldw_chatbook` had consumed 510,153 runner-minutes in August, approximately
  93% of the account total;
- `tldw_chatbook`'s `Tests` workflow creates 23 ordinary test jobs per run before
  the schedule-only matrix is considered.

The dominant work is duplicated at its trigger boundary. Every merge to `dev`
currently creates a `push` run and also updates permanent promotion PR #602
(`dev` to `main`), creating a second `pull_request` run. The workflow's nightly
schedule also launches the ordinary PR suite because those jobs have no
schedule exclusion. Job-level guards added for draft PR #602 do not prevent the
workflow run from entering GitHub's queue.

GitHub documents public standard runners as free, so this is not a paid-minute
exhaustion problem. GitHub also documents that Actions may be rate-limited as
usage scales. The repository-side recovery must therefore stop generating
duplicate work before asking GitHub to restore or increase account capacity.

## Goals

1. Preserve the full ordinary test suite for every pull request targeting
   `dev`.
2. Preserve a full post-release test run for pushes to `main`.
3. Make the daily schedule launch only the five-leg `nightly-deep` matrix.
4. Stop permanent `dev` to `main` PR #602 from creating duplicate ordinary CI
   workflow runs when `dev` advances.
5. Preserve manual dispatch behavior.
6. Leave short push guards on `dev` and `main` so direct pushes and merge commits
   still receive their focused checks.
7. Reduce account demand without changing application code, test selection, or
   shard coverage.

## Non-goals

- Reducing the six core or twelve UI shard coverage.
- Moving public-repository CI onto self-hosted runners.
- Changing `tldw_server` before the dominant `tldw_chatbook` source is removed.
- Upgrading the GitHub account or changing payment settings.
- Treating queued-run cancellation as the fix; cancellation is cleanup only.

## Considered Approaches

### 1. Filter at workflow triggers and exclude ordinary jobs from schedules

Selected. Restrict `pull_request` triggers to base branch `dev`, restrict the
heavy `Tests` push trigger to `main`, and add a schedule exclusion to ordinary
test jobs. This prevents duplicate work from being created while preserving the
approved PR and release coverage.

### 2. Keep broad triggers and skip jobs with `if`

Rejected. This is the current PR #602 mitigation. GitHub still creates workflow
runs and check runs before evaluating job conditions, so the account continues
to accumulate pending/queued workflow records during every `dev` update.

### 3. Split nightly tests into a separate workflow

Rejected for this recovery. It would make event ownership explicit, but would
duplicate setup and summary wiring or require a larger workflow refactor. A
schedule predicate on the existing ordinary jobs produces the same runner
contract with a smaller, independently testable diff.

## Trigger Contract

### Heavy `Tests` workflow

- `pull_request`: only base branch `dev`; retain the existing activity types.
- `push`: only branch `main`.
- `workflow_dispatch`: unchanged.
- `schedule`: unchanged at 08:30 UTC.
- Ordinary jobs (`core-tests`, artifact-lease jobs, `ui-tests`,
  `textual-minimum`, and `test-summary`) do not run for `schedule` events.
- `nightly-deep` continues to run only for `schedule` and manual dispatch.
- `all-tests` continues to run only for manual dispatch.
- The existing core/UI `max-parallel: 3` limits remain.

The PR #602-specific job conditions become unnecessary once `pull_request`
targets only `dev`; removing them prevents a stale exception from obscuring the
real trigger contract.

### Focused guard workflows

`Derived Artifacts`, `CSS Bundle Guard`, `Perf Guard`, and `Backlog Guard` retain
their existing `push` branches and path behavior, but their `pull_request`
triggers target only `dev`. Their PR #602-specific job conditions are removed.

### Label-driven evidence workflows

The TASK-598, TASK-601, TASK-602, and TASK-603 evidence workflows remain
unchanged. They accept only the `labeled` pull-request activity, so a normal
update to PR #602 does not trigger them. Narrowing their base branches would not
reduce synchronization fan-out and would unnecessarily remove explicitly
requested evidence from a labeled main-targeting PR. The TASK-2062 workflows
already target `dev` and retain that contract.

## Queue Recovery

After the trigger fix is pushed, queue cleanup uses current open-PR head SHAs
plus current `dev` and `main` SHAs as the authority boundary. Obsolete idle
queued/pending runs may be cancelled. A run with executing jobs, a current open
PR head, or a current protected-branch SHA is preserved. GitHub ghost records
that contain no jobs and return HTTP 409 to cancellation are recorded and left
alone.

The cleanup is successful only when a fresh pull-request run created after the
trigger change starts jobs and reaches a verdict. A reduced queue count alone
is not completion evidence.

## Verification

Focused contract tests must prove:

- the heavy workflow targets PRs into `dev` and pushes to `main` only;
- scheduled events skip every ordinary runner-consuming job and run
  `nightly-deep`;
- manual dispatch behavior remains intact;
- every synchronization-capable PR workflow changed by this recovery excludes
  base branch `main`;
- core and UI matrices remain bounded at three concurrent jobs;
- pull-request runs are not cancelled in progress;
- workflow YAML parses and repository diff checks pass.

Live verification must then prove both remaining TASK-22250 criteria: one PR's
test workflows complete without a simultaneous sweep, and `Tests` reports a
verdict on at least one PR.

## ADR Check

ADR required: no

ADR path: N/A

Reason: this changes operational GitHub Actions trigger and scheduling policy;
it does not alter application architecture, storage, security, dependencies, or
cross-module runtime contracts.
