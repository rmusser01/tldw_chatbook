# In-repo merge queue for `dev` — design

- **Date:** 2026-10-03
- **Status:** Draft for owner review
- **Program:** CI throughput (owner priority 1). This supersedes the 2026-09-28 drop of sub-project 5: the owner asked on
  2026-10-03 to "rebuild the CI workflows to be optimized and controlled so only one pr gets pushed at a time instead of all
  trying to be the one to merge".
- **Related:** ADR-103 (fast PR lane and required-gate aggregation), `backlog/docs/branch-protection-baseline.md`,
  `Docs/superpowers/specs/2026-09-27-ci-conflicts-and-waste-design.md`.
- **Revised 2026-10-06 (approve, never dispatch).** The first version dispatched the required check after a rebase, and no
  queue-rebased PR could ever merge: a `workflow_dispatch` run's check is not in the PR's status rollup, so the required
  check read as missing and the PR stayed `BLOCKED` with every check green until the stuck eviction (#2874, #3026; fact
  V4). The queue now **approves** the `pull_request` runs GitHub holds after its rebase (F4); those are ordinary
  `pull_request` runs whose check counts. A retry re-runs the failed run in its own check suite (V3). The queue was set
  `MERGE_QUEUE=off` on 2026-10-06 until this revision is merged and tested live on one PR.

## 1. Problem

`dev` is protected with strict status checks, so a PR can merge only when it is up to date with `dev` and its single
required check (`Derived artifacts reproduce from their sources`) is green on that up-to-date head. Each merge therefore
makes every other ready PR behind again. Today every armed PR is re-synced and re-tested after each merge, but only one of
them can win the next merge.

Evidence:
- 80 merged PRs (2026-09-21..28), 392 required-workflow runs: 186 runs (47%) were re-sync churn beyond one sync per PR,
  about 740 runner-minutes a day. The median run took 38 minutes.
- 2026-09-28 phase-2 backlog run: re-syncing every behind PR after each merge cost 5-7 extra full runs per merge, with no
  throughput gain.

## 2. Goals and non-goals

Goals:
1. PRs merge into `dev` one at a time, in the order they were armed. Only the PR at the front of the line is rebased and
   tested; every other armed PR is left untouched.
2. Every CI check that runs on a PR today still runs on the head that actually merges.
3. No new standing work for the owner: no scripts to run or maintain.

Non-goals:
- Raising the one-merge-per-CI-cycle ceiling. Strict protection already imposes it, and only GitHub's native merge queue
  (organization-owned repos only) can batch.
- Fork PRs (rare, for example #2826). Approving a fork's held runs would bypass GitHub's approval gate for outside
  contributors' workflows, so they stay manual.
- Priority lanes. Add them if FIFO proves wrong in practice.
- The nightly (sub-project 3), which resumes after this ships.

## 3. Constraints (owner rules)

- No PAT, GitHub App key or deploy key in any workflow. Only the built-in `GITHUB_TOKEN`.
- Nothing that depends on a new file or change on `main`. `schedule`, `workflow_run`, `check_suite`, `check_run` and
  `pull_request_target` all read the default branch, so none of them are used.
- PRs are rebased onto `dev`, never merged with `dev` (owner rebase rule).
- No local scripts for the owner to run.

## 4. Verified platform facts

Spike PR #2985 (2026-10-03, closed without merge) established:

| # | Fact | Result |
|---|------|--------|
| F1 | `workflow_dispatch` on a branch works when the workflow file exists on `main` and the branch's copy declares the trigger, even if `main`'s copy does not | Verified |
| F2 | A dispatched run's `Derived artifacts reproduce from their sources` check is the PR's required context (`isRequired(pullRequestNumber)` = true) | Verified, but **not sufficient**: it says the check *name* is required, not that a dispatched run satisfies it (V4) |
| F3 | `GITHUB_TOKEN` can call `updatePullRequestBranch(updateMethod: REBASE, expectedHeadOid)`, and auto-merge stays armed afterwards | Verified |
| F4 | A `GITHUB_TOKEN` rebase creates an `action_required` `pull_request` run for every PR workflow, held for approval (GitHub docs: "creates workflow runs in an approval-required state") | Verified. Approving them is the design (V4) |
| F5 | `pull_request` with type `auto_merge_enabled` fires a run from the PR's merge ref | Verified |

From GitHub's documentation:

| # | Fact |
|---|------|
| D1 | Events caused by `GITHUB_TOKEN` create no workflow runs, except `workflow_dispatch` and `repository_dispatch` |
| D2 | Auto-merge is disabled automatically only when someone without write access pushes to the head branch, or the base branch changes |
| D3 | `pull_request_target`, `schedule`, `check_suite` and `workflow_run` use the default branch's workflow file |

Facts verified in implementation task 1 (each has a fallback):

| # | Question | Result | Fallback if it fails |
|---|----------|--------|----------------------|
| V1 | Can `GITHUB_TOKEN` delete a workflow run (the empty `action_required` runs)? | Verified (run 37154004733) | Superseded 2026-10-06: the queue no longer deletes these runs, it approves them (V4) |
| V2 | Can `GITHUB_TOKEN` cancel a workflow run (runs on a superseded head)? | Verified (run 37154004733, attempt 1; attempt 2's 409 was the already-cancelled target) | Leave superseded runs to finish |

Facts verified live on 2026-10-06:

| # | Fact | Evidence |
|---|------|----------|
| V3 | A re-run (`POST /actions/runs/{id}/rerun-failed-jobs` or `/rerun`) adds a new attempt to the **same** check suite. `filter=latest` (what protection reads) then shows only the new result; `filter=all` (what `read_checks` reads) keeps the old failure, so a re-run that fails again still counts as the second failure | #3019 head `6a11342309`: one suite, failure 22:11, success 22:35, PR went `CLEAN` |
| V4 | A `workflow_dispatch` run's checks are **absent** from the PR's `statusCheckRollup`, so they never satisfy the required check. An approved held `pull_request` run's check is present and does | #3025 (pull_request): 13 rollup contexts, required present, auto-merged. #3026, #2874 (dispatched): 2 contexts, required missing, stuck. Probe #3033: `GITHUB_TOKEN` with the queue's permissions called `POST /actions/runs/37480728527/approve` -> success; the run ran; the rollup then held 9 contexts with the required check SUCCESS/pull_request and the merge state went `UNSTABLE` (mergeable) |

The re-run and approve endpoints are documented REST endpoints covered by `actions: write` (approve is documented for fork
PRs; V4 shows it also takes these held runs). GitHub only re-runs a completed run.

## 5. Architecture

Everything lives on `dev` and runs on the built-in token.

1. **`.github/workflows/merge-queue.yml` (new).**
   - Triggers:
     - `pull_request` with types `auto_merge_enabled`, `auto_merge_disabled` and `closed`, for base `dev`;
     - `push` to `dev`.
   - Job `if`: the queue mode is `dry` or `on`, and the event is a push or comes from a same-repo PR.
   - Steps: check out `dev` (never the PR) and run `python scripts/merge_queue.py`.
   - Permissions: `contents: write`, `pull-requests: write`, `actions: write`.
   - No concurrency group. Races are made safe by the action rules in section 7 instead, because concurrency groups cancel
     pending jobs and those show as red cancelled checks on PRs.

2. **`scripts/merge_queue.py` (new).**
   - Standard library plus the `gh` CLI that hosted runners already have.
   - Read layer: one GraphQL query that returns the open PRs into `dev`. For each PR it reads number, head SHA and repo,
     author type, draft flag, `autoMergeRequest.enabledAt`, `mergeStateStatus`, unresolved thread count, and the required
     check's state and completion time on the head. It also reads the head commit date and the `derived-artifacts.yml`
     runs on that head.
   - Every list read (the open PRs, the check runs and the workflow runs on a head) follows pagination to the end, up to
     10 pages of 100. A longer list fails the run instead of deciding on part of it, because an armed PR or a live run on
     a later page would be invisible.
   - A pure `decide(state, now) -> list[Action]` function holding all queue rules (section 6).
   - An action layer that performs the decided actions with the guards in section 7.

3. **`.github/workflows/derived-artifacts.yml` (changed).**
   - Add `workflow_dispatch` with a string input `pr`, and (2026-10-06 revision) a string input `wait_run`, the run a
     queue kick's tick waits for before deciding.
   - Define one condition for when the lanes run: `github.event_name == 'pull_request'`, or `workflow_dispatch` with a
     non-empty `pr`, or `workflow_dispatch` with no `pr` on any ref other than `refs/heads/dev`. Use it for both lanes and
     both verdict steps, so a manual dispatch with `pr`, or on a PR branch, runs the full fast lanes.
   - A dispatch with no `pr` on `dev` is a manual queue kick: the lanes and verdicts are skipped, and the required check is
     not reported red on `dev`. A dispatch with no `pr` on any other branch (the Actions UI "Run workflow" default, or
     `gh workflow run derived-artifacts.yml --ref <branch>`) runs the full gate instead, so the required check can never be
     greened on a PR head with zero tests run.
   - Add a non-required `queue-tick` job:
     - `needs: derived-artifacts`, with `if: '!cancelled()'` (not `always()`, so a run the queue itself cancelled is
       skipped, while a failed aggregate still lets the queue evict);
     - it runs only when the queue mode is `dry` or `on`, and the event is a dispatch, or a `pull_request` from a same-repo
       PR (armed or not: the payload's `auto_merge` is a trigger-time snapshot that the disarm-push-rearm flow leaves
       null, and the script itself acts only on the armed line);
     - job-level write permissions, so the top-level `contents: read` stays for every other job;
     - it checks out `dev` and runs the same script.
   - It is not a required check and not part of the aggregate's `needs`, so a queue failure can never turn the required
     check red. Pushes to `dev` do not run `queue-tick`; `merge-queue.yml` handles them.

4. **Other PR workflows (changed).** Each workflow that runs on `pull_request` must also run correctly from
   `workflow_dispatch`. The first version needed this so the queue could re-run them on the rebased head. Since the
   2026-10-06 revision the queue approves their held `pull_request` runs instead, and the rule keeps manual dispatches safe.
   - 8 of the 11 other PR workflows read PR-only context (`github.event.pull_request`, `head_ref`, or
     `event_name == 'pull_request'`).
   - perf-guard, task-19642 and task-32011 have no `workflow_dispatch` trigger yet.
   - Implementation audits every one. Each gets dispatch-aware conditions and tests.
   - A workflow that cannot be made to run safely from a dispatch is listed for an owner decision. None is dropped silently.

5. **Docs.**
   - CLAUDE.md "Merging into dev" and the same rules in AGENTS.md: they depend on the queue mode (section 10).
   - A new ADR for the queue, cross-referenced from ADR-103.
   - `branch-protection-baseline.md`.

## 6. Queue rules — `decide`

**The line.** Open PRs with base `dev`, auto-merge armed, not a draft, head repo equal to the base repo, and opened by a
user (GraphQL `author.__typename == "User"`). They are ordered by `autoMergeRequest.enabledAt`, oldest first.
- An armed fork PR gets one comment saying fork PRs are not queued and must be merged by hand. It is never part of the line.
- An armed PR opened by a bot or app (Dependabot, an agent such as the Copilot coding agent, or a deleted "ghost"
  account) gets one comment saying the same, for the security reason in section 9. It is never part of the line.

**Front PR.** Fresh state is read first. If `mergeStateStatus` is `UNKNOWN`, the queue re-reads it up to 12 times, 10
seconds apart (after each merge the next front is routinely `UNKNOWN` for a while); if it is still unknown, it does nothing.

| Front PR state | Action |
|---|---|
| `BEHIND` | **Rebase** (section 7). On success it approves the new head's held `pull_request` runs (waiting up to 10 x 3 s for the required workflow's to appear), cancels the old head's runs, and comments "rebased onto dev at `<sha>`" |
| `DIRTY` | **Evict:** conflicts with `dev` |
| Up to date, no required-check run on the head, or only cancelled ones; head commit older than 3 minutes | **Start:** approve a held run, else re-run a cancelled one. With neither, look again for up to 10 x 3 s (the head's age is its commit date, not its push, so its run may not be listed yet), standing down if a run appears; then evict ("push a commit, or close and reopen the PR"), since a dispatch would not count (V4) |
| Up to date, no run yet, head commit 3 minutes old or less | Wait (the author's own `pull_request` run may not be visible yet). In `on` mode the queue run waits out the 3 minutes once and decides again: a tick woken after a rebase whose held runs had not appeared may be the last event for that head |
| Up to date, required check queued or in progress | Wait (in flight) |
| Up to date, required check failed, first failure on this head | **Retry:** re-run the failed run in its own check suite (V3) and comment linking it. If the deciding run *is* the failed run (its queue-tick, while it is still in progress), wake a queue tick: a queue kick, `derived-artifacts.yml` dispatched on `dev` without `pr` and with input `wait_run`, whose tick waits for the run to complete, then re-runs it. (`derived-artifacts.yml`, because GitHub only dispatches a workflow whose file is on `main`; `merge-queue.yml` is not.) A run already live again is left alone, and one held again (its re-run's approval failed) is approved strictly. One retry per head: a second retry decision evicts (`evict-failed-twice`) once a re-read shows the run is not going again, because a re-run keeps its run id and a run that never reports the check stays a single stand-in. A retry attempt that was cancelled, not failed, is re-run in full without counting against the cap. Re-run errors: a refusal (HTTP 409/422, or a 403 for a run over a month old) evicts (`evict-rerun`) unless a re-read shows a racing run re-ran it; a transient error (5xx, 429, rate limit, network) re-raises; any other error re-raises once with a `rerun-error` comment and counts as a refusal the second time on the same head |
| Up to date, required check failed, second failure on this head | **Evict:** CI failed twice, linking both runs |
| `CLEAN` or `UNSTABLE` (or `BLOCKED` with no unresolved review threads, usually the state lagging the check), required check green, completed 15 minutes ago or less | Wait (auto-merge is about to fire) |
| `CLEAN` or `UNSTABLE` (or `BLOCKED` with no unresolved review threads), required check green, completed more than 15 minutes ago | **Evict:** auto-merge did not fire, re-arm to retry |
| `BLOCKED`, required check green, one or more unresolved review threads | **Evict:** blocked by unresolved conversations |

A completed `derived-artifacts.yml` run on the head that failed (`failure`, `startup_failure` or `timed_out`) without
reporting the required check in its check suite counts as a failed required check. A startup failure (for example a broken
workflow file on the branch) is therefore retried once and then evicted. GitHub may answer that re-run with a 403 the queue
cannot classify; the first one fails the queue run with a `rerun-error` comment and the second evicts (`evict-rerun`), so a
broken branch can never fail every queue run forever.

After an eviction the queue re-evaluates the new front PR in the same run, at most 10 times. PRs behind the front are
never rebased, dispatched or commented on.

**Modes**, from the repository variable `MERGE_QUEUE`:
- unset or `off`: do nothing;
- `dry`: compute decisions and write them to the job summary, with no side effects;
- `on`: act.

## 7. Actions and safety invariants

- **Rebase** uses `updatePullRequestBranch(updateMethod: REBASE, expectedHeadOid: <head the queue read>)`. On any failure the
  queue re-reads the PR. If that shows the same head and not `DIRTY`, it waits 3 seconds and re-reads once more: a racing
  run's accepted rebase moves the branch about a second after its mutation returns, and a refusal inside that window must
  not count as this run's own failure. Then:
  - head moved: do nothing (someone else acted);
  - now `DIRTY`: evict;
  - anything else (same head, still not `DIRTY`): the first time, post a `rebase-failed` comment quoting the error (first
    200 characters) and saying the queue will retry once; if that comment already exists for this head, evict ("rebase
    onto dev keeps failing"). A rebase that keeps failing while the PR stays `BEHIND` would otherwise stall the line.
- The mutation returns the pre-rebase head, and the branch moves about a second later. After a successful rebase the queue
  re-reads the PR up to 10 times, 3 seconds apart, until the head moves, and uses that new head for the comment. If it never
  moves, the queue logs it and does nothing more; the young-head and no-run rules recover it on a later tick.
- Once the new head appears, the queue approves its held runs first, then cancels the old head's live runs except its own
  run and any merge-queue.yml run.
- **Approve.** Every queue run, in `on` mode, first approves the front PR's held runs (best-effort per run), so the decision
  sees them as the live runs they become. Only runs whose event is `pull_request` and whose triggering actor is
  `github-actions[bot]` (the queue's own token) are approved, and only for the front PR, which is same-repo, user-authored
  and armed by someone with write access. The queue never dispatches the required check (V4).
- **Start and retry** re-read the head's live `derived-artifacts.yml` runs first; if one is live (a racing queue run acted
  first), they do nothing. On the start and retry rows, approving a held run is not best-effort: a GitHub error
  re-raises, so an outage fails the run instead of falling through to an eviction. A re-run refusal (HTTP 409/422, or a
  403 for a run over a month old) evicts, after one re-read of the run: if it is live or held again, a racing queue run
  re-ran it and this one stands down. A transient error (5xx, 429, rate limit, network) re-raises (section 8), so an
  outage never disarms the fronts it touches. Any other error re-raises once with a `rerun-error` comment, then counts as
  a refusal on the same head. A re-run's actor is the queue's token, so GitHub may hold it again; the queue then
  approves it.
- Only held runs that pass one predicate are ever approved, on every path: event `pull_request`, triggering actor
  `github-actions[bot]`, head repository this repository.
- **Evict** means `disablePullRequestAutoMerge` plus one comment. The disarm is best-effort: two racing runs (a merge fires
  both `push` to `dev` and `closed`) can evict the same PR, and the second disarm hits an already-disarmed PR. Re-arming
  puts the PR at the back of the line.
- **Comments** carry a hidden marker `<!-- merge-queue:<kind>:<head-sha> -->`. The queue never posts a kind twice for the same
  head. An eviction's kind names its cause (`evict-conflict`, `evict-failed-twice`, `evict-blocked`, `evict-stuck`,
  `evict-rebase`, `evict-rerun`, `evict-no-run`), so a re-armed PR evicted again on the same head for a different reason is still told
  why.
- **Forbidden**, and enforced by a guard test:
  - enabling auto-merge;
  - merging a PR (GraphQL `mergePullRequest`, REST `PUT .../merge`, or `gh pr merge` without `--disable-auto`);
  - any git push.

  A merge done with `GITHUB_TOKEN` pushes to `dev` without triggering any workflows (D1), which would silently stop both the
  queue and dev's post-merge checks.
- Every action is safe to repeat. Two queue runs racing produce at most one rebase: the pinned head makes the second
  mutation fail, and the post-failure re-read (above) makes the losing run see the moved head and do nothing, instead of
  counting the refusal as its own failure. Approving a run another queue run already approved is refused: the approval
  pass logs it (best-effort), and a start or retry approval re-raises, so that queue run fails and the next event decides
  afresh. Two runs re-running the same run: the second sees it live and stands down, or GitHub refuses its re-run and its
  re-read finds the run going again, so it stands down too. Evictions are idempotent.

## 8. Failure handling

- **API error or rate limit:** the queue run fails as a non-required job. The next event retries it.
- **A held run left unapproved** (the post-rebase wait ran out, or an approval failed): the next queue run's approval pass
  approves it, and a retry decision on that head approves it strictly instead of counting it as a second failure.
- **A woken tick's wait** (`wait_run`) is bounded at 5 minutes, inside queue-tick's 10-minute timeout; a run that cannot be
  read, or is still live after that (a GitHub fault), is decided on anyway, and the next event recovers it.
- **A queue run interrupted between rebase and approval:** the same pass recovers it.
- **Runner starvation:** the front PR simply waits in flight, as it does today.
- **Liveness:** the queue wakes on arm, disarm and close events, pushes to `dev`, and finished CI runs for same-repo PRs.
  - Manual kick: `gh workflow run derived-artifacts.yml --ref dev`, with no `pr` input. It works because that file already
    exists on `main` (F1).
  - Known gap: a stall during a stretch with no activity waits for the next event or a manual kick. There is no periodic
    timer, because `schedule` reads `main`.
- **Owner or agent merges by hand while a PR is in flight:** `dev` moves, the front PR becomes behind, and the queue rebases
  it again and cancels the superseded run.

## 9. Security

- `pull_request` runs for same-repo PRs get the permissions declared above. Those authors already have write access, so
  there is no escalation.
- A queue-approved held run executes the PR branch's own copy of `derived-artifacts.yml` (as the old queue dispatch did), so the branch controls every job in that run,
  `queue-tick`'s write permissions included. Checking out `dev` inside a job cannot change that, because the branch can
  edit the job. This equals the existing `pull_request` exposure: a same-repo `pull_request` run also executes the
  branch's YAML with whatever `permissions:` it declares, and GitHub documents that anyone with write access can raise
  the token's permissions by editing the workflow file. A wake-up job moved into `dev`'s copy (by dispatching a kick on
  `dev`) would not help either: the branch could still declare write permissions on any job of the dispatched run, and
  the extra hop would cost a queued run per CI completion.
- The one difference is the actor. A queue-approved run runs as `github-actions[bot]`, so GitHub's actor-based gates on
  `pull_request` runs do not apply to it: Dependabot runs get a read-only token and no secrets, and pushes by agents such
  as the Copilot coding agent wait for a human's "Approve and run". The queue therefore only queues PRs opened by a user
  (section 6). With that rule, every PR whose runs the queue approves was opened by a user and armed by someone with
  write access, on a branch only write-access accounts can push to, and those pushes' own `pull_request` runs could
  already do anything an approved run can. The queue approves only held runs triggered by its own actor, on the front
  PR, whose head repository is this repository.
- Fork PRs get a read-only token and are skipped explicitly.
- Both entry points check out and run `dev`'s copy of the script, never the PR's copy.
- PR titles, bodies and branch names are passed only as API arguments, never interpolated into shell. Workflow expressions
  feed values through `env:`.
- Write permissions are granted at job level, only to the queue jobs.

## 10. Agent rules (CLAUDE.md and AGENTS.md)

The queue runs entirely in GitHub Actions and reacts only to GitHub events. Any number of machines, sessions, tools or
people can open and arm PRs: the order is GitHub's server-side `autoMergeRequest.enabledAt`, and racing queue runs are made
safe by section 7. What every machine must share is the rules below.

They go into both `CLAUDE.md` (read by Claude Code sessions) and `AGENTS.md` (read by Codex sessions). As of 2026-10-03,
`AGENTS.md` contains no merge rules at all.

The rules depend on the mode, checked with `gh variable get MERGE_QUEUE`. A rollback therefore needs no docs change.

- **Queue `on`:**
  - Arm auto-merge once Qodo has reviewed the current head and every thread is resolved, then walk away.
  - Never `update-branch` an armed PR.
  - Never merge an armed PR by hand.
  - Run `gh pr merge <n> --disable-auto` before pushing more work, and `git pull --rebase` before pushing, using
    `--force-with-lease`.
  - A conflict-free queue rebase does not need a fresh Qodo review, because CI tests the combined result. New Qodo threads
    on the rebased head block the merge, and the PR is evicted with the reason.
  - The queue approves a rebased PR's held runs itself; approving one by hand is harmless. Never dispatch the required
    check by hand to unblock a PR: a dispatched run's check never counts (V4).
- **Queue `off` or `dry`:** today's rules apply. Rebase-sync only the PR about to merge, with
  `gh pr update-branch --rebase <n>`.

## 11. Testing

- **`decide` table tests**, one per row of the section 6 table, plus:
  - empty line;
  - FIFO ordering;
  - PRs behind the front are untouched;
  - fork and bot-author skip, and comment dedup;
  - a line, check-run list or workflow-run list that spans two pages;
  - `UNKNOWN` re-read;
  - the eviction loop bound;
  - each mode.
- **Action-layer tests** against a fake `gh`:
  - a pinned-head rebase failure never evicts;
  - held runs are approved only after a successful rebase, and only the queue's own, on the front PR, from this
    repository; a post-rebase wait that ends with nothing approved wakes a tick;
  - comments deduplicated by marker;
  - `dry` makes zero mutating calls;
  - a retry re-runs the failed run in place; the failed run's own tick wakes a tick instead; a live run is left alone;
    a second retry on the same head evicts;
  - a refused re-run evicts and the line moves on, unless a re-read shows a racing run re-ran it; a 5xx, 429, rate
    limit or network error fails the run and disarms nobody; an unclassified error fails once with a comment, then
    evicts;
  - a held re-run whose approval failed is approved strictly, never evicted; a cancelled retry attempt is re-run without
    counting against the cap;
  - start looks again before a no-run eviction and approves only held runs the queue caused; a young head with no run is
    decided again after the window (`on` mode only); a `wait_run` that cannot be read is decided on anyway;
  - `main()` passes `WAIT_RUN` through;
  - dispatched required checks never count, and a live dispatched run is not waited on;
  - a live run that appeared after the decision stops the action;
  - a refused rebase is re-read once more before it counts as a failure.
- **Boundary test:** `main()` through the real `Gh` class down to the `gh` argv, with canned JSON, for a `BEHIND` front
  PR.
- **Guard test:** fails if the script can enable auto-merge, merge or push.
- **Workflow-shape tests:**
  - `merge-queue.yml` has exactly the triggers above, and none of `pull_request_target`, `schedule`, `workflow_run` or
    `check_suite`;
  - permissions are job-level;
  - `queue-tick` is outside the aggregate's `needs`;
  - the lane condition is shared by both lanes and both verdict steps.
- **Pin updates:** the 5 pins that the spike tripped (`Tests/CI/test_ci_queue_pressure_contract.py`,
  `Tests/CI/test_derived_artifacts_workflow.py`), plus a fresh untruncated search for any pin on the other changed
  workflows.
- **Live:** the trial in section 12.

## 12. Rollout and rollback

1. One implementation PR, merged the old way. It contains sections 5 and 10, the tests, the ADR and the measurement script
   (section 13). It ships with `MERGE_QUEUE` unset, which means off.
2. The owner sets `MERGE_QUEUE=dry`. For about a day, the logged decisions are compared with what actually happens to armed
   PRs.
3. The owner sets `MERGE_QUEUE=on`, first with 2-3 low-risk PRs, then for all traffic.
4. **Rollback:** `gh variable set MERGE_QUEUE --body off` takes effect immediately. The implementation PR can be reverted.

Setting the variable needs repository admin, so it is an owner action.

## 13. Success measures

A small committed script measures required-workflow runs per merged PR, split by cause (content, sync or rebase, queue
dispatch). It replaces the evidence script lost with the scratchpad.

After 7 days on, compared with the 2026-09-21..28 baseline:
- sync and rebase runs per merged PR have a median of at most 1, against a baseline of 186 avoidable runs per 80 PRs;
- no PR merges out of FIFO order, except after evictions or manual merges;
- no line stall longer than 60 minutes while an eligible PR exists and runners are free;
- zero queue actions that enabled auto-merge or merged a PR.

## 14. Owner decisions recorded

- Build an in-repo queue, not an organization transfer for GitHub's native merge queue (2026-10-03).
- Run the spike before redesigning (2026-10-03), then accept the revised section 1 and section 2, with the 12 review fixes
  folded in.
- The queue must not depend on any single machine making PRs (owner question, 2026-10-03). It doesn't: GitHub events and
  server-side ordering only. The shared rules go into both CLAUDE.md and AGENTS.md (section 10).
- Still open:
  - any PR workflow the section-5 audit finds cannot run safely from a dispatch;
  - setting `MERGE_QUEUE`.
