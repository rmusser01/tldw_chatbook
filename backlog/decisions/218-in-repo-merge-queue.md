# ADR-218: In-repo merge queue for dev

Status: Accepted — owner decisions 2026-10-03 (build in-repo, not an org transfer; spec and plan approved)
Date: 2026-10-03

## Context

`dev` is strict-protected with one required check. Only one PR can merge per CI cycle, yet every armed PR was re-synced and
re-tested after each merge. Over 80 merged PRs, 47% of required runs were re-sync churn, about 740 runner-minutes a day.
GitHub's native merge queue needs an organization-owned repository, and this one is user-owned.

## Decision

- A queue workflow in this repository, `merge-queue.yml` plus the `queue-tick` job in `derived-artifacts.yml`, runs
  `scripts/merge_queue.py` on the built-in token.
- It rebases only the front armed PR, dispatches its CI, and lets auto-merge land it.
- It evicts PRs that conflict, fail twice, stay blocked by unresolved conversations, get stuck green, keep failing to
  rebase, or whose branch refuses the CI dispatch.
- It never arms, merges or pushes.
- The mode is set by the `MERGE_QUEUE` variable (off, dry or on).

## Consequences

### Amendment 2026-10-29: cancel only obsolete PR-head CI (TASK-38413)

The owner approved a separate synchronize-triggered cleanup job in the existing
queue workflow. It is independent of queue mode and outside Derived Artifacts'
concurrency group, so an obsolete run cannot hold its cleanup behind itself.
The queue-advance job continues to run only on its original events.

Use checkout-free github-script with job-local actions-write/PR-read permissions.
For an open, same-repository, human-authored dev PR, require its live head to match
the event head, and match each candidate's workflow, branch, repository, PR number,
event and active status. Only a distinct nonempty head SHA is obsolete. Re-read
the live PR before each cancellation; stop on changed identity/head or API error.
Require a nonempty branch filter. Preserve runs once their required verdict or
queue-tick job appears: the old run's controller may still be approving replacement
CI, including this cleanup job, after a queue rebase. Job-inventory errors stop
cleanup rather than risking cancellation of that controller.
Concurrent completion (HTTP 409) is tolerated. Preserve same-head runs regardless
of base movement, all manual/fork/unrelated/completed runs, and existing approvals,
retry, arming and protection rules. No force cancellation or new credentials.

GitHub provides no atomic PR-head-conditioned cancellation, so an immediate live
read narrows rather than eliminates that race. A blanket cancel-in-progress rule
cannot express these constraints and remains rejected. This cleanup adds no
main-dependent trigger or repository-wide admission controller.

Spec: `Docs/superpowers/specs/2026-10-29-ci-preflight-and-superseded-runs.md`.
Plan: `Docs/superpowers/plans/2026-10-29-ci-preflight-and-superseded-runs.md`.

- One CI cycle per merge, without the churn. The ceiling of one merge per cycle is unchanged.
- Nothing depends on `main`. A stall with no activity waits for the next event, or a manual
  `gh workflow run derived-artifacts.yml --ref dev`.
- Every pull_request workflow must stay dispatch-safe (`Tests/CI/test_pr_workflows_dispatch_safe.py`).
- Fork PRs stay manual, and so do PRs opened by a bot or app: a queue dispatch runs as `github-actions[bot]` and would
  skip GitHub's actor-based gates (Dependabot's read-only token, agent-push approval). Spec section 9.

## Amendment 2026-10-06: approve, never dispatch

- **Why:** the queue's dispatched CI never counted. A `workflow_dispatch` run's checks are absent from the PR's status
  rollup, so every queue-rebased PR stayed `BLOCKED` with all checks green until the stuck eviction (#2874, #3026). The
  queue was set `MERGE_QUEUE=off` that day.
- **Change:** after its rebase, the queue approves the PR's `pull_request` runs that GitHub holds because its token caused
  them; those count. Verified live on probe PR #3033: the queue's `GITHUB_TOKEN` approved the held run, and its required
  check then appeared in the rollup (spec V4). The queue no longer dispatches the required check.
- A retry re-runs the failed run in its own check suite (spec V3), at most once per head. When the deciding queue-tick is
  inside the failed run, it wakes a tick through a queue kick (`derived-artifacts.yml` on `dev`, input `wait_run`; GitHub
  only dispatches a workflow whose file is on `main`), and that tick waits for the run to complete.
- Eviction causes: "branch refuses the CI dispatch" is replaced by a refused re-run on a retry (`evict-rerun`, whether
  the run failed or its retry attempt was cancelled) and "no CI run the queue can start" (`evict-no-run`, also what a
  refused re-run of a cancelled run gives on the start path, at once). A strike (a rebase that did not take effect, no
  CI run the queue can start, an unclassified re-run error, or a refused approval) is warned about once per head and
  evicts (`evict-rebase`, `evict-rebase-unmoved`, `evict-no-run`, `evict-rerun`, `evict-approve`) only 10 minutes after
  the warning. A transient error (5xx,
  429, rate limit, network) never evicts. A PR whose conversation is locked leaves the line without a comment, because
  the queue keeps its counts in comments.
- Amendment 2026-10-11 (found on the first day of real traffic): when a PR and dev have both changed the same
  workflow file, GitHub refuses the built-in token's rebase; the queue evicts that PR at once (`evict-workflow-file`)
  and the author rebases by hand. (A PR that changes a workflow file dev left alone rebases normally: #3062. The rule is inferred from the cases seen; GitHub does not document it.)
  The loser of the two queue runs a merge wakes stands down instead of failing, and a comment both posted is reduced
  to one.
- The dispatch-safe rule above no longer serves the queue. It still keeps manual dispatches safe.
- The bot and fork rule is unchanged: approving a bot PR's held runs would skip the same actor-based gates. Only held runs
  triggered by the queue's own actor, on the front PR, from this repository, are approved.

Spec: `Docs/superpowers/specs/2026-10-03-merge-queue-design.md`. Plan: `Docs/superpowers/plans/2026-10-03-merge-queue.md`.
Extends: ADR-103.
