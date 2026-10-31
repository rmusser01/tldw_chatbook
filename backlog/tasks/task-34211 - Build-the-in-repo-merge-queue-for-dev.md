---
id: TASK-34211
title: Build the in-repo merge queue for dev
status: In Progress
assignee:
  - '@claude'
created_date: '2026-10-03 14:06'
labels:
  - ci-throughput
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Under strict protection only one PR can merge into dev per CI cycle, yet every armed PR is re-synced and re-tested after each merge: 47% of required runs over 80 merged PRs (2026-09-21..28) were re-sync churn, about 740 runner-minutes a day. The owner asked (2026-10-03) for CI that pushes one PR at a time. Spec: Docs/superpowers/specs/2026-10-03-merge-queue-design.md. Mechanics verified by spike PR #2985.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Armed PRs merge into dev one at a time in arming order; PRs behind the front are never rebased, dispatched or commented on
- [x] #2 The front PR is rebased with the built-in token, and its required check and other PR workflows run on the rebased head
- [x] #3 Conflicting, twice-failed, blocked or stuck PRs are evicted with a comment; a single CI failure is retried once
- [x] #4 The queue never enables auto-merge, merges or pushes (guard test)
- [x] #5 MERGE_QUEUE unset/off has no effect; dry logs decisions with no side effects; on acts
- [x] #6 CLAUDE.md and AGENTS.md carry the same mode-dependent merge rules
- [x] #7 Re-sync runs per merged PR are measurable before and after with a committed script
<!-- AC:END -->

## Implementation Notes

**Approach:** a one-at-a-time merge queue for dev, running as a GitHub Actions workflow
(`.github/workflows/merge-queue.yml`) on the built-in `GITHUB_TOKEN`, plus a non-required
`queue-tick` job added to `derived-artifacts.yml`. The decision logic is pure
(`decide_front` in `scripts/merge_queue.py`) with a thin action layer (`Gh` protocol) around
it; the queue never arms, merges or pushes — it only rebases the front PR and re-dispatches
its workflows. Mode is read from the repo variable `MERGE_QUEUE` (`off`/`dry`/`on`); unset
behaves as `off`.

**Spike evidence:** PR #2985 established, against main's copy of the workflow, that
`workflow_dispatch` works without a PR event trigger, that the dispatched check is the PR's
required context (`isRequired=true`), that a `GITHUB_TOKEN` rebase keeps auto-merge armed,
that a token rebase produces `action_required` runs with no check runs, and that
`auto_merge_enabled` fires a `pull_request` event. Task 1 (throwaway spike PR #2991, run
37154004733) verified V1 (delete a stuck run) and V2 (cancel a stuck run) live with
`GITHUB_TOKEN` — V2 on attempt 1; attempt 2's 409 was the already-cancelled target, not a
refusal. Both kept, no omissions in the cleanup/cancel paths.

**Fixes found in review:**
- The read layer now also treats a *live* `derived-artifacts.yml` run on the head as an
  in-progress check: the required check is a needs-gated aggregate with no check run while
  its lanes are running, so without this the queue would re-dispatch on every tick.
- After a rebase, the queue dispatches the required workflow immediately, then cancels
  old-head live runs — excluding `merge-queue.yml` runs and its own run id — so it never
  cancels itself mid-flight.
- A `derived-artifacts.yml` dispatch without `pr` now skips the fast/full lanes only on
  `dev` (the manual queue kick); on any other branch it runs the full gate. This closes a
  fail-open where a bare dispatch could green the required check with zero tests.
- `queue-tick` uses `!cancelled()` (so it still runs after the queue cancels a stuck
  aggregate) and `persist-credentials: false` on checkout.

**Dispatch-safety audit:** 8 PR-triggered workflows were already `workflow_dispatch`-safe.
`perf-guard.yml` and `task-19642-smoke-clock-matrix.yml` gained a bare `workflow_dispatch`
trigger. `task-32011-linux-storage-evidence.yml` gained the trigger, a dispatch branch
added to its job `if:`, and a `|| github.sha` checkout-ref fallback. Pinned by
`Tests/CI/test_pr_workflows_dispatch_safe.py`.

**Pre-queue baseline** (`scripts/measure_required_runs_per_merge.py --prs 80`, 80 merged
PRs 2026-09-21..28): `merged PRs: 80  required runs: 531`; re-sync runs per PR
(sync+rebase+queue): `median 2.0, beyond one: 218`.

**Docs:** CLAUDE.md and AGENTS.md carry identical mode-dependent merge rules (AC #6);
ADR-218 (new, in-repo merge queue) added; ADR-103 (fast-lane + required-gate aggregation)
cross-referenced; `backlog/docs/branch-protection-baseline.md` gained a merge-queue section.

**Final whole-branch review fixes** (each pinned by a test that fails when the fix is reverted):
- After `updatePullRequestBranch` (which returns the *pre*-rebase head), the queue polls the PR up to
  10 x 3 s for the new head and dispatches only then; no move = no dispatch, a later tick recovers it.
- A rebase that fails while the PR stays BEHIND posts a `rebase-failed` comment once, then evicts.
- `queue-tick` no longer gates on the payload's `auto_merge` (a trigger-time snapshot the
  disarm-push-rearm flow leaves null); it wakes on every same-repo PR run.
- BLOCKED + green evicts only with unresolved review threads; otherwise the 15-minute stuck-green window.
- Eviction markers name the cause (`evict-conflict`, `evict-failed-twice`, `evict-blocked`,
  `evict-stuck`, `evict-rebase`); the disarm is best-effort (racing evictions).
- A required-workflow run that failed without reporting the required check (startup failure) counts
  as a failed check: one retries, two evict.
- `UNKNOWN` re-reads raised to 12 x 10 s; the measure script ignores the token rebase's bot-triggered
  pull_request runs; docs: unset-variable wording, lessons-file supersession line.

**Verification:** `pytest Tests/CI` — 430 passed after the final review fixes (421 before), rc=0. `./scripts/preflight.sh` — rc=0, all
derived-artifact checks (CSS bundle, profile-owned-path census, production diagnostic
inventory, backlog task-id census, index plan pins, etc.) clean, no drift.

**Files changed** (`git diff --name-only origin/dev...HEAD`):
`.github/workflows/derived-artifacts.yml`, `.github/workflows/merge-queue.yml`,
`.github/workflows/perf-guard.yml`, `.github/workflows/task-19642-smoke-clock-matrix.yml`,
`.github/workflows/task-32011-linux-storage-evidence.yml`, `AGENTS.md`, `CLAUDE.md`,
`Docs/superpowers/plans/2026-10-03-merge-queue.md`,
`Docs/superpowers/specs/2026-10-03-merge-queue-design.md`,
`Tests/CI/test_ci_queue_pressure_contract.py`, `Tests/CI/test_derived_artifacts_workflow.py`,
`Tests/CI/test_measure_required_runs_per_merge.py`, `Tests/CI/test_merge_queue_actions.py`,
`Tests/CI/test_merge_queue_rules.py`, `Tests/CI/test_merge_queue_workflow.py`,
`Tests/CI/test_pr_workflows_dispatch_safe.py`,
`backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md`,
`backlog/decisions/218-in-repo-merge-queue.md`,
`backlog/docs/branch-protection-baseline.md`,
`backlog/tasks/task-34211 - Build-the-in-repo-merge-queue-for-dev.md`,
`scripts/measure_required_runs_per_merge.py`, `scripts/merge_queue.py`.

**Deferred follow-ups** (recorded, not blocking; none touch the queue's core invariants):
- Rules/tests: UNSTABLE + green >15min eviction has no regression test; the 15-minute
  boundary (`>`) is unpinned; an unused `from dataclasses import replace` import in
  `test_merge_queue_rules.py`; `HAS_HOOKS` is grouped with CLEAN/UNSTABLE (GitHub's
  CLEAN-with-hooks state) rather than named separately in the spec.
- Action layer races/limits: concurrent merge-triggered runs (`push:dev` +
  `pull_request:closed`) can double-comment/double-dispatch (`comment_once` is
  read-then-write); `pullRequests(first:100)` is unpaginated (27 open today); a crash right
  after a rebase loses the non-required re-dispatch until the next tick recovers the
  required one; a new head's `action_required` runs are only cleaned on a later tick, so
  they're orphaned if the PR merges first; the `own_run_id` one-liner is duplicated between
  `live_required_runs` and `_rebase`.
- Cosmetic: Task 1's report prose loosely described the V-table match (committed content is
  correct); Task 5's commit trailer says "Claude Haiku 4.5" instead of the plan's line
  (truthful, just inconsistent); plan's Task 4 LANES text is stale vs. the shipped condition
  (the spec was updated — the plan is the argument, not the authority).

AC #1 (live one-at-a-time ordering) is left unticked pending the owner's dry/live trial
(Steps 5-6); the queue ships dark (MERGE_QUEUE unset) per Step 3.
