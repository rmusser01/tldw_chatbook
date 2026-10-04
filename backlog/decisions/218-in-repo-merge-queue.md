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

- One CI cycle per merge, without the churn. The ceiling of one merge per cycle is unchanged.
- Nothing depends on `main`. A stall with no activity waits for the next event, or a manual
  `gh workflow run derived-artifacts.yml --ref dev`.
- Every pull_request workflow must stay dispatch-safe (`Tests/CI/test_pr_workflows_dispatch_safe.py`).
- Fork PRs stay manual, and so do PRs opened by a bot or app: a queue dispatch runs as `github-actions[bot]` and would
  skip GitHub's actor-based gates (Dependabot's read-only token, agent-push approval). Spec section 9.

Spec: `Docs/superpowers/specs/2026-10-03-merge-queue-design.md`. Plan: `Docs/superpowers/plans/2026-10-03-merge-queue.md`.
Extends: ADR-103.
