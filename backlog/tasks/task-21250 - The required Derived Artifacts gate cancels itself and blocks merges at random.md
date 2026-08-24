---
id: TASK-21250
title: >-
  The required Derived Artifacts gate cancels itself and blocks merges at random
status: Done
assignee: []
created_date: '2026-08-23'
labels:
  - ci
  - process
  - dev-red
priority: high
dependencies: []
---

## Description

TASK-19572 introduced `Derived Artifacts` as one small, stdlib-only, ~33-second job so that
the repo would finally have a required status check that *always reports* — replacing a
comprehensive suite that had produced no verdict since 2026-06-26. Branch protection now
requires exactly that one context.

It is not reporting. Measured over 60 consecutive runs on 2026-08-23:
**45 cancelled / 14 success / 1 failure — a 23% success rate.** Cancellations hit every
branch, including other contributors' work and direct pushes to `dev`, so merges to `dev`
are blocked at random for everyone.

Root cause is the workflow's own concurrency rule:

```yaml
group: ${{ github.workflow }}-${{ github.event_name }}-${{ github.ref }}
cancel-in-progress: ${{ github.ref != 'refs/heads/main' }}
```

The rule's stated justification is sound as far as it goes — "a merge is gated on the HEAD
commit's checks, and the newest run is the one that reports for HEAD" — but it assumes the
newest run gets to **finish**. For a `pull_request` event `github.ref` is
`refs/pull/N/merge`, and GitHub recreates that merge ref whenever the **base** branch moves.
On a repo absorbing 23–50 merges/day, every open PR's gate is therefore cancelled repeatedly
without anyone touching the PR. Because queue time on the shared pool measured 4–59 minutes
against ~33 s of actual work, the newest run rarely survives long enough to report.

Observed directly while landing the 2026-08-22 perf burn-down: three PRs sat unmergeable
through repeated reruns and rebases, each fresh run cancelled within ~2 minutes, while the
same workflow succeeded on a quiet branch.

## Acceptance Criteria

- [x] `pull_request` runs of the Derived Artifacts workflow are not cancelled by base-branch
      movement, so an open PR's required check reaches a conclusion without intervention
- [x] Push-event cancellation on `dev` is preserved, so the 50-obsolete-run pileup the rule
      was written to prevent does not return
- [x] `main` continues never to cancel, keeping its history complete
- [x] The reasoning and the measured evidence are recorded in the workflow itself, so the
      next person to tune concurrency sees why pull_request is exempt

## Implementation Notes

One-line change to the concurrency condition:

```yaml
cancel-in-progress: ${{ github.event_name == 'push' && github.ref != 'refs/heads/main' }}
```

`push` to `dev` still supersedes (the original pileup fix); `push` to `main` still never
cancels; `pull_request` now always runs to completion.

The trade-off is explicit: a rapidly-updated PR may stack a few redundant ~90-second ubuntu
runs. That is far cheaper than a required check that blocks every merge 77% of the time, and
it is the shape 19572 was reaching for — "small enough to run on every PR, which is what
makes it usable as a REQUIRED status check".

The workflow comment carries the 45/14/1 measurement and the merge-ref mechanism so the
justification travels with the code.

Modified: `.github/workflows/derived-artifacts.yml`.
