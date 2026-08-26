---
id: task-22250
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

- [ ] The agent performing the cancellations is identified (account concurrency
      ceiling, manual queue clearing, or workflow configuration)
- [ ] A PR's test workflows can run to completion without being swept
- [ ] `Tests` produces a verdict on at least one PR

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
