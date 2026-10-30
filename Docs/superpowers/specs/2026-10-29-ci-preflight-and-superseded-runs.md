# CI preflight and superseded runs

The owner approved the CI audit's first two changes on 2026-10-29: cancel
demonstrably superseded PR runs and reject invalid gates before expensive tests.
The audit found 17 armed PRs, 532 queued checks, a median 209-minute required
gate, and an obsolete head's 34 queued jobs holding up its replacement.

## Design

1. Add a separate cleanup job to `merge-queue.yml` on `pull_request.synchronize`.
   It is outside Derived Artifacts' concurrency group, independent of queue mode,
   and does not invoke the queue on synchronize. Run checkout-free JavaScript
   through the existing GitHub Actions API, with only actions-write and PR-read.
   Same-repository, human-authored open PRs targeting dev are eligible. Match the
   triggering head to the live PR, require a nonempty branch, paginate Derived Artifacts PR runs for its
   branch, then validate each run's repository, workflow, PR number, event, active
   status and distinct nonempty SHA. Inspect the current job inventory and preserve
   any run whose required aggregate or queue-tick has appeared: it may be approving
   replacement CI. Re-read the PR immediately before cancelling;
   stop if its head or identity changed. Preserve same-head runs even if the base
   changes. A concurrent completion (HTTP 409) is harmless; other API failures are
   visible and stop cleanup. Never cancel manual, fork, unrelated or completed
   runs. Cleanup does not approve, re-run, merge, disarm or change protection.
2. Move the 13 existing artifact guards into a `preflight` job, retaining their
   order and report-all-failures behavior. Also run the existing two static PR
   target-overlap/node-resolution tests without application conftests. This job
   installs only pytest and PyYAML; application test environments are unchanged.
   Every expensive lane needs successful preflight. Keep the required context
   `Derived artifacts reproduce from their sources` as an always-running verdict
   job which explicitly rejects unsuccessful preflight and every selected lane.
   Push and manual queue-kick behavior still validates artifact guards; full
   manual gate dispatches still require all lanes. No workflow path filtering.

## Limits and evidence

There is no repository-wide admission controller in this change. Shard counts,
timeouts, test selectors, retry policy and ordinary dependency sets stay fixed.
Cancellation cannot reserve runner capacity or make GitHub's read/cancel API
atomic; it narrows races with an immediate live read and never guesses identity.
Local tests execute the actual inline cleanup script with mocked API responses,
including races and failures, and verify the real workflow dependency graph.
Hosted performance and cancellation effects must be reported separately from
these deterministic local checks. No broad suite sweep is authorized.

Governance: amendments to ADR-103 and ADR-218; TASK-37800.
