---
id: TASK-16073
title: Restore latest dev test-suite health
status: To Do
assignee: []
created_date: '2026-08-14 02:10'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reproduce the complete test suite on a pinned latest-dev snapshot with durable failure reporting, then repair every reproducible failure and error discovered by that run so the same suite completes without unexpected failures. Keep fixes minimal, preserve product contracts, and do not hide defects by deleting coverage or weakening assertions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A frozen collection manifest and resumable checkpointed baseline on the pinned dev commit record every collected node exactly once with durable node IDs, terminal outcomes, logs, environment/package fingerprints, and machine-readable reports; collection errors and bounded minimized node/sequence process failures are retained as explicit red outcomes, while missing, duplicate, corrupt, unowned, or uncollected outcomes cannot pass the completeness gate.
- [ ] #2 Every reproducible deterministic or intermittent product, stale-contract, order-dependence, isolation, flake/race, or environment-harness defect discovered by that baseline is fixed at its smallest shared root cause without deleting tests or weakening valid assertions; intermittent fixes repeatedly pass their original triggering sequence.
- [ ] #3 No baseline failure or error becomes skipped, xfailed, deselected, not collected, removed, or renamed without an explicit user-approved scope amendment, and the final non-executed outcome set is unchanged from baseline.
- [ ] #4 Every discovered failure node and directly affected suite passes, and the identical checkpointed complete-suite pipeline on the exact rebased ready-PR head accounts for every collected node exactly once with zero failures or errors.
- [ ] #5 Changes remain limited to test-health repairs, introduce no speculative abstractions or dependencies, and keep logs, fixtures, captured reports, and persistent diagnostics free of credentials and private user data.
- [ ] #6 Implementation notes document pinned baseline and final generations, the classified node inventory and cited stale-contract authorities, RED-GREEN and flake-repetition evidence, final suite counts/duration, static checks, review findings, lessons decision, and ADR decision.
<!-- AC:END -->

## References

- `Docs/superpowers/specs/2026-08-13-task-16073-dev-test-suite-health-design.md`
- Follow-up to TASK-2703 scoped verification; the new baseline is captured independently on latest `dev`.
