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
- [ ] #1 A complete baseline run on the pinned dev commit records every failure and error with durable node IDs, tracebacks, environment metadata, and machine-readable results.
- [ ] #2 Every deterministic product, test-contract, order-dependence, isolation, or environment-harness defect discovered by that baseline is fixed at its smallest shared root cause without deleting tests or weakening valid assertions.
- [ ] #3 Each discovered failure node and every directly affected suite passes, and a final identical complete-suite run finishes with zero unexpected failures or errors.
- [ ] #4 Changes remain limited to test-health repairs, introduce no speculative abstractions or dependencies, and keep logs, fixtures, and captured reports free of credentials and private user data.
- [ ] #5 Implementation notes document the pinned baseline, classified failure set, RED-GREEN evidence, final suite counts and duration, static checks, review findings, lessons decision, and ADR decision.
<!-- AC:END -->

## References

- `Docs/superpowers/specs/2026-08-13-task-16073-dev-test-suite-health-design.md`
- Follow-up to TASK-2703 scoped verification; the new baseline is captured independently on latest `dev`.
