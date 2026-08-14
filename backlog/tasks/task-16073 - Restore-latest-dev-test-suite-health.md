---
id: TASK-16073
title: Restore latest dev test-suite health
status: In Progress
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
- [ ] #1 A frozen collection manifest and resumable checkpointed baseline on the pinned dev commit record every collected node exactly once with durable node IDs, terminal outcomes, logs, environment/package fingerprints, and machine-readable reports; collection errors and bounded minimized node/sequence process failures are retained as explicit red outcomes, while, within ADR-065's cooperative retained-signal ownership boundary, missing, duplicate, corrupt, unowned, or uncollected outcomes cannot pass the completeness gate.
- [ ] #2 Every reproducible deterministic or intermittent product, stale-contract, order-dependence, isolation, flake/race, or environment-harness defect discovered by that baseline is fixed at its smallest shared root cause without deleting tests or weakening valid assertions; intermittent fixes repeatedly pass their original triggering sequence.
- [ ] #3 No baseline failure or error becomes skipped, xfailed, deselected, not collected, removed, or renamed without an explicit user-approved scope amendment, and the final non-executed outcome set is unchanged from baseline.
- [ ] #4 Every discovered failure node and directly affected suite passes; the reviewed/rebased executable-input tree has a complete zero-red candidate generation, and the identical checkpointed pipeline on the exact documentation-only closeout HEAD repeats that accounting before merge.
- [ ] #5 Changes remain limited to test-health repairs, introduce no speculative abstractions or dependencies, and keep logs, fixtures, captured reports, and persistent diagnostics free of credentials and private user data.
- [ ] #6 Implementation Notes document the pinned baseline, reviewed executable-input generation/hash, stable `ready-pr-final` identity, classified inventory and authorities, RED-GREEN/flake/static/review/lessons/ADR evidence; the immutable `ready-pr-final` manifest and PR evidence document the post-commit exact-head counts, hashes, and duration.
<!-- AC:END -->

## References

- `Docs/superpowers/specs/2026-08-13-task-16073-dev-test-suite-health-design.md`
- `Docs/superpowers/plans/2026-08-13-task-16073-dev-test-suite-health.md`
- `backlog/decisions/065-checkpoint-harness-process-ownership.md`
- Follow-up to TASK-2703 scoped verification; the new baseline is captured independently on latest `dev`.

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Rebase onto current `dev`; prove a dependency-free checkpointed pytest harness; capture and classify every terminal red outcome; repair each root cause with focused RED-GREEN commits; rerun the identical complete pipeline and static/privacy checks; independently review; rebase and verify the exact PR head; then address Qodo/CI feedback and merge only when all authoritative gates are clear.

ADR required: yes. ADR path: `backlog/decisions/065-checkpoint-harness-process-ownership.md`. Negative harness testing exposed a Darwin process-ownership boundary; ADR-065 records the approved cooperative-subprocess contract, PID-version-safe cleanup, and fail-closed capability gate.
<!-- SECTION:PLAN:END -->
