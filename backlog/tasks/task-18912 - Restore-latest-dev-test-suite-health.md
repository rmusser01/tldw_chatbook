---
id: TASK-18912
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
Reproduce the complete test suite on a pinned latest-dev snapshot with durable failure reporting, then repair every actionable failure and error discovered by that run. Keep fixes minimal, preserve product contracts, and do not hide defects by deleting coverage or weakening assertions. The pre-existing Console size-ratchet failure remains owned by the approved TASK-3070 decomposition series rather than expanding this repair PR into seven architecture PRs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A frozen collection manifest and resumable checkpointed baseline on the pinned dev commit record every collected node exactly once with durable node IDs, terminal outcomes, logs, environment/package fingerprints, and machine-readable reports; collection errors and bounded minimized node/sequence process failures are retained as explicit red outcomes, while, within ADR-072's cooperative retained-signal ownership boundary, missing, duplicate, corrupt, unowned, or uncollected outcomes cannot pass the completeness gate.
- [ ] #2 Every actionable deterministic or intermittent product, stale-contract, order-dependence, isolation, flake/race, or environment-harness defect discovered by that baseline is fixed at its smallest shared root cause without deleting tests or weakening valid assertions; intermittent fixes repeatedly pass their original triggering sequence. The inherited Console size ratchet remains red and is delegated unchanged to TASK-3070.5 through TASK-3070.11.
- [ ] #3 No baseline failure or error becomes skipped, xfailed, deselected, removed, or silently uncollected. Two upstream-renamed nodes are verified through their current exact replacements, and the restored Watchlists compatibility node preserves the original baseline identity.
- [ ] #4 Every actionable discovered failure node and directly affected suite passes. On the reviewed executable-input tree, the exact 107-node discovered-failure set reports 106 passes and only the unchanged TASK-3070 architecture ratchet; final verification is limited to this exact set and touched functionality, as approved by the user, rather than another complete-suite pipeline.
- [ ] #5 Changes remain limited to test-health repairs, introduce no speculative abstractions or dependencies, and keep logs, fixtures, captured reports, and persistent diagnostics free of credentials and private user data.
- [ ] #6 Implementation Notes document the pinned baseline, classified inventory and authorities, RED-GREEN/flake/static/review/lessons/ADR evidence, final exact-node counts, the TASK-3070 limitation, and PR/CI/review evidence.
<!-- AC:END -->

## References

- `Docs/superpowers/specs/2026-08-13-task-18912-dev-test-suite-health-design.md`
- `Docs/superpowers/plans/2026-08-13-task-18912-dev-test-suite-health.md`
- `backlog/decisions/072-checkpoint-harness-process-ownership.md`
- Follow-up to TASK-2703 scoped verification; the new baseline is captured independently on latest `dev`.

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Rebase onto current `dev`; prove a dependency-free checkpointed pytest harness; capture and classify every terminal red outcome; repair each actionable root cause with focused RED-GREEN commits; verify the exact discovered-failure set and touched functionality; independently review; rebase and verify the exact PR head; then address Qodo/CI feedback and merge only when all authoritative gates are clear. TASK-3070 retains ownership of the Console size ratchet.

ADR required: yes. ADR path: `backlog/decisions/072-checkpoint-harness-process-ownership.md`. Negative harness testing exposed a Darwin process-ownership boundary; ADR-072 records the approved cooperative-subprocess contract, PID-version-safe cleanup, and fail-closed capability gate.
<!-- SECTION:PLAN:END -->
