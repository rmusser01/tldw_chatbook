---
id: TASK-16264
title: Reconcile Settings probe and responsive contracts
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 17:52'
labels:
  - testing
  - ui
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore deterministic Settings evidence for endpoint probes, responsive footer priority, model-catalog provider inventory, and asynchronous provider saves.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Settings provider probe tests stub endpoint I/O and await completion.
- [x] #2 Responsive footer and catalog tests assert current prioritized/provider inventory behavior.
- [x] #3 Screen pre-import isolation restores both module-cache and parent-package identities.
- [x] #4 Affected Settings modules pass without blocked network attempts.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve and classify checkpoint 61 failures.
2. Update stale test contracts and stub live probe seams.
3. Reproduce and eliminate cross-module module-identity leakage.
4. Run affected modules and static checks.

ADR required: no
ADR path: N/A
Reason: test-harness reconciliation only; no production architecture or behavior change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

- Reconciled the responsive Settings footer with the current highest-priority-prefix behavior and added QwenCloud to the model-catalog expectation inventory.
- Stubbed the two clickable provider endpoint probes and waited for their completed results, keeping the test suite's no-network boundary non-vacuous.
- Fixed the load-order root cause: the screen pre-import fixture restored `sys.modules` but left replacement child-module attributes on the parent package, so later string monkeypatches targeted a different module object than the collected screen classes.
- Verification: the direct ordered reproducer passed 132 tests and the complete 25-file checkpoint passed 730 tests with no blocked egress.
- Targeted Ruff lint and `git diff --check` passed. Ruff format reports the same four files as unformatted on `HEAD`; their existing format state was preserved to avoid unrelated churn.
- No production code, dependency, or ADR changes were required.
