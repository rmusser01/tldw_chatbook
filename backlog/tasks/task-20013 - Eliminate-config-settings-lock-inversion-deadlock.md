---
id: TASK-20013
title: Eliminate config settings lock inversion deadlock
status: Done
assignee:
  - '@codex'
created_date: '2026-08-23 19:07'
updated_date: '2026-08-23 19:27'
labels:
  - config
  - console
  - concurrency
  - regression
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent Console and background settings persistence from deadlocking when runtime settings rebuild and config-file mutation overlap.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Concurrent settings load and config mutation cannot hold the config-file and settings-rebuild locks in opposite order.
- [x] #2 Background onboarding persistence cannot block the Textual thread indefinitely through a config/settings lock inversion.
- [x] #3 Deterministic concurrency regression coverage reproduces the former lock inversion without timing-only sleeps.
- [x] #4 Existing atomic config persistence, runtime publication, cache rebuild, and reentrant behavior remain correct.
- [x] #5 Focused config/Console tests, full affected config modules, and the required four-suite Console aggregate pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the two-thread lock inversion deterministically with event-controlled interleaving and captured thread stacks.
2. Establish one lock-order contract for settings rebuild and config-file mutation without weakening atomic persistence.
3. Apply the smallest production fix after the regression test fails for the expected deadlock.
4. Run focused config/Console concurrency tests, full affected config modules, and the required four-suite aggregate.
5. Review the exact diff, run static gates, and complete task hygiene.

ADR required: no
ADR path: N/A
Reason: this is a lock-order bug fix within the existing configuration persistence boundary; it introduces no new storage, ownership, or cross-module contract.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Established the single process lock order as config-file lock → settings-rebuild lock → settings-cache lock. The slow `load_settings` path now enters the reentrant config-file lock before serializing a rebuild; cache hits retain their prior short path. This preserves the writer's atomic file replacement and runtime publication while preventing a background Console onboarding save and a concurrent Textual settings load from retaining opposite locks.

Added an isolated subprocess regression that uses events, not sleeps, to force the former two-thread cycle and captures both thread stacks on failure. The test failed before the production change with the writer in `_publish_runtime_config_unlocked -> load_settings` and the reader in `load_settings -> _load_settings_uncached -> _load_cli_config_bootstrap`; it passes after the lock-order change and also exercises same-thread config-lock reentrancy.

Verification: 171 full config-module tests passed; 41 focused config concurrency/mutation tests passed; 2 Console onboarding victim tests passed; the required four-suite Console/agent bridge/Change Review aggregate passed 602 tests. Ruff check, Ruff format check, Python compilation, and `git diff --check` passed for the touched implementation and regression files.

ADR required: no. ADR path: N/A. Reason: the fix documents and enforces the existing configuration transaction boundary rather than introducing a new architectural contract.
<!-- SECTION:NOTES:END -->
