---
id: TASK-1477
title: >-
  Surface run-level failure summary in the Evals results grid
status: To Do
assignee: []
created_date: '2026-07-30 10:00'
labels:
  - evals
  - word-bench
  - ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by live UAT (2026-07-30). On a fresh install the config template pre-fills `api_settings.llama_cpp.api_url` (config.py:2482), so the sample-bench gate is always true and the invited one-click golden path runs against a dead server — producing a grid of em-dashes headed "4 failed" with no visible explanation. The reason ("Failed: unreachable — All connection attempts failed") appears only if the user discovers cell-focus. There is no run-level banner, and no next step at the moment of failure.

The grid already loads every cell's failure reason; deriving a run-level summary needs no engine change. Related: task-703 (preflight verdicts through the API) — note `WordBenchRunner.run()` now returns the per-target `PreflightResult` map, so part of 703's first AC already exists at the engine seam.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] A run group whose cells all failed renders an always-visible callout naming the dominant failure reason and a concrete next step (start the server, then run the bench again)
- [ ] A partially failed run states how many cells failed and the dominant reason
- [ ] The callout follows the readiness vocabulary (`.ds-recovery-callout`, no hover-only content, no color-only signal)
- [ ] Tests cover the all-failed and mixed cases
<!-- AC:END -->
