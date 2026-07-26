---
id: TASK-707
title: >-
  Implement word bench concurrency, or hide the control in PR 3
status: To Do
assignee: []
created_date: '2026-07-26 14:30'
labels:
  - evals
  - word-bench
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the whole-branch review of PR 2 of the Evals rebuild (the word bench engine). Not a defect introduced by that PR unless stated; each is a seam the engine leaves for the screen that consumes it.

`BenchConfig.concurrency` is defined, validated into storage, and reloaded — and the runner never reads it. Execution is always sequential.

The spec makes it a requirement rather than an option: "Parallelism is opt-in through a `concurrency` field on the bench, so the setting travels with the bench that was tuned for it." Once PR 3 renders the bench editor, a user gets a control that does nothing.

This is a gap in PR 2's plan, not an implementer error — the plan defined the field in its first task and never scheduled the work. `BenchConfig.__post_init__` also validates `top_k >= 1` but not `concurrency >= 1`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] Either the runner honours `concurrency` (an `asyncio.Semaphore` over the row, preserving row-major completion order), or the field is removed and PR 3 renders no control
- [ ] If implemented, `concurrency >= 1` is validated in `BenchConfig.__post_init__`
- [ ] If implemented, a test asserts a concurrency of 1 still produces strict row-major order
<!-- AC:END -->
