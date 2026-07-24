---
id: TASK-496
title: Enforce evaluation execution and run-state contracts
status: To Do
assignee: []
created_date: '2026-07-24 13:17'
updated_date: '2026-07-24 13:19'
labels:
  - evals
  - reliability
  - workers
dependencies: []
references:
  - backlog/decisions/024-bounded-evaluation-and-tool-worker-execution.md
documentation:
  - Docs/superpowers/specs/2026-07-24-evaluation-execution-contracts-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make evaluation execution honor the synchronous provider boundary, configured request bounds, deterministic sample accounting, and truthful terminal run states.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Synchronous production provider calls execute without blocking the event loop or being incorrectly awaited
- [ ] #2 Configured request timeouts and retry policy govern each provider attempt
- [ ] #3 Configured maximum concurrency bounds sample execution while returned results remain in input order
- [ ] #4 Progress reports every settled sample exactly once and does not hide callback or persistence failures
- [ ] #5 Runs with one or more sample errors retain their results and finish failed with an actionable summary
- [ ] #6 Cancelled runs persist cancelled state, unregister cleanly, and propagate cancellation
- [ ] #7 Regression tests reproduce the production dispatcher contract, timeout, concurrency, partial failure, and cancellation behavior
- [ ] #8 Invalid concurrency, timeout, retry-count, and retry-delay bounds fail before evaluation begins
<!-- AC:END -->
