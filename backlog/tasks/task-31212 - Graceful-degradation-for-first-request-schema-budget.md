---
id: TASK-31212
title: Graceful degradation for first-request schema budget
status: To Do
assignee: []
created_date: '2026-09-03 11:45'
labels:
  - agents
dependencies: []
priority: medium
---

## Description

`build_first_request_schema_plan`'s `validated_fallback` is binary: when the discovery plan misses the context budget, it collapses to the empty `no_tools` plan — every tool vanishes at once. Measured incident during TASK-28238 phase 2 (Task 7): adding two ~300-token runtime schemas tipped the unrecognized-model 4096-token fallback (2048 reserve) from a 47-token fit into total collapse, silently dropping the run-log tools; only two indirectly-related tests caught it. Replace the cliff with staged degradation: drop optional runtime schemas in a defined priority order (e.g. worktree merge/discard first, then skill extras) and retry the fit before falling to `no_tools`, and/or surface a visible signal when tools were dropped for budget reasons.

## Acceptance Criteria

- [ ] Under the 4096-token fallback with fleet + run-log active, core tools (spawn/wait/check/send + run-log) survive even when optional schemas do not fit.
- [ ] The drop order is explicit and tested, and a dropped-for-budget outcome is observable (plan field or log record), not silent.
- [ ] No behavior change when everything fits.
