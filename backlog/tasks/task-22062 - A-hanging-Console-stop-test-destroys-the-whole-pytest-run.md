---
id: TASK-22062
title: A hanging Console stop test destroys the whole pytest run
status: To Do
labels:
  - tests
  - ci
  - console
priority: high
---

## Description

`Tests/Chat/test_console_local_citation_boundary.py::test_direct_user_stop_does_not_seal_and_clears_terminal_state`
never completes. Its coroutine awaits something that is never resolved: the
faulthandler dump shows a single MainThread parked in the asyncio selector with
no application frames on the stack at all.

This is worse than an ordinary failing test. `pyproject.toml` sets
`timeout_method = "thread"`, which cannot interrupt a blocked thread — so when
the 300s default timeout fires, the whole pytest process is destroyed. No JUnit
XML is written, no test is named as failing, and under `pytest-xdist` the worker
dies ("worker crashed while running ..."), taking its remaining queued tests
with it as setup errors.

Measured on clean `origin/dev` `8ef5bf12e` with no local edits, so this is not
branch-specific. Worth checking against the programme's open CI question: a
`core-tests` shard that dies at exactly its `timeout-minutes` looks the same
from the outside as a shard that hit one of these.

## Acceptance Criteria

- [ ] The test completes, or fails, within its timeout
- [ ] The root cause is identified as a product deadlock or a stale test await
- [ ] A hanging test cannot silently destroy a run's reporting (timeout method or a guard)
- [ ] The module runs to completion under xdist without worker loss

## Implementation Notes

Quarantined with an explicit `pytest.mark.skip` naming this task, not because a
skip is acceptable long-term but because the alternative is strictly worse: left
in place it deletes the results of every other test sharing its process. The
skip is the visible form of the same problem.
