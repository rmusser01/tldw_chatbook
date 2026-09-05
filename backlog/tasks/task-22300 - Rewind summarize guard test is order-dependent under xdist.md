---
id: TASK-22300
title: Rewind summarize guard test is order-dependent under xdist
status: To Do
labels:
  - tests
  - flake
priority: low
---

## Description

`Tests/UI/test_console_rewind_restore.py::test_summarize_choice_guards_against_changed_active_session`
passes or fails depending on what else runs alongside it, not on the code under
test.

Evidence — the same unmodified baseline commit produced both outcomes across two
separate `-n 2` runs of the same file set:

| run | tree under test | result |
|---|---|---|
| `newDevBase` | clean `dev` | FAIL |
| `findBase` | clean `dev` | pass |

In isolation it passes 5/5, and its own file passes 22/22. Only the parallel
run flips it.

This wasted real time twice. It was first dismissed as a flake on the strength
of "passes in isolation" alone — which was wrong, it had a genuine cause then
(a missing durable DB, fixed in #2078). It now flips on identical code, which
is what actually establishes non-determinism.

## Acceptance Criteria

- [ ] The test produces the same result regardless of what runs beside it
- [ ] The shared state or timing assumption behind the order-dependence is named
- [ ] It passes 10/10 consecutive `-n 2` runs of its file set

## Implementation Notes

Do not "fix" this by re-running until green or by marking it flaky. "Passes in
isolation" is not evidence of no cause; the evidence that matters is the same
code producing different outcomes. Look for state shared across the worker —
the active-session pointer this test guards is a plausible candidate.
