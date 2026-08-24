---
id: TASK-21566
title: Deferred startup tests break whenever startup gains a step
status: Done
assignee: []
created_date: ''
updated_date: '2026-08-24 15:03'
labels:
  - testing
  - test-integrity
  - performance
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two tests pin a policy gate: a citation reconciliation and a legacy migration must be
scheduled after the first interactive frame, and only when their write switch is on. That
is a real contract and worth pinning.

They exercise it by calling the deferred-startup method unbound, passing a hand-built
namespace in place of the application. The namespace has to name every attribute the method
touches, so the tests fail whenever deferred startup gains an unrelated step — most recently
when Actor Pack crash recovery moved off the construction path and introduced a worker call
the namespace did not have.

They are also brittle in a second way that had not surfaced yet: each asserts on the total
number of scheduled tasks and on which one came first. Deferred startup schedules unrelated
work, and one such task is now created unconditionally ahead of both, so those assertions
describe the shape of the whole method rather than the gate under test.

The failure mode is the same either way — an unrelated change to startup reddens two tests
about citation policy, and the message names an attribute neither test mentions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Adding an unrelated step to deferred startup does not break these tests
- [x] #2 A dependency that does not exist on the application class is still rejected, so the stand-in cannot drift into accepting anything
- [x] #3 Each test asserts on the task it is about, not on the total scheduled or on ordering
- [x] #4 The policy gate is still enforced in both directions, demonstrated by breaking it
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaced the hand-built `SimpleNamespace` stand-in with `Mock(spec=TldwCli)`, and changed
both assertions to count the task under test by name.

**AC#1.** The namespace had to enumerate every attribute
`_schedule_deferred_startup_work` touches, so the tests died whenever deferred startup gained
an unrelated step. The one that broke them was `run_worker`, added by task-21106 to move
Actor Pack crash recovery off the construction path — nothing to do with citations, yet it
reddened two citation-policy tests with a message naming an attribute neither test mentions.

**AC#2.** `spec=` is what makes this safe rather than merely permissive: verified directly
that `Mock(spec=TldwCli).run_worker` is satisfied while
`Mock(spec=TldwCli).definitely_not_a_real_attribute` still raises `AttributeError`. The
stand-in tracks the class without becoming a thing that accepts anything.

**AC#3 — the brittleness that had not surfaced yet.** Both tests asserted
`len(scheduled) == expected_tasks` and `scheduled[0][1] == "<the task>"`. Deferred startup
also schedules unrelated work, and `deferred_subscription_interrupt_reconcile` is now created
**unconditionally, ahead of both** — so those assertions pinned the shape of the whole method
rather than the policy gate. Even after adding `run_worker` they would have failed, for a
second and unrelated reason. They now filter `scheduled` by name.

**AC#4 — the gate is still enforced.** Forcing the write switch open in
`app.py` (`if True or ...`) makes the `enabled=False` parametrization fail (1 failed,
1 passed); restoring gives 16 passed. So the tests still detect a gate that stops gating,
which is the property they exist for.

**Evidence.** `Tests/Performance/test_app_startup_performance.py`: **16 passed**, from
4 failed / 12 passed.

Note on method: the product mutation was restored by file copy, not `git checkout --`, and
the change was committed before mutating. Earlier in this session a `git checkout --` restore
silently wiped an uncommitted rewrite — the standing rule about that is correct.

Modified: `Tests/Performance/test_app_startup_performance.py`.
<!-- SECTION:NOTES:END -->
