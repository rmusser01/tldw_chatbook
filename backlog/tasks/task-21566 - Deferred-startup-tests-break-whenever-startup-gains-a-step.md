---
id: TASK-21566
title: Deferred startup tests break whenever startup gains a step
status: In Progress
assignee: []
labels: [testing, test-integrity, performance]
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
- [ ] #1 Adding an unrelated step to deferred startup does not break these tests
- [ ] #2 A dependency that does not exist on the application class is still rejected, so the stand-in cannot drift into accepting anything
- [ ] #3 Each test asserts on the task it is about, not on the total scheduled or on ordering
- [ ] #4 The policy gate is still enforced in both directions, demonstrated by breaking it
<!-- AC:END -->
