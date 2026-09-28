---
id: TASK-24302
title: >-
  A partial() wrapper made a boot worker anonymous and blinded the boot-worker census
status: Done
assignee: []
created_date: '2026-08-28 23:30'
labels:
  - performance
  - guards
  - dev-red
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/Performance/test_boot_worker_census.py` fails on pristine dev `3a3383123e`: a worker starts during
boot as `('', 'console-persisted-browser-cache')` and misses its own allowlist row, which is
keyed on the name `_refresh_console_persisted_rows_cache`.

The census derives a worker's identity as `kwargs.get("name") or getattr(work, "__name__", "")`.
The call site now wraps the coroutine function in `functools.partial`, and a partial has no
`__name__`, so the name resolves to the empty string.

Two consequences, and the second is the important one. The guard is red. And the guard is now
partially BLIND: any future `run_worker(partial(...))` on the boot leg collapses into the same
anonymous `('', group)` identity, so the reviewed allowlist can no longer distinguish one from
another. A guard that cannot tell two workers apart has lost the property it exists to protect.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 The boot-worker census passes on an otherwise-unmodified dev checkout
- [x] #2 A worker wrapped in functools.partial is recorded under its underlying function's name, not the empty string
- [x] #3 A test proves the census distinguishes two different partial-wrapped workers in the same group, and fails if the unwrapping is removed
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Name the boot-leg worker explicitly at its call site.
2. Make the census unwrap `functools.partial` so it cannot go blind again.
3. Mutation-test both halves.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Two-part fix, because the call site was a symptom and the census was the defect.

An AST census found **34 partial-wrapped `run_worker` sites in this repo, 22 of
them passing no explicit `name`** -- so the anonymity was systemic, not one
site's slip. The census recorded `kwargs.get("name") or getattr(work,
"__name__", "")`, and a partial has no `__name__`, so every one of those 22
would register as `('', group)` and become indistinguishable from the others.

`Tests/Performance/test_boot_worker_census.py` now unwraps `partial.func`
(recursively -- a partial of a partial is legal) before giving up, and
`UI/Console_Modules/workspace.py` names its worker explicitly as well.

**Mutation-tested both ways.** Dropping the explicit `name=` leaves the guard
GREEN (the unwrapping covers it -- that is the resilience the fix is for);
dropping the unwrapping as well turns it RED with the exact original message.
Three unit tests pin the identity rule, and they execute the rule's real source
out of `_CENSUS_SCRIPT` rather than restating it, so a copy cannot drift.

**Trap worth recording:** the helper lives inside a triple-quoted script
string, so a docstring -- and later a `"""` inside a comment -- closed the
string early. Comments only, no nested triple quotes.

Files: `UI/Console_Modules/workspace.py`, `Tests/Performance/test_boot_worker_census.py`.
<!-- SECTION:NOTES:END -->
