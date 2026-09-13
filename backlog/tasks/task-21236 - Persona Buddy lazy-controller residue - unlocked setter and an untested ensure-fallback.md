---
id: TASK-21236
title: >-
  Persona Buddy lazy-controller residue - unlocked setter and an untested
  ensure-fallback
status: To Do
assignee: []
created_date: '2026-08-23'
labels:
  - personas
  - concurrency
  - test-coverage
dependencies: []
priority: low
---

## Description

Source: close-out of the 2026-08-22 holistic performance review burn-down; the two Minors the
TASK-21103 review ledgered rather than fixed. Both confirmed still present on dev `b2b1e2e0d`.

TASK-21103 (PR #2002, `6c0abdba7`) made the Persona Buddy controller lazy, taking PIL and 93%
of `Persona_Visual` off the boot path — **−80 modules**, and the PIL chain alone was
**1.276 s of the 3.10 s cold import** measured by the review.

1. `_build_persona_buddy_controller` guards its check-then-build with
   `_persona_buddy_controller_lock` (`app.py:6213`). The public property setter
   (`@persona_buddy_controller.setter`, `app.py:6289`) writes `self._persona_buddy_controller`
   with **no lock**, so a setter call concurrent with a build can be lost, or can replace a
   controller another caller already holds a reference to. Today the setter's only callers are
   tests and skeletal doubles on the UI thread — so this is theoretical, but it is theoretical
   only because of a caller property nothing enforces.
2. The ensure-fallback arm added for explicit Buddy actions
   (`personas_screen.py:6666-6668` — read the passive property, and when it is None call
   `ensure_persona_buddy_controller()`) has **no screen-level test**. That arm is the entire
   reason "Use for Buddy" works from a profile whose preferences still say disabled.

## Acceptance Criteria

- [ ] The controller slot cannot be written outside the lock that guards its construction, or the setter is shown unreachable from any concurrent context and that constraint is enforced rather than assumed
- [ ] A screen-level test drives "Use for Buddy" from a profile whose preferences say disabled and asserts the action completes through the constructed controller
- [ ] That test fails when the ensure-fallback arm is removed
- [ ] TASK-21103's import-closure guard stays green
