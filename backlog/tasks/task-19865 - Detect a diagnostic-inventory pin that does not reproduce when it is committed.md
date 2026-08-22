---
id: TASK-19865
title: >-
  Detect a diagnostic-inventory pin that does not reproduce when it is committed
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - process
  - tooling
  - diagnostics
priority: medium
dependencies:
  - TASK-19191
  - TASK-19572
---

## Description

Source: surfaced during **TASK-19572**'s work on the persistent diagnostic
inventory checker.

The persistent diagnostic inventory is a committed pin: a per-file census of
production diagnostic calls that a checker rebuilds and compares against, so
that a newly-added diagnostic has to be looked at by a human before it lands.
The whole method rests on one assumption — that the committed pin is what the
scanner actually produces from the tree it was committed against.

That assumption has been violated at least once, and it was silent. Rebuilding
the inventory from the blobs of the pin's own commit `0b112ab1e` does not
reproduce the pin:

- `Client_Media_DB_v2.py` — rebuild says 339, the pin says 338
- `library_screen.py` — rebuild says 111, the pin says 109

A recovery diff bounded at the pin's own commit shows **nothing** for those two
rows. That is the damaging part: the rows differ, the diff that is supposed to
explain the difference is empty, and so the calls behind them entered the
codebase without ever being reviewed — which is the single thing the pin exists
to prevent. Both were later traced by hand and found benign, but "benign after
the fact" is not the same as "reviewed", and nothing in the process
distinguished them.

The checker gained a warning about this after the fact (TASK-19572 added
`--statements <path> [--since REV]` to recover what actually changed). That
helps whoever is standing in front of a confusing report *later*. It does not
help at the moment the damage is done, which is when a pin is written that does
not describe the tree it is being committed with.

The outcome wanted here is a check that runs **at pin time**: rebuild from the
tree being committed, and refuse a pin that does not reproduce from it.

## Acceptance Criteria

- [ ] Committing a diagnostic-inventory pin that does not reproduce from the
      tree it is committed against is detected and reported, not silently
      accepted
- [ ] The detection names which rows fail to reproduce and by how much, so the
      operator can act without re-deriving it
- [ ] The check is reachable from `scripts/preflight.sh` and from the CI
      derived-artifacts job, so it is not a command someone has to remember
- [ ] The check is mutation-verified: hand-editing one count in the committed
      inventory makes it red, and restoring the value makes it green
- [ ] The historical non-reproducing pin at `0b112ab1e` is used as the
      regression case, or the reason it cannot be is recorded
- [ ] The check does not require an installed environment (the install-free
      contract TASK-19572's workflow establishes)

## Notes

The incident, stated plainly so this does not decay into a rule nobody believes:
two inventory rows rode into the codebase unexamined because the pin that was
supposed to gate them did not describe the tree it shipped with, and the
recovery tooling — asked afterwards what had changed — answered "nothing". The
review method has a lower bound, and this task is about making that bound
visible at the moment it is crossed rather than months later.
