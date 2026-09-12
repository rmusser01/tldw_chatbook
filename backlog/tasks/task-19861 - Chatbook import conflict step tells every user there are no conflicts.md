---
id: TASK-19861
title: >-
  Chatbook import conflict step tells every user there are no conflicts
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - ux
  - honesty
  - chatbooks
  - bug
priority: high
dependencies:
  - TASK-19550
  - TASK-19734
---

## Description

Source: surfaced by the reviewer of **TASK-19734**, who drove awkward import
cases through the real wizard against real SQLite. Re-verified at `3605bd52d`
(after 19734 merged as `cf38ee6f8`).

Step 3 of the chatbook import wizard asks the user to choose how conflicts
should be resolved — Skip, Rename, Replace, or Merge. Directly beneath that
choice, the step displays:

> ✅ No conflicts detected - all content is new

This message is **unconditional**. It is a plain `Static` yielded in
`compose()` (`UI/Wizards/ChatbookImportWizard.py:639-644`) with no condition
attached, `display=True` on every render, and no code anywhere ever hides it.
**No conflict check exists anywhere in the wizard or the importer.** Driven
against a destination where all four of the chatbook's items were already
present and would all be skipped, it still says "all content is new".

Every import passes through this step, so every user picks their conflict
strategy on a premise the application invented. A user who reads it and leaves
the default in place has been told that their choice does not matter, which is
exactly backwards in the case that matters most.

The lesser half of the same defect: `#conflict-container`
(`:635-637`), the "Potential Conflicts Detected:" panel and its
`#conflict-list`, carries the `hidden` class in `compose()` and is never
unhidden — no Python outside `compose()` references either id. So the wizard
ships both the true panel (permanently invisible) and the false one
(permanently visible).

This is the same family as **TASK-19550** (a "Create backup" checkbox that
writes a ✓ and does nothing) and **TASK-19734** (per-type ticks the import did
not earn): *the app asserts an outcome it did not produce*. Given that the
false claim here is what the user's next decision is based on, it is arguably
higher-impact than either.

Either half is a valid fix — perform a real conflict check and drive both
elements from it, or say nothing at all about conflicts — but shipping a
reassurance nothing computed is not.

## Acceptance Criteria

- [ ] The conflict step never claims that no conflicts were detected unless a
      check actually ran and actually found none
- [ ] When the destination already contains items the chatbook would import,
      the step either names them or says nothing — it does not assert that all
      content is new
- [ ] No element in the step is rendered permanently-hidden with no code path
      that can reveal it (either `#conflict-container` becomes reachable, or it
      is removed)
- [ ] A test drives the step against a destination pre-loaded with every item
      in the chatbook and asserts the "no conflicts" copy is absent, and is
      mutation-checked (restoring the unconditional `Static` makes it red)
- [ ] A test drives the step against a genuinely empty destination and asserts
      whatever the step now says is true of that case
- [ ] The user-facing documentation page for chatbook import matches whatever
      the step now claims

## Notes

Scope note for the implementer: if a real conflict check is out of scope for
this task, removing the false claim is a complete and shippable outcome on its
own — an honest silence is strictly better than a confident lie, and the
strategy selector still works. Do not close this by making the claim *more
specific* without computing it.
