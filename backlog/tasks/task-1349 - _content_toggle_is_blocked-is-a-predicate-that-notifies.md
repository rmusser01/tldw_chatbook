---
id: TASK-1349
title: _content_toggle_is_blocked is a predicate that notifies
status: To Do
assignee: []
created_date: '2026-07-29 05:30'
labels:
  - watchlists
  - code-health
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`watchlists_collections_screen.py:1544-1569`. The name promises a pure query, but the function
calls `self.notify(...)` before returning `True`.

Low risk as written — two call sites, both user-initiated gestures where the toast is wanted. It is
filed because this codebase has already been bitten by exactly this shape: `provider_is_configured()`,
a predicate, wrote an `eval_models` row from `compose()`, so opening the Evals screen mutated the
database on every fresh install. A side-effecting predicate is safe until someone calls it from a
render path, and the name gives no warning.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The predicate is pure, with the notify moved to its callers, or renamed to state that it acts
- [ ] #2 A grep confirms no other predicate-named function in the Watchlists modules has side effects
<!-- AC:END -->
