---
id: TASK-1349
title: _content_toggle_is_blocked is a predicate that notifies
status: In Progress
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
- [x] #1 The predicate is pure, with the notify moved to its callers, or renamed to state that it acts
- [x] #2 A grep confirms no other predicate-named function in the Watchlists modules has side effects
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
AC#1 (rename arm): `_content_toggle_is_blocked` -> `_refuse_content_toggle_off_read_tab`. The
notify stays in the helper (three call sites all want the same toast; moving it would duplicate the
string 3x), so the fix is to name it as the ACTION it is — a verb signals the side effect that the
innocent predicate name hid. Docstring states the naming rule and cites the `provider_is_configured()`
render-path incident. All 5 call sites + the doc cross-reference renamed.

AC#2: an AST scan (`Tests/Watchlists/test_no_side_effecting_predicates.py`) confirms
`_content_toggle_is_blocked` was the ONLY predicate-named function with a side effect across
`UI/Watchlists_Modules/` + `watchlists_collections_screen.py`; the other five (`_is_url_family_source`,
`_is_sole_expanded_centre_region`, `_is_markdown`, `_has_local_wc_context`, `_can_generate_briefing`)
are pure. Kept as a permanent regression: a new side-effecting predicate now fails this test rather
than waiting to be found from a render path. Mutation-verified (rename back -> guard reds).

Files: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`,
`Tests/Watchlists/test_no_side_effecting_predicates.py` (new).
<!-- SECTION:NOTES:END -->
