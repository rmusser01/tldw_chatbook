---
id: TASK-2765
title: 'Home ''Import Library sources'' opens Library without the ingest context it promises'
status: Done
assignee: ['@t2-home']
created_date: '2026-08-06'
updated_date: '2026-10-11'
labels: [home, bug, ui]
dependencies: []
---

## Description (the why)

The fresh-install primary action `Import Library sources`
("Library content makes Console and RAG more useful.") posts
`NavigateToScreen("library", screen_context={})` —
`_home_primary_action_context` returns `{}` for `import_sources`
(`UI/Screens/home_screen.py:97-113`). Library only opens its ingest canvas
when `LIBRARY_NAV_CONTEXT_INGEST` is present (`library_screen.py`), which
this path never sets — compare `app.py`'s ingest deep link, which does.

So the one button a brand-new user sees lands on Library's default rail
row, not an import surface — the label promises an importer and delivers
navigation.

## Acceptance Criteria (the what)

- [x] `import_sources` passes the ingest navigation context so Library opens
      its import surface.
- [x] A test pins the context for the import_sources action.

## Implementation Plan

1. Premise-verify at origin/dev 9f8533686c: `_home_primary_action_context`
   has no `import_sources` branch (falls through to `{}`); Library's
   `LIBRARY_NAV_CONTEXT_INGEST` consumer and app.py's identical deep link
   confirmed.
2. Red-first: pin the fresh-install press as
   `{"ingest_media": True}` context on route `library` (observed `{}` pre-fix).
3. Add the branch returning `{LIBRARY_NAV_CONTEXT_INGEST: True}`.

## Implementation Notes

- `_home_primary_action_context` now returns
  `{LIBRARY_NAV_CONTEXT_INGEST: True}` for `import_sources` — the exact
  key/value app.py's ingest deep link and the Meetings screen use, consumed
  by `library_screen.py` to open the ingest canvas. No Library-side change
  needed.
- Test: `test_home_import_sources_primary_action_opens_library_ingest_canvas`
  (RED pre-fix: `{}`; GREEN: `{LIBRARY_NAV_CONTEXT_INGEST: True}` on route
  `library`).
- Files: `tldw_chatbook/UI/Screens/home_screen.py`,
  `Tests/UI/test_home_screen.py`.
- Verified: full `Tests/UI/test_home_screen.py` green (72 passed) + neighbors
  (`test_home_console_resume.py`) at the fix commit.

ADR required: no
ADR path: N/A
Reason: reuses an existing navigation-context contract (same key as two
other callers); no new cross-module interface.
