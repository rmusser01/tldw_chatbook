---
id: TASK-2765
title: 'Home ''Import Library sources'' opens Library without the ingest context it promises'
status: To Do
assignee: []
created_date: '2026-08-06'
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

Found in the guide-G5 verification (code re-verified at dev @ 84e4b33f0;
module unchanged since the live-probed survey).

## Acceptance Criteria (the what)

- [ ] `import_sources` passes the ingest navigation context so Library opens
      its import surface.
- [ ] A test pins the context for the import_sources action.
