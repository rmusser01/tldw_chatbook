---
id: TASK-1347
title: Watchlists Overview first-run tests assert a container, not its copy
status: To Do
assignee: []
created_date: '2026-07-29 05:30'
labels:
  - watchlists
  - testing
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by mutation during Phase D Task 7: blanking the Overview pane's first-run title leaves **all
four** of its tests green. They assert that a container exists, not that it says anything, so the
first-run guidance could be emptied and CI would not notice.

The first-run affordance is what a brand-new user sees when they have no watchlists — the one
screen state where copy is the entire feature. This is the same shape as the ten-plus
green-for-the-wrong-reason tests found across the Phase D and TASK-1240 branches.

Also in this area: `Tests/UI/test_watchlists_content_pane.py`'s `_render_to_console` helper prints
the rendered article to stdout during the run (`console.print(renderable)` with `record=True`),
which is cosmetic noise in test output.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Overview first-run tests assert the actual guidance copy, and blanking that copy makes at least one of them fail
- [ ] #2 The same check is applied to the tree's first-run affordance
- [ ] #3 _render_to_console no longer writes to stdout during a normal test run
<!-- AC:END -->
