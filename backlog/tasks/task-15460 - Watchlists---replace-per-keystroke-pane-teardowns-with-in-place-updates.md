---
id: TASK-15460
title: Watchlists: replace per-keystroke pane teardowns with in-place updates
status: To Do
assignee: []
created_date: '2026-08-11 12:05'
labels:
  - perf
  - watchlists
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the audit: `UI/Watchlists_Modules/article_list.py:188` declares `search_query = reactive("", recompose=True)` set directly in `on_input_changed` (`:355`) — every character typed tears down and rebuilds ~220 widgets (rows + day headers + toolbar), with a `recompose()` override (`:371`) existing solely to re-focus the destroyed search box; `status_filter`/`runtime_backend` share the blast radius. Same family: `items_pane.py:79/:295` (whole DataTable; its own docstring at `:311-314` admits the teardown-per-keystroke) and `sources_pane.py:131-135/:792` (toolbar + 8-Input create form + table). The downstream DB fetch is already debounced 0.3 s — only the recompose is per-keystroke.

Fix direction: plain reactives + in-place row repaint (surgical helpers already exist, e.g. `article_list._repaint_row`); filter via display toggles or diffing. Stability constraint: replace the re-focus hack with real focus preservation and pin it — focus must never leave the search box while typing. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Typing in Watchlists search/filter boxes causes no pane recompose (evidence)
- [ ] #2 Filtering results identical, including the debounced DB reload path (tests)
- [ ] #3 Focus stays in the input while typing (regression test replacing the re-focus override)
<!-- AC:END -->
