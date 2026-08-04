---
id: TASK-2306
title: Run selection populates the run detail region
status: To Do
assignee: []
created_date: '2026-08-04'
labels:
  - watchlists
  - bug
  - uat-2026-08-04
dependencies: []
priority: high
---

## Description (the why)

UAT: clicking a run row (and click+Enter) never populates "Run detail" — it
stays "No run selected", leaving the detail/Items/Logs sub-regions
unreachable. Dead interaction on the primary object of the tab.

UAT finding F34 (high).

## Acceptance Criteria (the what)

- [ ] Selecting a run row (mouse and keyboard) populates Run detail, its
      Items list and Logs.
- [ ] A regression test drives selection through the real table and asserts
      the detail region updates.
- [ ] Verified live in a real terminal.
