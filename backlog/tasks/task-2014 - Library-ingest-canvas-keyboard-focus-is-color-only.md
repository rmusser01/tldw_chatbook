---
id: TASK-2014
title: >-
  Library ingest canvas keyboard focus is color-only and mostly invisible
status: In Progress
assignee: []
created_date: '2026-08-02 21:30'
labels:
  - library
  - ingest
  - ux
  - uat
  - accessibility
  - css
priority: high
dependencies: []
---

## Description (the why)

Tabbing through the ingest canvas, 8 of 10 focus stops produce no
monochrome-visible change: escape-code diffs show the only change is a
background-color run. That violates DESIGN.md's focus contract
(`outline: heavy $accent`, keyboard-first, "color must never be the only
carrier of meaning") and compounds TASK-2010 — users cannot even see that a
job-tick recompose stole their focus. Found in the 2026-08-02 ingest UAT
(critique snapshot 2026-08-02T21-04-04Z).

## Acceptance Criteria (the what)

- [ ] Every focusable widget on the ingest canvas (path/title/author/keywords
      inputs, Browse/Start/row-action buttons, option checkboxes/selects/
      inputs, collapsible headers) shows a visible, non-color-only focus
      indicator, verified in a plain monochrome `tmux capture-pane -p` dump.
- [ ] Focus styling causes no dimension change (DESIGN.md: hover/focus must
      be dimensionally stable).
