---
id: TASK-1563
title: 'Settings: collapse view-only stub categories; fix Follow-up: doubling'
status: Done
assignee: []
created_date: '2026-07-31 02:00'
labels: [settings, ux, P2]
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique finding (P2): seven categories (Artifacts, Skills, Schedules,
Watchlists, Workflows, MCP Defaults, ACP Defaults) are full navigation peers
whose entire page says "change this elsewhere" -- a third of the rail is
non-actionable prose. Each also renders the shipped copy bug "Follow-up:
Follow-up: ..." (root cause: `_detail_row("Follow-up", contract.follow_up)`
at settings_screen.py:10433 re-prefixes strings already prefixed at
540-665).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 "Follow-up:" renders exactly once per stub page.
- [x] #2 View-only categories are visually distinguished in the rail (dimmed and/or badged "view") or collapsed under one "Owned elsewhere" group.
- [x] #3 Rail remains navigable to their detail pages.
<!-- AC:END -->

## Implementation Notes

- Doubling root cause fixed at the data: seven contract `follow_up` strings
  carried a literal "Follow-up: " prefix that `_detail_row("Follow-up", ...)`
  re-labeled; the prefixes are stripped (single render verified live on
  Schedules).
- Rail treatment: every category whose ownership record has
  `writes_allowed=False` now carries a " (view)" badge in its rail label via
  `_category_button_label` -- honest for the seven Domain Defaults stubs AND
  the always-view categories (Overview, Diagnostics, Privacy); one exact-label
  test updated accordingly. Categories stay navigable. Chose badge-only over
  collapsing into an "Owned elsewhere" group to keep discoverability and the
  smaller diff; grouping remains open as a future IA change if the rail grows.
