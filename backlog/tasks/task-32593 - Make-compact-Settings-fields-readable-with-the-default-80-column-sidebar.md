---
id: TASK-32593
title: Make compact Settings fields readable with the default 80-column sidebar
status: To Do
assignee: []
created_date: '2026-09-15 00:18'
labels:
  - design-system
  - ui
  - audit
dependencies: []
references:
  - Docs/superpowers/reports/2026-09-14-component-first-ui-audit.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Shared Settings rows reserve 24 columns for labels at 80-column width. Providers model and endpoint fields have only about seven editable content cells, while Network policy names truncate to Verif or Cust. Users can type values but cannot comfortably inspect them in the default layout.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With the normal sidebar visible at 80x24, Providers Model and Endpoint and Network CA bundle path have at least 20 editable content columns.
- [ ] #2 The closed Network Certificate verification selector displays each complete selected policy name at 80 columns.
- [ ] #3 Field labels remain complete and unambiguous, and keyboard focus stays visible through row adaptation and 120-to-80-to-120 resizing in both themes.
- [ ] #4 Provider values and typed CA paths survive resizing; Network validation still reports a specific actionable error for an invalid path.
<!-- AC:END -->
