---
id: TASK-1495
title: >-
  Wizard step viewport clips without scrolling; primary inputs and CTAs
  invisible
status: To Do
assignee: []
created_date: '2026-07-31 00:22'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UX UAT on merged dev (PR-1095): at 120x40 the Provider step shows only the provider RadioSet — the API-key input, discovery banner, and Use-this-server button are clipped below a non-scrolling fold with no indicator; the Full-track Summary exit buttons are never visible. Root cause: the wizard step region lacks overflow-y and internal lists are uncapped. Supersedes/expands TASK-1375 (which assumed long-path edge case).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Provider step shows the API key field without scrolling at 120x40
- [ ] #2 Step content scrolls with a visible scrollbar when it overflows
- [ ] #3 Every step's primary action zone (incl. Summary exits) is visible at 120x40 and reachable at 80x24
- [ ] #4 Pilot test asserts compositor visibility of key input and Summary exits
<!-- AC:END -->
