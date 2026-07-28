---
id: TASK-1142
title: 'Browser section headers are dead affordances'
status: To Do
assignee: []
created_date: '2026-07-27 18:05'
labels: [console, ui, uat]
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UAT (Docs/superpowers/qa/parallel-agents-uat-2026-07-27, F4): the conversation browser's top-level section headers (Starred/Workspaces/Chats) render collapse carets (▾/▸) but do not respond to clicks (caret column, caret+1, and label all inert), while workspace group rows toggle fine. This is a misleading affordance and makes TASK-912's collapsed-section marker aggregation unreachable through live interaction. Either wire the headers for click-toggle (persisting like group state) or remove the caret glyph from non-interactive headers.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Section headers either toggle on click (state persisted, aggregate glyph shown when collapsed) or carry no collapse affordance.
- [ ] #2 A mounted test drives the chosen behavior through the real click path.
<!-- AC:END -->
