---
id: TASK-99
title: 'Tokenize hardcoded #6f6f6f borders on Schedules/Workflows/ACP panes'
status: To Do
assignee: []
created_date: '2026-06-11 17:32'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
css/components/_agentic_terminal.tcss uses hardcoded #6f6f6f borders for schedules/workflows/acp pane ids, breaking theme portability; replace with $ds-grid-line like the personas fix.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No hardcoded hex borders remain for those screens,Bundle regenerated
<!-- AC:END -->
