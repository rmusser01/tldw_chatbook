---
id: TASK-2075
title: 'Library: uniform count policy and honest footer (F-014)'
status: To Do
assignee: []
created_date: '2026-08-03 17:24'
labels:
  - ux-review
  - library
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
(0) appears on 3 rows but not others; footer shows DB telemetry ('Prompts: N/A | Chats/Notes: N/A | Media: N/A') in user chrome. Evidence: db_status_manager.py:69. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Counts follow one policy (dim dash while loading, count when known, none when source off),DB-size telemetry is out of the main footer (Details/Logs),Tests updated
<!-- AC:END -->
