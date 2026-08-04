---
id: TASK-2078
title: >-
  Library: reason tooltips on disabled export/skill-trust buttons and editor
  shortcut hints (F-018, F-019)
status: To Do
assignee: []
created_date: '2026-08-03 17:24'
labels:
  - ux-review
  - library
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Export selected (3 canvases), skill-trust Unlock/Review/Approve, and editor Discard are disabled without reason tooltips; editor ctrl+s/escape are advertised nowhere. Evidence: library_conversations_canvas.py:100-105, library_skills_canvas.py:1154-1175, library_screen.py:877-884. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every disabled button on Library surfaces a reason tooltip,Skill editor shows its ctrl+s/escape hints inline,Tests updated
<!-- AC:END -->
