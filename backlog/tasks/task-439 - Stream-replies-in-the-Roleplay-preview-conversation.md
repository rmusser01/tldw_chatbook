---
id: TASK-439
title: Stream replies in the Roleplay preview conversation
status: To Do
assignee: []
created_date: '2026-07-21 09:38'
labels:
  - roleplay
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the RP/character-card UX review (Docs/superpowers/qa/rp-ux-review-2026-07-21/report.md). Observed live: a ~20s generation showed only a static "Running" status and then the full reply at once. Console streams; the preview should too (with the existing status line as fallback for non-streaming providers).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Preview replies render incrementally for providers that support streaming
- [ ] #2 Non-streaming providers keep a working status indicator
<!-- AC:END -->
