---
id: TASK-32392
title: 'Library Media reader: stored analyses render through an unsanitized Markdown sink'
status: To Do
assignee: []
created_date: '2026-09-11 10:30'
labels:
  - library
  - media
  - security
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Read and Analysis tabs render stored item text and stored analyses through a Markdown sink that has not been sanitized since LIB-13 -- the condition predates task-32365, which only changed which view is shown, and the review of that task declined the finding as needing its own task covering both tabs rather than being folded into a rendering fix. Stored analyses and imported item text are attacker-influenced content in the ingest sense: the bytes come from a file or a URL the user pointed at, not from the user's own typing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Stored item text and stored analyses are sanitized before they reach the Markdown renderer, on both the Read and the Analysis tab
- [ ] #2 A stored analysis carrying markup that would otherwise be interpreted renders as inert text, pinned by a test on each tab
- [ ] #3 The sanitization uses the project's existing input-sanitizing seam rather than a new one
<!-- AC:END -->
