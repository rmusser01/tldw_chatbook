---
id: TASK-140
title: Support Markdown character card imports
status: To Do
assignee: []
created_date: '2026-06-28 02:43'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Allow the ds-native Personas character import flow to import Markdown files that embed existing supported character-card data, without introducing a new free-form Markdown schema.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Personas import picker offers Markdown files alongside existing JSON and PNG character-card imports
- [ ] #2 Markdown files with YAML frontmatter import through the existing character-card parser
- [ ] #3 Markdown files with fenced JSON card data import through the existing character-card parser
- [ ] #4 Invalid Markdown fails without creating or selecting a character and shows the existing import failure path
- [ ] #5 Existing JSON and PNG import behavior remains unchanged
<!-- AC:END -->
