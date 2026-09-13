---
id: TASK-32551
title: >-
  Library Notes: Preview repeats the title for a body that starts with "#
  Title", and shows a literal "[note]" callout marker
status: To Do
assignee: []
created_date: '2026-09-13 06:47'
labels:
  - library
  - notes
  - critique-3
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 (dev 5fd502dbac), both assessors, persona researcher. Residual of task-32142 (title added above the Preview body); #2's P3 "the preview leaks the Obsidian callout marker" was recorded and never filed.

**What happened.** An imported note whose body starts with `# Library ▸ Notes review` shows the title as the title line and again as the rendered H1 (A 39; B 34). Alex's seeded note shows a literal "[note]" at the top of Preview — the callout is not rendered (A 50). Captures: A 39, 50; B 34.

**Cause.** INFERRED.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 When the body's first line is an H1 equal to the note title, Preview shows it once
- [ ] #2 Obsidian callouts (> [!note] …) render as a styled block in Preview, or the marker line is hidden
- [ ] #3 A test pins both renderings
<!-- AC:END -->
