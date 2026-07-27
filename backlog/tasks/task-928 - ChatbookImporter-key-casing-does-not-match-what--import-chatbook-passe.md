---
id: TASK-928
title: >-
  ChatbookImporter key casing does not match what _import_chatbook passes
status: To Do
assignee: []
created_date: '2026-07-27 09:00'
labels:
  - settings
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while fixing TASK-899.

`ChatbookImporter` expects capitalized database keys (`"ChaChaNotes"`, `"Prompts"`, `"Media"`), but `Tools_Settings_Window._import_chatbook` builds its dictionary with lowercase keys (`"chachanotes"`, `"prompts"`, `"media"`).

Pre-existing and independent of the path-resolution work, so it was deliberately left alone there. The effect is that the importer does not receive the database paths under the names it looks for; confirm whether it silently falls back, imports nothing, or raises, and fix the mismatch at whichever end is the real contract.

Worth pinning with a test once resolved, since a casing mismatch between two modules is invisible to type checking and to any test that stubs one side.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] The key casing agrees between `_import_chatbook` and `ChatbookImporter`
- [ ] The real behaviour of the current mismatch is established and recorded in the task notes
- [ ] A test fails if the two sides disagree again
<!-- AC:END -->
