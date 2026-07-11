---
id: TASK-172
title: Wire prompt/character ingest-import completion callbacks
status: To Do
assignee: []
created_date: '2026-07-11 23:53'
labels:
  - follow-up
  - ingest
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
T167 found the notes-import success/failure callbacks were never dispatched (no worker handler matched the group), and fixed it for notes only. The sibling prompt (prompt_ingest_events.py) and character (character_ingest_events.py) import handlers have the same latent gap — their on_import_success/failure callbacks are defined but never invoked. Wire them the same way notes was fixed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Prompt import completion invokes its success/failure callback,Character import completion invokes its success/failure callback
<!-- AC:END -->
