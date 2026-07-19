---
id: TASK-196
title: Prompt version-history UI in the Library editor
status: To Do
assignee: []
created_date: '2026-07-12 13:16'
labels:
  - ux
  - library
  - prompts
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deferred from the 2026-07-12 Library Prompts spec: LocalPromptService already supports list_prompt_versions/restore_prompt_version (rebuilt from sync_log); the v1 editor shows only the vN meta line. Add a history disclosure with per-version preview and Restore.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Editor exposes version history for a prompt,Restoring a version updates the prompt with a distinct outcome message,History survives normal edits (sync_log-based)
<!-- AC:END -->
