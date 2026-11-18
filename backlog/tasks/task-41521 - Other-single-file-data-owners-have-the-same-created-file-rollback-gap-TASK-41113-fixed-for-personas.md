---
id: TASK-41521
title: "Other single-file data owners have the same created-file rollback gap TASK-41113 fixed for personas"
status: To Do
assignee: []
created_date: 2026-11-17 05:27
labels:
- follow-up
- backup
- data-loss
priority: high
references:
- tldw_chatbook/Backup_Recovery
- tldw_chatbook/Personas
dependencies:
- TASK-41113
- TASK-41114
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the actor stream (round 15). TASK-41113 fixed the later-rollback refusal after a restore created the persona store file, for personas only. Other single-file data owners (dictionary history, RAG context, and any other owner that a restore can create) have the same gap, so a later rollback may refuse or leave the created file behind. Also noted: mid-session quarantine (`_compensate_failed_creation`) updates only `_blocked_intent_ids`, so the cached summary stays unblocked until a clear; this is pre-existing and harmless.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every backup-enabled single-file owner is listed in the notes with whether a restore can create its file
- [ ] #2 Each owner that can is covered by a rollback test that fails before the fix
- [ ] #3 The mid-session quarantine summary updates immediately, or the notes record why it stays
<!-- AC:END -->
