---
id: TASK-14901
title: One reversibility story across Library destructive actions
status: To Do
assignee: []
created_date: '2026-08-10 17:20'
labels:
  - library
  - ux
  - recritique-2026-08-09
dependencies: []
priority: medium
---

## Description

Filed from task-4023's cross-task observation (2026-08-09, task-4022 review round 2)
and the re-critique's heuristic #4 score of 1: the Library ships three different
reversibility stories on one screen. Blank/session notes are silently GC'd with no
undo; bulk media delete gets an in-place Undo receipt (task-4022); single media
delete gets nothing at all (confirm, then silence). Prompt/skill drafts discard,
notes persist. One consistent contract — what is undoable, for how long, and what
receipt destruction leaves — needs a design decision that spans notes, media
(single and bulk), prompts, and skills, so it could not ride task-4023's
grammar/copy batch.

## Acceptance Criteria

- [ ] A written rule states which Library destructive actions are undoable and what receipt each leaves; the rule is recorded in backlog/docs or a decision file
- [ ] Single media delete and bulk media delete follow the same receipt/undo pattern
- [ ] Notes deletion/GC behavior matches the written rule (or the rule explicitly names notes as the exception, with the reason)
