---
id: TASK-32378
title: Prompt lane summary reads four different ways across Library and Console
status: To Do
assignee: []
created_date: '2026-09-11 08:47'
labels:
  - library
  - console
  - prompts
  - copy
  - critique-10
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-32364 changed the Library prompts row's 'System + User' to 'has system and user text' but left its three siblings ('System only', 'User only', 'Empty') in the older register, and Console keeps a separate copy of the same logic with a fourth vocabulary ('System + User' / 'System' / 'User' / 'No compiled lanes'). The same prompt now describes its lanes differently depending on which screen you are on. Evidence: critique #10 fix-wave review of task-32364, findings 6 and the implementer's own open follow-up.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One vocabulary describes a prompt's lanes on every surface that shows it
- [ ] #2 Console's prompt row consumes the Library vocabulary rather than repeating the branch
<!-- AC:END -->
