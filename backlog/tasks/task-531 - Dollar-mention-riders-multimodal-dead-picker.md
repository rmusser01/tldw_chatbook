---
id: TASK-531
title: >-
  $-mention riders: multimodal draft support + retained dead picker machinery (PR #801)
status: To Do
assignee: []
created_date: '2026-07-24 14:10'
updated_date: '2026-07-24 14:10'
labels:
  - skills
  - console
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deferred from the $-mention layer (PR #801): (1) multimodal drafts deliberately skip $-mention expansion (send-path limitation - expansion under inline-replace/fork needs an attachment-preservation design), so mentions in image-bearing drafts go literal; (2) _console_command_run_skill and the skills picker modal were retained dead with honest docstrings - either remove them or wire the picker to compose $name into the composer.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A design decision (or implementation) for $-mention expansion on multimodal drafts, preserving attachments.
- [ ] #2 Dead picker machinery is removed or re-wired; no retained-dead code paths remain for skill invocation.
<!-- AC:END -->
