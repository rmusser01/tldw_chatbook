---
id: TASK-616
title: >-
  $-mention riders: multimodal draft support + retained dead picker machinery (PR #801)
status: Won't Do
assignee: []
created_date: '2026-07-24 14:10'
updated_date: '2026-10-11 12:00'
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


## Closeout audit 2026-10-11

Board-hygiene flip (cohort A, Jun-Jul 2026 status sweep). **Verdict: WON'T DO** — evidence verified against origin/dev (a3c47c6c72).

Superseded: the $-mention layer this task deferred against no longer exists in that form — _console_command_run_skill and the skills picker modal were removed (the skill path was rebuilt around the extracted skill controller, a315c92206) and source attachment moved to the staged-sources Context design (task-400). Neither rider has a live code path on origin/dev.
