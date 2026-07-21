---
id: TASK-418
title: >-
  Skills copy pass - self-referential empty state, approve modal title, jargon
  labels
status: To Do
assignee: []
created_date: '2026-07-21 15:18'
labels:
  - skills
  - ux
  - copy
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
P2 from the 2026-07-21 Skills UX/NNG review. Copy problems verified live: (1) list empty state says 'No skills yet - create them in Library > Skills' which is where the user already is and never names Create > New skill or Import or the on-disk skills directory; (2) pressing Approve opens a modal titled 'Unlock Local Skill Trust' - task/dialog mismatch; (3) jargon: 'disable model invocation: no' double negative, 'context: inline/fork', row flags 'user - agent' with no legend, 'trusted baseline'/'trust-blocked'. NNG heuristics 2 (match with real world) and 10 (help).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Empty state names the actual creation and import paths and no longer points at itself,Approve flow modal title and message describe approving the reviewed version,Toggle and cycle labels read as plain statements without double negatives,Row flags line is either self-explanatory or accompanied by a legend/tooltip,Copy changes covered by snapshot or unit tests where such tests exist
<!-- AC:END -->
