---
id: TASK-32215
title: >-
  Library Notes: the Sort chooser exists only on the empty list; folder verbs
  are ungated
status: To Do
assignee: []
created_date: '2026-09-10 14:53'
labels:
  - library
  - notes
  - ux
  - critique-9
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The empty Notes toolbar reads `New · Sort: Newest · ○ Select`; with seven notes it reads `New · Select · Add from files… · Export` + `New folder · Add to folder · Move note · Remove placement` and Sort is gone, while three of the eight controls are meaningless without a selection and carry no `○` or reason. Media, Prompts and Skills keep their sort. (The Notes wave-2 branch may touch this toolbar; coordinate.) Evidence: critique #9 snapshot (.impeccable/critique/2026-09-10T*__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 02374bf66a on fresh + seeded profiles, 235x52/100x30/60x24). Register row 12.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Sort is available on a populated Notes list in the same slot as its siblings
- [ ] #2 Selection-scoped folder verbs appear only with a checked row or are gated with the inline-reason grammar
<!-- AC:END -->
