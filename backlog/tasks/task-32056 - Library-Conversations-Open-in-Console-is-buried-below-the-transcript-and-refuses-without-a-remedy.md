---
id: TASK-32056
title: >-
  Library Conversations: 'Open in Console' is buried below the transcript and
  refuses without a remedy
status: To Do
assignee: []
created_date: '2026-09-08 18:23'
labels:
  - library
  - conversations
  - console
  - ux
  - critique-8
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The action sits at the bottom of the reader (row 49 of 52 under a 30-message thread, beside a clipped pager) and pressing it toasts 'Copy or link this conversation into workspace workspace-default before using it in Console.' with nothing on screen that performs the copy or link. Seeded (unlinked) rows hit this; imported or restored conversations will too. Evidence: critique #8 snapshot `.impeccable/critique/2026-09-08T16-05-41Z__tldw-chatbook-ui-screens-library-screen-py.md` (dev 1c022378cb, 2026-09-08, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile). Register row 7.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 'Open in Console' lives in the reader header beside Read/Info, where Media places 'Use in Console', with a keyboard route
- [ ] #2 An ineligible conversation shows a disabled action with the reason inline and an adjacent action that links or copies it into the active workspace, after which the hand-off works
- [ ] #3 No refusal is delivered only as a toast
<!-- AC:END -->
