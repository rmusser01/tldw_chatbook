---
id: TASK-32216
title: >-
  Library Import: 'Show details' on a failed row throws focus to the Keywords
  field
status: To Do
assignee: []
created_date: '2026-09-10 14:53'
labels:
  - library
  - import
  - keyboard
  - critique-9
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Pressing Show details expands the errno line in place, but focus lands on the `Keywords (optional)` input near the top of the form, so a keyboard user who wanted Retry next is typing into a metadata field for the next import; once per row on a folder import. Evidence: critique #9 snapshot (.impeccable/critique/2026-09-10T*__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 02374bf66a on fresh + seeded profiles, 235x52/100x30/60x24). Register row 13.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After Show details toggles, focus returns to the pressed control (the discipline the reader's More strip already applies)
- [ ] #2 Pinned by a test
<!-- AC:END -->
