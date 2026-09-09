---
id: TASK-32067
title: >-
  Library Conversations reader: raw UUID and ISO timestamps; one-page pager
  painted
status: To Do
assignee: []
created_date: '2026-09-08 18:25'
labels:
  - library
  - conversations
  - ux
  - critique-8
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The reader header reads 'Loaded bf20fab2-0474-… · 30 of 30 messages' and each message shows a raw ISO timestamp, while the list shows '27m'; the pager renders '1-6 of 6 · Page 1 of 1 / Already on the first page. · No more re… / ○ Previous ○ Next' on a one-page list, where Media hides it. Evidence: critique #8 snapshot `.impeccable/critique/2026-09-08T16-05-41Z__tldw-chatbook-ui-screens-library-screen-py.md` (dev 1c022378cb, 2026-09-08, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile). Register row 18.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The reader identifies the conversation by title, not UUID, and shows relative or short timestamps
- [ ] #2 Pagers are hidden (or inert and unlabelled) when there is a single page, consistently across Media, Conversations and Prompts
<!-- AC:END -->
