---
id: TASK-32228
title: Library Conversations footer offers neither 'esc focus rail' nor a '/' hint
status: In Progress
assignee: []
created_date: '2026-09-10 14:56'
updated_date: '2026-09-10 17:20'
labels:
  - library
  - conversations
  - footer
  - critique-9
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Conversations canvas footer is nearly empty while every sibling list advertises Escape and `/`. Evidence: critique #9 snapshot (.impeccable/critique/2026-09-10T*__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 02374bf66a on fresh + seeded profiles, 235x52/100x30/60x24). Register row 27.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Conversations footer advertises the same list keys as its siblings, and they work
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Measure the Conversations footer against its siblings.
2. Give the branch the list keys it honours without weakening the pinned two-step 'focus Items'/'focus Library' Escape grammar.
3. Docs + live-verify.
<!-- SECTION:PLAN:END -->
