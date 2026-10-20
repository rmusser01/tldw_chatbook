---
id: TASK-34753
title: Study notifications repeat their title in the toast
status: To Do
assignee: []
created_date: '2026-10-10 19:17'
labels:
  - study
  - notifications
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Creating a local Study deck shows the toast 'Local study deck created: Local study deck created: <name>'. LocalStudyService passes a title and a message that both carry the same prefix, and NotificationDispatchService joins them as '<title>: <message>', so the user reads the phrase twice. Seen live during TASK-34751 (deck 'bd9st Bio [/b] deck'); unchanged on dev 4de437a5c5 (2026-10-19: local_study_service.py passes title 'Local study deck created' and message 'Local study deck created: <name>'); other local services that dispatch title + message (quizzes, watchlists, media, research, evaluations) may compose the same way.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The deck-created toast reads the action once, followed by the deck name
- [ ] #2 Every local service notification shown as a toast states its action once (title and message are not both the same phrase)
- [ ] #3 A test pins the toast text for deck creation
<!-- AC:END -->
