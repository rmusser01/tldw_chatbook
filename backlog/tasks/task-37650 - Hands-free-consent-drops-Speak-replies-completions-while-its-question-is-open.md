---
id: TASK-37650
title: Hands-free consent drops Speak replies completions while its question is open
status: To Do
assignee: []
created_date: '2026-10-29 12:46'
labels:
  - console
  - voice
  - speech
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in review of TASK-33626.1. The Hands-free consent reuses the Speak replies coordinator's operation generation and its open-modal hold. Raising it cancels a Speak replies completion that is still resolving its destination, and while the Hands-free question is open the coordinator records completions as observed instead of speaking them. With Speak replies On, a reply that completes around a Ctrl+Shift+H press (or a spoken "hands free") is silently never spoken. The window is narrow and matches the coordinator's existing 'an open speech question holds completions' policy, but the user loses a reply they asked to hear without any notice.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With Speak replies On, a reply that completes while the Hands-free consent is open is spoken (or queued and spoken) after the question closes, whichever answer the user gives
- [ ] #2 Raising the Hands-free consent does not cancel a Speak replies completion that is already resolving its destination
- [ ] #3 A mounted ChatScreen test drives a reply completion during the open Hands-free consent and fails on the TASK-33626.1 code
<!-- AC:END -->
