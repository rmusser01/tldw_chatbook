---
id: TASK-22453
title: Make older local character conversations discoverable in Roleplay
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-26 07:29'
labels:
  - roleplay
  - ux
  - conversations
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Roleplay currently shows only the 20 most recent saved conversations for a selected local character. Users with longer-running character histories need to find and open older conversations from the same Roleplay surface instead of detouring through Library.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A user can browse beyond the initial 20 saved conversations for the selected local character without leaving Roleplay.
- [ ] #2 Conversation ordering remains stable and no unchanged conversation is skipped or repeated while loading additional results; creations and ordering-key modifications made after browsing begins take effect when the character is reselected, while deletions may be reflected by the next read.
- [ ] #3 Every discovered conversation can be previewed and offers the same Resume chat, Send to Console draft, and Open in Library actions.
- [ ] #4 Loading, empty, exhausted, and retryable failure states are explicit and keyboard accessible.
<!-- AC:END -->
