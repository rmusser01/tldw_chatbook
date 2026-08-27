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

## Implementation Plan

1. Add deterministic `(last_modified, id)` seek pagination to the existing local character-conversation DB query, preserving legacy offset callers.
2. Add presentation-only conversation tail states and append behavior to the Roleplay inspector.
3. Orchestrate 21-record sentinel reads, retry, deduplication, and stale-result ownership in the conversations controller and screen.
4. Verify database, keyboard, focus, layout, preview-action parity, and isolated live behavior with targeted checks.

ADR required: no
ADR path: N/A
Reason: This is a routine extension of the existing local Roleplay discovery query and inspector list; it does not change storage, ownership, synchronization, security, or service boundaries.

Detailed plan: [2026-08-27-task-22453-older-roleplay-conversations-implementation.md](../../Docs/superpowers/plans/2026-08-27-task-22453-older-roleplay-conversations-implementation.md)
