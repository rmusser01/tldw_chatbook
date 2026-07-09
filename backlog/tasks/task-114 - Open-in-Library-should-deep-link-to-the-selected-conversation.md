---
id: TASK-114
title: Open in Library should deep-link to the selected conversation
status: Done
assignee: []
created_date: '2026-06-11 04:19'
updated_date: '2026-06-26 15:59'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Personas workbench 'Open in Library' navigates to the conversations route root; NavigateToScreen carries no payload. Extend navigation or post-navigation state so the chosen conversation_id opens directly.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Selecting Open in Library lands on the specific conversation
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: This is a narrow user-navigation bug fix using existing app routing patterns; it does not change storage, service contracts, or long-lived architecture.

1. Locate the Personas Open in Library action and the main navigation message contract.
2. Locate how the Library conversations view loads and selects conversations.
3. Add a failing UI/unit test proving Open in Library carries a selected conversation and Library lands on it.
4. Implement the minimal navigation target handoff without changing unrelated Library routes.
5. Add/adjust focused regression coverage for fallback behavior when no target is provided.
6. Run the targeted UI tests and diff hygiene checks.
7. Mark acceptance criteria complete and add implementation notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented the Personas Open in Library handoff as a direct Library content-hub deep link. The Personas conversations controller now sends `NavigateToScreen("library", {"mode": "conversations", "conversation_id": ...})` only when a conversation is open, and warns without navigating when there is no selected transcript.

`LibraryScreen` now accepts navigation context, switches to Conversations mode, and preserves the requested conversation id so the local source snapshot selects it when records load. Regression coverage verifies the emitted navigation context, no-selection fallback, and Library selection of a non-first conversation from route context.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
