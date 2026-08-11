---
id: TASK-15423
title: >-
  Library "Open in Console" does not navigate or load the conversation
status: To Do
assignee: []
created_date: '2026-08-11 12:00'
labels:
  - library
  - console
  - bug
  - uat
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Observed twice during TTS UAT on `origin/dev` `82b595049` (2026-08-11), clean
profile, 235x52 tmux, conversation seeded directly into ChaChaNotes
(`add_conversation` + two `add_message`, no character).

Library ▸ Conversations ▸ conversation detail shows an "Open in Console"
button. Clicking it (button visibly takes focus) and pressing Enter did
nothing: the app stayed on the Library screen, and after manually switching to
Console the conversation was not loaded there. Reproduced in two app sessions —
one with the Console provider-setup gate active (no LLM configured) and one
with Console fully unlocked and a session live.

Two adjacent observations from the same UAT, possibly the same root or worth
splitting during triage:

- Console's rail "Search conversations" for the seeded title returned
  "0 matches" while Library search/browse found it fine — so Console could not
  reach the conversation by either route.
- No error, toast, or log line was observed for the failed open (in-app Logs
  Errors count did not change).

Needs verification of the intended contract first: if "Open in Console" is
expected to attach the DB conversation to a Console session, it silently does
nothing; if Console deliberately only lists its own session-created chats, the
button (and the search asymmetry) misleads.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] Activating "Open in Console" on a conversation either opens it in Console or surfaces a visible, accurate reason it cannot
- [ ] Console's conversation search finds DB conversations that Library finds, or the two surfaces' differing scopes are made explicit in the UI
<!-- AC:END -->
