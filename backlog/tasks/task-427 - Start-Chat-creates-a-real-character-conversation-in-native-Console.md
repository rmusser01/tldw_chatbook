---
id: TASK-427
title: Start Chat creates a real character conversation in native Console
status: To Do
assignee: []
created_date: '2026-07-21 09:38'
labels:
  - roleplay
  - console
  - ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the RP/character-card UX review (Docs/superpowers/qa/rp-ux-review-2026-07-21/report.md). P0. Start Chat / Open in Console currently degrade to invisible "staged live work" (chat_screen.py:8490-8497): blank transcript, no greeting, no character identity on the session (no character_id; chat_screen.py:5795-5853), composer prefilled with a meta-instruction. Observed live: sending the prefilled text routed through the agent harness, spawned a sub-agent, and the model replied "Please provide the previous conversation and the character details you'd like me to use" - the staged context never reached it. The RP core loop is a dead end. Start Chat must produce a conversation that (a) shows the character greeting as the first assistant message, (b) is identified with the character (title + session identity), (c) applies the card system prompt/definition on the send path, and (d) sends via the plain provider path, not the agent harness, by default.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Start Chat on a character opens Console with the character greeting visible as the first assistant message
- [ ] #2 The created conversation is titled/labelled with the character name and retains that identity across app restarts
- [ ] #3 The first user send replies in character (card system prompt and definition applied) without the user re-describing the character
- [ ] #4 Character sends do not route through the agent/sub-agent harness unless the user has explicitly enabled an agent profile
<!-- AC:END -->
