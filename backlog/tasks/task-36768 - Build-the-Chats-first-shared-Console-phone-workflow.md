---
id: TASK-36768
title: Build the Chats-first shared Console phone workflow
status: To Do
created_date: 2026-10-25 17:39
dependencies:
- TASK-36767
labels:
- console
- web-server
- mobile
- adr-244
priority: medium
documentation:
- Docs/superpowers/plans/2026-10-25-shared-console-mobile-supervision-implementation.md
- Docs/superpowers/specs/2026-10-25-shared-console-mobile-supervision-design.md
- backlog/decisions/244-shared-console-ownership-and-mobile-supervision.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give phone users a usable view of existing chats, decisions and individual agents with clear command targets and reconnect state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The phone opens on shared Chats with workspace, activity and running/waiting/finished/unavailable state, then provides live transcript, expandable tool activity, decisions and agent details.
- [ ] #2 Send, Steer, Redirect and Stop name the exact supported target; subagents expose guidance and Stop, and queued acceptance is distinguished from consumption or completed cancellation.
- [ ] #3 Per-chat and per-worker drafts survive transient disconnects while the page remains open, newer edits survive late acknowledgements, and switching phone views never changes desktop selection.
- [ ] #4 Disconnected or uncertain state disables mutation, reconnect refreshes authoritative state without replay, and expired authentication follows the existing sign-in flow.
- [ ] #5 Existing plain assets, semantic browser CSS variables, visible focus and control states support phone touch, keyboard, contained long content and soft-keyboard viewport; no new framework, service worker or localStorage is added.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
