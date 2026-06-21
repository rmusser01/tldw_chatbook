---
id: TASK-129
title: Console chat lifecycle UAT completion
status: To Do
assignee: []
created_date: '2026-06-21 00:36'
labels:
  - console
  - chat
  - uat
dependencies:
  - TASK-128
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify and harden the core Console chat lifecycle so users can create chats, send messages, recover from blocked sends, close tabs, and return to saved conversations without losing context.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 User can create a new chat tab from Console and clearly identify the active tab.
- [ ] #2 User can close a chat tab with the compact close control without accidentally activating the wrong tab.
- [ ] #3 User can type, paste, and send a normal message while seeing all composer text during focus.
- [ ] #4 Blocked-send states produce visible, actionable recovery feedback without silently failing.
- [ ] #5 Sent user messages and completed assistant responses render in the transcript with readable separation.
- [ ] #6 Chat lifecycle state survives leaving and returning to Console within the same app session.
- [ ] #7 Focused regression coverage verifies new chat, close tab, send, blocked-send feedback, transcript rendering, and session return behavior.
- [ ] #8 Rendered CDP/Textual-web evidence is captured before approval and PR completion.
<!-- AC:END -->

## Parallel Ownership

Owns chat tabs, composer lifecycle, send/blocked-send feedback, transcript baseline rendering, and basic in-session persistence. Avoid provider configuration internals, workspace ownership rules, and per-message action semantics beyond selection hooks required for transcript stability.

ADR required: no, unless lifecycle persistence changes database/schema or long-lived chat ownership contracts.
ADR path: N/A unless implementation planning identifies a contract change.
Reason: Expected work is UI and controller hardening around existing chat/session seams.
