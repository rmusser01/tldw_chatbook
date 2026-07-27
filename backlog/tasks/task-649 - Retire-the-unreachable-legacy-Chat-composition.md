---
id: TASK-649
title: Retire the unreachable legacy Chat composition
status: To Do
assignee: []
created_date: '2026-07-26 23:50'
updated_date: '2026-07-27 00:16'
labels:
  - architecture
  - state
  - chat
  - cleanup
dependencies:
  - TASK-648
references:
  - backlog/decisions/011-chatbook-workbench-ui-system.md
  - backlog/decisions/026-application-session-state-ownership.md
  - >-
    Docs/superpowers/specs/2026-07-26-tldwcli-reactive-state-decomposition-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove the dormant ChatWindow and ChatWindowEnhanced production surface instead of preserving dead UI with a second application-state owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An import and reachability manifest proves that no registered production route constructs or imports the deleted legacy Chat composition.
- [ ] #2 ChatScreen removes chat_window and _ensure_chat_window branches while native Console composition and routing remain unchanged.
- [ ] #3 Legacy composition, exclusive helpers, handlers, styles, and tests are deleted; shared modules remain only for live consumers with legacy-only branches removed.
- [ ] #4 No LegacyChatState, compatibility root state, or adapter is introduced, and direct import of the retired surface is not supported.
- [ ] #5 Normal production Console route, action, and snapshot checks plus focused static, formatting, compile, and authorized integration checks pass.
<!-- AC:END -->
