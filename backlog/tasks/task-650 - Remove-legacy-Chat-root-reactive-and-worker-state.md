---
id: TASK-650
title: Remove legacy Chat root reactive and worker state
status: To Do
assignee: []
created_date: '2026-07-26 23:50'
labels:
  - architecture
  - state
  - chat
  - reliability
dependencies:
  - TASK-648
  - TASK-649
references:
  - backlog/decisions/011-chatbook-workbench-ui-system.md
  - backlog/decisions/026-application-session-state-ownership.md
  - >-
    Docs/superpowers/specs/2026-07-26-tldwcli-reactive-state-decomposition-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Delete the legacy Chat root session, sidebar, prompt, character, widget, worker, and debounce state after the dormant composition no longer consumes it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every legacy Chat root reactive named by the approved specification and every writer, watcher, and dynamic reference to it are removed.
- [ ] #2 _chat_state_lock, current_ai_message_widget, current_chat_worker, current_chat_is_streaming, related accessors, timers, and legacy note identifiers are removed.
- [ ] #3 ChatScreen saves and restores only native Console session and rail owners and performs no root sidebar writes.
- [ ] #4 Native Console worker, cancellation, transcript, and session behavior remains unchanged without a legacy streaming bridge.
- [ ] #5 Normal production TldwCli Chat checks plus focused ownership, privacy, static, formatting, compile, and authorized integration checks pass.
<!-- AC:END -->
