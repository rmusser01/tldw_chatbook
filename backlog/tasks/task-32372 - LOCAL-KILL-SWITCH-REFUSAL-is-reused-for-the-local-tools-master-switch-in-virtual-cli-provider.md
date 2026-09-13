---
id: TASK-32372
title: LOCAL_KILL_SWITCH_REFUSAL is reused for the local-tools master switch in virtual_cli_provider
status: To Do
assignee: []
created_date: '2026-09-11 01:55'
labels:
  - mcp
  - console
dependencies: []
priority: low
---

## Description

`virtual_cli_provider.invoke` returns `LOCAL_KILL_SWITCH_REFUSAL` both when `[console] local_tools_enabled` (the master switch task-32286 toggles) is off and when the real kill switch is on. After task-32285 unified the sentence to "tool call blocked: the chat tool kill switch is on", and task-32280 tags it `denied-killswitch`, a master-switch refusal would name the wrong control. Unreachable today (`_compose_virtual_cli_provider` returns no provider when the master is off), so this is defence-in-depth copy.

Source: approval-card / MCP-permissions fix wave 2026-09-10/11 (plan `Docs/superpowers/plans/2026-09-10-approval-card-fix-wave.md`, review snapshot `.impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md`); rider recorded in the lane ledger, not fixed in the wave.

## Acceptance Criteria

- [ ] A master-switch-off refusal (if reachable) carries its own sentence naming the master switch and its own audit token
- [ ] The kill-switch sentence and `denied-killswitch` token are reserved for the kill switch
- [ ] One test pins each path
