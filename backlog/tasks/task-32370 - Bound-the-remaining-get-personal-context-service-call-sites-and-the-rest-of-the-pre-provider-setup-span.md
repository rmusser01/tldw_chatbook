---
id: TASK-32370
title: Bound the remaining get_personal_context_service() call sites and the rest of the pre-provider setup span
status: To Do
assignee: []
created_date: '2026-09-11 01:55'
labels:
  - console
  - performance
dependencies: []
priority: medium
---

## Description

task-32344 bounded the lazy Personal Context bootstrap inside the Console send path to 10 s (the macOS Keychain auth UI could otherwise block a send for minutes). About five direct `get_personal_context_service()` call sites in `app.py` remain unbounded, and the rest of the setup span (profile-tool compose, canvas registration) has no budget either.

Source: approval-card / MCP-permissions fix wave 2026-09-10/11 (plan `Docs/superpowers/plans/2026-09-10-approval-card-fix-wave.md`, review snapshot `.impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md`); rider recorded in the lane ledger, not fixed in the wave.

## Acceptance Criteria

- [ ] Every direct `get_personal_context_service()` caller in `app.py` goes through the bounded/in-flight-guarded seam or is explicitly documented as off the send path
- [ ] The pre-provider setup span has one budget that covers profile-tool compose and canvas registration, with the same WARNING when exceeded
- [ ] A test pins that a wedged keychain never blocks a send for longer than the budget
