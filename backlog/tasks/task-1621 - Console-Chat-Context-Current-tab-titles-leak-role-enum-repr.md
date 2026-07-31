---
id: task-1621
title: 'Console Chat Context viewer: Current-tab fold titles leak the role enum repr'
status: To Do
assignee: []
created_date: '2026-07-31'
labels: [console, polish, ui]
dependencies: []
---

## Description (the why)

The "Chat Context" viewer's **Current** tab titles each message fold with
its role and status, but the role renders as the raw enum repr:
`[ConsoleMessageRole.USER] complete` / `[ConsoleMessageRole.ASSISTANT]
complete` instead of a user-facing `[user]` / `[assistant]`. Internal
identifiers leaking into a user-facing surface; observed live on dev @
ff435772c during the G1 user-guide capture session (2026-07-31), headless
and real-terminal alike.

## Acceptance Criteria (the what)

- [ ] Current-tab fold titles show a human role word (e.g. `[user]
      complete`), never the enum repr.
- [ ] Covered by a test that would fail if an enum repr reappears in the
      fold title.
