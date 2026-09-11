---
id: TASK-32373
title: Permissions matrix: Allow (session) reads ambiguously on a non-ask row
status: To Do
assignee: []
created_date: '2026-09-11 01:55'
labels:
  - mcp
  - copy
dependencies: []
priority: low
---

## Description

task-32291 appends ` (session)` to a matrix cell that has a live session approval. On a row whose effective state is already Allow the cell reads "Allow (session)", which suggests the allow is session-scoped when it is the persistent policy plus a redundant stamp. Needs a copy decision (suppress the suffix on Allow rows, or word it as "also approved this session").

Source: approval-card / MCP-permissions fix wave 2026-09-10/11 (plan `Docs/superpowers/plans/2026-09-10-approval-card-fix-wave.md`, review snapshot `.impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md`); rider recorded in the lane ledger, not fixed in the wave.

## Acceptance Criteria

- [ ] A persistent Allow row with a session stamp renders copy that does not imply the allow is temporary
- [ ] The legend explains the chosen wording
- [ ] The matrix test covers Allow + session and Ask + session
