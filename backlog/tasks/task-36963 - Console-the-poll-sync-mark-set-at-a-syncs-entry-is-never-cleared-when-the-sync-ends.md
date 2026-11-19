---
id: TASK-36963
title: >-
  Console: the poll-sync mark set at a sync's entry is never cleared when the
  sync ends
status: To Do
assignee: []
created_date: '2026-10-26 03:23'
labels:
  - console
  - tests
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the TASK-33620.15.4 review (2026-10-25). config_sync.begin_console_sync sets the _POLL_SYNC context variable at every sync's entry and nothing resets it when the sync finishes. Today only project_console_tick reads it, and settle_stopped_poll passes force=True, so no caller observes a stale mark. Any future inline caller of project_console_tick running in the poll task's context after a poll sync would read the mark as still set and skip the explicit-sync deferral it should apply. Resetting it needs an end-of-sync hook in ChatScreen's sync (screen-size ratcheted), which is why it was not folded into TASK-33620.15.4.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The poll-sync mark reads False once the sync that set it has finished, on success and on failure
- [ ] #2 A deterministic test pins it and fails on the pre-fix head
- [ ] #3 ChatScreen's screen-size ratchet row does not grow, or grows with a dated owner-decision comment
<!-- AC:END -->
