---
id: TASK-21121
title: >-
  The 0.2s run tick gained a full per-tick message-snapshot pass for changed-files scope
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - performance
  - console
priority: medium
dependencies: []
---

## Description

Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21121).

`_console_changed_files_scope()` (chat_screen.py:11467-11497) runs on every 0.2 s run tick and
calls `messages_for_session` - a full shallow-snapshot of every session message
(console_chat_store.py:2858-2865) - then reverse-scans; its own docstring concedes worst-case
O(messages) per tick when the session has no change-review marker, which is the common case.
Combined with the cost path this makes >=2 full snapshot passes per tick during a run.

## Acceptance Criteria

- [ ] The newest run-id (or marker presence) is memoized on the store and bumped on marker append (pattern: the token-estimate cache), so the no-marker common case is O(1) per tick
- [ ] A counter probe during a streamed reply in a large session shows the snapshot-pass reduction; run-tick behavior unchanged
