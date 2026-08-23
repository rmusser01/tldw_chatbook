---
id: TASK-21133
title: >-
  Retire the dead 10s token-count timer - its entire consumer surface was removed
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - cleanup
  - performance
priority: low
dependencies: []
---

## Description

Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21133).

The app-global 10 s interval (app.py:11746) resolves the active footer and, on the chat tab,
attempts four widget queries that ALL fail - `#chat-log` no longer exists anywhere, footers
compose `show_token_count=False` since task-17653, and the estimator result is only
debug-logged (chat_token_events.py:103-181; the file's own comments say the counter is
retired). The producer ticks forever for nothing.

## Acceptance Criteria

- [ ] The interval, update_token_count_display, and the periodic path are deleted; the estimator remains for on-demand callers (input-changed / model-changed)
- [ ] No footer or token-display regression - existing tests green
