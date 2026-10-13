---
id: TASK-34923
title: Stop-copying-a-discarded-message-snapshot-per-stream-chunk
status: To Do
assignee: []
created_date: '2026-10-13 14:49'
labels:
  - performance
  - ui-perf
  - wave-2-streaming
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
append_stream_chunk returns self._snapshot(message) — a full ~50-field instance copy via dict comprehension — on every chunk (console_chat_store.py:14495), yet every hot-path caller (console_chat_controller.py:25888, console_agent_bridge.py:3904/3911) discards the return value. Pure per-token allocation and GC pressure at high token rates.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Hot path no longer pays a message copy per chunk,Any caller relying on the returned copy is updated and tests pass
<!-- AC:END -->
