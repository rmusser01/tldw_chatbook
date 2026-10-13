---
id: TASK-34923
title: Stop-copying-a-discarded-message-snapshot-per-stream-chunk
status: Done
assignee: []
created_date: 2026-10-13 14:49
labels:
- performance
- ui-perf
- wave-2-streaming
dependencies: []
priority: low
updated_date: 2026-10-13 16:52
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
append_stream_chunk returns self._snapshot(message) — a full ~50-field instance copy via dict comprehension — on every chunk (console_chat_store.py:14495), yet every hot-path caller (console_chat_controller.py:25888, console_agent_bridge.py:3904/3911) discards the return value. Pure per-token allocation and GC pressure at high token rates.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Hot path no longer pays a message copy per chunk
- [x] Any caller relying on the returned copy is updated and tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Verify append_stream_chunk callers discard the returned snapshot
2. Drop the per-chunk copy from the hot path
3. Targeted test run + notes
<!-- SECTION:PLAN:END -->

## Implementation Notes

- `append_stream_chunk` (console_chat_store.py) now returns `None` on both paths (normal tail and the stopped-message drop), removing the per-chunk `~50-field `_snapshot`` copy. Docstring records the contract change; callers wanting state read `get_message` / `messages_for_session`.
- Production callers verified: controller/realtime/agent-bridge sites all discarded the return; the hands-free tap wrapper (hands_free.py:1694) merely forwarded it — unchanged and harmless. The only return consumers were two store tests and one trace-transforms wrapper, all updated to assert state via store reads (buffer-not-materialized semantics preserved via the live `message.content == ""` check).
- Verification: `Tests/Chat/test_console_chat_store.py` 422/423 — the single failure (`test_hidden_db_legacy_cas_miss_with_unavailable_reader_marks_live_stale`) is pre-existing on this branch (the 4-suite batch — store, trace transforms, hands-free wiring, turn attention — scores an identical 61 failed / 496 passed with this change stashed).
- ADR required: no — hot-path allocation removal, no boundary change.
