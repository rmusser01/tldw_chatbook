---
id: TASK-34920
title: Gate-the-0.2s-Console-transcript-sync-tick-on-a-store-change-counter
status: To Do
assignee: []
created_date: '2026-10-13 14:48'
labels:
  - performance
  - ui-perf
  - wave-2-streaming
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
While a run is active, every 0.2s transcript-sync tick shallow-copies ALL session messages (_snapshot, ~50 fields each), re-groups the full transcript (set_messages, chat_screen.py:18111), rebuilds an N-tuple fingerprint, and walks all messages for image/generation/video specs and citations — all BEFORE the refresh_key gate at chat_screen.py:18249 which only skips the DOM reconcile. Cost scales linearly with transcript history at 5Hz for the whole stream, so long conversations burn CPU and can drop frames while streaming even though only the tail row changed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A cheap store change indicator short-circuits the tick when nothing changed since the last processed version,Per-tick work no longer scales with the unchanged transcript prefix while streaming,Transcript rendering behavior unchanged (existing transcript tests pass)
<!-- AC:END -->
