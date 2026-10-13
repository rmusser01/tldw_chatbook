---
id: TASK-34922
title: Reuse-the-transcript-grouping-pass-between-set_messages-and-row-building
status: To Do
assignee: []
created_date: '2026-10-13 14:49'
labels:
  - performance
  - ui-perf
  - wave-2-streaming
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
group_console_transcript_messages runs twice per refresh over the same message list — once in set_messages (console_transcript.py:4393) and again in _transcript_rows (:6947) — doubling the O(N) transcript-length-proportional work on every refresh tick during streaming. The second pass could reuse the units/spans the first pass already cached.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Units computed in set_messages are reused by the row-builder path (single grouping walk per refresh),No duplicate grouping walk in the same refresh cycle
<!-- AC:END -->
