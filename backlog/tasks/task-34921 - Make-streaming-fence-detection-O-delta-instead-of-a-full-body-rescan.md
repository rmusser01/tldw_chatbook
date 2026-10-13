---
id: TASK-34921
title: Make-streaming-fence-detection-O-delta-instead-of-a-full-body-rescan
status: To Do
assignee: []
created_date: '2026-10-13 14:48'
labels:
  - performance
  - ui-perf
  - wave-2-streaming
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On every 0.2s sync tick for every streaming markdown row, the open-fence check (console_transcript.py:1946 -> :1248-1313) splits the ENTIRE accumulated body into a list and regex-matches line-by-line — even for replies that contain no fence at all (the scan is how it finds out). Quadratic per message over the stream; the docstring itself calls the cost unbounded with message size. A running fence-parity counter updated from the appended delta would make it O(delta).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Fence parity tracked incrementally from appended deltas or scanning limited to the unexamined tail,No per-tick full-body split+regex over unchanged text,Open-fence flush behavior identical (existing fence tests pass)
<!-- AC:END -->
