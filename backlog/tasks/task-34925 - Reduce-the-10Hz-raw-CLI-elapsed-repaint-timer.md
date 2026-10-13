---
id: TASK-34925
title: Reduce-the-10Hz-raw-CLI-elapsed-repaint-timer
status: To Do
assignee: []
created_date: '2026-10-13 14:50'
labels:
  - performance
  - ui-perf
  - wave-2-streaming
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
ConsoleTranscript arms set_interval(0.1, _tick_raw_cli_elapsed) per message widget while a raw-CLI run is running/stopping (console_transcript.py:1546-1552) — 10Hz repaints of elapsed-text Statics for the whole run duration, when a 1Hz display (or coalesced tick) is the convention elsewhere in the app.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Elapsed repaint cadence reduced or coalesced (e.g. 1Hz) while run state changes land immediately,Test covers elapsed display during a raw-CLI run
<!-- AC:END -->
