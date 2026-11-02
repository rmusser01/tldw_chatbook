---
id: TASK-38461
title: >-
  Console: a send's start ends at its first provider request even when that is
  an auxiliary one
status: To Do
assignee: []
created_date: '2026-10-31 09:09'
labels:
  - console
  - performance
dependencies:
  - TASK-36610
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-36610 ends a send's start at the first gateway adapter entry made in that send's own work (console_send_diagnostics.record_send_stage at provider_entry). That is normally the send's main request, but an auxiliary request its work makes first (an automatic compaction's summary, a response rule's evaluation) ends the start there, and the rest of the start's display reads go off the loop before the main request. This costs first-send time only; no wrong data is shown. The base build's process-wide mark ended the start at the same request. The behaviour is documented on console_send_start.note_provider_request. Found by TASK-36610's review.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A send whose work makes an auxiliary provider request before its main request keeps its start open until the main request, or measured evidence shows the early end costs no first-send time
- [ ] #2 A deterministic test fails on the base build for that case if the behaviour changes
<!-- AC:END -->
