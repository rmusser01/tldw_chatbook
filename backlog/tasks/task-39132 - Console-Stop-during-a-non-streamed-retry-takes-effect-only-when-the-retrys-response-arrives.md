---
id: TASK-39132
title: >-
  Console: Stop during a non-streamed retry takes effect only when the retry's
  response arrives
status: To Do
assignee: []
created_date: '2026-11-07 02:47'
labels:
  - console
  - streaming
  - local-models
  - p3
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When a self-hosted server or Custom OpenAI-compatible endpoint refuses a streamed request or streams nothing, Console asks once more without streaming (TASK-37421; named custom endpoints since TASK-37762). That retry is one blocking request, and Stop pressed while it is in flight ends the run only once its response arrives. Measured in TASK-37762's review against a server that delays its non-streamed answer by 2.5 s: both a Custom slot and a named custom endpoint ended cancelled at about 2.6 s, with no third request and no saved reply. The outcome is right, but a slow local model can take minutes to answer without streaming, and during that time Stop appears to do nothing. Pre-existing; separate from TASK-37760, which is about the first-token window.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Stop pressed while a non-streamed retry is in flight ends the run within about a second, without waiting for the response
- [ ] #2 No reply text from the abandoned request is saved or shown, and a later response does not change the stopped turn
- [ ] #3 A test with a server that delays its non-streamed answer asserts the run ends cancelled before the delay elapses, for a Custom slot and a named custom endpoint; it fails before the change
<!-- AC:END -->
