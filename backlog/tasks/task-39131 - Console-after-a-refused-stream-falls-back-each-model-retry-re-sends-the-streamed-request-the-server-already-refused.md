---
id: TASK-39131
title: >-
  Console: after a refused stream falls back, each model retry re-sends the
  streamed request the server already refused
status: To Do
assignee: []
created_date: '2026-11-07 02:45'
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
Since TASK-37421 (and TASK-37762 for a named custom endpoint), a self-hosted server or Custom OpenAI-compatible endpoint whose streamed request is refused, or ends empty, is asked once more without streaming inside the same provider call. When that non-streamed request then fails with a retryable error (a 5xx), the agent runtime's model retry (Agents/model_retry.py, three attempts for a 5xx) starts the whole call again, so every attempt first re-sends the streamed request the server has already refused and only then the non-streamed one. Measured in TASK-37762's review against a loopback server that refuses stream=true and answers stream=false with 503: a named custom endpoint made 12 requests and a Custom slot 9, where 3 non-streamed attempts were all that could help. On a paid Custom endpoint each refused or empty streamed request can still cost a request. Found by the TASK-37762 review; pre-existing TASK-37421 design that named custom endpoints now share.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After a run has fallen back from a refused or empty stream to a non-streamed request, its later model-retry attempts against the same provider and model send only the non-streamed request
- [ ] #2 A new run, or a different provider or model, still streams first
- [ ] #3 A test with a server that refuses stream=true and answers stream=false with a retryable 5xx counts the requests for a Custom slot and a named custom endpoint; it fails before the change
<!-- AC:END -->
