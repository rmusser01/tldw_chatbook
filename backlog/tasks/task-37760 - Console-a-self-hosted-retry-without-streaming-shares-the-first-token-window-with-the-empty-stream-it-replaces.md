---
id: TASK-37760
title: >-
  Console: a self-hosted retry without streaming shares the first-token window
  with the empty stream it replaces
status: To Do
assignee: []
created_date: '2026-10-29 17:10'
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
Since TASK-37421 a self-hosted server or Custom OpenAI-compatible slot whose stream is refused or ends empty is asked once more without streaming, inside the same `stream_chat` call. The agent bridge's stall watchdog (`watch_content_stalls` around `stream_chat`, `first_item_timeout_seconds` from `first_token_timeout_seconds`, 300 s by default for keyless providers) waits for the call's first item, and nothing is yielded until the retry's whole reply has been generated. So one window covers the empty stream and the entire non-streamed retry: a server that spends most of the window on an empty stream leaves the retry only the rest, and a long reply from a slow local model can be cut although it would have arrived. Found by the TASK-37421 review from reading the code (`console_agent_bridge.py`, the `watch_content_stalls` call); not reproduced. The llama.cpp fallback has the same shape.

A related, accepted cost: a model that legitimately answers with nothing (an immediate end of sequence) is asked a second time, doubling the local compute for that turn. This matches the llama.cpp fallback and the owner decision for TASK-37421; it is noted here so a fix to the window does not also change it unknowingly.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A non-streamed retry after a refused or empty stream gets a full first-item window of its own, whatever the first stream used
- [ ] #2 A test drives an empty stream that takes most of the window followed by a retry that answers within a full window, and the reply completes; it fails before the change
<!-- AC:END -->
