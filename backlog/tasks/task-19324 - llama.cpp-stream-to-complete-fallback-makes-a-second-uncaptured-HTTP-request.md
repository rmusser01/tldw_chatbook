---
id: TASK-19324
title: llama.cpp stream-to-complete fallback makes a second uncaptured HTTP request
status: To Do
assignee: []
created_date: '2026-08-21 06:03'
labels:
  - console
  - exchange-capture
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Console Conversation Inspector's exchange-capture feature (task-18300 and its subtasks) captures every provider call so a user can see exactly what was sent and received on a past turn. The llama.cpp gateway branch in console_provider_gateway.py has a documented fallback path where a streaming request that fails mid-stream retries as a non-streaming complete_llamacpp_chat call. That second HTTP request is made outside the capture seam -- the retry's request/response never gets its own ExchangeCapture, so a turn that actually made two HTTP calls to the local server shows only one in the inspector, understating what was actually sent. Found during the final whole-branch review of task-18300 (M2), explicitly deferred rather than fixed at merge time.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The llama.cpp stream-to-complete fallback path's second HTTP request is captured with its own ExchangeCapture (or the existing capture is amended to reflect both calls), so the Exchange tab shows every HTTP request a turn actually made.
- [ ] #2 A test exercises the fallback path (streaming request fails mid-stream, retried as non-streaming) and asserts the retry's request/response appears in exchange_captures().
<!-- AC:END -->
