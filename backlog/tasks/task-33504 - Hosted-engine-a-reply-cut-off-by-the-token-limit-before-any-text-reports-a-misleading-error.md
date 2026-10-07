---
id: TASK-33504
title: >-
  Hosted engine: a reply cut off by the token limit before any text reports a
  misleading error
status: Done
assignee:
  - '@Robert'
created_date: '2026-09-29 19:24'
updated_date: '2026-09-29 20:02'
labels:
  - providers
  - engine
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When a reasoning model spends its whole max-tokens budget thinking, the provider ends the turn with finish reason `length` and no visible text. The engine reports 'finish state is inconsistent', which reads as a provider bug and gives the user no way forward.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A turn that ends on the token limit with no reply text and no tool calls fails with a message saying the token limit was reached before a reply and to raise it
- [x] #2 Other inconsistent finish states still fail exactly as before
- [x] #3 Tests pin the message for streaming and non-streaming turns
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. In the engine finish policy, a length finish with no text and no tool calls raises a ChatProviderError (status 400, so the agent runtime does not retry it) naming the token limit and telling the user to raise it.
2. Other inconsistent finish states unchanged.
3. Tests for streaming and non-streaming turns, plus the retry classification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
HostedPresetFinishPolicy.validate_finish: a length finish with no text and no calls (non-tolerant records) now raises ChatProviderError(status 400) "<Provider> reached the max-tokens limit before writing a reply. Raise Max tokens and try again." Two reasons it is not a protocol error: the non-streaming handler rewrites HostedChatProtocolError into a generic "malformed successful response" 502 (the message would be lost), and the agent runtime retries 5xx (model_retry._RETRYABLE_STATUS) -- the old error was retried up to 3 times, each spending the same tokens to fail the same way. Other inconsistent finishes (stop with no text, length with calls, tool_calls without calls) are unchanged; the old test that grouped length with stop was split. Handler-level tests prove the message reaches the caller on both the streaming and non-streaming paths and that the error is non-retryable.
<!-- SECTION:NOTES:END -->
