---
id: TASK-266
title: Native tool-calls for Google/Gemini (functionDeclarations conversion)
status: To Do
assignee: []
created_date: '2026-07-17 02:55'
labels:
  - agents
  - providers
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-263 converted Anthropic; Google/Gemini remains fence-only because chat_with_google normalizes responses and drops function-call parts. Same containment pattern as Anthropic: all conversion inside the handler (OpenAI tools -> functionDeclarations; assistant tool_calls -> functionCall parts; role=tool -> functionResponse parts; response functionCall parts -> OpenAI message.tool_calls; streaming equivalent), NATIVE_TOOLS_PROVIDERS flip gated on a real API round-trip.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 chat_with_google accepts OpenAI-format tools and converts request/response/streaming shapes end-to-end,google joins NATIVE_TOOLS_PROVIDERS only after a live round-trip against the real API,Fence fallback intact until the flip (no partial states)
<!-- AC:END -->
