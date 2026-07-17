---
id: TASK-266
title: Native tool-calls for Google/Gemini (functionDeclarations conversion)
status: In Progress
assignee:
  - '@claude'
created_date: '2026-07-17 02:55'
updated_date: '2026-07-17 03:48'
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Plan at Docs/superpowers/plans/2026-07-17-google-native-tool-calls.md — task-263 containment pattern: (1) request-side (functionDeclarations wrapping w/ Gemini-shape passthrough; assistant.tool_calls -> functionCall parts; role=tool -> coalesced functionResponse user turn with id->name map + positional fallback — Gemini has no call ids); (2) streaming functionCall parts -> whole OpenAI delta.tool_calls fragments (accumulator-compatible) + pins for the ALREADY-EXISTING non-streaming parsing; (3) coordinator live gate (user-provided key) THEN flip + close-out. Executed via SDD.
<!-- SECTION:PLAN:END -->
