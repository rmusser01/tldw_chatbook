---
id: TASK-266
title: Native tool-calls for Google/Gemini (functionDeclarations conversion)
status: Done
assignee:
  - '@claude'
created_date: '2026-07-17 02:55'
updated_date: '2026-07-17 04:31'
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
- [x] #1 chat_with_google accepts OpenAI-format tools and converts request/response/streaming shapes end-to-end,google joins NATIVE_TOOLS_PROVIDERS only after a live round-trip against the real API,Fence fallback intact until the flip (no partial states)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Plan at Docs/superpowers/plans/2026-07-17-google-native-tool-calls.md — task-263 containment pattern: (1) request-side (functionDeclarations wrapping w/ Gemini-shape passthrough; assistant.tool_calls -> functionCall parts; role=tool -> coalesced functionResponse user turn with id->name map + positional fallback — Gemini has no call ids); (2) streaming functionCall parts -> whole OpenAI delta.tool_calls fragments (accumulator-compatible) + pins for the ALREADY-EXISTING non-streaming parsing; (3) coordinator live gate (user-provided key) THEN flip + close-out. Executed via SDD.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
task-263 containment pattern: all conversion in chat_with_google. Request: OpenAI tools -> functionDeclarations (Gemini-shape passthrough; blank-name dropped); assistant.tool_calls -> functionCall parts (list-content text preserved after a review Important — same bug class as the anthropic sibling); role=tool -> functionResponse parts coalesced into ONE user turn, name via id->name map from echo turns + positional fallback (Gemini has no call ids; response must be a JSON object — dict-parseable content used directly else {result: str}). Non-streaming functionCall->tool_calls parsing already existed — pinned incl. parse_native_tool_calls round-trip. Streaming: whole functionCall parts -> complete OpenAI fragments (running cross-stream index; malformed-part skip after a reviewer-repro'd stream-abort Important; blank-name consumes no position). LIVE-GATE DISCOVERY: Gemini 3 models 400 without thoughtSignature round-trip — now carried opaquely (google_thought_signature) through parser/fragments/echo/request-converter + gateway accumulator extra-key preservation, 3 tests. Gate PASS on gemini-flash-latest (streaming): single round-trip + genuine parallel two-tool batch. Evidence Docs/superpowers/qa/google-native-2026-07/. Cohere remains fence-only (task-267). Out-of-scope observed: google map 'temperature':'temp' keys a nonexistent generic name (temp silently dropped for google).
<!-- SECTION:NOTES:END -->
