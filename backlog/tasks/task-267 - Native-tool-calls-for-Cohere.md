---
id: TASK-267
title: Native tool-calls for Cohere
status: Done
assignee:
  - '@claude'
created_date: '2026-07-17 02:55'
updated_date: '2026-07-17 21:42'
labels:
  - agents
  - providers
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Sibling of the Google conversion task: chat_with_cohere normalizes responses and drops tool-call data, keeping Cohere fence-only after task-263. Convert inside the handler (OpenAI tools/history <-> Cohere tool shapes, both response paths), flip gated on a real API round-trip.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 chat_with_cohere accepts OpenAI-format tools and converts request/response/streaming shapes end-to-end
- [x] #2 cohere joins NATIVE_TOOLS_PROVIDERS only after a live round-trip against the real API
- [x] #3 Fence fallback intact until the flip (no partial states)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Plan at Docs/superpowers/plans/2026-07-17-cohere-native-tool-calls.md. DECISION (user-approved): migrate handler v1->v2 /chat FIRST (v1 parameter_definitions cannot express nested JSON Schema; tool_results is out-of-history; no call ids), then near-1:1 OpenAI tool mapping. 6 tasks: v2 text migration byte-compat-pinned -> request tools+tool history -> non-streaming tool_calls -> streaming fragments + cohere_tool_plan extra (mirrors google_thought_signature) -> service pin behind override -> live gate then flip.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Migrated chat_with_cohere v1->v2 /chat (user-approved: v1 parameter_definitions cannot express nested JSON Schema, out-of-history tool_results, no call ids; v2 is OpenAI-shaped end-to-end) with text byte-compat pinned first, then native tools: request-side passthrough w/ guards + tool-role history (document-content shape, positional tool_call_id fallback), non-streaming message.tool_calls, streaming tool-call-start/delta -> accumulator-contract fragments, tool_plan round-tripped as cohere_tool_plan via _PRESERVED_FRAGMENT_EXTRAS (google_thought_signature mechanism). 34 handler tests + real-accumulator cross-layer pin + service-level fence-suppression pin. Task review: Approved/spec PASS. LIVE GATE (real api.cohere.com/v2/chat, command-a-03-2025, streaming): case A single round-trip PASS; case B parallel two-tool FOUND A REAL BUG - no-arg streamed calls accumulate arguments='' and Cohere 400s the echo ('must be a stringified JSON object'); fixed at the request boundary (empty->'{}' + regression test), rerun PASS with two tool_calls in one stream. Flip commit bae6fd6a only after PASS; registry docstring rewritten; service pin now against the real registry. Evidence: Docs/superpowers/qa/cohere-native-2026-07/. Branch claude/cohere-native-267 f3c1b227..bae6fd6a.
<!-- SECTION:NOTES:END -->
