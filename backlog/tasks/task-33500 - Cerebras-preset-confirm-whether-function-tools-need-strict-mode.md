---
id: TASK-33500
title: 'Cerebras preset: confirm whether function tools need strict mode'
status: Done
assignee:
  - '@Robert'
created_date: '2026-09-29 19:23'
updated_date: '2026-09-29 20:02'
labels:
  - providers
  - engine
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
oh-my-pi's provider notes claim Cerebras needs `strict: true` on every function tool. If that is true, Chatbook's Cerebras preset sends tool requests Cerebras degrades or rejects; if it is false, adding strict would break most tool turns, because Chatbook's tool schemas are not strict-conformant. Settle it from Cerebras's own documentation before changing anything.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The claim is checked against Cerebras's current public API documentation and the finding is recorded with sources
- [x] #2 The Cerebras preset sends function tools only in a form Cerebras documents as valid
- [x] #3 If no code change is needed, the registry records why, so the claim is not re-litigated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Verify the strict-mode claim against Cerebras's API reference, tool-use and structured-output docs (and the oh-my-pi/hermes code).
2. If strict is optional: no payload change; record the contract and why strict is not sent in the CEREBRAS registry comment.
3. Pin that the Cerebras tools payload carries no strict key.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
No code change needed: the claim is wrong. Cerebras's API reference (inference-docs.cerebras.ai/api-reference/chat-completions, read 2026-09-29) documents `tools[].function.strict` as optional, default false; leaving it off only loosens argument conformance. The one related rule (kimi-k2.7-code needs the SAME strict value on every tool, or none) is satisfied by never sending it. Under strict, API version 2 (default since 2026-07-22) rejects schemas missing `additionalProperties: false` or using pattern/format/minLength/oneOf -- most of Chatbook's own tool schemas (DateTime/Calculator, local_tool_provider) -- so opting in would 400 most tool turns. oh-my-pi's "all_strict" is an all-or-nothing reliability rule with schema rewriting and a retry fallback, not a requirement; hermes-agent sends Cerebras tools without strict.

Recorded in the registry comment above TOGETHER/FIREWORKS/CEREBRAS; pinned by Tests/LLM_Calls/test_preset_reasoning_and_tools.py (tools go out without strict; a caller-supplied strict is still refused). If strict is ever wanted: a conformance-gated all-or-none record flag, no schema rewriting.
<!-- SECTION:NOTES:END -->
