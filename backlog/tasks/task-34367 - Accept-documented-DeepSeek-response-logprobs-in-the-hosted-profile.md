---
id: TASK-34367
title: Accept documented DeepSeek response logprobs in the hosted profile
status: Done
created_date: 2026-10-04 18:05
assignee:
- '@codex'
updated_date: 2026-10-04 19:01
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live DeepSeek UAT receives HTTP 200 but the strict hosted parser rejects the documented choice logprobs field, preventing both complete and streamed replies.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Nonstreaming and streaming DeepSeek responses with null or object logprobs produce replies and retain usage.
- [x] #2 Undeclared response fields and malformed required fields still fail closed.
- [x] #3 Targeted regressions and a live three-message captured conversation pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add realistic response fixtures through the actual DeepSeek adapter and observe RED. 2. Declare the choice-level logprobs allowance in the DeepSeek profile only. 3. Verify complete modules and real captured UAT. ADR required: no. ADR path: backlog/decisions/179-generic-hosted-provider-engine-and-preset-registry.md. Reason: use the existing strict per-provider response allowance contract.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
DeepSeek's documented choice-level logprobs field was rejected even on HTTP 200; the adapter exposed that protocol error as a retryable provider failure. Added only the DeepSeek profile's existing optional choice-key allowance, shared by complete and SSE parsing. Required field checks and unknown-field rejection remain strict; optional logprobs values are ignored, not given a new validation contract. Six actual-adapter response regressions pass, including null/object in both modes and negative controls. The final affected 12-module run passed 216 tests; three unchanged-dev settlement failures were explicitly excluded and their baseline evidence retained. Real setup and three subsequent DeepSeek/deepseek-chat exchanges completed with capture enabled and persisted response links; original owner config hash is unchanged. No new ADR: implements ADR-179. Modified DeepSeek preset and its focused test module. Task IDs moved above a concurrently created main-checkout task before finalization.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
