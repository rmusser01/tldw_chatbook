---
id: TASK-38609
title: LLM stack hygiene batch from the 2026-11-02 audit
status: Done
created_date: 2026-11-02 14:30
labels:
- cleanup
- llm
priority: low
updated_date: 2026-11-03 02:56
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Small latent items in the LLM stack: add_model_capability leaves the is_vision_capable lru cache stale, local engine sessions bypass the ADR-222 session registry, a double-brace debug log never evaluates, the embeddings error body is logged raw, chat_with_openai streaming timeout ignores api_timeout, and _MAX_RETRY_AFTER_SECONDS is defined three times. See bug-sweep-2026-11-02-dev.md LLM stack findings.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 add_model_capability clears is_vision_capable lru entries
- [x] #2 Local engine sessions route through the provider_sessions registry
- [x] #3 Debug log fixed and embeddings error body routed through safe_llm_error_detail
- [x] #4 Streaming timeout honors api_timeout and the duplicate constant is consolidated
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Six hygiene items: lru cleared on add, local sessions via the ADR-222 registry, debug log fixed, embeddings errors via safe detail, streaming timeout honors api_timeout, one retry-after constant. Commit 248a204ebb.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
