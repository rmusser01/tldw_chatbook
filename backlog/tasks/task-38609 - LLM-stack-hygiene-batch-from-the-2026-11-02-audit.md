---
id: TASK-38609
title: LLM stack hygiene batch from the 2026-11-02 audit
status: To Do
created_date: 2026-11-02 14:30
labels:
- cleanup
- llm
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Small latent items in the LLM stack: add_model_capability leaves the is_vision_capable lru cache stale, local engine sessions bypass the ADR-222 session registry, a double-brace debug log never evaluates, the embeddings error body is logged raw, chat_with_openai streaming timeout ignores api_timeout, and _MAX_RETRY_AFTER_SECONDS is defined three times. See bug-sweep-2026-11-02-dev.md LLM stack findings.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 add_model_capability clears is_vision_capable lru entries
- [ ] #2 Local engine sessions route through the provider_sessions registry
- [ ] #3 Debug log fixed and embeddings error body routed through safe_llm_error_detail
- [ ] #4 Streaming timeout honors api_timeout and the duplicate constant is consolidated
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
