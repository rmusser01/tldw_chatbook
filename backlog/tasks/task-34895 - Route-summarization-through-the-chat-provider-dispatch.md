---
id: TASK-34895
title: Route summarization through the chat provider dispatch
status: To Do
created_date: 2026-10-12 17:52
labels:
- simplification
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Summarization carries a stale second copy of the provider layer: 19 per-provider summarize_with_* functions with a hand-rolled if/elif dispatch alias table per-provider credential resolution (from legacy config sections) and 12 hand-rolled SSE loops. A summarization call is just one chat completion; routing through chat_api_call deletes ~3.9k LOC and unlocks every engine preset for summarization.

Evidence (verified on dev df4985e): Files: LLM_Calls/Summarization_General_Lib.py (9 cloud functions ~1.9k LOC dispatched by _dispatch_to_api at :333 with _CHAT_DISPATCH_NAME_ALIASES at :318) and LLM_Calls/Local_Summarization_Lib.py (9 local functions ~2k LOC). The alias-map comment at Summarization_General_Lib.py:326-330 admits engine presets are unusable today. Transport was already shared via owned_json_post (TASK-32853); everything above transport is per-provider. Keep model_capabilities predicates per TASK-19021/19051. No ADR blocks this.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Single summarize core builds the prompt and calls chat_api_call,Per-provider summarize_with_* functions plus the dispatch chain and alias table removed,Hand-rolled SSE loops in the summarization libs removed,Ingest-consumed error strings preserved or ingest updated alongside,Engine-preset providers (moonshot zai etc) become selectable for summarization,model_capabilities sampling-param gating preserved on the generic path
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
