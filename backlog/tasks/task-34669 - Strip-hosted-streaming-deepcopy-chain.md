---
id: TASK-34669
title: Strip hosted-streaming deepcopy chain
status: Done
created_date: 2026-10-07 02:40
updated_date: 2026-10-07 06:17
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave 3 / F8a: each streamed chunk on six hosted providers is deep-copied 5-9 times and JSON round-tripped twice
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Mutation isolation test green,Per-chunk deepcopy count drops to 0-1,Perf microbenchmark recorded,Hosted provider streaming tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation.md Task 5 (T5)
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Replaced the hosted-streaming deepcopy chain with fresh per-level dict construction at all three sites: engine wrapper __next__ copy deleted (docstring records rationale + pinning test), _filtered_event/_filtered_choice/_filtered_tool_call rebuilt as fresh per-level dicts, _normalize_messages tool-result and _normalize_tools shallow where provably safe (shape-locked all-string leaves; type==function enforced). Two retained deepcopies with verified justification and pinning tests: usage (Mapping-only validation admits nested mutable payloads like OpenRouter prompt_tokens_details/cost) and tool parameters (unbounded JSON schema). The DROP contract (validated-then-dropped extras, annotations covered) closes unknown-key leaks. Evidence: deepcopy/chunk 5-14 -> 0 hot path (usage-bearing cold frames 2: 1 documented exception + 1 pre-existing internal accounting copy at hosted_chat.py:321/399, out of scope); 10k-chunk benchmark 13.07 -> 5.53 us/chunk (~2.3x), 104,256 -> 5 deepcopy calls; visible frames byte-identical (golden pin); mutation-isolation contract (witness/victim lockstep deep poisoning) passes pre and post. 446 passed targeted (baseline 440 + 6 new). Files: hosted_provider_engine.py, hosted_chat.py, Tests/LLM_Calls/test_hosted_streaming_copy_tax.py. Report: .superpowers/sdd/task-5-report.md
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->

## Renumbering provenance

This is the same completed hosted-streaming task previously identified as
TASK-34417, not newly scheduled work. Its creation date, Done status, acceptance
criteria and implementation history are preserved. During PR #3050's rebase the
older Console empty-MCP-catalog task keeps that ID under the older-arrival rule.
The Console claimant was created 2026-10-06 07:22 and first arrived in commit
5d7c244a0a at 2026-10-06 08:50-0700. This younger streaming claimant was created
2026-10-07 02:40 and first arrived in commit c10a106645 at
2026-10-06 19:46-0700. Its replacement ID is TASK-34669. Existing dependencies
follow this same historical task; the higher replacement number does not make
that dependency newly future work.
