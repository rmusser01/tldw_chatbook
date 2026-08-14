---
id: TASK-16307
title: Isolate summarization migration tests from tokenizer download
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 00:16'
updated_date: '2026-08-14 00:16'
labels:
  - test-health
  - prompts
  - network
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep summarization prompt-migration tests focused on prompt precedence by preventing their incidental token counter from loading GPT-2 over the network.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Summarization migration tests use the existing offline fallback tokenizer.
- [x] #2 Prompt precedence and payload assertions remain unchanged and pass.
- [x] #3 Focused, containing-chunk, mutation, static, and diff evidence pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: this isolates tests at an existing optional-dependency seam without changing runtime behavior.

1. Preserve the four Hugging Face blocked-egress failures as RED evidence.
2. Make the test module's tokenizer dependency resolve unavailable so the production fallback path is used.
3. Prove removing that isolation reintroduces egress, then run focused/chunk/static/diff gates.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Made the summarization migration module resolve the optional Transformers dependency unavailable, which drives the existing word-count fallback while leaving prompt resolution, chunking, and fake LLM payloads intact. RED: four prompt tests each attempted Hugging Face GPT-2 metadata requests and were rejected by the network guard. GREEN: all seven module tests plus the complete 25-file containing chunk passed; removing the override restores the blocked egress. Scoped Ruff/format and diff checks passed. ADR required: no; runtime tokenizer behavior is unchanged.
<!-- SECTION:NOTES:END -->
