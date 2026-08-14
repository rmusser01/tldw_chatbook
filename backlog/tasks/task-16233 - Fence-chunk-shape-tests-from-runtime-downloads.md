---
id: TASK-16233
title: Fence chunk-shape tests from runtime downloads
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 09:23'
updated_date: '2026-08-14 09:24'
labels:
  - testing
  - chunking
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the all-method chunk-shape contract deterministic by exercising token and semantic fallbacks without contacting model or corpus hosts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Chunk-shape tests make zero network attempts
- [x] #2 Token and semantic cases still exercise their built-in fallback output paths
- [x] #3 Dedicated tokenizer/corpus fallback and download-remediation tests remain green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the Hugging Face and NLTK egress failures as RED evidence.
2. Patch only the tokenizer and NLTK acquisition seams in the broad shape test.
3. Retain and run the dedicated fallback/download unit coverage and static checks.

ADR required: no
ADR path: N/A
Reason: This isolates an offline shape test without changing production chunking behavior.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The broad all-method shape test now substitutes only the built-in word-approximation tokenizer and NLTK fallback seam for token/semantic cases. That prevents GPT-2 and corpus downloads while preserving the exact chunk-shape assertions; the separate missing-corpus/download remediation tests remain unchanged and green. Verification: all 11 chunking tests passed with zero network-guard errors; Ruff lint and py_compile passed; git diff --check passed. Ruff format remains the identical pre-task file baseline.
<!-- SECTION:NOTES:END -->
