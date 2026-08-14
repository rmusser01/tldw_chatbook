---
id: TASK-16218
title: Isolate ebook chunking from tokenizer download
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 00:52'
updated_date: '2026-08-14 00:56'
labels:
  - test-health
  - chunking
  - network
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep ebook chunking contract tests deterministic and offline by using the existing tokenizer fallback.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Ebook chunking tests do not attempt a Hugging Face tokenizer download.
- [x] #2 Real chapter, sentence, and word chunking output assertions remain exercised.
- [x] #3 The complete module, containing chunk, static, and diff gates pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: this isolates tests at the existing optional tokenizer fallback seam without changing runtime behavior.

1. Preserve the four blocked GPT-2 metadata requests as RED evidence.
2. Resolve the optional Transformers tokenizer unavailable for this test module only.
3. Prove ebook chunking still executes, then run module/chunk/static/diff gates.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Made the ebook chunking module resolve the optional Transformers dependency unavailable, selecting the production fallback tokenizer without changing the real chapter, sentence, or word chunking implementations under test. Before the repair, the default EPUB case attempted four Hugging Face connections; afterward all ebook tests ran offline. The focused three-module gate passed 11 tests and final chunk 23 passed 342 tests. Ruff lint/format and diff checks passed.
<!-- SECTION:NOTES:END -->
