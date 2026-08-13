---
id: TASK-2151
title: RAG answer display clamp should cut at a word boundary with ellipsis
status: To Do
assignee: []
labels:
  - library
  - rag
  - polish
dependencies: []
priority: low
---

## Description

LIBRARY_RAG_ANSWER_DISPLAY_MAX_LENGTH (8,000 chars, an estimate) hard-cuts the rendered RAG answer mid-sentence with no ellipsis, unlike the evidence snippets' _clamp_display_text (word boundary + "..."). Measure real model output lengths first (task-2150's smoke is the natural moment), then either raise/confirm the cap or route the answer through the word-boundary clamp. The all-bracket escape already handles a cut-mid-citation tail safely, so this is polish, not safety.

## Acceptance Criteria

- [ ] Clamp decision made from measured real output (cap value recorded)
- [ ] If clamping stays: word-boundary cut with ellipsis, pinned by a test
