---
id: TASK-16237
title: Keep Kokoro real-model validation offline by default
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 09:54'
updated_date: '2026-08-14 09:55'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent the ordinary TTS validation suite from attempting live model and tokenizer downloads when local Kokoro ONNX prerequisites are absent.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The real Kokoro integration test skips before initialization unless its local ONNX dependency and model artifacts exist.
- [x] #2 The network guard observes no Kokoro egress in the ordinary suite.
- [x] #3 Focused Kokoro validation and static checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce and classify the unexpected NLTK/Hugging Face egress.
2. Add a local-prerequisite admission check before real backend initialization.
3. Run the focused integration node, complete Kokoro validation module, and scoped static checks.
4. Record verification and close the task.

ADR required: no
ADR path: N/A
Reason: This is an offline test-admission fix and does not change the TTS runtime boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added fail-before-initialization admission for the optional real Kokoro ONNX test: the local kokoro_onnx dependency and both model artifacts must already exist, otherwise the node skips without model/tokenizer egress. The focused node skipped before network, the full module passed 13 tests with one prerequisite skip, Ruff check/format and git diff --check passed. ADR required: no (offline test admission only).
<!-- SECTION:NOTES:END -->
