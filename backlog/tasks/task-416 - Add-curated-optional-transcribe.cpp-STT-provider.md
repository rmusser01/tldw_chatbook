---
id: TASK-416
title: Add curated optional transcribe.cpp STT provider
status: To Do
assignee: []
created_date: '2026-07-24 01:04'
labels:
  - stt
  - gguf
  - native
dependencies:
  - TASK-407
  - TASK-411
  - TASK-412
  - TASK-413
references:
  - backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
documentation:
  - Docs/superpowers/specs/2026-07-23-stt-parakeet-onnx-transcribe-cpp-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Adopt transcribe.cpp as an optional GGUF STT adapter with a small representative catalog while keeping Parakeet ONNX and faster-whisper authoritative for semantic default routing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The optional extra pins transcribe.cpp 0.1.3 exactly, imports it lazily, and reports unavailable or incompatible runtime states without affecting application startup.
- [ ] #2 The curated catalog contains Whisper small, Canary 180M Flash, Moonshine tiny, and Qwen3-ASR 0.6B with reviewed immutable revisions, sizes, digests, licenses, and per-model capabilities.
- [ ] #3 Whisper, Canary, and Moonshine default to Q8_0 with F32 optional; Qwen3-ASR defaults to Q8_0 with BF16 optional and rejects explicit language constraints until the user selects auto.
- [ ] #4 The adapter consumes only managed compatible GGUF handles, normalizes required 16 kHz mono audio, allows one active inference, and records precise results and provenance.
- [ ] #5 No curated transcribe.cpp model participates in provider default routing or silent fallback, and unsupported capabilities fail before enqueue.
- [ ] #6 All curated families pass file and bounded-buffer contract smoke, cancellation where supported, crash containment, and platform wheel tests.
<!-- AC:END -->
