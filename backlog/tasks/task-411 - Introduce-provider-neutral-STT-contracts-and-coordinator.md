---
id: TASK-411
title: Introduce provider-neutral STT contracts and coordinator
status: To Do
assignee: []
created_date: '2026-07-24 01:03'
labels:
  - stt
  - architecture
  - routing
dependencies: []
references:
  - backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
documentation:
  - Docs/superpowers/specs/2026-07-23-stt-parakeet-onnx-transcribe-cpp-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Separate provider discovery, capability-aware routing, request and result normalization, error policy, and legacy compatibility from native STT implementations.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Typed request, result, segment, provenance, progress, cancellation, provider metadata, and stable error contracts are defined without importing native runtimes.
- [ ] #2 A sealed provider registry distinguishes declared from runtime-observed capabilities and fails closed on mismatches, duplicate IDs, and unsupported composed pipelines.
- [ ] #3 Semantic default routing resolves omitted language to en, explicit en to Parakeet v2, validated non-English to Parakeet v3, and auto, unsupported languages, or translation to faster-whisper.
- [ ] #4 Parakeet v3 metadata declares routing-only caller assertion rather than an enforced language hint; exact manual providers are honored only when compatible.
- [ ] #5 Cross-engine fallback is never automatic, while the one same-provider accelerator-to-CPU initialization retry remains representable as policy.
- [ ] #6 TranscriptionService remains a thin compatibility facade and retained providers can use an isolated temporary bridge.
- [ ] #7 Dependency-free contract and routing tests cover every policy row, language field, warning, error code, and action eligibility.
<!-- AC:END -->
