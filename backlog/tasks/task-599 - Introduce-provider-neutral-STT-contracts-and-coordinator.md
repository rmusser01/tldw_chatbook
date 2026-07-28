---
id: TASK-599
title: Introduce provider-neutral STT contracts and coordinator
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-24 01:03'
updated_date: '2026-07-28 14:27'
labels:
  - stt
  - architecture
  - routing
dependencies: []
references:
  - backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
documentation:
  - Docs/superpowers/specs/2026-07-23-stt-parakeet-onnx-transcribe-cpp-design.md
  - Docs/superpowers/plans/2026-07-28-provider-neutral-stt-coordinator.md
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
Reason: ADR-025 already governs the provider boundary, language routing, explicit fallback policy, and compatibility facade.

1. Land the previously reviewed dependency-free contract slice and finish stable failure/device-retry values test-first.
2. Add a sealed exact-ID provider registry with declared/runtime capability validation.
3. Add built-in Parakeet v2/v3 and faster-whisper metadata plus deterministic default/manual routing.
4. Add an injected coordinator that validates composed capabilities, normalizes results, and returns explicit retry policy without executing fallback.
5. Isolate retained providers behind an injected bridge and turn TranscriptionService into a thin compatibility facade without promoting defaults.
6. Run focused regression/static checks, map every acceptance criterion, review, and record evidence.

Detailed plan: Docs/superpowers/plans/2026-07-28-provider-neutral-stt-coordinator.md
<!-- SECTION:PLAN:END -->
