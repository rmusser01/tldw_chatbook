---
id: TASK-599
title: Introduce provider-neutral STT contracts and coordinator
status: Done
assignee:
  - '@codex'
created_date: '2026-07-24 01:03'
updated_date: '2026-07-28 17:30'
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
- [x] #1 Typed request, result, segment, provenance, progress, cancellation, provider metadata, and stable error contracts are defined without importing native runtimes.
- [x] #2 A sealed provider registry distinguishes declared from runtime-observed capabilities and fails closed on mismatches, duplicate IDs, and unsupported composed pipelines.
- [x] #3 Semantic default routing resolves omitted language to en, explicit en to Parakeet v2, validated non-English to Parakeet v3, and auto, unsupported languages, or translation to faster-whisper.
- [x] #4 Parakeet v3 metadata declares routing-only caller assertion rather than an enforced language hint; exact manual providers are honored only when compatible.
- [x] #5 Cross-engine fallback is never automatic, while the one same-provider accelerator-to-CPU initialization retry remains representable as policy.
- [x] #6 TranscriptionService remains a thin compatibility facade and retained providers can use an isolated temporary bridge.
- [x] #7 Dependency-free contract and routing tests cover every policy row, language field, warning, error code, and action eligibility.
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented provider-neutral STT contracts, sealed exact-ID registry, deterministic language routing, explicit coordinator failure/action policy, synchronized cancellation/progress handling, retained-provider bridge, and an explicit TranscriptionService compatibility facade. Production defaults remain unchanged; semantic default promotion remains gated to TASK-605. The facade preserves the zero-argument constructor, public signatures, mutable config compatibility, exact provider arguments, and legacy results/exceptions.

ADR check: ADR-025 applies; no new ADR was required. The faster-whisper base declaration conservatively omits float16 because independent device/precision sets cannot truthfully express CPU rejection; exact model declarations may add it when safe.

Acceptance evidence: AC1 Tests/STT/test_contracts.py and test_boundaries.py; AC2 test_registry.py and coordinator composed-capability cases; AC3-AC4 test_routing.py semantic/manual matrices; AC5 test_coordinator.py action/device-retry/cancellation cases; AC6 test_legacy_bridge.py and test_transcription_service_facade.py; AC7 the dependency-free STT policy suites. Final post-rebase evidence: 570 STT/routing/vertical tests passed; 44 faster-whisper tests passed and 2 skipped; 65 Audio/Dictation/Diarization tests passed, 3 skipped, with one live uncached model-download case deselected. Ruff format/check, mypy tldw_chatbook/STT, and git diff --check passed. Final review also added silence-safe timestamp results, fail-closed diarization/timestamp composition, matching fallback eligibility, exhaustive language-metadata invariants, and serialized retained-backend cleanup.

PR review follow-up made a fully default request executable by matching its timestamp request to Parakeet v2's declared no-timestamp capability, preserved explicit non-English routing assertions when translating v3 requests into retained-backend arguments, rejected undeclared automatic language detections, and completed the facade cleanup/type documentation. The complete STT and lazy-MLX slice passed 523 tests after these changes; targeted mypy for the dependency-free contracts/bridge, compileall, and git diff checks also passed.

Task status remains In Progress because the mandatory repository-wide suite is not green on current dev: two unrelated collection tests import removed StreamDone/TabState names; the checked diagnostic inventory is already stale on origin/dev; and the live diarization test requires an uncached Hugging Face download. These base/environment failures were reproduced or directly compared against origin/dev and are outside TASK-599 scope.
<!-- SECTION:NOTES:END -->
