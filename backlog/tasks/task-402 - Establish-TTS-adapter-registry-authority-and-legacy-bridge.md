---
id: TASK-402
title: Establish TTS adapter registry authority and legacy bridge
status: Done
assignee:
  - '@codex'
created_date: '2026-07-24 00:42'
updated_date: '2026-07-24 15:07'
labels:
  - tts
  - architecture
dependencies: []
references:
  - backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md
documentation:
  - Docs/superpowers/specs/2026-07-23-audio-cpp-tts-adapter-registry-design.md
  - Docs/superpowers/plans/2026-07-23-tts-adapter-registry-legacy-bridge.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace direct access to the wildcard TTS backend manager with one application-owned, sealed adapter registry while preserving the behavior of all six existing TTS providers through provider-scoped legacy adapters.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Application code owns one TTSService and one sealed TTSAdapterRegistry; the compatibility accessor returns only the bound service and can be explicitly reset.
- [x] #2 The registry uses exact canonical provider IDs with an empty initial alias map, lazily materializes at most one adapter per provider under concurrency, and rejects duplicate or post-seal registration.
- [x] #3 Operation leases keep adapter resources alive through complete or partial response consumption; identical configuration is a no-op, changed configuration retires only the selected provider, and shutdown is ordered, bounded, and idempotent.
- [x] #4 OpenAI, ElevenLabs, Kokoro, Chatterbox, Higgs, and AllTalk remain available through isolated provider-scoped legacy hosts without exposing TTSBackendManager or concrete backends outside the bridge.
- [x] #5 The enumerated legacy resolver covers every internal-model form used by current callers, and the existing generate_audio_stream signature routes through the registry and closes its response on success, failure, cancellation, and partial consumption.
- [x] #6 Per-internal-backend legacy locks serialize construction, initialization, progress callback installation, stream consumption, and callback clearing; progress-sink failures do not fail synthesis while different providers may operate concurrently.
- [x] #7 Focused registry, bridge, application-binding, lifecycle, concurrency, and compatibility tests pass without changing visible STTS behavior.
- [x] #8 New registry and bridge diagnostics log neither configuration values nor synthesis text, and regression coverage removes the existing OpenAI API-key-prefix disclosure.
- [x] #9 STTS settings post canonical Textual select values, support explicit non-secret reset/clear semantics, serialize one checked atomic read/write/compare/refresh transaction, refresh only changed live adapters, attempt all affected providers, and report persistence and partial-refresh outcomes truthfully.
- [x] #10 TTSService shutdown uses one retained bounded-close task followed by definitive cleanup joins; sealing rejects synthesis, catalog, and reconfiguration calls, wakes blocked synthesis admissions, guarantees abandoned-response cleanup despite failures, atomically owns late-produced responses, and leaves no waiter blocked after wait_closed completes.
- [x] #11 Refreshed legacy adapters consume the existing nested Higgs settings and OpenAI endpoint/organization settings, while configuration comparison values and credentials remain absent from diagnostics.
- [x] #12 STTS initialization retrieves the application-bound service without constructing or passing compatibility configuration that the accessor intentionally ignores.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Define provider-neutral adapter contracts and idempotent response cleanup using TDD.
2. Implement the sealed exact-ID registry, lazy materialization, operation leases, reconfiguration, retirement, and bounded shutdown.
3. Quarantine the existing class registry and add six provider-scoped legacy adapters with enumerated routing, catalogs, locks, and progress translation.
4. Move TTSService onto the registry while retaining the compatibility byte generator and explicit application binding.
5. Bind construction and teardown to TldwCli, route STTS progress through the service, remove direct manager access, and close the OpenAI key-prefix leak.
6. Run focused and compatibility suites, static/boundary checks, update documentation and task notes, then complete TASK-402.

ADR required: yes
ADR path: backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md
Reason: ADR-023 governs the provider boundary, lifecycle, compatibility bridge, and ordered native-adapter migration.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the application-owned sealed TTSAdapterRegistry and TTSService with exact provider IDs, lazy provider-scoped legacy adapters, cancellation-safe operation leases, targeted retirement, and bounded/definitive shutdown. Final review amendments correct Textual display/value ordering, classify every real settings payload key through one atomic binding table, serialize save/compare/refresh transactions, provide truthful partial-refresh outcomes, and support safe OpenAI endpoint reset plus organization clearing. Live refresh now reaches nested Higgs settings and OpenAI endpoint/organization configuration without logging submitted values. Service shutdown seals every public operation, wakes semaphore admissions, owns late responses, releases leases independently from blocking stream close, attempts abandoned-response cleanup despite registry failures, and reports response failures deterministically in creation order. Verification: 68 changed-path regressions passed; the full TTS/STTS/audio-service/media-reading gate passed 268 tests with 14 optional skips; Ruff check/format, compileall, scoped mypy, boundary grep, ADR/scope audit, and git diff hygiene passed. ADR-023 remains governing; native audio.cpp transport and supervision remain deferred to later ordered tasks.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Automated unit and compatibility tests cover all new registry, bridge, lifecycle, concurrency, cancellation, and privacy behavior.
- [x] #2 Focused static analysis, compilation, and diff hygiene checks pass.
- [x] #3 The TTS module guide, accepted design, implementation plan, and ADR-023 are linked and consistent.
- [x] #4 A self-review confirms no concrete backend or manager access remains outside the legacy bridge.
- [x] #5 All acceptance criteria are checked and implementation notes summarize the completed change before status moves to Done.
<!-- DOD:END -->
