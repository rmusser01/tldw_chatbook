---
id: TASK-402
title: Establish TTS adapter registry authority and legacy bridge
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-24 00:42'
updated_date: '2026-07-24 01:05'
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
- [ ] #1 Application code owns one TTSService and one sealed TTSAdapterRegistry; the compatibility accessor returns only the bound service and can be explicitly reset.
- [ ] #2 The registry uses exact canonical provider IDs with an empty initial alias map, lazily materializes at most one adapter per provider under concurrency, and rejects duplicate or post-seal registration.
- [ ] #3 Operation leases keep adapter resources alive through complete or partial response consumption; identical configuration is a no-op, changed configuration retires only the selected provider, and shutdown is ordered, bounded, and idempotent.
- [ ] #4 OpenAI, ElevenLabs, Kokoro, Chatterbox, Higgs, and AllTalk remain available through isolated provider-scoped legacy hosts without exposing TTSBackendManager or concrete backends outside the bridge.
- [ ] #5 The enumerated legacy resolver covers every internal-model form used by current callers, and the existing generate_audio_stream signature routes through the registry and closes its response on success, failure, cancellation, and partial consumption.
- [ ] #6 Per-internal-backend legacy locks serialize construction, initialization, progress callback installation, stream consumption, and callback clearing; progress-sink failures do not fail synthesis while different providers may operate concurrently.
- [ ] #7 Focused registry, bridge, application-binding, lifecycle, concurrency, and compatibility tests pass without changing visible STTS behavior.
- [ ] #8 New registry and bridge diagnostics log neither configuration values nor synthesis text, and regression coverage removes the existing OpenAI API-key-prefix disclosure.
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

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Automated unit and compatibility tests cover all new registry, bridge, lifecycle, concurrency, cancellation, and privacy behavior.
- [ ] #2 Focused static analysis, compilation, and diff hygiene checks pass.
- [ ] #3 The TTS module guide, accepted design, implementation plan, and ADR-023 are linked and consistent.
- [ ] #4 A self-review confirms no concrete backend or manager access remains outside the legacy bridge.
- [ ] #5 All acceptance criteria are checked and implementation notes summarize the completed change before status moves to Done.
<!-- DOD:END -->
