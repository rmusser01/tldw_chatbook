---
id: TASK-402
title: Establish TTS adapter registry authority and legacy bridge
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-24 00:42'
updated_date: '2026-07-24 14:10'
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
- [ ] #3 Operation leases keep adapter resources alive through complete or partial response consumption; identical configuration is a no-op, changed configuration retires only the selected provider, and shutdown is ordered, bounded, and idempotent.
- [x] #4 OpenAI, ElevenLabs, Kokoro, Chatterbox, Higgs, and AllTalk remain available through isolated provider-scoped legacy hosts without exposing TTSBackendManager or concrete backends outside the bridge.
- [x] #5 The enumerated legacy resolver covers every internal-model form used by current callers, and the existing generate_audio_stream signature routes through the registry and closes its response on success, failure, cancellation, and partial consumption.
- [x] #6 Per-internal-backend legacy locks serialize construction, initialization, progress callback installation, stream consumption, and callback clearing; progress-sink failures do not fail synthesis while different providers may operate concurrently.
- [ ] #7 Focused registry, bridge, application-binding, lifecycle, concurrency, and compatibility tests pass without changing visible STTS behavior.
- [x] #8 New registry and bridge diagnostics log neither configuration values nor synthesis text, and regression coverage removes the existing OpenAI API-key-prefix disclosure.
- [ ] #9 Saving provider-affecting STTS settings reloads the effective configuration and reconfigures only the affected materialized provider adapters without restarting the application.
- [ ] #10 TTSService shutdown wakes and rejects blocked synthesis admissions, closes abandoned service-wrapped responses after the bounded drain, and leaves no synthesis waiter blocked after wait_closed completes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing regression coverage for the actionable PR review findings: missing legacy backend symbols and cleanup-failure detail preserved under cancellation.
2. Add failing STTS settings-save coverage proving recognized provider keys reload effective config once, reconfigure only candidate providers, leave defaults alone, and do not materialize unrelated adapters.
3. Add failing TTSService shutdown coverage proving admission seals, semaphore waiters fail closed, abandoned responses close after the registry drain deadline, and wait_closed leaves no blocked waiter.
4. Implement the minimal Google-style docstrings, narrow AttributeError handling, and cleanup callback contract needed for the review findings.
5. Implement targeted provider reconfiguration using the existing legacy config snapshot and bound TTSService.
6. Implement service-level close signaling and tracked response cleanup around the existing registry shutdown deadline.
7. Preserve the retained async STTS event-task design; document and resolve the worker-dispatch review finding as a false positive because the hook already dispatches same-loop async work with explicit cleanup ownership.
8. Run focused and broad tests, static checks, formatting, boundary checks, and diff hygiene; update implementation notes and acceptance criteria only from fresh evidence.
9. Reply to and resolve every current review thread, address stale summary findings without unrelated code changes, push the branch, wait for required GitHub checks, and merge PR #833.

ADR required: yes
ADR path: backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md
Reason: ADR-023 already governs the service lifecycle, provider reconfiguration, compatibility bridge, and shutdown behavior; no new ADR is required.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the app-owned sealed TTSAdapterRegistry and TTSService binding with exact canonical provider IDs, cancellation-safe operation leases, targeted retirement, and bounded definitive shutdown. Added six provider-scoped legacy adapters with enumerated routing, isolated managers, per-backend locking, operation-scoped progress, and stream-lifetime shutdown handles that drain, cancel, join, close abandoned partial responses, and preserve real cleanup errors without stale timeouts. STTS now uses the owned service, owns temporary/task cleanup, and logs settings and initialization outcomes without credential values or raw exception text. Published the provider-neutral API boundary and updated the TTS guide. Final verification: 129 focused tests passed; 241 broad regressions passed with 14 optional skips; compileall, scoped mypy, boundary grep, ADR/scope audit, diff hygiene, and Ruff check/format on 19 changed Python files passed. Independent correctness, quality, and whole-branch reviews found no remaining Critical or Important issues. ADR-023 remains governing; native audio.cpp transport and supervision remain deferred to later ordered tasks.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Automated unit and compatibility tests cover all new registry, bridge, lifecycle, concurrency, cancellation, and privacy behavior.
- [x] #2 Focused static analysis, compilation, and diff hygiene checks pass.
- [x] #3 The TTS module guide, accepted design, implementation plan, and ADR-023 are linked and consistent.
- [x] #4 A self-review confirms no concrete backend or manager access remains outside the legacy bridge.
- [ ] #5 All acceptance criteria are checked and implementation notes summarize the completed change before status moves to Done.
<!-- DOD:END -->
