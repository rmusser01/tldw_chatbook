---
id: TASK-561
title: Establish TTS adapter registry authority and legacy bridge
status: Done
assignee:
  - '@codex'
created_date: '2026-07-24 00:42'
updated_date: '2026-07-25 01:27'
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
- [x] #9 Saving provider-affecting STTS settings reloads the effective configuration and reconfigures only the affected materialized provider adapters without restarting the application.
- [x] #10 TTSService shutdown wakes and rejects blocked synthesis admissions, closes abandoned service-wrapped responses after the bounded drain, and leaves no synthesis waiter blocked after wait_closed completes.
- [x] #11 STTS settings emit canonical Textual select values, support explicit OpenAI reset/clear semantics, and defer success or failure notification to the persistence handler.
- [x] #12 Refreshed legacy adapters consume nested Kokoro and Higgs settings plus validated OpenAI endpoint and organization settings without exposing submitted values.
- [x] #13 STTS initialization retrieves the application-bound service without constructing or passing compatibility configuration that the accessor intentionally ignores.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing regression coverage for actionable PR review findings: missing legacy backend symbols and cleanup-failure detail under cancellation.
2. Add failing STTS settings-save coverage for atomic persistence, exact UI keys, provider-scoped effective configuration, raw/normalized/environment precedence, and targeted adapter retirement.
3. Add failing TTSService and legacy-host shutdown coverage for admission sealing, abandoned responses, one absolute deadline, uncooperative finalizers, post-seal cleanup observation, and cancellation precedence.
4. Implement Google-style docstrings, narrow AttributeError handling, and sanitized cleanup callback behavior.
5. Implement atomic settings persistence and targeted provider reconfiguration through the bound TTSService.
6. Implement service-level close signaling, tracked response cleanup, forceable lease/semaphore release, and a single propagated shutdown deadline.
7. Preserve the retained async STTS event-task design and resolve the worker-dispatch finding as a false positive because the hook already dispatches same-loop async work with explicit cleanup ownership.
8. Run focused and broad local tests, static checks, formatting, boundary checks, ADR/scope audit, and diff hygiene; update task evidence only from fresh results.
9. Reply to and resolve every PR review thread, push the corrected source branch, ignore GitHub CI status per the user's explicit instruction, and merge PR #833.

ADR required: yes
ADR path: backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md
Reason: ADR-023 governs the service lifecycle, provider-scoped reconfiguration, compatibility bridge, and single-deadline shutdown behavior; its update clarifies an existing decision, so no new ADR is required.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the app-owned sealed TTSAdapterRegistry and application-bound TTSService with exact provider IDs, cancellation-safe leases, targeted reconfiguration, and a single shutdown deadline. Preserved all six existing providers behind isolated legacy adapters with enumerated routing, per-backend locking, bounded cleanup, and value-free diagnostics. Reconciled the rebased PR review fixes by restoring canonical Textual Select values, complete settings-key persistence for OpenAI, Chatterbox, Higgs, and AllTalk, explicit OpenAI endpoint reset and organization clearing, validated endpoint/header configuration, serialized concurrent saves, and service retrieval without ignored compatibility configuration. Kept playground generation in its already-retained same-loop event task and documented why a nested worker would break ownership. Fresh post-format verification: 301 passed and 14 optional skips across TTS, STTS UI, audio-service, and media-reading regressions; Ruff check/format, compileall, scoped mypy, boundary isolation, ADR/scope audit, and git diff hygiene passed. ADR-023 remains governing. Native audio.cpp transport and process supervision remain intentionally deferred to later slices.

After merge, this task was renumbered from TASK-402 to TASK-561 because the
latest `dev` already contained an unrelated TASK-402. The filename,
frontmatter, ADR, design, Slice 1 plan, and Slice 2 dependency were updated
without changing the shipped implementation.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Automated unit and compatibility tests cover all new registry, bridge, lifecycle, concurrency, cancellation, and privacy behavior.
- [x] #2 Focused static analysis, compilation, and diff hygiene checks pass.
- [x] #3 The TTS module guide, accepted design, implementation plan, and ADR-023 are linked and consistent.
- [x] #4 A self-review confirms no concrete backend or manager access remains outside the legacy bridge.
- [x] #5 All acceptance criteria are checked and implementation notes summarize the completed change before status moves to Done.
<!-- DOD:END -->
