---
id: TASK-569
title: Complete external audio.cpp STTS Playground vertical
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-25 13:47'
updated_date: '2026-07-25 13:54'
labels:
  - tts
  - audio-cpp
  - stts
  - ui
dependencies:
  - TASK-560
references:
  - backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md
documentation:
  - Docs/superpowers/specs/2026-07-23-audio-cpp-tts-adapter-registry-design.md
  - Docs/superpowers/plans/2026-07-25-audio-cpp-external-stts-playground.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the STTS Playground catalog-driven through the application-owned TTS service so users can configure one external audiocpp_server, discover TTS models and voices, generate a validated complete WAV, and play or save the result without managed process behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The STTS provider selector is populated from sealed registry descriptors using canonical provider IDs without materializing every adapter, and all six legacy providers retain their visible behavior.
- [ ] #2 Selecting audio_cpp lazily resolves only that provider, performs bounded readiness and model discovery, discards stale provider/configuration/catalog results, and exposes safe unavailable, incompatible, stale, and reconfiguring states.
- [ ] #3 The external audio.cpp settings surface validates and persists only the approved external configuration, Save does not connect, changed configuration retires only audio_cpp, and Test Connection plus Refresh Models are explicit actions.
- [ ] #4 Audio.cpp model and lazy voice controls use catalog metadata, select a local Server default sentinel that becomes voice=None, render identifiers safely, and choose a valid announced fallback when refreshed metadata removes a selection.
- [ ] #5 Audio.cpp forces WAV and speed 1.0 with disabled explanatory controls, while switching to a legacy provider restores that provider's model, voice, format, speed, and provider-specific control state.
- [ ] #6 Generate uses an immutable provider-neutral request snapshot through TTSService, never falls back, prevents overlapping generation, and keeps discovery, generation, playback, and save worker ownership independent.
- [ ] #7 Successful complete-WAV results retain provider, model, voice, source-text snapshot, operation ID, and actual response metadata so later selector changes cannot relabel playback or saved filenames.
- [ ] #8 Stable adapter failures, retryability, and recovery actions produce safe actionable Playground state; cancellation remains cancellation, stale catalogs disable new generation, and existing generated audio remains playable and saveable.
- [ ] #9 The Playground communicates that external synthesis sends submitted text to the configured server while UI diagnostics and logs reveal neither synthesis text, configuration values, credentials, origins, nor unsafe remote identifiers.
- [ ] #10 Deterministic service-fake and Textual tests cover the external end-to-end flow without an audio.cpp binary, server, model download, or managed binary/server.json UI, and relevant legacy STTS regressions remain green.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add immutable request/artifact contracts and read-only service descriptor/configuration-revision APIs.
2. Add pure catalog selection, Server-default, restriction, fallback, and stale-token state with tests.
3. Add validated external audio.cpp settings persistence/reconfiguration plus explicit Test Connection and Refresh Models actions.
4. Make STTS Playground controls descriptor/catalog-driven with safe labels, lazy independent workers, audio.cpp restrictions, and legacy-state restoration.
5. Route audio.cpp generation through native TTSService.synthesize(), retain immutable artifact provenance, and keep legacy generation on the temporary bridge.
6. Harden worker ownership, cancellation, stale-result rejection, safe recovery, playback/export, cleanup, and privacy behavior.
7. Update ADR-023, the approved design, module guide, and user documentation while preserving managed-mode deferrals.
8. Run focused and broad tests, Ruff/format/compile/mypy/boundary/diff checks, self-review, record evidence, and finish TASK-569.

ADR required: yes
ADR path: backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md
Reason: ADR-023 already governs the adapter registry, catalog-driven Playground, external privacy boundary, complete-WAV contract, no-fallback policy, lifecycle, and ordered delivery slices; Slice 3 implements that accepted decision, so no new ADR is required.
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Automated unit, service-integration, and Textual tests cover the new catalog, settings, worker, generation, playback, save, stale-result, cancellation, error, and privacy behavior.
- [ ] #2 Ruff, formatting, compileall, scoped mypy, focused and broad regressions, boundary searches, and diff hygiene pass.
- [ ] #3 ADR-023, the approved design, TTS module guide, and user-facing configuration guidance describe the landed external Playground flow and preserve managed-mode deferrals.
- [ ] #4 Self-review confirms no binary handling, server.json ownership, process launch, supervision, restart, managed log display, or automatic fallback entered Slice 3.
- [ ] #5 Every acceptance criterion is checked and implementation notes record exact verification evidence before the task moves to Done.
<!-- DOD:END -->
