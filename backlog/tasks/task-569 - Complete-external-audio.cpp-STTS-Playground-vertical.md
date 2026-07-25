---
id: TASK-569
title: Complete external audio.cpp STTS Playground vertical
status: Done
assignee:
  - '@codex'
created_date: '2026-07-25 13:47'
updated_date: '2026-07-25 18:05'
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
- [x] #1 The STTS provider selector is populated from sealed registry descriptors using canonical provider IDs without materializing every adapter, and all six legacy providers retain their visible behavior.
- [x] #2 Selecting audio_cpp lazily resolves only that provider, performs bounded readiness and model discovery, discards stale provider/configuration/catalog results, and exposes safe unavailable, incompatible, stale, and reconfiguring states.
- [x] #3 The external audio.cpp settings surface validates and persists only the approved external configuration, Save does not connect, changed configuration retires only audio_cpp, and Test Connection plus Refresh Models are explicit actions.
- [x] #4 Audio.cpp model and lazy voice controls use catalog metadata, select a local Server default sentinel that becomes voice=None, render identifiers safely, and choose a valid announced fallback when refreshed metadata removes a selection.
- [x] #5 Audio.cpp forces WAV and speed 1.0 with disabled explanatory controls, while switching to a legacy provider restores that provider's model, voice, format, speed, and provider-specific control state.
- [x] #6 Generate uses an immutable provider-neutral request snapshot through TTSService, never falls back, prevents overlapping generation, and keeps discovery, generation, playback, and save worker ownership independent.
- [x] #7 Successful complete-WAV results retain provider, model, voice, source-text snapshot, operation ID, and actual response metadata so later selector changes cannot relabel playback or saved filenames.
- [x] #8 Stable adapter failures, retryability, and recovery actions produce safe actionable Playground state; cancellation remains cancellation, stale catalogs disable new generation, and existing generated audio remains playable and saveable.
- [x] #9 The Playground communicates that external synthesis sends submitted text to the configured server while UI diagnostics and logs reveal neither synthesis text, configuration values, credentials, origins, nor unsafe remote identifiers.
- [x] #10 Deterministic service-fake and Textual tests cover the external end-to-end flow without an audio.cpp binary, server, model download, or managed binary/server.json UI, and relevant legacy STTS regressions remain green.
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the catalog-driven external audio.cpp STTS Playground vertical through the application-owned TTS service. Added immutable request/artifact contracts, lazy descriptor/catalog/voice resolution, validated external settings and explicit discovery actions, native complete-WAV generation, provenance-safe playback/export, retained handler-owned generation, artifact leasing/cleanup, fixed safe recovery copy, and exact stale provider/configuration/catalog/model rejection. Preserved the six legacy providers behind the temporary bridge, including their original model/voice defaults and friendly labels. Hardened review edges with final-destination export validation, a strict frozen Pydantic audio.cpp configuration model, out-of-band UI sentinels that cannot collide with opaque remote IDs, synchronously reserved catalog/voice request generations, settings action generations, and playback-lifetime artifact leases. Ordinary current voice failures can use Server default while superseded, stale, and registry lifecycle results cannot mutate current state. Updated ADR-023, the approved design, the TTS module guide, and user-facing external-server/privacy guidance; no new ADR was needed.

Verification evidence:
- Focused Slice 3 matrix after the final rebase: 203 passed, 1 existing RequestsDependencyWarning.
- Broad TTS/STTS regressions on the final implementation: 896 passed, 14 expected optional skips, 6 existing dependency/SWIG deprecation warnings.
- Full audio.cpp Playground and pure catalog files: 54 passed.
- Ruff check passed and Ruff format confirmed 17 scoped files formatted.
- compileall passed for TTS, STTS Window/catalog, and STTS event handler.
- Scoped mypy passed for 5 changed typed source modules.
- Managed-mode added-line boundary search returned no matches; git diff --check passed.
- Rebased without conflicts onto latest origin/dev 3d401a1e709576273b7f542c4ea747fc93ce94bf; it is an ancestor of the feature head.
- The final base-only rebase changed unrelated Console/task-623 files; range-diff marked all 18 rewritten commits patch-identical.
- Final independent review at d4ccdaa004b7b53bc47bbc987c2dcdcf69e4121d found no Critical, Important, or Minor issues and declared the code merge-ready for correctness, cancellation ordering, races, lifecycle, security, privacy, compatibility, and integration with the current dev base.

Core files changed include TTS/playground_types.py, TTS/TTS_Generation.py, TTS/audio_cpp_config.py, TTS/legacy_catalogs.py, UI/stts_playground_catalog.py, UI/STTS_Window.py, Event_Handlers/STTS_Events/stts_events.py, focused TTS/UI tests, and the governing/user documentation. The deliberate tradeoff remains external-only, complete-WAV-first delivery through an async-stream-compatible interface; binary/server.json launch and supervision remain deferred.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Automated unit, service-integration, and Textual tests cover the new catalog, settings, worker, generation, playback, save, stale-result, cancellation, error, and privacy behavior.
- [x] #2 Ruff, formatting, compileall, scoped mypy, focused and broad regressions, boundary searches, and diff hygiene pass.
- [x] #3 ADR-023, the approved design, TTS module guide, and user-facing configuration guidance describe the landed external Playground flow and preserve managed-mode deferrals.
- [x] #4 Self-review confirms no binary handling, server.json ownership, process launch, supervision, restart, managed log display, or automatic fallback entered Slice 3.
- [x] #5 Every acceptance criterion is checked and implementation notes record exact verification evidence before the task moves to Done.
<!-- DOD:END -->
