---
id: TASK-13201
title: Generate and supervise guided audio.cpp configurations on POSIX
status: To Do
assignee: []
created_date: '2026-08-09 17:38'
labels:
  - tts
  - audio-cpp
  - backend
  - lifecycle
dependencies:
  - TASK-13200
references:
  - backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md
  - backlog/decisions/050-audio-cpp-generated-model-setup-ownership.md
documentation:
  - Docs/superpowers/specs/2026-08-09-audio-cpp-guided-model-setup-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Project accepted guided settings into immutable generation-local server configuration and run them through the existing managed audio.cpp lifecycle on macOS and Linux.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Saving guided settings remains side-effect free; only a deliberate Test, Start, Restart & Apply, or synthesis may validate launch inputs, allocate a loopback port, create an artifact, contact audio.cpp, or launch a child.
- [ ] #2 A deliberate launch projects the exact accepted settings and recipes into one private immutable generation-local server.json using safe top-level fields, absolute accepted model paths, loopback-only/no-CORS/no-body-log defaults, lazy loading, and one generated endpoint.
- [ ] #3 The existing app-scoped supervisor launches exactly one no-shell audiocpp_server child for all accepted models, publishes only the generated loopback endpoint, and definitively reaps the exact child and removes the exact artifact on failure, replacement, shutdown, and app exit.
- [ ] #4 Saved, applied, process, recipe/file, catalog, and capability generations remain distinct; a live child keeps its accepted snapshot despite source-file changes, and the next deliberate replacement revalidates before mutation.
- [ ] #5 Backend Auto and explicit overrides select only recipe-supported tuples, and fallback occurs only for allowlisted backend-unavailable failures after the failed child and artifact are definitively cleaned up.
- [ ] #6 The native catalog admits only upstream tts and clone speech tasks, preserves typed capabilities including clone-only packages, and rejects ASR, VC, Music, and other task types from the TTS adapter.
- [ ] #7 Launch, validation, probe, catalog, backend, and cleanup failures use stable recovery phases and context-free sanitized errors that expose no executable, config, model, temporary, environment, prompt, credential, or raw upstream detail.
- [ ] #8 Hermetic lifecycle tests and pinned macOS/Linux real-process evidence cover multi-model lazy registration, first synthesis, saved-while-running replacement, crash recovery, exact shutdown, and zero orphan/artifact leakage.
<!-- AC:END -->
