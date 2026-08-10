---
id: TASK-13201
title: Generate and supervise guided audio.cpp configurations on POSIX
status: Done
assignee:
  - '@codex'
created_date: '2026-08-09 17:38'
updated_date: '2026-08-10 19:16'
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
- [x] #1 Saving guided settings remains side-effect free; only a deliberate Test, Start, Restart & Apply, or synthesis may validate launch inputs, allocate a loopback port, create an artifact, contact audio.cpp, or launch a child.
- [x] #2 A deliberate launch projects the exact accepted settings and recipes into one private immutable generation-local server.json using safe top-level fields, absolute accepted model paths, loopback-only/no-CORS/no-body-log defaults, lazy loading, and one generated endpoint.
- [x] #3 The existing app-scoped supervisor launches exactly one no-shell audiocpp_server child for all accepted models, publishes only the generated loopback endpoint, and definitively reaps the exact child and removes the exact artifact on failure, replacement, shutdown, and app exit.
- [x] #4 Saved, applied, process, recipe/file, catalog, and capability generations remain distinct; a live child keeps its accepted snapshot despite source-file changes, and the next deliberate replacement revalidates before mutation.
- [x] #5 Backend Auto and explicit overrides select only recipe-supported tuples, and fallback occurs only for allowlisted backend-unavailable failures after the failed child and artifact are definitively cleaned up.
- [x] #6 The native catalog admits only upstream tts and clone speech tasks, preserves typed capabilities including clone-only packages, and rejects ASR, VC, Music, and other task types from the TTS adapter.
- [x] #7 Launch, validation, probe, catalog, backend, and cleanup failures use stable recovery phases and context-free sanitized errors that expose no executable, config, model, temporary, environment, prompt, credential, or raw upstream detail.
- [x] #8 Hermetic lifecycle tests and pinned macOS/Linux real-process evidence cover multi-model lazy registration, first synthesis, saved-while-running replacement, crash recovery, exact shutdown, and zero orphan/artifact leakage.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the full guided Settings projection through provider publication while keeping Save pure, and extend the native catalog contract to exact tts/clone typed capabilities.\n2. Add a POSIX guided launch materializer that revalidates accepted packages, resolves only recipe-supported backends, selects a bounded loopback port, and atomically creates one private immutable server.json.\n3. Feed the generated launch snapshot into the existing AudioCppAdapter and AudioCppSupervisor, with exact artifact cleanup on pre-spawn failure, exit, restart, shutdown, and app close.\n4. Add generation-fenced catalog cross-checking, stable sanitized failures, and no fallback except an explicit allowlisted backend-unavailable classification after cleanup.\n5. Prove concurrent first use, lazy multi-model launch, staged replacement, source-change behavior, crash/shutdown cleanup, and real POSIX child execution; then update docs and close the task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Preserved the complete guided Settings snapshot through provider publication
  while keeping Save side-effect free, and added typed `tts`/`clone` catalog
  evidence at the native adapter boundary.
- Added a POSIX materializer that revalidates explicitly accepted package
  identities, intersects recipe-supported platform/backend tuples, selects a
  private loopback port, and creates one no-follow, owner-only, immutable
  generation-local `server.json`.
- Reused the existing app-scoped `AudioCppSupervisor` as the sole process
  authority. Generated artifacts now follow their exact process generation
  through startup failure, replacement, unexpected exit, shutdown, and close.
- Fenced generated catalogs against the accepted model/family/task/mode set,
  retained lazy multi-model operation, and kept backend fallback disabled in
  the absence of a stable allowlisted backend-unavailable classification.
- Corrected real-package revalidation so an aggregate ambiguous discovery does
  not erase two explicitly accepted exact candidates sharing one package root;
  added the corresponding regression and testing lesson.
- Addressed PR review by applying the centralized arbitrary-path policy to the
  selected executable, completing the public materializer docstring, and
  retaining bounded allowlisted internal-failure diagnostics without raw
  exception text or type names.
- Verification at implementation commit
  `29e4262d9d6a7abe107206bb4ac097e7c06a444e`: 1,109 affected tests passed;
  Ruff check/format, scoped mypy, compileall, privacy/boundary checks, and diff
  integrity passed; pinned audio.cpp 0.5.1 CPU journeys passed on native macOS
  arm64 and provisioned Linux arm64. The final WAV was byte-identical to the
  audibly confirmed retained macOS sample.
- ADR check: no new ADR was required. The implementation follows and extends
  the existing ownership decisions in ADR-023 and ADR-050.
<!-- SECTION:NOTES:END -->
