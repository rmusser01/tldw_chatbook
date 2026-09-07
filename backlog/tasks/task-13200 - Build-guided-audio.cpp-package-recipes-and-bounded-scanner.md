---
id: TASK-13200
title: Build guided audio.cpp package recipes and bounded scanner
status: Done
assignee:
  - '@codex'
created_date: '2026-08-09 17:38'
updated_date: '2026-08-09 19:01'
labels:
  - tts
  - audio-cpp
  - backend
  - scanner
dependencies:
  - TASK-3795
references:
  - backlog/decisions/050-audio-cpp-generated-model-setup-ownership.md
documentation:
  - Docs/superpowers/specs/2026-08-09-audio-cpp-guided-model-setup-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the typed guided-setup configuration, exact recipe registry, and bounded local package discovery foundation for the initial Supertonic and PocketTTS packages.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Guided Managed setup has a typed structured configuration whose source, binary, accepted packages, default model, backend preference, safety limits, and dormant values round-trip without changing existing External or user-provided server.json configurations.
- [x] #2 An immutable recipe registry identifies the pinned audio.cpp release, exact family/package variant, supported speech tasks, required files, safe model projection, backend tuples, compatibility state, and recipe revision without admitting arbitrary JSON fields.
- [x] #3 Every pinned Supertonic and PocketTTS package variant is classified by an exact reviewed recipe or an explicit unsupported/blocking result; ambiguous, heuristic-only, unknown-version, and incomplete packages are never silently selected.
- [x] #4 Local discovery scans only user-approved roots off the event loop, obeys finite file/depth/time/result limits, supports cancellation, does not follow symlinks or reparse points, and reports partial and permission outcomes truthfully.
- [x] #5 Discovery deduplicates canonical package identities, preserves all ambiguous candidates for review, and exposes only sanitized names and bounded relative evidence in normal UI/error surfaces.
- [x] #6 The accounting surface lists all 21 pinned families and 67 declared packages with exact support states, while user-facing support claims include only evidenced recipe/platform/backend tuples.
- [x] #7 Pure tests cover configuration round trips, exact positive and negative recipe fixtures, ambiguity, scanner bounds/cancellation/path attacks, and prove this foundation performs no process launch, socket bind, HTTP request, model download, or model-file write.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a frozen full-settings model for Guided, External, and dormant user-JSON values without changing `AudioCppConfig`.
2. Add the sealed release-0.5.1 Supertonic/PocketTTS recipe registry, exact matcher, accepted-projection validator, and 21-family/67-package accounting.
3. Add an explicit-root, no-follow, bounded, cancellable off-loop scanner with sanitized evidence and canonical deduplication.
4. Prove the joined foundation is side-effect free, run focused/static verification, self-review, document results, and close the task.

ADR required: no new ADR.

ADR path: `backlog/decisions/050-audio-cpp-generated-model-setup-ownership.md`

Reason: Direct implementation of ADR-050's approved structured-settings, sealed-recipe, and explicit-root scanner boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the guided audio.cpp foundation without changing the existing runtime `AudioCppConfig` or adding launch, generated-JSON, UI, download, or Model Library behavior.

- Added frozen full-settings and accepted-package models, including dormant External/manual values, bounded compute/safety fields, durable package UUIDs, immutable recipe projections, and defensive JSON round trips.
- Added a sealed release-0.5.1 registry for all 4 Supertonic and 11 PocketTTS packages, exact fail-closed matching, accepted-snapshot validation, tuple-scoped evidence, and complete 21-family/67-package accounting (15 Approved, 52 explicit Open gaps, no unsupported claims).
- Added an explicit-root scanner with finite depth/entry/candidate/result/metadata/time/detail budgets, cancellation/off-loop execution, canonical identities, path-safe evidence, per-candidate permission/partial truth, and symlink/reparse race fencing.
- Added pure positive, negative, mutation-style, privacy, and joined side-effect regressions. Recorded the queued-directory symlink-race lesson in `backlog/docs/lessons-testing-evidence.md`.
- PR review hardened the scanner to validate selected-root syntax before filesystem access, fail closed before file open when no-follow support is unavailable, and reject conflicting same-path recipe validation contracts. Public APIs now carry complete Google-style parameter/result/error documentation.

ADR required: no new ADR. This task directly implements the structured-settings, sealed-recipe, accepted-snapshot, and explicit-root scanner boundaries in `backlog/decisions/050-audio-cpp-generated-model-setup-ownership.md`.

Verification:

- `pytest` foundation set: 96 passed.
- `pytest Tests/TTS` excluding six sandbox-blocked loopback cases: 2,600 passed, 16 skipped, 6 deselected.
- The six deselected real-socket/process cases rerun outside the sandbox: 6 passed.
- Ruff check and format check passed for every changed Python file; compileall and `git diff --cached --check` passed.
<!-- SECTION:NOTES:END -->
