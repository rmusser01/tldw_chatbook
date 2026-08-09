---
id: TASK-13200
title: Build guided audio.cpp package recipes and bounded scanner
status: To Do
assignee: []
created_date: '2026-08-09 17:38'
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
- [ ] #1 Guided Managed setup has a typed structured configuration whose source, binary, accepted packages, default model, backend preference, safety limits, and dormant values round-trip without changing existing External or user-provided server.json configurations.
- [ ] #2 An immutable recipe registry identifies the pinned audio.cpp release, exact family/package variant, supported speech tasks, required files, safe model projection, backend tuples, compatibility state, and recipe revision without admitting arbitrary JSON fields.
- [ ] #3 Every pinned Supertonic and PocketTTS package variant is classified by an exact reviewed recipe or an explicit unsupported/blocking result; ambiguous, heuristic-only, unknown-version, and incomplete packages are never silently selected.
- [ ] #4 Local discovery scans only user-approved roots off the event loop, obeys finite file/depth/time/result limits, supports cancellation, does not follow symlinks or reparse points, and reports partial and permission outcomes truthfully.
- [ ] #5 Discovery deduplicates canonical package identities, preserves all ambiguous candidates for review, and exposes only sanitized names and bounded relative evidence in normal UI/error surfaces.
- [ ] #6 The accounting surface lists all 21 pinned families and 67 declared packages with exact support states, while user-facing support claims include only evidenced recipe/platform/backend tuples.
- [ ] #7 Pure tests cover configuration round trips, exact positive and negative recipe fixtures, ambiguity, scanner bounds/cancellation/path attacks, and prove this foundation performs no process launch, socket bind, HTTP request, model download, or model-file write.
<!-- AC:END -->
