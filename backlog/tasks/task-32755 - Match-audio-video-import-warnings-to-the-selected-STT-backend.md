---
id: TASK-32755
title: Match audio video import warnings to the selected STT backend
status: Done
assignee:
  - '@codex'
created_date: '2026-09-17 20:19'
updated_date: '2026-09-17 20:45'
labels:
  - bug
  - ingestion
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Audio/video dependency warnings list mutually exclusive STT backends including retired MLX packages. This makes a valid Fedora YouTube import look blocked despite its installed selected backend.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 YouTube imports never advertise retired MLX transcription providers on Fedora or other platforms.
- [x] #2 Import warnings and start consent name only the selected supported STT backend while preserving genuine missing dependencies.
- [x] #3 The selected backend reaches transcription execution and focused regression tests pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
Reason: Correct stale capability metadata and warning projection under the existing supported provider and no-silent-fallback policy.
1. Reproduce YouTube warnings with a faster-whisper-only installation and selected-provider transitions.
2. Remove retired MLX providers from ingest capability inventory, add transcribe.cpp, and filter captured missing-backend warnings by the actual selected provider for both display and consent.
3. Verify URL parsing reaches the selected transcription runner, run targeted state/UI/routing regressions, document findings and open PR against dev.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Removed retired MLX providers from audio/video capability inventory, added transcribe.cpp metadata and projected warnings through the selected STT before both display and consent. Existing URL execution already preserves selection; Auto stays faster-whisper under ADR-025. Added 8 warning/state and 4 real media-pipeline routing regressions and updated retired display fixtures plus user guidance. Targeted validation has 657 distinct local passes and 13 Linux reruns; 19 failures reproduced on untouched dev are recorded in Docs/superpowers/reviews/2026-09-17-optional-install-stt-verification.md. No new ADR required.
<!-- SECTION:NOTES:END -->
