---
id: TASK-32755
title: Match audio video import warnings to the selected STT backend
status: Done
assignee:
  - '@codex'
created_date: '2026-09-17 20:19'
updated_date: '2026-09-17 21:15'
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
- [x] #4 Parakeet setup explains that its Python runtime and model files are installed separately and points to the existing model-download action.
- [x] #5 Invalid persisted transcription providers remain visible as option errors and cannot hide STT warnings or enable import.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md; backlog/decisions/050-external-parakeet-roots-with-managed-vad.md
Reason: Correct stale capability metadata and warning projection, then clarify the existing explicit model-install workflow without changing acquisition or source selection.
1. Reproduce YouTube warnings with a faster-whisper-only installation and selected-provider transitions.
2. Remove retired MLX providers from ingest capability inventory, add transcribe.cpp, and filter captured warnings for both display and consent.
3. Verify URL parsing reaches the selected runner, run targeted state/UI/routing regressions and document findings.
4. Follow-up: distinguish the Parakeet runtime package from separately installed model files beside the optional folder field; document the existing managed install action, verify its tests and rendered guidance, and update the open PR.
5. Qodo follow-up: validate persisted provider values, preserve warnings for invalid selections, block Start and expose an inline repairable selector; verify state and mounted UI regressions.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Audio/video warnings and consent now project captured dependency evidence through the selected supported STT backend; retired MLX entries are removed and transcribe.cpp is included. Existing execution preserves selection and Auto remains faster-whisper under ADR-025. Invalid persisted values retain warnings, block Start and show a repairable selector error through shared validation. Parakeet guidance distinguishes the Python runtime from its separately acquired English model and points to the existing managed installer, which clears the folder override. Final combined validation: 692 passes with 19 independently reproduced dev baseline exclusions, plus 20 Linux reruns. Parakeet setup also passed 151 scoped checks and both localhost acquisition assertions with a stable temporary profile; its full hint was checked in the rendered canvas. All seven preflight guards pass; no added lint diagnostics. No new ADR; ADR-025/ADR-050 unchanged. Details: Docs/superpowers/reviews/2026-09-17-optional-install-stt-verification.md.
<!-- SECTION:NOTES:END -->
