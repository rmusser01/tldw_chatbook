---
id: TASK-931
title: Install verified Parakeet v2 INT8 from Library
status: Done
assignee:
  - '@codex'
created_date: '2026-07-27 15:29'
updated_date: '2026-07-27 15:37'
labels:
  - stt
  - ingestion
  - downloads
  - ui
dependencies: []
references:
  - backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
documentation:
  - Docs/superpowers/specs/2026-07-23-stt-parakeet-onnx-transcribe-cpp-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let a user explicitly install the pinned Parakeet v2 INT8 ONNX bundle from the Library ingestion options and immediately use the verified local folder.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Library audio/video options offer an explicit Parakeet v2 INT8 install action with immutable source revision, CC-BY-4.0 license, download size, and destination shown before confirmation.
- [x] #2 The installer streams the four pinned files into isolated staging, verifies exact size and SHA-256, and atomically publishes only a complete bundle.
- [x] #3 Failures leave no loadable partial bundle, an already-valid bundle is reused without network access, and providers cannot trigger installation.
- [x] #4 Installation runs off the Textual event loop and successful completion fills the local Parakeet model folder for the current batch form.
- [x] #5 Focused downloader and Library UI tests plus a real verified-bundle transcription smoke pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing tests for immutable v2 metadata, verified staged installation, corruption cleanup, valid-bundle reuse, and the Library install event/result path.
2. Implement one curated v2 installer with stdlib HTTP streaming, exact byte/digest checks, free-space preflight, a verification receipt, and atomic publication.
3. Add a Parakeet install button and confirmation modal to the existing Library panel; run the install on a thread worker and populate transcription_model_dir on success.
4. Run focused tests, lint, a local fixture install, and a real ONNX transcription from the installed result.

ADR required: yes
ADR path: backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
Reason: ADR-025 already establishes explicit consent, pinned verified acquisition, staging, and provider-download prohibition. This task implements only the v2 user path and does not create a generic artifact framework.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added one curated Parakeet v2 INT8 installer directly to the Library audio/video options. The consent modal shows the immutable Hugging Face repository/revision, CC-BY-4.0 license, 630.6 MiB download, destination, and verification behavior. The stdlib installer streams the four pinned files into same-filesystem staging, checks free space, exact byte counts and SHA-256, writes a verification receipt, and atomically renames only a complete bundle. Existing verified installs are reused offline; corrupt downloads and invalid existing destinations fail without a loadable partial. The installer is invoked only by the explicit UI action and runs in a Textual thread worker; successful completion selects the installed directory for the current batch.

Verification: 154 affected installer, Library UI, ingestion, and ONNX tests passed; Ruff and git diff --check passed. A real localhost HTTP smoke streamed all 661,191,781 bytes from the pinned bundle, verified and atomically published it to a fresh directory, then transcribed the smoke WAV from that new install in 1.14 seconds with the exact expected result.

ADR: Reused backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md; no generic artifact framework or provider-initiated acquisition was introduced.
<!-- SECTION:NOTES:END -->
