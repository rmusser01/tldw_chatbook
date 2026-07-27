---
id: TASK-867
title: macOS never gets its preferred transcription engine
status: To Do
assignee: []
created_date: '2026-07-27 01:43'
labels:
  - audio
  - packaging
  - macos
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On macOS the app prefers parakeet-mlx as its speech-to-text provider and falls back to faster-whisper only when it is absent, but nothing installs parakeet-mlx: the audio and video extras list faster-whisper and mention the Apple Silicon engines only in a comment suggesting a second manual install. The preference can therefore never engage on a normal macOS install. Found while verifying audio ingest end to end, which failed with 'faster-whisper is not installed' on a machine whose audio dependencies were otherwise complete.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A standard macOS install of the audio or video extra provides the engine the app prefers
- [ ] #2 Audio ingest succeeds on macOS without a second manual install step
- [ ] #3 Non-macOS installs are unaffected and still use faster-whisper
<!-- AC:END -->
