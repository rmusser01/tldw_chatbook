---
id: TASK-31586
title: >-
  Wire process_audio on ParakeetMLXStreamingTranscriber so the dictation
  streaming regime works
status: To Do
assignee: []
created_date: '2026-09-05 04:40'
labels:
  - audio
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The lazy dictation service requires process_audio; the MLX transcriber only has add_audio, so streaming partials are dead for Console and Meetings (meeting spec §9 item 1).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Console dictation on Apple Silicon shows word-level partials with parakeet-mlx
<!-- AC:END -->
