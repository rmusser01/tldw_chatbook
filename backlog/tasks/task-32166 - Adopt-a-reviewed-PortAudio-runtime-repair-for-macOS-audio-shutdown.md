---
id: TASK-32166
title: Adopt a reviewed PortAudio runtime repair for macOS audio shutdown
status: To Do
assignee:
  - '@codex'
created_date: '2026-09-09 07:22'
updated_date: '2026-09-09 07:23'
labels:
  - audio
  - tts
  - runtime
dependencies:
  - TASK-32165
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The bundled PortAudio library selected by sounddevice 0.5.6 in these macOS test environments contains the CoreAudio lock inversion reproduced during Higgs speech on macOS 26.5.2. Isolated qualification does not update installed user environments. Adopt a reviewed, distributable repair once its runtime ownership and upstream status are resolved.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The upstream repair status and a pinned redistributable release or owned build recipe are reviewed and the runtime/dependency decision is recorded in an ADR.
- [ ] #2 The default supported installation selects an identified repaired PortAudio library without replacing unrelated system libraries or silently changing user settings.
- [ ] #3 Fresh-install macOS ARM Console playback, natural drain, active Stop, successor and process exit pass with actual library hashes and targeted regression evidence.
- [ ] #4 Upgrade instructions and remaining platform support limits are documented without advertising the experimental library as an upstream release.
<!-- AC:END -->
