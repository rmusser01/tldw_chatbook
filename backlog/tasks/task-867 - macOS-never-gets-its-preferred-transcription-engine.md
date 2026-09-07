---
id: TASK-867
title: macOS never gets its preferred transcription engine
status: Done
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
- [x] #1 A standard macOS install of the audio or video extra provides the engine the app prefers
- [x] #2 Audio ingest succeeds on macOS without a second manual install step
- [x] #3 Non-macOS installs are unaffected and still use faster-whisper
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The packaging half had already shipped under TASK-839 (parakeet-mlx in the `audio`/`video` extras on
darwin, faster-whisper as the cross-platform fallback) -- verified, not redone. The remaining half was
runtime: `config.py`'s `CONFIG_TOML_CONTENT` template hardcoded `default_provider = "faster-whisper"`
into every generated config, directly beneath a comment claiming macOS defaults to parakeet -- so the
platform preference in `load_settings` could never engage, and a real macOS user with parakeet-mlx
installed had dictation resolve to whisper. Extracted `_default_stt_provider_for_platform()` and
interpolate it into the template at import time (approach (a): zero reader changes, smallest blast
radius). Existing configs are respected and never rewritten; non-darwin unchanged; an uninstalled
preferred engine still falls back through `installed_local_providers()`. False comment corrected.
7 new tests in `Tests/test_config_stt_provider_probe.py`; RED verified by reverting config.py
(reproduces the live symptom `provider: 'faster-whisper'`). Commit 02e1e9b05.
<!-- SECTION:NOTES:END -->
