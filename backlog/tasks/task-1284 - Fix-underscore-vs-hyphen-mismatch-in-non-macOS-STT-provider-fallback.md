---
id: TASK-1284
title: 'config.py non-macOS STT provider fallback uses "faster_whisper" (underscore) instead of the hyphenated provider id'
status: To Do
assignee: []
created_date: '2026-07-28 15:00'
labels: [config, dictation, bug]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`config.py` (currently line 920) initializes `default_stt_provider = "faster_whisper"` (underscore) before the macOS-only branches below it override it with correctly hyphenated ids (`"parakeet-mlx"`, `"lightning-whisper-mlx"`). Every provider id actually used for dispatch elsewhere in the codebase -- `console_voice_input.py`'s `LOCAL_PROVIDER_MODULES` (`"faster-whisper"`), and `transcription_service.py`'s provider-branch matching -- is hyphenated. On a non-macOS platform (where neither `if sys.platform == "darwin"` branch runs), `default_stt_provider` keeps the underscored value and gets written into `STT_settings.default_stt_provider`, which downstream code (`console_voice_input.resolve()`'s `STT_settings` fallback path, `transcription_service` dispatch) does not recognize as any installed provider id -- it fails the "is this the configured provider actually installed" check silently rather than matching.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 `config.py`'s non-macOS `default_stt_provider` fallback is the hyphenated id `"faster-whisper"`, matching every other provider id in `LOCAL_PROVIDER_MODULES` and `transcription_service.py`.
- [ ] #2 On a non-macOS platform with no `[transcription].default_provider` configured, `STT_settings.default_stt_provider` resolves to a value that matches an installed/dispatchable provider id rather than silently falling through to the "not installed" branch.
<!-- AC:END -->
