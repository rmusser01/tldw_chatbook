---
id: TASK-1282
title: 'Fix "lightning-whisper" provider id in the two legacy Dictation Window dropdowns'
status: To Do
assignee: []
created_date: '2026-07-28 15:00'
labels: [dictation, bug]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`tldw_chatbook/UI/Dictation_Window_Improved.py` (the two provider `Select` option lists, currently at lines 351 and 359) and `tldw_chatbook/UI/Dictation_Window.py` (currently line 227) both offer `("Lightning Whisper", "lightning-whisper")` as a dropdown option. Every place that actually dispatches on a provider id -- `transcription_service.py`, and `console_voice_input.py`'s `LOCAL_PROVIDER_MODULES` -- uses `"lightning-whisper-mlx"` instead. The Console dictation service allowlist had this exact id bug and was already fixed there; these two legacy windows still carry it.

`Dictation_Window_Improved.py`'s `ImprovedDictationWindow` is live: `STTS_Window.py` imports and mounts it as the Dictation tab, so selecting "Lightning Whisper" there currently sends an id nothing recognizes. `Dictation_Window.py` (the non-"Improved" version) has no production importer today, but carries the identical bug and is worth fixing at the same time so it isn't resurrected wrong.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 `Dictation_Window_Improved.py`'s two provider dropdown option lists use the id `"lightning-whisper-mlx"`, matching what `transcription_service.py` and `console_voice_input.py`'s `LOCAL_PROVIDER_MODULES` actually dispatch on.
- [ ] #2 `Dictation_Window.py`'s provider dropdown option list uses the same corrected id.
- [ ] #3 Selecting "Lightning Whisper" in the STT/TTS settings Dictation tab (`STTS_Window.py`) results in the `lightning-whisper-mlx` provider actually being invoked, not a silently ignored/unmatched value.
<!-- AC:END -->
