---
id: TASK-1280
title: 'Delete dead Widgets/voice_input_button.py (zero callers, unmarshalled cross-thread widget access)'
status: To Do
assignee: []
created_date: '2026-07-28 15:00'
labels: [console, dictation, cleanup, tech-debt]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Console dictation streaming upgrade (`feat/console-voice-dictation`) replaced this widget's role with the `console_voice_input` controller wired directly into `ChatScreen`. `VoiceInputButton` and `FloatingVoiceInput` in `tldw_chatbook/Widgets/voice_input_button.py` now have zero production callers: nothing outside the file itself imports either class (verified with a repo-wide grep excluding `__pycache__`; only design docs still mention it).

Beyond being dead weight, the widget is also unsafe as written. `LazyLiveDictationService.start_dictation()` invokes its `on_partial_transcript`/`on_error` callbacks from its own recognizer/processing thread, not from the `run_worker()` call that starts it. `VoiceInputButton._on_partial()` and `_on_error()` call `self._set_status()`, which does `self.query_one("#voice-status", Static)` and mutates widget state directly from that foreign thread, with no `call_from_thread`/`post_message` marshalling. Deleting the dead file removes that latent crash risk along with the maintenance burden of an unused, semi-duplicate dictation UI.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 `tldw_chatbook/Widgets/voice_input_button.py` is removed from the repository.
- [ ] #2 No import references `tldw_chatbook.Widgets.voice_input_button`, `VoiceInputButton`, or `FloatingVoiceInput` anywhere in the codebase.
- [ ] #3 Any macOS microphone-permission remedy copy still relied on by other dictation surfaces (e.g. `Audio/dictation_service_lazy.py`, `Widgets/audio_troubleshooting_dialog.py`, `Chat/console_voice_input.py`) is preserved and unaffected by the removal.
- [ ] #4 The full test suite collects with no failures attributable to the removal.
<!-- AC:END -->
