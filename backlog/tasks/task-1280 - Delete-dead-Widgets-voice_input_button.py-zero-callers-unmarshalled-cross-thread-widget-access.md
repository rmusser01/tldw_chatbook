---
id: TASK-1280
title: >-
  Delete dead Widgets/voice_input_button.py (zero callers, unmarshalled
  cross-thread widget access)
status: Done
assignee:
  - '@claude'
created_date: '2026-07-28 15:00'
updated_date: '2026-07-29 15:07'
labels:
  - console
  - dictation
  - cleanup
  - tech-debt
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
- [x] #1 `tldw_chatbook/Widgets/voice_input_button.py` is removed from the repository.
- [x] #2 No import references `tldw_chatbook.Widgets.voice_input_button`, `VoiceInputButton`, or `FloatingVoiceInput` anywhere in the codebase.
- [x] #3 Any macOS microphone-permission remedy copy still relied on by other dictation surfaces (e.g. `Audio/dictation_service_lazy.py`, `Widgets/audio_troubleshooting_dialog.py`, `Chat/console_voice_input.py`) is preserved and unaffected by the removal.
- [x] #4 The full test suite collects with no failures attributable to the removal.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-verify zero callers repo-wide\n2. Preserve macOS permission copy if uniquely held\n3. Delete file + any dedicated tests\n4. Import sweep + targeted suites green
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Re-verified zero production callers repo-wide before touching anything:
`grep -rn "voice_input_button\|VoiceInputButton\|FloatingVoiceInput" tldw_chatbook Tests --include="*.py"`
matched only the file itself (its own class definitions/CSS selectors) --
no importer anywhere in tldw_chatbook or Tests, and no dedicated test file
existed for it either. Only design docs under Docs/ still mention it.

Checked the macOS microphone-permission remedy copy before deleting:
voice_input_button.py itself carries NONE -- no "permission"/"microphone
access"/"System Settings" text anywhere in the file; it only ever surfaces
whatever message start_dictation()/stop_dictation() hand back via
on_error/_set_status. The actual remedy copy ("Open System Settings >
Privacy & Security > Microphone...") lives in Audio/dictation_service_lazy.py
(two occurrences, verified present and untouched) and is what
Widgets/voice_input_widget.py already relies on the same way -- it also
carries no copy of its own. So there was nothing uniquely held by the
deleted file to preserve; the removal loses no remedy text.

Deleted tldw_chatbook/Widgets/voice_input_button.py (VoiceInputButton +
FloatingVoiceInput, including the latent cross-thread widget-mutation bug
described in the task: _on_partial/_on_error calling self._set_status()'s
self.query_one() directly from LazyLiveDictationService's foreign
recognizer thread, with no call_from_thread/post_message marshalling). No
dedicated test file existed to delete alongside it.

Post-deletion import sweep: `grep -rn "voice_input_button" tldw_chatbook
Tests --include="*.py"` returns nothing. `python -m compileall tldw_chatbook
-q` succeeds (exit 0; only pre-existing unrelated SyntaxWarnings in
Utils/Splash_Screens/environmental/train_journey.py). Did not run the full
Tests/UI or Tests/Chat suites per this session's standing constraints
(whole-directory runs there take ~2h); instead ran
`pytest --collect-only` over Tests/Widgets/ (253 tests) and Tests/Audio/
(excluding the two hardware-hang files; 128 tests) -- both collect cleanly
with zero errors, and the repo-wide grep already rules out any reference
(including a dynamic/importlib-string one) anywhere under Tests/, so
nothing under Tests/UI or Tests/Chat can be referencing the deleted module
either. Tests/Audio/test_voice_input_widget.py (the OTHER widget) still
passes in full: 21/21.
<!-- SECTION:NOTES:END -->
