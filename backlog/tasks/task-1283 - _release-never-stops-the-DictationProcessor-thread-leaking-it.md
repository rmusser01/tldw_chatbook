---
id: TASK-1283
title: '_release() frees the microphone but never stops the DictationProcessor thread, leaking it on every abandon'
status: To Do
assignee: []
created_date: '2026-07-28 15:00'
labels: [console, dictation, bug, resource-leak]
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`ConsoleVoiceInputController._release()` (`tldw_chatbook/Chat/console_voice_input.py`) is the no-join teardown path used by `abandon()` (mid-`preparing`, mid-`listening`, and after a mid-session service error) and by `_run_begin()` when `abandon()` wins a race. It only calls `service._audio_service.stop_recording()` -- it never touches `service.stop_processing`, the `threading.Event` that `LazyLiveDictationService._processing_loop()` polls to know when to exit.

That loop only ever gets told to stop inside `LazyLiveDictationService.stop_dictation()` (`self.stop_processing.set()`, followed by a 2-second `processing_thread.join()`), which is exactly the blocking path `_release()`/`abandon()` exist to skip. So every abandoned or mid-session-failed dictation leaves its `DictationProcessor` daemon thread (`threading.Thread(target=self._processing_loop, daemon=True, name="DictationProcessor")`) running forever, polling `processing_queue.get(timeout=0.1)` in a loop that never terminates. Because the thread's target is the bound method `self._processing_loop`, the thread holds a live reference to the `LazyLiveDictationService` instance, so the whole service object -- and whatever it's still holding -- is kept alive by that thread indefinitely, not just leaked memory but a permanently running background thread per abandoned capture.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Abandoning a Console dictation session (mid-`preparing`, mid-`listening`, or via a mid-session service error) stops the `LazyLiveDictationService`'s `DictationProcessor` daemon thread, not just the audio capture stream.
- [ ] #2 No `DictationProcessor` thread remains alive (`thread.is_alive()` is `False`) after an abandoned or error-terminated dictation session, allowing the service object to be garbage collected.
- [ ] #3 The existing 2-second-join `stop_dictation()` path used by the normal `stop()` flow is unaffected.
<!-- AC:END -->
