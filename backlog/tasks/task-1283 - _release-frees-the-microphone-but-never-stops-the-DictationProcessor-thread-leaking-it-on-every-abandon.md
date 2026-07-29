---
id: TASK-1283
title: >-
  _release() frees the microphone but never stops the DictationProcessor thread,
  leaking it on every abandon
status: Done
assignee:
  - '@claude'
created_date: '2026-07-28 15:00'
updated_date: '2026-07-29 14:52'
labels:
  - console
  - dictation
  - bug
  - resource-leak
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
- [x] #1 Abandoning a Console dictation session (mid-`preparing`, mid-`listening`, or via a mid-session service error) stops the `LazyLiveDictationService`'s `DictationProcessor` daemon thread, not just the audio capture stream.
- [x] #2 No `DictationProcessor` thread remains alive (`thread.is_alive()` is `False`) after an abandoned or error-terminated dictation session, allowing the service object to be garbage collected.
- [x] #3 The existing 2-second-join `stop_dictation()` path used by the normal `stop()` flow is unaffected.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce: thread survives abandon\n2. Set stop_processing in _release() (no join — abandon stays non-blocking)\n3. Test: real _processing_loop thread exits after abandon\n4. Mutation-check
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed: _release() now sets service.stop_processing (a threading.Event) via
getattr, defensively, right after the existing audio.stop_recording() call,
in its own try/except so a raise in either step never escapes and never
disturbs the other. No join is added -- abandon()'s whole point is staying
non-blocking; the daemon thread's own `while not self.stop_processing.is_set()`
loop exits on its next 0.1s poll and self-drains/flushes on the way out.

Added test_release_stops_the_real_processing_thread in
Tests/Chat/test_console_voice_input.py: builds a real
LazyLiveDictationService via __new__ (real Event/Queue/locks, no fakes for
the loop itself, following the Tests/Audio/test_dictation_capture_release.py
pattern), starts a real _processing_loop thread, calls controller._release(),
and asserts the thread exits within a 2s join. Mutation-checked: reverting
the stop_processing.set() addition makes the new test fail (thread stays
alive); restored byte-identical afterward.

stop_dictation()'s existing 2s-join path is untouched (AC3); the new
try/except is purely additive in _release().

Verification: Tests/Chat/test_console_voice_input.py (98 tests, incl. the
new one) + Tests/UI/test_console_dictation.py + the Audio dictation_* suites
listed in the task's suggested set, all green (151 passed). ruff check clean
on both changed files.
<!-- SECTION:NOTES:END -->
