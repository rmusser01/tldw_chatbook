---
id: TASK-15772
title: STTS Select widgets compose options in the wrong tuple order, so set-value calls fail
status: Done
assignee: []
created_date: '2026-08-13 12:31'
labels:
  - bug
  - stts
priority: medium
---

## Description

Found during task-15478's review (input-latency burn-down) for one Select
and confirmed here to be a repeated pattern across `UI/STTS_Window.py`.
Textual's `Select.options` expects `(label, value)` tuples, but multiple
Selects in this file are composed with `(id, label)` — the reverse:

- `#import-source-select` (`options=[("file", "Text File"), ...]`, flagged
  in task-15478's notes) — `_import_content`'s
  `if import_source == "file":` branches check the id, but the widget's
  actual `.value` after selection is the display label ("Text File"), never
  the lowercase id. All four "Import From" dispatch branches
  (file/notes/conversation/paste) are non-functional today via this Select.
- `#audiobook-provider-select` (`options=[("openai", "OpenAI"),
  ("elevenlabs", "ElevenLabs"), ("kokoro", "Kokoro (Local)"),
  ("chatterbox", "Chatterbox (Local)")]`) and `#audiobook-format-select`
  (`options=[("mp3", "MP3"), ...]`) — same reversed shape, confirmed by
  reading the compose call. `_initialize_audiobook_defaults` (STTS_Window.py,
  scheduled via `set_timer(0.1, ...)` on mount) then does
  `provider_select.value = "openai"` / `format_select.value = "m4b"` inside a
  bare `try/except Exception: logger.debug(...)` — since "openai"/"m4b" are
  never present among the Select's actual values (the labels are), Textual's
  illegal-value validation fires and the default-selection attempt silently
  no-ops into the debug log on every STTS window mount.

## Acceptance Criteria

- [x] `#import-source-select`, `#audiobook-provider-select`, and
      `#audiobook-format-select` all compose `options=` in Textual's real
      `(label, value)` order
- [x] Every `.value` comparison and assignment against these three Selects
      (`_import_content`'s branch checks;
      `_initialize_audiobook_defaults`'s provider/format assignment) is
      updated to match, and actually selects the intended default on mount
      (not silently swallowed by the surrounding `try/except`)
- [x] All four "Import From" dispatch branches (file/notes/conversation/paste)
      are reachable and functional through the Select, not just through a
      direct method call (test drives the Select, not `_import_content`
      called directly)
- [x] `_initialize_audiobook_defaults` no longer logs an illegal-select-value
      warning on a fresh STTS window mount (test asserts no such log line)

## Implementation Plan

1. Sweep every `Select(` construction under the STTS surface
   (`UI/STTS_Window.py`, `UI/stts_profile_library.py`,
   `UI/stts_playground_catalog.py`, `UI/Screens/stts_screen.py`,
   `Widgets/TTS/*.py`) and classify each as backwards `(id, label)` or
   correct `(label, value)` by reading the compose call directly -- do not
   trust the task description's enumeration alone.
2. For every backwards Select, grep every consumer of its `.value`
   (comparisons, assignments, dict lookups keyed by the value) across the
   whole repo, not just the same file, to see whether the consumer was
   already written expecting the id (in which case only the compose call
   needs fixing) or was written to match the backwards label (in which case
   the consumer needs a matching fix so the net behavior does not change
   twice).
3. Write born-red tests against current HEAD driving the real widget/Select
   (not calling private dispatch methods directly) that fail showing the
   label-vs-value confusion: (a) compose-time option shape assertions, (b)
   setting the Select's `.value` to the real id raises
   `InvalidSelectValueError` pre-fix, (c) `_initialize_audiobook_defaults`
   swallows an exception into a debug log pre-fix instead of landing the
   default.
4. Fix the tuple order on every backwards Select found in step 1; update any
   consumer identified in step 2 that assumed the backwards shape.
5. For the import-source dispatch test (AC #3), mock/stub at the
   `_import_from_notes`/`_import_from_conversation` method boundary rather
   than exercising the real DB helper imports underneath them --
   task-16471 (filed, not this task) covers those imports being missing
   entirely, so this task's dispatch-layer coverage stays honest about that
   residual dependency instead of pretending to close it.
6. Re-run the born-red tests to confirm green, run ruff check/format on
   touched files, update the task file (ACs, Implementation Notes, status).

## Implementation Notes

Full sweep of every `Select(` construction under the STTS surface
(`UI/STTS_Window.py`, `UI/stts_profile_library.py`,
`UI/stts_playground_catalog.py`, `UI/Screens/stts_screen.py`,
`Widgets/TTS/chapter_editor_widget.py`,
`Widgets/TTS/character_voice_widget.py`) found six `Select(` construction
sites total. Three were backwards exactly as the task described:
`#import-source-select`, `#audiobook-provider-select`,
`#audiobook-format-select` (all in `AudioBookGenerationWidget.compose`,
`UI/STTS_Window.py`). The other three were already correct
`(label, value)`: `#narrator-voice-select` (same compose method, e.g.
`("Alloy", "alloy")`), `#stts-bundle-review-choice`
(`stts_profile_library.py`, built from a `{choice_id: "Display text"}`
dict then swapped to `(display, choice_id)`), and the two dynamic
`Select`s in `Widgets/TTS/chapter_editor_widget.py` /
`character_voice_widget.py`.

Consumer-side sweep (every `.query_one("#import-source-select"/
"#audiobook-provider-select"/"#audiobook-format-select", Select).value`
read, plus the `Select.Changed` handlers and the `costs_per_1k` /
`_get_model_for_provider` id-keyed dict lookups) showed every consumer was
already written assuming `.value` returns the id ("openai", "file", "mp3",
...) -- the backwards compose call was the *only* bug. No consumer code
needed touching; fixing the three `options=` lists made the existing
comparisons correct for the first time.

Born-red evidence (captured before the fix, see commit history): a probe
script confirmed `provider_select.value = "openai"` and
`import_select.value = "file"` both raised
`InvalidSelectValueError` against the pre-fix tuples (the real values were
the labels, e.g. `"OpenAI"`/`"Text File"`). The new test file
`Tests/UI/test_stts_select_tuple_order.py` encodes this as permanent
regression coverage: compose-shape assertions for all three Selects, a
Select-driven (button-press, not direct `_import_content()` call) dispatch
test per import branch with `_import_from_file`/`_import_from_paste`
mocked directly and `_import_from_notes`/`_import_from_conversation`
mocked at the method boundary (task-16471, not this task, owns making
those two dialogs' underlying DB imports real), and a log-capture test
proving `_initialize_audiobook_defaults` no longer emits
"Could not set audiobook provider/format" debug lines and lands both
selects on their intended default value.

Residual dependency: selecting "Notes" or "Conversation" from
`#import-source-select` and pressing "Import Content" now correctly calls
`_import_from_notes`/`_import_from_conversation` (this task's fix), but
those two methods still import four nonexistent `ChaChaNotes_DB` helpers
and will raise `ImportError` before either dialog opens, until task-16471
lands. That import failure happens *outside* the `try/except Exception`
block in both methods (it is a top-of-function import, not inside the
`try:`), so it propagates rather than being toasted -- also task-16471's
concern, not touched here. Verified live (probe script, not just static
reading): calling `_import_from_notes()` directly raises
`ImportError: cannot import name 'fetch_all_notes' from
'tldw_chatbook.DB.ChaChaNotes_DB'`.

Fixing `#audiobook-provider-select` surfaced one genuine regression in an
existing test, not a new bug: `_initialize_audiobook_defaults`'s
`provider_select.value = "openai"` now actually succeeds (previously
silently swallowed), so it fires a real `Select.Changed` on mount that
cascades into `_update_voice_options("openai")`. Before the fix, the STTS
window's on-mount default provider selection was a complete no-op, so
existing tests never had to account for that mount-time cascade.
`Tests/UI/test_stts_profile_library.py::
test_audiobook_kokoro_blend_group_is_not_a_keyboard_select_option` set up
kokoro voice options with a direct `_update_voice_options("kokoro")` call
immediately after mount, then used `pilot.press(...)` (which advances the
clock) to drive keyboard selection -- letting the still-armed 0.1s mount
timer fire in between and overwrite its kokoro setup back to OpenAI's
voice list. Fixed by adding `await pilot.pause(0.15)` before the test's
own setup, so the mount-time default settles first, matching what a real
user experiences (they cannot interact with the widget before it finishes
mounting). Full `Tests/UI/test_stts_profile_library.py` and
`Tests/UI/test_speech_audiobook_chapter_detection.py` reruns (162 and 10
tests respectively) are green with this fix; two failures in
`Tests/UI/test_stts_capability_state.py`
(`test_speech_rail_exposes_and_opens_voice_profiles`,
`test_speech_capability_summary_stays_visible_at_minimum_terminal`) were
confirmed pre-existing at baseline (same failures with this task's diff
reverted) and unrelated to Select tuple order -- left untouched.

Also swept `UI/Speech/*.py` (imported directly by `STTS_Window.py`:
`speech_effects_pane`, `speech_playground_pane`, `speech_profile_mixin`,
etc.) for the same bug class, since the task's "and its modules" scope
covers anything STTS_Window pulls in. Every `Select(`/`PruneSafeSelect(`/
`.set_options(` call site there already uses correct `(label, value)`
tuples; `speech_settings_group.py`'s `SELECT_OPTIONS` dict even carries an
in-code comment describing an earlier, already-fixed instance of this
exact bug ("Illegal select value 'openai'"). Nothing further to fix there.

Modified files: `tldw_chatbook/UI/STTS_Window.py` (three `options=` tuple
orders); `Tests/UI/test_stts_select_tuple_order.py` (new, 10 tests);
`Tests/UI/test_stts_profile_library.py` (one test given a settle-pause to
stay correct against the newly-working mount-time default).
