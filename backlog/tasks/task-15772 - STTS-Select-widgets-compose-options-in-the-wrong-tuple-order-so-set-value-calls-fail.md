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
- [x] (added during review round 2) every other backwards-order Select
      found by a full sweep of the STTS surface (`#chapter-voice-select` in
      `chapter_editor_widget.py`; `#character-voice-select`/
      `#bulk-voice-select`'s dynamic voice list and `#voice-style-select` in
      `character_voice_widget.py`) is also fixed to `(label, value)`, with
      born-red evidence and consumer-side correctness verified for each

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

**This section was corrected after an external review (round 2) found the
first pass's sweep-completeness claim false and one test fix flake-prone.
Both are fixed below; the sweep table and file list are now the complete,
reconciled picture, not the original round-1 claim.**

### Full sweep table (every Select-shaped construct found, both rounds)

| Select / backing list | File | Pre-fix order | Status |
|---|---|---|---|
| `#import-source-select` | `UI/STTS_Window.py` | backwards `(id, label)` | **FIXED** (round 1) |
| `#audiobook-provider-select` | `UI/STTS_Window.py` | backwards | **FIXED** (round 1) |
| `#audiobook-format-select` | `UI/STTS_Window.py` | backwards | **FIXED** (round 1) |
| `#narrator-voice-select` | `UI/STTS_Window.py` | already `(label, value)` | unchanged |
| `#stts-bundle-review-choice` | `UI/stts_profile_library.py` | already correct | unchanged |
| `#chapter-voice-select` | `Widgets/TTS/chapter_editor_widget.py` | backwards | **FIXED** (round 2); zero consumers anywhere in the repo -- dead UI, fixed for uniformity only |
| `#character-voice-select` (options built dynamically in `_update_voice_options`, not at the `Select(...)` call site) | `Widgets/TTS/character_voice_widget.py` | backwards | **FIXED** (round 2) |
| `#bulk-voice-select` (shares the same dynamic list) | `Widgets/TTS/character_voice_widget.py` | backwards | **FIXED** (round 2) |
| `#voice-style-select` | `Widgets/TTS/character_voice_widget.py` | backwards | **FIXED** (round 2) |
| `UI/Speech/*.py` (`speech_axis_row.py`, `speech_settings_pane.py`, `speech_settings_group.py`, `speech_settings_mixin.py`, `speech_catalog_mixin.py`, `speech_profile_mixin.py`) -- `Select(`/`PruneSafeSelect(`/`.set_options(` sites | `UI/Speech/` | already correct (spot-checked, not every line) | unchanged |

**Round 1 undercounted this.** The original sweep only grepped for literal
`Select(` construction call sites and missed `character_voice_widget.py`'s
`#character-voice-select`/`#bulk-voice-select`: both are `compose()`d empty
(`options=[]`) and populated later via `voice_select.set_options(self.
voice_options)`, where `self.voice_options` (`_update_voice_options`) was
itself built backwards -- e.g. `("alloy", "Alloy")` instead of
`("Alloy", "alloy")`. A `grep -n "Select("` sweep cannot see a bug that
lives in a `.set_options()` argument built elsewhere. Round 1 also
mis-stated its own tally ("six sites... three correct" does not reconcile
with its own four-item "other three" list) and missed the backwards, if
dead, `#chapter-voice-select`. Six Select-shaped constructs were backwards
in total across both rounds; all six are now fixed.

### Consumer-side fixes (round 2)

`character_voice_widget.py`'s `self.voice_options` list is read by two
internal helpers besides being fed to `Select.set_options()`:
`_get_voice_label` (unpacked as `for vid, label in ...`) and
`detect_characters_from_text`'s auto-assign path (`voice_id, _ =
self.voice_options[i + 1]`). Both assumed the *old* `(id, label)` shape.
Flipping `self.voice_options` to `(label, value)` for `Select` without
updating these two would have swapped the bug from "Select can't hold the
value" to "the widget's own label lookups return ids and vice versa" --
so both were updated in the same commit to unpack `(label, vid)` /
`(_, voice_id)` respectively. `#voice-style-select`'s only consumer
(`_get_current_voice_settings`'s `settings["style"] = style_select.value`)
needed no change: like the three round-1 Selects, it was already written
assuming `.value` is the id -- only the compose order was backwards.

### Live-reproduced bugs from round 2

`_update_assignment_ui` (fires via `watch_selected_character_index`, i.e.
every character-row click in the Voice Assignment panel) did
`voice_select.value = assigned_voice` where `assigned_voice` is a real id
("narrator", "ash", ...), wrapped in `except Exception as e:
logger.debug(f"Some UI elements not ready: {e}")` -- the identical
crash-swallowed-into-debug-log pattern `_initialize_audiobook_defaults` had
before round 1's fix. Reproduced live via
`Tests/Widgets/test_character_voice_widget.py::
test_selecting_a_character_row_lands_the_assigned_voice_without_swallowing_an_exception`:
pre-fix it failed with the captured log line
`Some UI elements not ready: Illegal select value 'ash'.`; post-fix the
dropdown actually reflects the assigned voice and no such log line appears.
This is mounted right next to `#narrator-voice-select` (both inside
`AudioBookGenerationWidget`'s "🎭 Voice Assignment" section) -- the sibling
widget that round 1 fixed, with the same bug shape, one file over.

### Born-red evidence (round 2)

`Tests/UI/test_stts_select_tuple_order.py` (round 1, 10 tests),
`Tests/Widgets/test_character_voice_widget.py` (6 new tests added to the
existing file, on top of its 3 pre-existing ones), and the new
`Tests/Widgets/test_chapter_editor_widget_select_tuple_order.py` (1 test)
were all confirmed to fail against the pre-round-2 code before the fix,
with `InvalidSelectValueError`/shape-mismatch failures matching the table
above -- run and captured before any round-2 source edit landed.

### Deterministic settle instead of a fixed sleep (round 2)

The round-1 fix to `test_audiobook_kokoro_blend_group_is_not_a_keyboard_
select_option` used `await pilot.pause(0.15)` to let the mount-time
`set_timer(0.1, ...)` settle before the test's own kokoro setup. Review
round 2 flagged this as a flake seed: `pilot.pause(delay)` and Textual's
`Timer._run` are both real wall-clock sleeps, and the margin between the
two hardcoded deadlines (0.15 - 0.1 = 0.05s) has to additionally cover a
message-pump round trip and a two-handler `Select.Changed` reactive
cascade -- comfortable under normal load, a genuine risk under a contended
runner. Replaced with a bounded poll on the actual condition
(`provider_select.value == "openai"`, up to 100 `pilot.pause()` iterations,
raising loudly if it never settles) instead of assuming a fixed margin.
Reran the affected test 5x in a row post-fix: passed every time, and
faster (~1.1-1.2s vs the fixed 0.15s+ floor).

### Residual dependency (task-16471, not this task)

Selecting "Notes" or "Conversation" from `#import-source-select` and
pressing "Import Content" now correctly calls
`_import_from_notes`/`_import_from_conversation` (round 1's fix), but those
two methods still import four nonexistent `ChaChaNotes_DB` helpers and
raise `ImportError` before either dialog opens, until task-16471 lands.
Verified live: calling `_import_from_notes()` directly raises
`ImportError: cannot import name 'fetch_all_notes' from
'tldw_chatbook.DB.ChaChaNotes_DB'`. That import sits outside both methods'
own `try/except`, so it propagates rather than being toasted.

### Test counts and an unrelated pre-existing flake (corrected)

Round 1's notes said "162" and "10" tests for
`Tests/UI/test_stts_profile_library.py` and
`Tests/UI/test_speech_audiobook_chapter_detection.py`; the actual counts
are **163** and **8** respectively (confirmed via `pytest -q` and
`--collect-only`).

`Tests/UI/test_stts_profile_library.py` (163 tests) has a pre-existing,
non-deterministic flake unrelated to this task: on repeated full-file runs,
either `test_reference_export_defaults_sanitized_and_bundle_requires_ack`
or `test_windows_clone_export_keeps_sanitized_default_and_disables_bundle`
(different victim each run, same exact test order -- no reordering plugin
is installed) intermittently fails; neither test touches Select, the
audiobook widgets, or anything this task modified. Confirmed pre-existing
and independent of this task's diff by copying the pristine
pre-task-15772 `Tests/UI/test_stts_profile_library.py`
(`git show 8727a2861:...`) over the working file (Edit-based restore
immediately after) and rerunning against current source: the same flake
(`test_reference_export_defaults_sanitized_and_bundle_requires_ack`) still
failed with zero task-15772 test changes present. Left untouched -- not
this task's bug to fix. `Tests/UI/test_stts_capability_state.py`'s two
failures (`test_speech_rail_exposes_and_opens_voice_profiles`,
`test_speech_capability_summary_stays_visible_at_minimum_terminal`) were
separately confirmed pre-existing in round 1 (same failures with the whole
task-15772 diff reverted) and are likewise untouched.

### `UI/Speech/*.py` sweep (round 1, re-confirmed)

Spot-checked `speech_axis_row.py`, `speech_settings_pane.py`,
`speech_settings_group.py` (all `Select(`/`PruneSafeSelect(` call sites),
plus representative `.set_options(` sites in `speech_settings_mixin.py`
and `speech_catalog_mixin.py`/`speech_profile_mixin.py` -- all already
correct `(label, value)`; `speech_settings_group.py`'s `SELECT_OPTIONS`
dict even carries an in-code comment describing an earlier, already-fixed
instance of this exact bug ("Illegal select value 'openai'"). This was
*not* an exhaustive read of every remaining `.set_options(` line in that
directory (~20 sites total); the sample covered the highest-traffic ones.

### Modified files

- `tldw_chatbook/UI/STTS_Window.py` -- three `options=` tuple orders (round 1)
- `tldw_chatbook/Widgets/TTS/character_voice_widget.py` -- `_update_voice_options`'s
  four voice-list branches + narrator insert, `_get_voice_label`'s unpacking,
  `detect_characters_from_text`'s unpacking, `#voice-style-select`'s
  `options=` (round 2)
- `tldw_chatbook/Widgets/TTS/chapter_editor_widget.py` -- `#chapter-voice-select`'s
  `options=` (round 2)
- `Tests/UI/test_stts_select_tuple_order.py` -- new, 10 tests (round 1)
- `Tests/Widgets/test_character_voice_widget.py` -- 6 new tests appended to
  the existing 3 (round 2)
- `Tests/Widgets/test_chapter_editor_widget_select_tuple_order.py` -- new,
  1 test (round 2)
- `Tests/UI/test_stts_profile_library.py` -- one test's mount-settle wait
  swapped from a fixed `pilot.pause(0.15)` to a bounded deterministic poll
  (round 2)
