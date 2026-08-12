---
id: TASK-15478
title: STTS audiobook paste box queries a switch that is never composed
status: Done
assignee:
  - '@claude'
created_date: '2026-08-11 12:05'
labels:
  - bug
  - stts
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the latency audit: `UI/STTS_Window.py:555-565` handles TextArea.Changed by materializing the full text then querying `#auto-chapters-switch` — an id that is composed nowhere in the repo (4 query sites at `:388/:443/:528/:561`, zero compose sites) — so the handler raises NoMatches on every keystroke. If the switch were restored as-is, the design would run `ChapterDetector.detect_chapters` over the entire pasted book plus a notify toast per keystroke (`:630-670`).

Decide: restore the switch with detection moved to Submit or a debounced worker, or remove the dead queries. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Typing in the audiobook paste box raises no exceptions (evidence)
- [x] #2 If chapter detection is kept, it runs off the keystroke path (Submit or debounced worker)
- [x] #3 The chosen behavior is covered by a test
<!-- AC:END -->

## Implementation Plan

**Decision: keep chapter detection, drop the dead switch, debounce the
keystroke path.** Evidence (`git log --all --oneline -S'auto-chapters-switch'`):
the switch WAS composed (`Switch(id="auto-chapters-switch", value=True)`,
commit `74ec9b62b`) and was removed by commit `256911ea6` ("audiobook work",
2025-07-22) when the whole "Chapter Settings" collapsible was replaced by the
`ChapterEditorWidget` (its own pattern input + manual "Detect" button). The
four `query_one("#auto-chapters-switch", ...)` guards were never updated in
that same commit -- an oversight, not a deliberate kill of the auto-detect
feature. `_detect_chapters()` itself (STTS_Window.py:700) is fully wired to
the still-composed `#chapter-editor-widget` and works; only the phantom
switch guard is dead. The switch defaulted to `value=True` and the app now
exposes no UI to turn it off, so the faithful reading of "current intended
behavior" is "detection always runs" for the three one-shot import paths
(file/notes/conversation) -- no keystroke repetition there, so AC #2 is moot
for them. Only `on_text_area_changed` (the paste box) fires per keystroke;
that path keeps detection but moves it behind a debounce timer instead of
running synchronously per keystroke, per AC #2's explicit "debounced worker"
option. Not restoring the switch UI itself avoids re-growing the collapsible
group count that `Tests/UI/test_speech_audiobook_layout.py`'s docstring
documents as deliberately fixed ("the grouping itself is unchanged").

1. `_handle_file_selection`, `_import_from_notes`, `_import_from_conversation`:
   drop the dead `if self.query_one("#auto-chapters-switch", Switch).value:`
   guard, call `self._detect_chapters()` unconditionally (matches the
   pre-refactor default).
2. `on_text_area_changed`: drop the dead guard; queue `_detect_chapters()` on
   a cancel-and-restart `self.set_timer(...)` debounce (same idiom as
   `library_screen.py`'s `_queue_library_prompts_search`), so a burst of
   keystrokes runs detection once, ~1s after the user stops typing, off the
   message-pump path. Stop the pending timer `on_unmount`.
3. Tests in `Tests/UI/test_speech_audiobook_chapter_detection.py`, mounting
   `AudioBookGenerationWidget` directly in a minimal `App` host (pattern from
   `Tests/UI/test_stts_settings_widget.py`'s `_Host`):
   - typing/loading text into `#content-preview` raises no exception (the
     repro of the current bug, born red against the unmodified code);
   - the debounce timer is armed (not an immediate synchronous call) on
     `TextArea.Changed`, and `_detect_chapters` runs once after it elapses,
     not once per keystroke;
   - a one-shot import path (`_handle_file_selection`) still populates
     `detected_chapters` without needing any switch.
4. Run the STTS window suites (`grep -rl STTS_Window Tests/`) plus the new
   file; read the pass counts.

## Implementation Notes

Implemented exactly the plan above: kept auto-detect, deleted the four dead
`query_one("#auto-chapters-switch", ...)` guards, did not restore any switch
UI.

- `_handle_file_selection`, `_import_from_notes`'s `handle_note_selection`,
  `_import_from_conversation`'s `handle_conversation_selection`: the dead
  guard is gone; each now calls `self._detect_chapters()` unconditionally
  after loading content into `#content-preview`. These are one-shot,
  user-triggered paths (file pick / note pick / conversation pick), not
  keystroke-repeated, so AC #2 does not apply to them and no debounce was
  needed.
- `on_text_area_changed` (the paste box): now calls
  `_queue_debounced_chapter_detection()`, which (re)arms a
  `self.set_timer(_CHAPTER_DETECT_DEBOUNCE_SECONDS, ...)` (1.0s) on every
  `TextArea.Changed`, cancelling any prior pending timer first. Detection
  (`ChapterDetector.detect_chapters` over the full text + a notify toast)
  now runs at most once per pause in typing, never synchronously inside the
  message handler. The timer is stopped in a new `on_unmount` to avoid a
  stray callback after the widget is gone.
- Confirmed via `git log --all -S'auto-chapters-switch'` that the switch WAS
  composed once (`74ec9b62b`) and was dropped by `256911ea6`
  ("audiobook work", 2025-07-22) when the "Chapter Settings" collapsible was
  replaced by `ChapterEditorWidget` -- an oversight in that refactor, not a
  deliberate feature removal, since `_detect_chapters()` itself stayed fully
  wired to the still-composed `#chapter-editor-widget`.
- Mutation-tested the regression test: temporarily restored the pre-fix file
  content (via `git show HEAD:...`, since nothing was committed yet) and
  confirmed all three new tests fail red with the exact reported symptom
  (`NoMatches: No nodes match '#auto-chapters-switch'` raised out of
  `on_text_area_changed`), including confirming the three import paths were
  ALSO silently swallowing the same exception through their outer
  `except Exception` blocks (e.g. `_handle_file_selection` showed a false
  "Failed to import file: ..." error toast on an import that had actually
  succeeded). Restored the fix afterward and reconfirmed green.

**Tests**: new file `Tests/UI/test_speech_audiobook_chapter_detection.py`
(3 tests, all born red against the pre-fix code, green after): typing raises
no exception; a burst of keystrokes inside the debounce window calls
`_detect_chapters` zero times until the window elapses, then exactly once;
a one-shot file import still populates `detected_chapters` with no switch.
Also ran the full STTS/Speech suite (`grep -rl STTS_Window Tests/` plus
`test_speech_audiobook_layout.py`, which asserts the collapsible-group
layout is unchanged -- confirming no switch UI was re-added): 339 passed, 0
failed.

**Files changed**:
- `tldw_chatbook/UI/STTS_Window.py`
- `Tests/UI/test_speech_audiobook_chapter_detection.py` (new)
