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
  `on_text_area_changed`). Restored the fix afterward and reconfirmed green.
- **Correction (review round 2):** the original version of this note claimed
  all three pre-fix import paths "silently swallowed" the same `NoMatches`
  into a false toast. That is only true for `_handle_file_selection` --
  verified per-path severity below.

**Per-path pre-fix failure severity** (corrected; the code fix itself did
not change based on this -- all four dead queries were removed
unconditionally either way):

- `_handle_file_selection`: its own `try/except Exception` wraps the entire
  callback body, so the dead-switch `NoMatches` WAS caught there, producing
  a false `"Failed to import file: ..."` error toast on an import that had
  actually already succeeded. Confirmed live in the mutation-test run above
  (captured log: `ERROR ... Failed to import file: No nodes match
  '#auto-chapters-switch' ...`).
- `_import_from_notes`'s `handle_note_selection` and
  `_import_from_conversation`'s `handle_conversation_selection`: **not** a
  toast. Both are passed as the `callback` to `self.app.push_screen(dialog,
  callback)`. Textual delivers a screen-dismiss callback via
  `ResultCallback.__call__` -> `self.requester.call_next(self.callback,
  result)` (`textual/screen.py`), i.e. as a `Callback` message processed on
  a LATER message-pump cycle of the widget -- by which point the `try`
  block that lexically surrounds the `push_screen()` call has already
  returned and is no longer on the stack. Neither nested callback has a
  `try/except` of its own. Verified with a minimal standalone repro
  (`push_screen(dialog, callback)` where `callback` raises `NoMatches`,
  no per-callback try/except, matching this file's exact shape): the outer
  `try` printed as having "completed normally" (never entered its
  `except`), and the `NoMatches` instead surfaced through
  `message_pump._flush_next_callbacks` all the way out as an unhandled
  exception -- a crash-class error, not a caught-and-converted toast.

## Review follow-up (round 2)

Two Important findings from review, both addressed without changing the
chosen resolution (keep detection, no switch UI):

1. **Per-fire cost still on the loop.** `_detect_chapters()`'s CPU-bound
   part (`ChapterDetector.detect_chapters`, O(len(content)) regex scanning)
   still ran synchronously wherever it was called from -- including the
   debounce timer's callback, which runs on the event loop. Benchmarked
   locally: ~30ms/90k words, ~48ms/300k, ~150ms/1,000,000 words (~5MB) --
   consistent with the reviewer's own ~19/60/200ms figures and well past the
   repo's 100ms worker budget, for exactly the pastes an audiobook feature
   invites. Fixed by splitting `_detect_chapters()` into a thin dispatcher
   plus `_detect_chapters_worker` (a `@work(thread=True, exclusive=True)`
   method) that runs the detector off the event loop and marshals the
   result back to `_apply_detected_chapters` via `self.app.call_from_thread`
   -- for all four call sites, not just the debounced one. The three
   one-shot import paths do not have genuinely bounded content either (a
   file/note/conversation import can be just as large as a paste), so they
   route through the same threaded path rather than being special-cased as
   "small enough."
   - Minor also fixed here: `_notify_chapter_count` now only pops the
     "Detected N chapters" toast when N changed since the last toast, so a
     debounced re-paste that re-runs detection several times does not spam
     one toast per settle.
2. **Notes accuracy** -- see the corrected per-path section above.

## Verification (round 2)

Heartbeat-seam pattern (same idiom as
`Tests/UI/test_llm_screen_ollama_probe_nonblocking.py`, task-15473): a
concurrent `asyncio` task ticking every 5ms, bracketed tightly around
`_detect_chapters()` + `await app.workers.wait_for_complete()` for a
~3,000,000-word (~15MB) paste. Mutation-tested by temporarily forcing a
synchronous call (bypassing the worker): heartbeats dropped from ~25-35 to
**0** over the same window, confirming the test actually discriminates the
regression -- an earlier draft of this test bracketed the debounce timer's
own ~1s of unrelated real-time sleep too, which was proven (empirically) to
swamp the signal: a deliberately-reintroduced synchronous call still
cleared 173 heartbeats in that wider window, a false pass.

**Tests**: `Tests/UI/test_speech_audiobook_chapter_detection.py` now has 5
tests (all born red against the applicable pre-fix code, green after):
typing raises no exception; a burst of keystrokes inside the debounce window
calls `_detect_chapters` zero times until the window elapses, then exactly
once; a one-shot file import still populates `detected_chapters` with no
switch (now also awaits `app.workers.wait_for_complete()` since detection is
threaded); the event loop stays responsive (heartbeats >= 10, mutation-tested
per above) during a large-paste detection; a toast fires only when the
detected chapter count changes (mutation-tested: removing the dedup check
made the test fail red).

Re-ran: the 5-test file alone (5 passed), plus the full STTS/Speech suite
(`grep -rl STTS_Window Tests/` plus `test_speech_audiobook_layout.py`, which
asserts the collapsible-group layout is unchanged -- confirming no switch UI
was re-added): 341 passed, 0 failed (339 baseline + 2 new tests this round).
`ruff check` on both changed files: all checks passed.

**Files changed**:
- `tldw_chatbook/UI/STTS_Window.py`
- `Tests/UI/test_speech_audiobook_chapter_detection.py` (new; 5 tests)

## Review follow-up (round 3)

Three findings from a re-review of round 2, all addressed -- see
`Tests/UI/test_speech_audiobook_chapter_detection.py` and
`tldw_chatbook/UI/STTS_Window.py` for the full detail; summary below.

1. **Stale-result overwrite (Important).** `exclusive=True` on
   `_detect_chapters_worker` cancels a *queued* worker in its group but
   cannot interrupt one already executing on its OS thread
   (`Worker.cancel()` cancels the wrapping asyncio Task, not the thread) --
   the reviewer reproduced 3/3 a slower, superseded dispatch's
   `call_from_thread` overwriting a newer result, realistic because the
   three one-shot import paths have no debounce between them and detection
   can run up to ~700ms. Fixed with a monotonically-increasing
   `_chapter_detect_generation` id, captured in `_detect_chapters` and
   threaded through the worker; `_apply_detected_chapters` now applies a
   result only if its `generation` still matches the latest dispatched one
   -- the guard runs entirely on the main thread (both the increment and
   the check), so there is no cross-thread race on the counter itself.
   Deliberately did NOT also add the suggested optional
   `get_current_worker().is_cancelled` early-exit: adding it during
   implementation caused the end-to-end reproduction test to pass even with
   the generation guard *disabled* (mutation-tested), because in that
   scenario the worker's own cancelled flag was already true by the time it
   resumed -- i.e. it would have quietly masked whether the "real" guard
   actually mattered. Left out in favor of one unambiguous correctness
   mechanism.
2. **Toast reset semantics.** `_last_notified_chapter_count` never reset,
   so a genuinely new import that happened to detect the same count as a
   previous session was silently un-toasted (reviewer reproduced). Reset
   seam: `_import_content` -- the single dispatcher all four source types
   (file/notes/conversation/paste) funnel through, and the app's own signal
   that "a new bring-in-content action began" (for "paste" specifically,
   this is what unlocks the previously-disabled content-preview box).
   Detections re-run later within the same session with no further
   `_import_content` call in between -- e.g. the paste box's debounced
   re-detection -- still dedupe against each other, since this is the only
   reset point.
3. **Flaky gate.** The heartbeat test asserted an absolute count (`>= 10`),
   which failed 4/6 runs under real machine load despite the fix being
   structurally sound. Redesigned as a same-run, load-independent
   comparison: a synchronous control arm (`ChapterDetector.detect_chapters`
   called directly inside an `async def` wrapper with no internal `await`)
   is *guaranteed* zero heartbeats by construction -- a coroutine with no
   await point cannot yield, so a concurrently scheduled heartbeat task
   cannot run even once during it, independent of machine speed -- and the
   real threaded call only has to beat that guaranteed zero
   (`threaded_heartbeats > sync_heartbeats`). Also reduced
   `_make_large_book`'s chapter-header density (2000 -> 60,000 words per
   chapter): the original density produced ~999 chapters for a 3M-word
   book, which intermittently tripped an unrelated, pre-existing race in
   `ChapterEditorWidget`/`Select`'s mount sequence when the chapter table
   populated that many rows in one reactive update (observed once in a
   full-file run, reproducible 0/4 afterward at the reduced density -- a
   real but out-of-scope flake, not owned by this task).

**Also found and worked around, out of scope:** the "Import From" `Select`
(`#import-source-select`) is composed with `options=[(id, label), ...]`
(e.g. `("file", "Text File")`), but Textual's `Select.options` order is
`(renderable, value)` -- so the widget's actual `.value` is the display
label ("Text File"), never the lowercase id `_import_content`'s
`if import_source == "file":` branches check against. This means the
"Import Content" button's source dispatch is non-functional today for all
four sources, regardless of this task's changes. Not fixed here (separate,
pre-existing bug); the round-3 dedup-reset test calls `_import_content`
directly rather than driving it through the Select, since the reset line
runs unconditionally before the (currently dead) branching.

**Verification (round 3):**
- Mutation-tested both new guards: disabling the generation check in
  `_apply_detected_chapters` failed both
  `test_apply_detected_chapters_rejects_a_stale_generation` and
  `test_a_slower_superseded_detection_never_overwrites_a_faster_one` red;
  disabling the `_import_content` reset failed
  `test_notify_dedup_resets_on_a_new_import_action` red. Restored, all
  green.
- Ran `Tests/UI/test_speech_audiobook_chapter_detection.py` 6x consecutively
  with `-s`: **all 6 green** (8 tests each), heartbeat-seam counts recorded
  each run: `sync=0 threaded=24`, `sync=0 threaded=25` (x4), `sync=0
  threaded=24` -- sync is deterministically 0 every run as designed;
  threaded consistently clears 24-25, far above the old flaky threshold.
- Full STTS/Speech batch (same 9 files): **344 passed**, 0 failed (341
  baseline + 3 new tests this round).
- `ruff check` on both changed files: all checks passed.

**Files changed (round 3, on top of rounds 1-2):**
- `tldw_chatbook/UI/STTS_Window.py` -- generation guard, `_import_content`
  dedup reset.
- `Tests/UI/test_speech_audiobook_chapter_detection.py` -- 3 new tests
  (stale-generation unit test, end-to-end slow/fast dispatch test,
  dedup-reset test), heartbeat test redesigned load-robust, `_make_large_book`
  chapter density reduced, notify tests updated for the new `generation`
  parameter.
