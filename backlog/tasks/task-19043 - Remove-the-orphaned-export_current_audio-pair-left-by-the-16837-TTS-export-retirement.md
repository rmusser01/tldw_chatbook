---
id: TASK-19043
title: >-
  Remove the orphaned export_current_audio pair left by the 16837 TTS export
  retirement
status: Done
assignee:
  - '@claude'
created_date: '2026-08-20 08:40'
labels:
  - cleanup
  - dead-code
  - tts
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-16837 (PR #1742) retired the never-dispatched `TTSExportEvent` path and
pinned it out (`Tests/TTS/test_tts_improvements.py::
test_per_message_export_surface_stays_retired`). That pin's docstring claims
"the user-reachable audio export lives on the S/TT/S playground path
(`STTSEventHandler.export_current_audio`)" — verified stale at dev `1bf7f234e`:
the playground actually exports via `UI/Speech/speech_playback_mixin.py::
_export_audio`/`_handle_audio_export` (`#audio-export-btn`, FileSave + direct
copy), which never touches the handler.

The surviving pair is orphaned: `app.py::export_current_audio` (:11413) has
zero callers anywhere in the tree, and it is the only production caller of
`Event_Handlers/STTS_Events/stts_events.py::STTSEventHandler.
export_current_audio` (:2786). Outside those two definitions, the only
references are `Tests/TTS/test_stts_export_security.py` (drives the handler
directly to prove destination-path validation) and the stale pin docstring.
Whole-tree grep for `export_current_audio` confirms — no dynamic dispatch,
no UI call site.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The dead pair (`app.py` wrapper + `STTSEventHandler.export_current_audio`) is removed — or wired to a live surface only if one genuinely needs it (none found; owner ruling prefers durable removal over speculative wiring)
- [x] #2 The 16837 pin's docstring no longer asserts the stale reachability claim
- [x] #3 `test_stts_export_security.py`'s destination-path-validation coverage is handled intentionally: retired with the code or re-pointed at the live playground export path's validation — not silently dropped
- [x] #4 TTS suites green; whole-tree grep for the removed names returns nothing
<!-- AC:END -->

## Implementation Plan

1. Re-verify orphanhood at the branch base with fresh whole-tree greps,
   including dynamic-dispatch shapes (string literals, getattr, action_ names,
   concat fragments like "current_audio").
2. Check the live playground export path's validation
   (`speech_playback_mixin._handle_audio_export`) and its existing test
   coverage to decide retire-vs-repoint for the security test.
3. Baseline `Tests/TTS/` to a file (failure SET, not count).
4. Delete the `app.py` wrapper (~:11413) and
   `STTSEventHandler.export_current_audio` (~:2786); retire
   `Tests/TTS/test_stts_export_security.py` if step 2 shows equivalent live
   coverage; fix the 16837 pin docstring and the stale
   `Docs/Features/TTS-To-Do.md` line to point at the mixin path (without
   naming the removed symbol).
5. Prove the kept live-path validation coverage is non-vacuous with one
   representative mutation (break the mixin's filename validation, watch the
   dangerous-destination test go red; Edit-based restore).
6. Re-run `Tests/TTS/` + the lifecycle export tests; repo-wide
   `--collect-only -q` sweep; ruff on touched files; final whole-tree grep for
   the removed name.

## Implementation Notes

**Removed the orphaned pair durably** (base `25500ad87`, branch
`task/19043-burn`):

- `tldw_chatbook/app.py`: deleted the 8-line `export_current_audio` wrapper
  (was :11413). Re-verification at the base confirmed zero callers and no
  dynamic-dispatch shape (no string literal, no getattr, no action_ name, no
  concat fragment of `current_audio` beyond the definitions themselves).
- `tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py`: deleted
  `STTSEventHandler.export_current_audio` (was :2786). Its function-local
  imports (`shutil`, `validate_filename`, `validate_path_simple`) die with
  it; `_current_playground_audio_path` survives with a live caller
  (`play_current_audio`). No new orphans created by the deletion
  (dead-graph walked from both ends).

**Security coverage resolved as RETIRE-with-the-code:** deleted
`Tests/TTS/test_stts_export_security.py` (3 tests). The destination-path
validation it drove lived inside the deleted handler. The live playground
export (`UI/Speech/speech_playback_mixin.py::_export_audio` /
`_handle_audio_export`, `#audio-export-btn`) carries the same validation
stack (`validate_path_simple` on dest + parent, `validate_filename`) and
already has equivalent coverage in
`Tests/UI/test_speech_playground_pane_lifecycle.py`: happy-path copy
(`test_export_uses_artifact_captured_before_dialog_completes`), cancel
release, and dangerous-destination rejection
(`test_audio_export_rejects_unsafe_dialog_destination` — file not created,
error severity, "dangerous pattern" message). That kept coverage was proven
non-vacuous by mutation: bypassing the mixin's whole validation block turned
the rejection test red on exactly the load-bearing assert (`assert not
unsafe_destination.exists()`); a first, narrower mutation (only
`validate_filename`) SURVIVED because the validation is layered —
`validate_path_simple` on the full dest catches the `;` first. Mixin
restored Edit-based, shasum-verified byte-identical to HEAD.

**Docstring/doc fixes:** the 16837 pin
(`test_per_message_export_surface_stays_retired`) and the
`Docs/Features/TTS-To-Do.md` line now point at the real export path
(speech_playback_mixin / `#audio-export-btn`) and record that the
previously-cited handler pair was itself orphaned and removed here —
without naming the removed symbol, so the whole-tree grep stays clean.

**Evidence:** `Tests/TTS/` before = 10 failed / 4079 passed / 16 skipped;
after = 9 failed / 4077 passed / 16 skipped (-3 retired tests, +1 rotating
flake), identical command (`-q -n 8`). Failure-set diff fully accounted
for: the union of both arms' 12 unique failures rerun serially left 3
reds, ALL present in the PRE-change baseline (2 pass in isolation =
xdist-load flakes; 1 — `test_tts_connection_error_copy`'s openai typed
error — fails isolated for want of `OPENAI_API_KEY` in the env, pre-
existing both arms). Touched pin suite `test_tts_improvements.py` 25/25.
UI speech consumers of the touched handler: 144 passed, 1 failed —
`test_speech_result_delivery.py::test_delivery_comes_from_the_shared_mixin`,
proven red on PRISTINE base `25500ad87` (pinned-SHA probe worktree) and
already fixed on current dev by `ab468a4a2`; not this branch's doing.
Repo-wide `--collect-only -q`: 51,467 collected, zero errors. `ruff check`
clean on touched Python files. Final whole-tree grep for the removed name:
no hits outside the two backlog task records (this one and 16837's).

**Filing candidate (out of AC scope, left in place):** the sibling
`play_current_audio` pair (`app.py:11404` -> handler `:2767`) is orphaned
by the same shape — zero callers of the app wrapper anywhere in the tree.

**Lesson recorded** in `lessons-testing-evidence.md`: a pristine probe
worktree cut at the ref name `origin/dev` ran DIFFERENT code than this
branch's base (the shared checkout's ref had moved; the moved-to dev had
already fixed the red), which briefly misread an upstream-fixed red as a
regression of mine. Probes must pin the base SHA and print their commit.
