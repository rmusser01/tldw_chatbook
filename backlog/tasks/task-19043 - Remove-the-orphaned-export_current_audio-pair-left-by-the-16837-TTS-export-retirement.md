---
id: TASK-19043
title: >-
  Remove the orphaned export_current_audio pair left by the 16837 TTS export
  retirement
status: To Do
assignee: []
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
- [ ] #1 The dead pair (`app.py` wrapper + `STTSEventHandler.export_current_audio`) is removed — or wired to a live surface only if one genuinely needs it (none found; owner ruling prefers durable removal over speculative wiring)
- [ ] #2 The 16837 pin's docstring no longer asserts the stale reachability claim
- [ ] #3 `test_stts_export_security.py`'s destination-path-validation coverage is handled intentionally: retired with the code or re-pointed at the live playground export path's validation — not silently dropped
- [ ] #4 TTS suites green; whole-tree grep for the removed names returns nothing
<!-- AC:END -->
