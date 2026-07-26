---
id: TASK-634
title: Media analysis shows false success with sentinel text on LLM failure
status: Done
assignee: []
created_date: '2026-07-25 22:40'
labels:
  - media
  - bug
  - error-handling
dependencies: []
---
## Description

Found by the task-577 PR2 T3 review's live-path trace (2026-07-25). **Pre-existing
bug, not introduced by the retirement** (the exception-path contract is
byte-identical before and after; `MediaWindow_v2.py` untouched).

The Media library's Analysis tab (`media_viewer_panel.py` "Generate Analysis" →
`MediaAnalysisRequestEvent` → `MediaWindow_v2.py:~1569`) calls
`app.chat_wrapper(...)` synchronously (`asyncio.to_thread`, `streaming=False`)
and consumes the RETURN VALUE. On any exception from the underlying LLM call,
`worker_events.chat_wrapper_function` returns the sentinel string
`"STREAMING_HANDLED_BY_EVENTS"` (worker_events.py ~:425) — a contract designed
for the retired event-driven chat path. `MediaWindow_v2.py:~1627-1628` treats
any `str` return as valid response text, so on a real LLM failure the user sees
the literal text "STREAMING_HANDLED_BY_EVENTS" as their analysis plus an
"Analysis generated successfully" notification.

Fix direction: give the media-analysis call a sane failure contract — either
(a) `chat_wrapper_function` raises/returns None on the non-streaming exception
path (check for other return-value consumers first; after task-577 PR2 the
media path is the ONLY live caller), or (b) MediaWindow_v2 rejects the sentinel
and error-notifies. (a) is cleaner now that the event-driven consumers are
retired; the `StreamDone` post in the exception handler is unhandled/harmless
either way.

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A failing LLM call during media analysis surfaces an error to the user (no success toast, no sentinel text rendered as analysis)
- [x] #2 A successful analysis is unaffected (return-value contract for the success path unchanged)
- [x] #3 A regression test drives the failure path (LLM call raising) and pins the error surfacing
<!-- AC:END -->

## Implementation Plan

1. `worker_events.chat_wrapper_function`'s single `except Exception` block (non-streaming
   path) currently posts `StreamDone` and unconditionally returns the
   `"STREAMING_HANDLED_BY_EVENTS"` sentinel, regardless of whether the caller requested
   streaming. Branch on the already-bound `streaming` kwarg: keep the streaming branch's
   `StreamDone` post + sentinel return byte-identical (nothing else consumes it, but it's
   the documented legacy contract); for the non-streaming branch, `raise` (bare, to
   preserve the traceback) so the caller gets the real exception instead of a string that
   gets rendered as if it were a valid LLM response.
2. `MediaWindow_v2.py`'s `perform_analysis()` outer `except Exception` (the only live
   non-streaming caller of `chat_wrapper`) already notifies the user with
   `f"Error: {str(e)[:100]}"`, but leaves the analysis display untouched — so a failure
   right after a prior successful analysis would leave stale content on screen. Mirror the
   existing falsy-response else-branch exactly: reset `#analysis-display` to
   `"*Analysis generation failed - no valid response text*"`, guarded by its own
   try/except.
3. Add regression tests first (TDD): a unit test pinning the non-streaming failure now
   raises (RED against current code, which returns the sentinel), a twin test pinning the
   streaming branch's untouched contract, and a media-seam test driving
   `handle_analysis_request` with `chat_wrapper` patched to raise, asserting an error
   notification, no success notification, no sentinel text anywhere, and the analysis
   display reset to a failure message.

## Implementation Notes

Approach taken: exactly the two production edits described in the plan, TDD'd with a RED
proof first for both new/extended test homes.

- `tldw_chatbook/Event_Handlers/worker_events.py` (`chat_wrapper_function`'s single
  `except Exception as e:` block, non-streaming path): the error logging/metrics
  (`logger.exception`, `log_counter`, `log_histogram`) stayed byte-identical. Below that,
  branched on `streaming` (already bound before the `try`, so safe to read in the `except`):
  the streaming branch keeps the original `StreamDone` post + `"STREAMING_HANDLED_BY_EVENTS"`
  return byte-identical (pinned by
  `test_chat_wrapper_function_streaming_failure_keeps_sentinel_contract`); the non-streaming
  branch now does a bare `raise` instead of swallowing the exception into the sentinel
  string (pinned by `test_chat_wrapper_function_nonstreaming_failure_raises`, which was RED
  against the pre-fix code with `Failed: DID NOT RAISE <class 'RuntimeError'>`).
- `tldw_chatbook/UI/MediaWindow_v2.py` (`handle_analysis_request`'s `perform_analysis()`
  outer `except Exception as e:`): after the existing `self.app_instance.notify(f"Error:
  {str(e)[:100]}", severity="error")` (left unchanged), added a try/except-guarded
  `#analysis-display` reset to `"*Analysis generation failed - no valid response text*"`,
  mirroring the falsy-response else-branch a few lines above verbatim. New test
  `test_media_analysis_llm_failure_surfaces_error_not_sentinel` in
  `Tests/UI/test_media_window_v2_parity.py` drives `handle_analysis_request` with
  `app_instance.chat_wrapper` patched to raise; it was RED against pre-fix code
  (`AssertionError: Expected update to have been awaited` — the display reset didn't exist
  yet) and is GREEN post-fix. It also asserts no "Analysis generated successfully" notify
  and no literal sentinel text anywhere in the notify calls or the display update text.
- Both edits are strictly additive to the failure paths; the success-path return-value
  contract (`return result` for non-streaming, the streaming generator-consumption branch,
  and the `response_text` success branch in `MediaWindow_v2.py`) is untouched, satisfying
  AC #2. Verified via the full existing `Tests/UI/test_media_window_v2_parity.py` +
  `Tests/UI/test_media_handoffs.py` + `Tests/Event_Handlers/` + `Tests/test_smoke.py` suites
  passing, plus pyflakes and `import tldw_chatbook.app` clean.
- Files modified: `tldw_chatbook/Event_Handlers/worker_events.py`,
  `tldw_chatbook/UI/MediaWindow_v2.py`,
  `Tests/UI/test_media_window_v2_parity.py` (new test + import).
- File added: `Tests/Event_Handlers/test_worker_events_contract.py` (new test home — none
  existed for `worker_events.py` before this task).
