---
id: TASK-634
title: Media analysis shows false success with sentinel text on LLM failure
status: To Do
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
- [ ] #1 A failing LLM call during media analysis surfaces an error to the user (no success toast, no sentinel text rendered as analysis)
- [ ] #2 A successful analysis is unaffected (return-value contract for the success path unchanged)
- [ ] #3 A regression test drives the failure path (LLM call raising) and pins the error surfacing
<!-- AC:END -->
