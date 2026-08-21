---
id: TASK-19560
title: >-
  Kokoro model download blocks the event loop with no timeout, and two live
  summarization POSTs are unbounded
status: To Do
assignee: []
created_date: '2026-08-21 20:10'
labels:
  - concurrency
  - tts
  - networking
priority: high
dependencies: []
---

## Description

Source: 2026-08-21 holistic review, Lane 4 (concurrency / async / workers) —
its **#1** (the lane's top-ranked finding) and **#7**. Grouped: both are
unbounded synchronous network calls on live paths. Re-verified at this branch
base.

**A — Kokoro TTS model download freezes the whole TUI.** CONFIRMED.
`TTS/backends/kokoro.py` performs `requests.get(url, stream=True)` at lines
**281, 329, 1068 and 1093** (`_download_model_if_needed` at 1060,
`_download_voice_if_needed` at 1084). These are dispatched via
`asyncio.create_task` from `Event_Handlers/TTS_Events/tts_events.py`, and the
TTS backend layer contains **no `to_thread` offload at all**.

Two distinct harms:
- A 327 MB model download runs **on the event loop** — the entire TUI is frozen
  for its duration, on **first Kokoro use**, which is a reachable ordinary path.
- `requests` defaults to `timeout=None`. A stalled connection freezes the app
  **permanently**, with no cancel available — the user's only recourse is to
  kill the process.

**B — two live summarization POSTs have no timeout.** CONFIRMED.
`LLM_Calls/Summarization_General_Lib.py:1125` (Anthropic) and
`LLM_Calls/Local_Summarization_Lib.py:84`. Both are reachable from PDF and
document ingestion and from web search.

**Scope discipline — what this task is NOT.** The lane explicitly checked and
cleared two adjacent suspicions, and re-auditing them would be wasted work:
`httpx` defaults to a 5 s timeout, so the **27 bare `AsyncClient()` sites are
NOT unbounded**; and `get_openai_embeddings` is unbounded but has **zero
callers**.

## Acceptance Criteria

- [ ] Kokoro model and voice downloads run off the event loop — the TUI stays
      responsive while a model is fetching
- [ ] Every download in `TTS/backends/kokoro.py` carries a connect and read
      timeout; a stalled connection surfaces an error instead of hanging the
      app forever
- [ ] The download is cancellable from the UI, and cancelling it leaves no
      partial model that a later run treats as complete
- [ ] The user sees progress for a multi-hundred-megabyte download rather than
      an unexplained freeze
- [ ] `Summarization_General_Lib.py:1125` and `Local_Summarization_Lib.py:84`
      carry explicit timeouts
- [ ] A guard test fails on a `requests.get`/`requests.post` without a timeout
      in the packages covered here, so the next one is caught at review time
- [ ] Verified live on first Kokoro use, not only by unit test — the freeze is a
      runtime property
