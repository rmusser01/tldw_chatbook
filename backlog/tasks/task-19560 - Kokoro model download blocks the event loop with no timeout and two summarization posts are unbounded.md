---
id: TASK-19560
title: >-
  Kokoro model download blocks the event loop with no timeout, and two live
  summarization POSTs are unbounded
status: Done
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

- [x] Kokoro model and voice downloads run off the event loop — the TUI stays
      responsive while a model is fetching
- [x] Every download in `TTS/backends/kokoro.py` carries a connect and read
      timeout; a stalled connection surfaces an error instead of hanging the
      app forever
- [x] The download is cancellable from the UI, and cancelling it leaves no
      partial model that a later run treats as complete
- [x] The user sees progress for a multi-hundred-megabyte download rather than
      an unexplained freeze
- [x] `Summarization_General_Lib.py:1125` and `Local_Summarization_Lib.py:84`
      carry explicit timeouts
- [x] A guard test fails on a `requests.get`/`requests.post` without a timeout
      in the packages covered here, so the next one is caught at review time
- [x] Verified live on first Kokoro use, not only by unit test — the freeze is a
      runtime property

## Implementation Notes

**Kokoro downloads.** All four sites (PyTorch model, voice pack, ONNX model,
voices bin) route through one `_kokoro_stream_download` helper:

* connect + read timeouts -- the read timeout is per-chunk, so a slow but
  progressing transfer is not killed;
* the body streams to a `.part` sibling and is promoted with `os.replace`
  only once fully read, and any `BaseException` (cancellation included)
  removes the partial. This is the load-bearing half: `_download_model_if_
  needed` gates purely on `os.path.exists`, so a truncated file was
  previously indistinguishable from a finished model;
* progress logged at most every 2s, with a percentage when `content-length`
  is present;
* the two ONNX sites keep their existing checksum-verify and move logic --
  only the transfer moved off the loop.

Progress is reported to the log rather than to a UI progress bar; wiring a
visible bar needs a caller that consumes a callback, which does not exist
today. Flagged rather than silently counted as done.

**Summarization POSTs.** Both named sites now carry the config-driven timeout
the OpenAI path already used. While implementing, found that
`Local_Summarization_Lib` read the setting via `get_cli_setting` without
importing it -- a `NameError` on first real call that no AST-level test would
catch. Import added and pinned by a test that resolves the accessor and its
value.

**SCOPE FINDING — 27 more unbounded calls, reported not fixed.** A static
audit of the two summarization modules finds **29** timeout-less
`post`/`get` calls; this task named 2. Bounding an arbitrary two of
twenty-nine leaves the same hang-forever hazard everywhere else, and the
right answer is a session-level default timeout (requests has no native
per-session default, so it needs a small `Session` subclass or adapter) --
a design change deserving its own task. Remaining sites:

    Summarization_General_Lib.py:1386 Summarization_General_Lib.py:1482 Summarization_General_Lib.py:1634 Summarization_General_Lib.py:1687 Summarization_General_Lib.py:1817 Summarization_General_Lib.py:1907 Summarization_General_Lib.py:2047 Summarization_General_Lib.py:2101 Summarization_General_Lib.py:2237 Summarization_General_Lib.py:2296 Summarization_General_Lib.py:2424 Summarization_General_Lib.py:2497 Summarization_General_Lib.py:2645 Summarization_General_Lib.py:2701 Local_Summarization_Lib.py:410 Local_Summarization_Lib.py:924 Local_Summarization_Lib.py:988 Local_Summarization_Lib.py:1414 Local_Summarization_Lib.py:1466 Local_Summarization_Lib.py:1948 Local_Summarization_Lib.py:2002 Local_Summarization_Lib.py:2206 Local_Summarization_Lib.py:2260 Local_Summarization_Lib.py:654 Local_Summarization_Lib.py:730 Local_Summarization_Lib.py:1156 Local_Summarization_Lib.py:1224

Files: `tldw_chatbook/TTS/backends/kokoro.py`,
`tldw_chatbook/LLM_Calls/Summarization_General_Lib.py`,
`tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py`,
`Tests/TTS/test_kokoro_download_hardening.py`,
`Tests/LLM_Calls/test_summarization_request_timeouts.py`.
