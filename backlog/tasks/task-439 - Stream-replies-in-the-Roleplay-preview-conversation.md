---
id: TASK-439
title: Stream replies in the Roleplay preview conversation
status: Done
assignee: []
created_date: '2026-07-21 09:38'
updated_date: '2026-07-24 02:29'
labels:
  - roleplay
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the RP/character-card UX review (Docs/superpowers/qa/rp-ux-review-2026-07-21/report.md). Observed live: a ~20s generation showed only a static "Running" status and then the full reply at once. Console streams; the preview should too (with the existing status line as fallback for non-streaming providers).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Preview replies render incrementally for providers that support streaming
- [x] #2 Non-streaming providers keep a working status indicator
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Investigation-first outcome: already satisfied — no code change needed.**

The preview reply worker (`PersonasPreviewController._run_reply`) already streams: it iterates `gateway.stream_chat(...)` (an async generator) and renders each chunk via `pane.begin_reply()` / `append_reply_chunk()` / `finalize_reply()`, with a streaming→non-streaming retry, mid-stream cancellation (`_stale()` + `discard_partial_reply`), and the status line ("Running"/"Ready"/…) intact for the non-streaming path. This was true **at the RP-review baseline `dc196563f`** (the streaming was added by the pre-review "Extract Personas preview controller" commit) — it is not a missing feature.

The gateway genuinely yields incrementally: `stream_llamacpp_chat` does `async for line in response.aiter_lines(): yield chunk` (SSE), and `_stream_generic_chat` yields per item.

**Why the review saw non-streaming ("~20s of static 'Running' then the full reply at once"):** the preview's provider *resolution* at review time — the P0-1 `[character_defaults]` bug (ships `anthropic`/`claude-3-haiku`, normally unconfigured), so the preview resolved an unready provider and errored/fell back instead of streaming. **TASK-425 (fallback to `chat_defaults`) and TASK-426 (provider readout), both merged earlier this campaign, fixed exactly that.** With the current code the preview reaches a ready streaming provider and streams.

**AC#1 — verified:**
- Existing integration test `test_reply_streams_progressively_into_one_line` asserts the transcript shows the first chunk *while the stream is held open* (real pane).
- **Live-verified end-to-end** against a real SSE server through the production path (real `ConsoleProviderGateway` → real network SSE at `/v1/chat/completions` → real `PersonasPreviewPane`), the transcript grew token-by-token over ~6.5s: `"character: Greetings"` (t=0.06s) → … → `"character: Greetings, detective. The fog rolls thick over Baker Street tonight."` (t=6.56s).

**AC#2 — verified:** the status line updates independently of streaming (`personas_preview_pane.set_status`); non-streaming providers keep it working — covered by `test_unready_character_provider_falls_back_to_chat_defaults`, `test_fallback_provider_survives_non_streaming_retry`, `test_blocked_provider_shows_readable_status`.

No production files changed. The RP review's streaming symptom was a downstream effect of the provider-resolution defect already fixed by TASK-425/426.
<!-- SECTION:NOTES:END -->
