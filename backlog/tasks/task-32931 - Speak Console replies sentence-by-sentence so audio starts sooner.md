---
id: TASK-32931
title: Speak Console replies sentence-by-sentence so audio starts sooner
status: Done
assignee:
  - '@dsh'
created_date: '2026-09-27 20:10'
updated_date: '2026-10-10 16:13'
labels:
  - tts
  - speech
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A Console reply is synthesized as ONE provider request, so on a locally hosted
TTS server that generates at roughly speaking pace the user waits for the whole
message before hearing anything. Measured against this project's Qwen3-TTS
setup: 515 characters = ~19s of silence before the first sound, and a
2099-character message exceeded the backend's 60s HTTP timeout and failed
outright -- after which the abandoned generation kept the server's single
inference lock and every later request (down to five characters) timed out too,
for ~15 minutes.

Playback cannot start earlier on the current shape, for a reason that is
structural rather than incidental: the streaming sink only accepts a WAV
response whose COMPLETE body has arrived (`TTS/pcm_stream.sink_plan` validates
the whole RIFF body before it will produce a plan), so time-to-first-sound
equals total generation time.

The fix is to stop synthesizing the utterance as one lump: split it on sentence
boundaries, issue one request per piece, strip each piece's WAV header, and feed
the pieces to ONE sink as a single continuous PCM stream -- so audio starts
after the first piece while the rest generate during playback.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A long Console reply begins playing after its first sentence-sized piece, not after the whole message
- [x] #2 Every piece of the utterance is played, in order, exactly once, with each piece's WAV header stripped before the sink
- [x] #3 Text short enough to be a single piece, a non-WAV format, and a machine with no audio sink each keep the existing single-request behaviour unchanged
- [x] #4 A failure after playback has started is reported to the user and never silently replayed from the beginning
- [x] #5 A sink that fails to open before any audio plays still delivers the utterance through the existing fallback path
- [x] #6 Targeted TTS and TTS_Events suites pass
- [x] #7 Sentence chunking preserves cancellation, exact response contracts, utterance progress, and non-Console file delivery; long unspaced text is bounded.
<!-- AC:END -->

## Implementation Plan
<!-- SECTION:PLAN:BEGIN -->
1. Hoist the provider call in `_generate_tts` into a local `request_piece(text)`
   seam, so the same selection/authorization/destination rules serve both the
   ordinary whole-utterance request and the per-piece requests. The existing
   call becomes `request_piece(text)` called once -- no behaviour change.
2. Add `_stream_text_pieces_if_applicable`: gate on a WAV-bound request plus an
   available sink, split with the existing `TextChunker` at a small token
   budget, generate the FIRST piece before opening the sink (its validated
   header supplies rate/channels), and hand `_stream_response_via_sink` a
   raw-PCM `SinkPlan` over a generator that yields each piece in turn. `None`
   from any of those gates leaves the untouched single-request path to run.
3. Reuse the existing sink helper rather than writing a second one -- it already
   owns sink open/teardown, event forwarding, lifecycle transitions, underrun
   accounting, and the outcome codes.
4. Tests: whole-utterance completeness in request order (the TASK-32027 lesson:
   chunked synthesis must not keep only the first chunk, and must not feed
   repeated headers as audio), plus the unchanged-path and failure-path pins.
5. ADR check: no ADR. No schema, no new module boundary, no provider or runtime
   change -- this reuses the existing sink seam and the existing chunker. The
   rejected alternative (making the local TTS server stream its response
   instead) is recorded in the Implementation Notes.

Review follow-up (2026-10-10):
1. Rebase onto current dev and preserve both lesson additions.
2. Reproduce and fix import failure, cancellation during response cleanup, per-piece response validation, overall progress, and accidental non-Console chunking.
3. Bound oversized text pieces while preserving word boundaries; document one provider call per piece and custom metered-endpoint quota limits.
4. Exercise long speech through the public entry, run targeted speech regressions and derived-artifact preflight, and review the final diff.
ADR required: no
ADR path: N/A
Reason: repair existing speech lifecycle and response contracts without changing module boundaries or storage. Existing ADRs: backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md and backlog/decisions/028-character-tts-generation-profile-ownership.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
**What changed.** A long Console utterance is now synthesized as several
sentence-sized provider requests and fed to ONE streaming sink as a single
continuous PCM stream, so playback begins after the first piece instead of after
the last.

- `Event_Handlers/TTS_Events/tts_events.py`
  - The provider call inside `_generate_tts` is hoisted into a local
    `request_piece(text)` seam, so the selection/authorization/destination rules
    serve both the ordinary whole-utterance request (now one call to it) and the
    per-piece requests. Behaviour for the single-request path is unchanged.
  - `_stream_text_pieces_if_applicable` gates on a WAV-bound request plus an
    available sink, splits with the existing `TextChunker` at
    `_SPEECH_PIECE_MAX_TOKENS = 35`, generates the FIRST piece before opening the
    sink (its validated header supplies rate/channels), then hands the existing
    `_stream_response_via_sink` a raw-PCM `SinkPlan` over
    `_iter_speech_pieces_pcm`, which strips each piece's WAV header and requests
    the next piece only after the previous one has been handed over.
  - `_collect_speech_piece` owns one piece's request/drain/lease-release.
- `Tests/TTS/test_sentence_chunked_speech.py` (new, 6 tests).

**Decisions and trade-offs.**

- *Reused the existing sink helper instead of adding a second one.*
  `_stream_response_via_sink` already owns sink open/teardown, event forwarding,
  lifecycle transitions, underrun accounting and the outcome codes, so the new
  path produces them by construction rather than by imitation.
- *`fallback_on_failure=False` for the piecewise pump.* Once audio has been
  heard, falling back would replay the utterance from the top. A mid-utterance
  failure is therefore reported as `streaming_failed` (with the composed error
  copy) rather than silently retried. A sink that fails to OPEN is different --
  nothing has played yet, so `None` falls through to the ordinary whole-text
  request, which is what AC #5 pins.
- *No artifact file on this path.* Parity with the existing successful sink
  branch, which discards its artifact for the same reason (nothing downstream
  should auto-play audio that was already played live).
- *Only WAV-bound requests are chunked.* Pieces are joined by stripping each
  WAV header and concatenating samples; a compressed format has no equivalent
  in-stream join, so it keeps the existing path.
- *Rejected alternative: make the local TTS server stream its response.* It
  would need the sink's WAV eligibility rule relaxed as well (the sink requires
  a complete validated RIFF body), so it changes an app-side contract for a
  server-side win, and it would leave the app dependent on one server build.
  Chunking works against any provider that returns WAV.

**Verification.**

- `Tests/TTS/test_sentence_chunked_speech.py`: 6 passed. The completeness test
  asserts the bytes fed to the sink equal every piece's own payload, in request
  order, which is the TASK-32027 failure mode (chunked synthesis that plays only
  the first chunk, or feeds repeated WAV headers as audio).
- Regression baseline, speech-path subset (10 files covering `_generate_tts`,
  the sink, format adaptation, logging privacy, and `Tests/TTS_Events/`):
  **11 failed / 339 passed with the change and 11 failed / 339 passed with
  `tts_events.py` reverted to HEAD** -- identical, so the failures are
  pre-existing/environmental (they are path-ownership and subprocess
  assertions unrelated to this module).
- Live, against the real local Qwen3-TTS server (`qwen3-tts.service` on
  127.0.0.1:10203), same 461-character reply, only the audio device replaced:
  the old whole-utterance request needed **18.7s before any audio existed**;
  the shipped chunked path put its **first audio into the sink at 2.8s**, in 4
  pieces (81/93/119/165 chars), one completion event, no error. Total time was
  essentially unchanged (17.6s vs 18.7s) -- the win is entirely in when playback
  can start.
- Live again through the REAL `StreamingPcmSink` (an actual `sounddevice`
  output stream, not a recorder -- `lessons-testing-evidence.md`: a stand-in is
  not the shipped sink), real server, 416 characters: **first audible at 4.1s**,
  4 pieces, and the real sink's own lifecycle reported
  `SinkStarted -> SinkDrained` with `complete(error=None)`. A 156-character
  reply on the same run correctly stayed a single piece, confirming the gate.
- Persistent-diagnostic inventory: this task adds exactly one diagnostic call
  (`logger.warning("TTS piece response close failed")`), and the pin was
  regenerated for it on this branch. The required review
  (`--statements ... --since <pin commit>`) reported **1 added, 0 removed, 0
  moved/re-indented**, and the statement interpolates nothing -- no user
  content, secret, path or URL. The pin diff is two lines (call_count 51 -> 52
  plus the digest); no other owner's row moved.
- Regression baseline on the PR base (`origin/dev` 97b16d4fb6), speech subset
  (the sink/console-speech/format/logging-privacy files plus all of
  `Tests/TTS_Events/`): **6 failed / 242 passed with this change and 6 failed /
  242 passed with it stashed** -- identical, so those failures are pre-existing
  on `origin/dev`, not caused here.

**Review follow-up (2026-10-10).** Fixed the missing `Any` import that broke test collection; shared the whole-response contract checks with every piece; retained cancellation during cleanup, including after a primary failure; scaled per-piece progress to the utterance; and restricted chunking to Console playback. Oversized pieces are capped at 200 characters, prefer word boundaries, and preserve delimiters. Tests now distinguish every PCM payload and cover the public Console snapshot/event route, exact-selection rejection on first/later pieces, manual file delivery, and multilingual/oversized text.

One provider request is made per piece. A custom OpenAI-compatible endpoint backed by tldw_server consumes one `max_calls` unit per request; a finite quota may reject a later piece after playback begins. This is an accepted tradeoff of client-side chunking, reported as a playback failure without replay. No server quota or batching contract changes are included.

Fresh local verification: 29 chunked-speech tests passed; the targeted Console speech, format adaptation, logging privacy and TTS_Events run passed 268 tests with four excluded tests. Those four config-source lifetime failures were independently reproduced unchanged on dev `0c3ebc6ed76e5753385897c958f80b1c282abaa3` (4 failed, 19 deselected in the logging-privacy file), before reaching the modified speech path. Ruff formatting and undefined-name checks passed. All derived-artifact preflight checks passed after rebasing onto dev `75159f8843a10cf761114defc2d49554a8772bb3`; the final targeted run was 268 passed / the same four independently reproduced baseline failures. Independent final code review found no remaining concrete defects. Historical live-server/device measurements above were not repeated locally because that server is unavailable.

ADR required: no; existing ADR-023 and ADR-028 govern these lifecycle/selection repairs. No storage or boundary changes.

<!-- SECTION:NOTES:END -->
