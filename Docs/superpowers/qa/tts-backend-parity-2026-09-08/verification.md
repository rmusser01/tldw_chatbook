# TTS backend delivery parity — 2026-09-08

TASK-32076 follows the Kokoro repair in PR #2512. It checks all seven registered
providers for the same effective-selection, whole-utterance, format, and
operation-ownership failures. This is delivery verification, not a claim that
all optional neural models or hosted accounts were available for live inference.

## Confirmed defects repaired

- Switching from a legacy provider to audio.cpp copied foreign model/voice IDs
  into native selection and disabled Generate. Saved same-provider exact pins
  still refuse unavailable models/voices; provider switching preserves ownership.
- Chatterbox concatenated WAV containers for long and streaming replies, so
  decoders stopped at the first chunk. It also called a nonexistent conversion
  method, ignored `stream=False`, reused a subprocess's canceled response, and
  retained changed fallback settings after cancellation. WAV encoding no longer
  depends on TorchCodec. Retried attempts cannot replay partially delivered PCM.
- ElevenLabs WAV responses were raw PCM; AAC/FLAC/Opus used unsupported wire
  identifiers. WAV/AAC/FLAC now convert documented 24 kHz PCM; Opus uses its
  documented wire enum. AllTalk's requested PCM is converted from server WAV to
  signed 16-bit 24 kHz mono audio.
- Shared conversion used invalid FFmpeg output format names for raw PCM and AAC.
- Higgs published error strings or empty results as audio, silently skipped
  missing dialogue sections, and released task ownership while native threads
  were still loading or generating. Failures now propagate; cleanup retains
  threads and loading/generation serialize appropriately.
- Explicit legacy PCM lacked rate metadata and could reach a file player as raw
  bytes. Known-rate PCM now has honest metadata and a temporary WAV playback
  copy. Speech Lab retains the raw export; unknown custom OpenAI PCM is refused
  for playback. TASK-1880's default/caller-scoped PCM selection remains separate.
- macOS afplay returned exit code 0 after about two seconds for a valid
  5.717-second Opus file. Opus now uses the existing FFplay player. Completion
  checks both the exact process owner and exit status.
- Buffered operations are bounded and return nonretryable shortening guidance;
  HTTP success with an empty body is rejected by the remote backends.

## Backend evidence and limits

| Provider | Exercised evidence | Live provider inference in this follow-up |
| --- | --- | --- |
| Chatterbox | Repeated six-format single/chunked/streaming output; exact decoded frames; real subprocess protocol; cancellation, retry, limits | Unavailable: chatterbox-tts/model weights absent; inference replaced with deterministic tensors |
| Higgs | Repeated six-format output; complete dialogue; errors/nonfinite samples; retained loading/generation threads | Unavailable: Boson model/runtime absent; inference replaced with deterministic arrays |
| ElevenLabs | Documented wire codecs through HTTP transport; six decoded formats; repeated requests; empty/limit errors | No authenticated remote inference performed |
| AllTalk | Real WAV-to-PCM conversion/resampling; repeated requests; HTTP/error contracts | No configured/running AllTalk server |
| OpenAI / compatible | Existing real loopback HTTP server and endpoint tests; body delivery, empty errors, official-vs-custom PCM metadata | No authenticated OpenAI inference performed |
| audio.cpp | Real adapter/catalog/response validation with transport fixtures; fresh and repeated mounted admission; switching/pinned-selection controls | No running audio.cpp service or usable native model found |
| Kokoro | Existing encoded-output/limit regressions; provider-switch and explicit-language isolation; PCM metadata/playback | Prior PR #2512 supplied real local ONNX playback; this follow-up uses it as a recorded speech fixture |

All seven providers cross the mounted Speech Lab Generate admission path twice.
The UI matrix also checks native/legacy round trips, exact missing Global/Studio
pins, and default reply resolution. HTTP/model/device fixtures are identified in
test docstrings; those tests do not prove hosted-provider availability.

## Playback evidence

The local harness passes a previously generated Kokoro sentence through the real
backend delivery/encoding code and the actual macOS audio device. Model inference
and remote HTTP responses use that recorded speech as their only fixture. The
entire clip is decoded and independently transcribed after playback.

Expected sentence: “This second reply confirms that repeated speech generation
and playback complete successfully.” Source duration: 5.717 seconds.

| Backend delivery path | Format | Full decode | Player | Player elapsed / exit | Complete transcript |
| --- | --- | --- | --- | --- | --- |
| Chatterbox streaming tensors | WAV | 5.717 s | afplay | 6.608 s / 0 | Yes |
| Higgs arrays | MP3 | 5.717 s | afplay | 6.319 s / 0 | Yes |
| ElevenLabs PCM conversion | AAC (ADTS) | 5.760 s | afplay | 6.838 s / 0 | Yes |
| ElevenLabs PCM conversion | FLAC | 5.717 s | afplay | 6.558 s / 0 | Yes |
| AllTalk WAV conversion + playback wrapper | PCM / WAV | 5.717 s | afplay | 6.578 s / 0 | Yes |
| OpenAI-compatible body delivery | Opus | 5.717 s | ffplay | 6.133 s / 0 | Yes |

Local evidence: `/private/tmp/tts-backend-parity-playback/evidence.json`, audio
files in its `artifacts/` directory, and
`/private/tmp/validate-tts-backend-parity-playback.py`. These are validation
artifacts, not application data or bundled model weights.

## Automated checks

- Python 3.12 main targeted backend/Console/codec cohort: **744 passed, 1 skipped**
  (the skip is the optional installed-Chatterbox integration test).
- Final Chatterbox delivery/cancellation/empty-stream cohort: **30 passed**.
- Python 3.13 shared backend/provider UI/PCM/recovery cohort: **115 passed,
  16 skipped** (Higgs fixtures require Torch, absent in that environment).
- Python 3.13 independent PCM/ownership and adjacent UI/lifecycle cohorts:
  **116 + 242 + 129 passed**, with overlapping coverage relative to the above.
- Red-to-green evidence includes real container truncation, codec enum/muxer
  failures, cancellation ownership, typed limits, empty results, missing native
  selection IDs, raw PCM fallback, and macOS Opus/process completion.
- Targeted new Python files pass full Ruff; all changed files pass syntax and
  undefined-name checks and formatting. Legacy lint debt was compared with HEAD;
  no new diagnostics were introduced.
- Diagnostic statement review removed obsolete Chatterbox/Higgs messages and
  added only fixed Opus guidance plus a numeric player exit code. No new logging
  destination or user-content interpolation was added. The diagnostic inventory
  was regenerated after that review.

No full repository test sweep was requested or run. Existing dependency warnings
and the Studio test's unawaited coroutine warning are documented in the local
logs. Linux/Windows device playback and unavailable live provider inference
remain unverified.

ADR required: no. Existing ADRs 023, 039, and 040 govern these repaired delivery,
settings-ownership, and current-result contracts. No new provider/runtime or
storage boundary was introduced.
