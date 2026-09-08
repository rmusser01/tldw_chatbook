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
| Chatterbox streaming tensors | WAV | 5.717 s | afplay | 6.693 s / 0 | Yes |
| Higgs arrays | MP3 | 5.717 s | afplay | 6.870 s / 0 | Yes |
| ElevenLabs PCM conversion | AAC (ADTS) | 5.760 s | afplay | 6.882 s / 0 | Yes |
| ElevenLabs PCM conversion | FLAC | 5.717 s | afplay | 6.611 s / 0 | Yes |
| AllTalk WAV conversion + playback wrapper | PCM / WAV | 5.717 s | afplay | 6.575 s / 0 | Yes |
| OpenAI-compatible body delivery | Opus | 5.717 s | ffplay | 6.046 s / 0 | Yes |

Local evidence: `/private/tmp/tts-backend-parity-qodo-playback/evidence.json`, audio
files in its `artifacts/` directory, and
`/private/tmp/validate-tts-backend-qodo-playback.py`. These are validation
artifacts, not application data or bundled model weights. The table records the
repeat run after the Qodo fixes; all six complete transcripts passed again.

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

## PR boot-guard follow-up

CI exposed one extra boot-time module from the new PCM helper. Its imports now
run only when PCM playback is requested; the existing census also pins the helper
as absent at `_ui_ready`. The module budget is unchanged. All 4 census tests and 94 PCM/UI/streaming
regressions pass after this fix.

The fetched dev base also exceeded its CSS budget before this PR's changes:
804,241 bytes versus 804,000. The CSS files in the original PR matched dev.
Shortening an existing explanatory comment removes 497 bytes from the boot
bundle, yielding 803,744 bytes. Both the source and generated bundle have
identical non-comment CSS to HEAD. No rule or budget constant changed; ADR-097
permits shedding this existing boot cost. The CSS budget regression passes.


## Qodo review follow-up

All eight review comments were checked against the actual callers. Three runtime
defects were confirmed: a nonzero player exit was ignored by both completion
consumers; Higgs cleanup returned before native inference finished; and the Higgs
buffer check undercounted float64 arrays.

- Speech Lab treats player ERROR as terminal, releases the playback copy and
  lease, resets transport controls, and reports failure. The legacy completion
  poll rejects ERROR/IDLE immediately and verifies stop, current file, and state
  again at its deadline. Only a still-playing current clip retains the existing
  estimated-duration fallback; failed playback never invokes success.
- Higgs owns one retained cleanup task through caller cancellation. The adapter's
  foreground deadline stays bounded; cleanup releases the model after native
  work actually stops. Tests cross the real manager and host shutdown boundaries.
- Higgs checks both source nbytes and the proposed float32 buffer before
  finiteness/downmix/conversion, including exact float64 and float16 boundaries.
- Application-path tests cover PCM conversion, canceled copying, exact release
  ownership and copy deletion, plus mounted Opus playback through the actual
  asynchronous and synchronous players with only process/device seams replaced.
- Direct buffer-limit tests cover below/equal/above values and every public error
  field. PCM protocol values now have descriptive names, and Chatterbox's public
  streaming method documents its request and complete-file/raw-PCM yield contract.

Higgs regressions went from 4 failures to 8 passing cases; its affected
backend/manager/bridge cohort passed 148 tests. Shared limit/PCM checks passed
43 tests. Playback failure regressions went from 8 failures to 17 passing cases through
the real application paths. A fresh independent review of these fixes found no
further actionable issues. All six derived preflight checks pass. The diagnostic
statement review found no added or changed logging calls, and the inventory
already matches exactly; no regeneration was necessary.

Playback/lifecycle coverage passed all 304 distinct cases across the broad run
and focused rerun. The broad run first passed 303 cases and exposed one old
ordering test that relied on five event-loop yields while its fake player could
already finish. Explicit admission/finish events make that assertion
deterministic; the entire utterance module and new failure module then passed
38 tests. This fixture repair does not alter production behavior.
