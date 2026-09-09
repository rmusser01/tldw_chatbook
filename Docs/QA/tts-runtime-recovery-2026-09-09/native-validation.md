# macOS ARM native TTS qualification and recovery QA

Real CPU and Metal runs established complete PCM delivery and successful app recovery for the cases below. Strict ASR text matching remained imperfect, and deliberately terminating an active native request exposed crashes in audio.cpp. These are separate findings.

## Tested revisions and scope

- Application: `565dc499210491def757bed325c4014e324dc470`, imported from an immutable archive. All 2,574 source files matched the archive manifest before and after each final run; additional-model imports were also checked against that snapshot.
- audio.cpp: release `release-0.5.1`, commit `238ab6a9e321c17de8e120559f57efeedaeb1345`. Native source remained clean, and CPU/Metal executable hashes matched the original build manifest.
- Model repository: `audio-cpp/audio.cpp-gguf`, revision `597048d9a920592808d7d4e2acd7b9c4596a143a`.
- Host: macOS 26.5.2, ARM64. Runs used mounted production Speech widgets, trusted Console messages, destination resolution and consent fingerprints, actual managed servers and HTTP, and physical output-device callbacks. Temporary observers recorded evidence without replacing inference, HTTP, or audio output.

These results qualify that archived application revision. They do not independently qualify subsequent Kokoro changes or package installations.

## Additional models

Each configuration completed two Console replies, with a stopped managed child before each reply. All eight synthesis responses returned HTTP 200; playback reached natural drain. All twelve managed children, including initial readiness generations, exited zero.

| Model and configuration | Backend | Reply durations | Full PCM transfer | Exact ASR captures |
|---|---|---|---|---|
| PocketTTS English Q8 | CPU | 6.080 / 5.280 s | 2/2 | 4/4 |
| PocketTTS English Q8 | Metal | 5.920 / 5.440 s | 2/2 | 4/4 |
| Supertonic, runtime Q8 setting | CPU | 6.458 / 6.421 s | 2/2 | 3/4 |
| Supertonic, runtime Q8 setting | Metal | 6.445 / 6.434 s | 2/2 | 2/4 |

Independent FFmpeg decoding matched WAV PCM for all sixteen source/device captures. Each device capture matched generated PCM after removing only all-zero boundary silence. No internal alignment, deletion, or substitution was applied. The three Supertonic ASR differences were `playback complete successfully` → `playback completes successfully`; raw segments and timestamps are retained. The strict verifier exited one rather than normalizing these differences away.

PocketTTS used a 6.455-second synthetic neutral Supertonic M1 reference through a server preset. No personal recording or gated Alba asset was used. This exercised advanced managed JSON configuration, not Guided clone admission.

## Stop, cancellation, failure, and retry

Supertonic 3 F16 passed all six app lifecycle phases on both backends.

| Phase | CPU | Metal |
|---|---|---|
| Stop during real playback | Stopped in 119 ms | Stopped in 114 ms |
| Full successor after Stop | Complete drain | Complete drain |
| Cancel an in-flight generation | App request stopped in 1.29 ms | App request stopped in 1.41 ms |
| Full successor after cancellation | Complete drain | Complete drain |
| SIGTERM during active synthesis | Child SIGSEGV, exit −11; app reported failure | Child SIGABRT, exit −6; app reported failure |
| Retry the same trusted message | New child, complete drain, clean exit | New child, complete drain, clean exit |

The in-flight barrier observed an actually transmitted POST and pending response; CPU additionally showed native CPU activity. The failed request opened no playback sink. Rendered failure evidence showed `TTS failed: TTS generation failed; retry`. Retrying preserved the destination fingerprint, advanced process generation, and succeeded. Final handler owners, active/retained tasks, admitted operations, responses, and adapter leases were all cleared; children joined and cleanup errors were empty.

All six full successor/retry device recordings matched generated PCM after boundary-zero trimming. Exact ASR clauses matched only 2/12 source/device captures. Retained differences include `reply`/`replay`, `words`/`word`, `finish`/`finished`, and `amber`/`Amber's`. This supports complete PCM transfer and recovery, not exact pronunciation of every input word. Both transcript verifiers preserve raw output and apply only case/punctuation normalization.

## Native shutdown and quantization limits

The injected SIGTERM crashes remain a native-runtime defect despite successful app recovery. Metal diagnostics show `ggml_metal_free: deallocating` followed by `GGML_ASSERT(backend) failed`, with synthesis still on the request-thread backtrace. Detached request threads outliving the stack-owned server/model state are a source-derived likely mechanism, not a debugger-proven diagnosis. No upstream code was changed.

The published `supertonic-3-q8_0.gguf` is byte-identical to `supertonic-3-orig.gguf`: 454,072,836 bytes, SHA256 `af814486a0bc9513fb36afabd9b1155ad14fb2c36a107ac6ffe62ea9adafb662`. Header inspection found 698 F32 and 72 I64 tensors, with no Q8 tensors. This remained true at checked upstream revision `056144d2744697c9439bd32647279674dba0c964`.

The additional runs instead selected `session_options.supertonic.weight_type=q8_0`. Pinned source applies this setting to selected linear weights while retaining others as F32. Runtime tensor memory was not independently inspected; these are results with the supported runtime setting, not qualification of a genuinely quantized downloaded GGUF.

## Residual gaps and evidence

This work does not establish full Console shell navigation, Speech Lab Generate interaction, Guided cloning, microphone measurement of room audio, cross-platform behavior, long-duration stress, or immediate termination of native computation after app cancellation. Linguistic exactness and safe native shutdown during active inference remain unresolved. No final run left an owned inference/playback process running.

Canonical evidence and artifact hashes:

- [Additional-model report](/private/tmp/tts-runtime-recovery-validation/native-prep/REPORT.md) and [summary](/private/tmp/tts-runtime-recovery-validation/native-prep/summary.json).
- [Failure/recovery report](/private/tmp/tts-runtime-recovery-validation/fault-harness/REPORT.md) and [summary](/private/tmp/tts-runtime-recovery-validation/fault-harness/summary.json).

These link the exact run directories, raw WAVs, transcripts, lifecycle/HTTP observations, failure screens, cleanup evidence, and retained observer corrections. Runtime evidence and later content verification remain separate artifacts.
