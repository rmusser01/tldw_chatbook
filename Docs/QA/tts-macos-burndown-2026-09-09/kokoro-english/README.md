# English Kokoro qualification — 2026-09-09

The completed macOS matrix passed Speech Lab generation/playback, Console warmup, cancellation during observed inference, fresh successor playback, and repeated Console playback for all three tuples below. All 33 successful source clips matched the complete expected text after case/punctuation/whitespace normalization, with all beginning/middle/end anchors present. Earlier failures remain separate from this final denominator.

| Runtime | Output path | Successful clips | Console repeats | Observed Stop to return | Runtime exit |
| --- | --- | ---: | ---: | ---: | ---: |
| [PyTorch CPU](runs/pytorch-cpu-en-us-final-01/evidence.json) | WAV; Speech Lab file player and Console PCM sink | 15 | 12 | 3.061 s | 0 |
| [PyTorch MPS](runs/pytorch-mps-en-us-final-01/evidence.json) | WAV; Speech Lab file player and Console PCM sink | 9 | 6 | 2.466 s | 0 |
| [ONNX CPU](runs/onnx-cpu-en-us-final-01/evidence.json) | MP3; complete file playback | 9 | 6 | 3.069 s | 0 |

All runs used voice `af_heart`, language `en-us`, and speed `1.0`. [The original matrix receipt](matrix.json) records exact commands and runtime/content exits; [the derived summary](summary.json) retains timings, phase memory observations, versions, assets, cancellation and failure distinctions.

## Cancellation and playback evidence

Each final Stop overlapped exactly one observed inference call: PyTorch `KModel.forward` or ONNX `InferenceSession.run`. That call entered before Stop, exited before Stop returned, and no subsequent call entered for the cancelled request. At Stop, the generation owner, backend task, response, lease and native-call counts were live; all recorded owners were zero at settlement. Every successor began after settlement and played successfully. This records cooperative completion of already-running work, not instantaneous kernel interruption.

PyTorch receipts identify actual devices `cpu` and `mps:0`; the MPS run also records the `_CpuSTFT` Fourier component. ONNX records `CPUExecutionProvider`. These observations bind the requested devices to the exercised runtime path.

There were 11 completed `afplay` processes and 22 Console sink drains with nonzero audible callback frames. Playback elapsed time covered each decoded clip within the harness's 250 ms tolerance. All final resource counters and cleanup errors were clear; the worker processes exited 0. Scope was the mounted Speech pane and trusted Console store/admission/TTS handler with real models, codecs and output device. It did not test full-shell navigation or acoustic loopback.

## Complete content, including the raw difference

Expected text:

> Silver compass opens this reply. The middle sentence describes a quiet orchard. This complete reply ends with amber sunrise.

All 33 raw transcripts instead capitalize **Compass**. This is the only text difference; there are no lexical differences. The anchors `silver compass`, `quiet orchard`, and `amber sunrise` all appear in order. Thus 33/33 are normalized-exact, while 0/33 are case-sensitive raw-exact.

The unchanged [CPU content receipt](runs/pytorch-cpu-en-us-final-01/content.json), [MPS content receipt](runs/pytorch-mps-en-us-final-01/content.json), and [ONNX content receipt](runs/onnx-cpu-en-us-final-01/content.json) preserve every raw transcript, segment time, audio hash and comparison result. Local multilingual Whisper-small used CPU int8 with faster-whisper 1.2.1 and ctranslate2 4.8.2. The model file hashes and source-evidence hash are retained in each receipt. ASR covers generated source files; actual playback completion is independently recorded above.

The denominator is 15 + 9 + 9 successful phase clips, excluding the three cancelled phases. There are 25 distinct audio byte hashes: the nine ONNX outputs are byte-identical, while their receipts each record a fresh inference and file playback. These are repeated behavior observations, not 33 different utterances or voices.

## Earlier CPU failures retained

[CPU initial](runs/cpu-initial/evidence.json) failed before an observed inference call because the harness omitted the production `_seed_axis_defaults` initialization. Its mounted pane therefore selected `af_alloy` instead of configured `af_heart`; speed was `1.0`. The runner was subsequently corrected to seed the same defaults as production. This receipt records a harness setup failure, not a production voice-selection regression. Separate engine-switch/catalog defects are covered by the UI tests. Final cleanup joined with zero recorded owners.

The next run, named [cpu-fixed](runs/cpu-fixed/evidence.json), honored the configured voice and completed Speech Lab/Console warmup, but still failed: a second `KModel.forward` began 3.935 seconds after Stop. Both calls eventually joined before Stop returned, so this was unwanted continued generation rather than early ownership release. Its raw failure remains `Cancelled request started another native call`. The final CPU run passed with one overlapped call and zero post-Stop call entries. Both earlier receipts retain their original failure text, timings, source-map differences and runner hashes; neither is counted as a final matrix pass.

## Finite memory observations

| Runtime | Repeats | Repeat-settlement RSS range | RSS after cleanup |
| --- | ---: | ---: | ---: |
| PyTorch CPU | 12 | 1550.8–2508.9 MiB | 1342.4 MiB |
| PyTorch MPS | 6 | 1285.4–1286.0 MiB | 1286.0 MiB |
| ONNX CPU | 6 | 1329.0–1366.8 MiB | 1330.5 MiB |

The CPU observations fluctuate substantially. Across six MPS repeats, RSS increased by about 0.67 MiB, MPS allocated memory ranged from 538.0–539.0 MiB, and driver memory stayed at 2250.9 MiB. MPS allocator values remained nonzero after operational cleanup. These finite phase samples do not establish long-running leak freedom or complete allocator memory return. Exact before/phase/after values remain in the runtime receipts and [summary](summary.json).

## Provenance and package verification

The tested worktree was `tts-macos-burndown`, HEAD `2e3389e694e93592a1c66e5c3416bf29a1057d6c` plus the captured source changes. All six final before/after maps are identical across 2,253 source-file hashes. The [shared source manifest](provenance/source-hashes.json) binds those changes; HEAD alone is not the tested source identity. Final runner SHA256 is `159d70b3e56f64696715a6bf0e64b08d7c4f6324bb4b28c3d8c9324b9ecce4d2`. Runtime receipts retain Python 3.12.11, torch 2.14.0, kokoro 0.9.4, kokoro-onnx 0.6.1, onnxruntime 1.29.0, native-module hashes and exact model/voice/config asset hashes. The command's PyTorch model alias resolves to the shared local asset path recorded in the worker receipt.

To keep the package compact, runtime JSON preserves all original values while factoring only the two repeated source-hash maps into a relative shared manifest with explicit overrides/removals. Restoring those maps and removing the `packaging` field reproduces each original JSON object exactly. Content and matrix receipts are byte-preserving copies; their original evidence hashes still refer to the raw capture. [Package provenance](package-provenance.json) records both original and packaged hashes, and [verification](verification.json) records five semantic round trips, copy equality, content denominators, cancellation ordering, playback and resource checks.

Full raw runs remain local: [CPU final](/private/tmp/tts-macos-burndown/harness/pytorch-cpu-en-us-final-01), [MPS final](/private/tmp/tts-macos-burndown/harness/pytorch-mps-en-us-final-01), [ONNX final](/private/tmp/tts-macos-burndown/harness/onnx-cpu-en-us-final-01), [CPU initial](/private/tmp/tts-macos-burndown/harness/cpu-initial), and [cpu-fixed](/private/tmp/tts-macos-burndown/harness/cpu-fixed). Audio, models, binaries, private profiles, source archives and large logs are excluded. Packaging ran no inference or playback and changed no production code, scripts, tests or task status.
