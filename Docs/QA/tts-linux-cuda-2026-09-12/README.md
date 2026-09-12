# Linux CUDA and physical-device TTS qualification

The latest [rebased-source validation](rebased/README.md) repeats all three runtimes on dev `a766133fc4` plus TASK-32505, including upstream playback-format changes. Runtime/cleanup checks pass; Chatterbox has a retained ASR disagreement and human listening remains pending. The results below document the earlier source separately.

Kokoro CUDA, Kokoro ONNX CPU and the corrected Chatterbox CUDA wheel passed mounted Speech Lab playback, trusted Console delivery, Stop during native inference, successor playback and three repeated replies on Debian 13.6. Chatterbox also passed a separate synthetic-reference Lab audition. Every successful source clip passed independent full-text ASR; its retained small-recognizer discrepancy and medium-recognizer rerun are detailed in the Chatterbox report. PipeWire observed every playback stream on the Logi USB headset. Human listening confirmation is still pending; no acoustic recording was made.

| Runtime | Playback paths | Successful clips / full-text passes | Stop to return | Process exit |
| --- | --- | --- | --- | --- |
| [Kokoro PyTorch CUDA](runs/kokoro-cuda-01/evidence.json) | WAV: Lab ffplay, Console PortAudio/Pulse sink | [6 / 6](runs/kokoro-cuda-01/content.json) | 0.166 s | 0 |
| [Kokoro ONNX CPU](runs/kokoro-onnx-01/evidence.json) | MP3: Lab and Console ffplay | [6 / 6](runs/kokoro-onnx-01/content.json) | 2.905 s | 0 |
| [Chatterbox CUDA, corrected wheel](chatterbox/README.md) | WAV: default Lab/Console plus synthetic-reference Lab | [7 / 7, Whisper medium](chatterbox/runs/chatterbox-cuda-03/content-medium.json) | 0.137 s; child terminated | 0 |

These are finite checks of English speech at speed 1.0: Kokoro `af_heart`, Chatterbox `default`, and one synthetic reference. They do not establish full-shell navigation, other voices/languages, every output format, kernel-level profiling, long-duration leak freedom or acoustic quality. The existing Linux ARM headless qualification remains a separate result in [TASK-32153's earlier evidence](../tts-macos-burndown-2026-09-09/linux-arm/README.md).

## Source and runtime identity

The Kokoro runs used clean source revision `8ab21ecaf3`, built into wheel SHA256 `f07dd76f6fb4e5b6712bd61407a644de3432533ff175b83ada3da5eced3c83c1`. All 2,352 Python files across `tldw_chatbook` and `tldw_profile_core` matched source, wheel and installed package before execution; [the complete map](provenance/wheel-identity.json) retains the file hashes. Each runtime receipt also records package hashes before/after and the separately modified CUDA validator's hash. No production backend was altered for these Kokoro runs.

[Host provenance](provenance/host-before.json) records kernel `6.12.107+deb13-amd64`, NVIDIA driver `610.57.04`, and the RTX 3090's physical identity. The actual Kokoro model ran on `cuda:0`, compute capability 8.6, with Torch `2.6.0+cu124`. The ONNX session recorded `CPUExecutionProvider`. Python is 3.12.8, Kokoro 0.9.4, kokoro-onnx 0.6.1, onnxruntime 1.29.0 and sounddevice 0.5.6; [the environment freeze](provenance/kokoro-freeze.txt) preserves the complete selection.

[Kokoro assets](provenance/kokoro-assets.json) are pinned to public revision `f3ff3571791e39611d31c381e3a41a3af07b4987`. [Reused-asset hashes](provenance/reused-assets-manifest.json) bind the ONNX model, voice binary and Whisper-small snapshot transferred from the prior qualification. The ONNX hashes equal those in the earlier Linux ARM provenance. Models were prepared before runtime; inference denied network access and runtime installation. Each run used a new private app profile and data directory.

## Cancellation, playback and cleanup

Each Stop overlapped one real native call. CUDA `KModel.forward` entered before Stop; its host return and successful CUDA completion barrier preceded Stop's return. ONNX `InferenceSession.run` likewise finished before Stop returned. Neither cancelled request started another native call or opened an audio sink. All recorded request owners were zero at settlement, and each successor started afterward.

The CUDA native observations synchronize before entry and after host return. They include observation overhead and prove completion ordering, not that a GPU kernel occupied every instant of that interval. Actual model placement is checked separately. Three CUDA repeat settlements held approximately 585–590 MB of live tensor allocations with 1.571 GB reserved. Allocator caches remained after operational cleanup; the worker then exited and disappeared from NVIDIA's compute-process list. This is bounded retention evidence, not an assertion that closing the service frees every allocator block immediately.

Complete file-player elapsed times and Console PCM callback/drain observations are retained in the runtime receipts. [CUDA routing](provenance/kokoro-cuda-01-routing-summary.json) recorded one ffplay and five PortAudio/Pulse streams; [ONNX routing](provenance/kokoro-onnx-01-routing-summary.json) recorded six ffplay streams. All observed streams used PipeWire sink 55, mapped in host provenance to the Logi headset's stable USB sink name. The system's default sink was not changed. Routing observation began during the first CUDA Lab clip and covered all subsequent streams; raw quarter-second observations remain on the host.

Both runtime workers exited 0 with all final recorded resource counters zero: [CUDA process receipt](provenance/kokoro-cuda-01-process-exit.json), [ONNX process receipt](provenance/kokoro-onnx-01-process-exit.json). Both independent ASR commands exited 0. ASR had a positive-control audit denying access to the real user profile. CUDA ASR briefly overlapped ONNX CPU qualification; no throughput or uncontended-latency claim is made.

## Full content

Expected text:

> Silver compass opens this reply. The middle sentence describes a quiet orchard. This complete reply ends with amber sunrise.

All twelve complete clips matched after normalization, with beginning/middle/end anchors in order. Raw transcripts, segment times, encoded-audio hashes, recognizer versions and model hashes remain in the content receipts. Case/punctuation normalization is explicit; original transcripts are preserved. ASR checks source audio, while routing and drain evidence independently check device delivery.

## Setup, retention and remaining work

Current dev requires Python 3.12. Initially provisioned task-owned Python 3.11 environments were not used for application qualification. The first application dependency resolution failed because kokoro-onnx 0.6.1 requires NumPy 2 while the attempted pin was 1.26.4; [the failure log](provenance/runtime-install-attempt1.log) is retained. Kokoro and Chatterbox use separate environments to honor their incompatible NumPy constraints. No failed synthesis attempt is hidden behind these setup corrections.

[Package manifest](package-manifest.json) records 28 original paths, sizes and hashes; these are byte-preserving copies, including the runtime evidence consumed by ASR. [Independent review and targeted verification](verification.json) cover copy identity, content denominator, source identity, cancellation ordering and playback completion. Large models, wheels, binary audio, private profiles and raw routing logs remain under `/home/ml-user/tts-linux-cuda-20260912` on the authorized Linux host. Reproduction recipes are in `recipes/`; the reusable CUDA validator remains `scripts/validate_live_tts.py`.

[Chatterbox qualification](chatterbox/README.md) adds 51 verified records, a corrected wheel containing TASK-32505, seven full-content passes with Whisper medium, and the unchanged Whisper-small 6/7 discrepancy. Its final process/GPU/audio observations are empty and the user configuration/default sink are unchanged. These receipts qualify base `8ab21ecaf3` plus the documented fix; [rebased results](rebased/README.md) provide the later-source evidence. Human listening confirmation remains pending. Backlog completion must preserve these limits. ADR required: no; existing ADR-023, speech-settings ADR-039 and Speech Lab ADR-040 apply.
