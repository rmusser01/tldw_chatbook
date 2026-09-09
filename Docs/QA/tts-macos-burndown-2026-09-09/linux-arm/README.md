# Linux ARM Kokoro ONNX qualification — 2026-09-09

The fresh Linux ARM environment passed production `KokoroTTSBackend` WAV synthesis, cancellation during an actual ONNX session call, joined cleanup, and synthesis on the same backend afterward. Both complete source clips passed independent local ASR with the full expected text and all three content anchors. [The summary](summary.json) separates these achieved outcomes from the remaining physical Linux playback requirement in TASK-32153.

This is headless Linux ARM inference and decoding evidence. The Docker Desktop VM exposed no `/dev/snd`; no Linux Speech Lab or Console playback, output-device drain, shutdown after playback, or acoustic loopback was exercised. A Linux host with an actual output device remains necessary for that acceptance criterion.

| Phase | Outcome | Observed native duration | Complete audio |
| --- | --- | ---: | --- |
| Warmup | Success | 5.949 s | 8.405 s, mono PCM16 WAV, 24 kHz |
| Cancellation | Cancelled and joined | 9.290 s | Zero bytes emitted |
| Successor | Success | 5.802 s | 8.405 s, mono PCM16 WAV, 24 kHz |

## Actual cancellation and cleanup

The [passing runtime receipt](runs/run-02/evidence.json) records exactly one unchanged `onnxruntime.InferenceSession.run` call for the cancelled request. The observer delegates the original function and records entry/exit around its actual work; it adds no blocking gate, delay, tensor change, or early return. The call ran on a foreign worker thread with `CPUExecutionProvider` against the supplied model path.

Stop was requested 16.304 ms after native entry. The call exited before cancellation settled; Stop to settlement took **9.280 seconds**, with native exit 5.913 ms before settlement. The cancelled request started no later native call, emitted zero WAV bytes, and retained no worker or native interval at settlement. The successor began afterward on the same backend, and the cancelled phase record remained unchanged. Final backend close joined, cleared the model instance, and recorded zero native tasks, ONNX workers, and active calls. This demonstrates waiting for already-running native work to finish; it does not demonstrate immediate kernel interruption.

## Complete content

Expected text for warmup and successor:

> Silver compass opens this reply. The middle sentence describes a quiet orchard. This complete reply ends with amber sunrise.

The [ASR receipt](content/content.json) preserves all raw transcripts and segment times. Both transcripts capitalize **Compass**; that is the only text difference. The results are 2/2 exact after Unicode case, punctuation, and whitespace normalization, with `silver compass`, `quiet orchard`, and `amber sunrise` present in order. Lexical differences are not normalized away. ASR segment timestamps are decoder estimates; the WAV frame count determines duration.

The denominator includes both successful phase clips and excludes cancellation and the initial setup failure. Each success required a fresh model call, but the two audio files are byte-identical: 201,728 frames each, WAV SHA256 `8210f6331c1a3729049b9c3781877e722b673eccbfd7294b6459f67ac4967afe`. This covers one English utterance with voice `af_heart`, speed `1.0`; it is not a multilingual or endurance matrix.

Independent ASR ran afterward on macOS in the existing isolated Python 3.12 environment, using a supplied local multilingual Whisper-small snapshot, faster-whisper 1.2.1, and ctranslate2 4.8.2 with CPU int8. [Its command receipt](content/asr-command.json) records exit 0, 4.089 seconds elapsed, joined process ownership, and unchanged verifier hashes. The content receipt binds the original runtime receipt, source audio hashes, and every ASR model-file hash. It verifies generated source files; it does not establish physical playback.

## Initial setup failure and controlled repair

[Attempt 01](runs/run-01-setup-failure/evidence.json) failed before the first observed session call. Phonemizer copies the eSpeak shared library into temporary storage before loading it, and the private Docker tmpfs inherited `noexec`. Loading that copy failed with `failed to map segment from shared object`. The failed request still closed and joined, with all recorded resource counts zero.

The [library-loading controls](environment/tmpfs-library-controls.json) reproduced the failure without a model: the default mount reported `noexec` and failed, while adding `exec` allowed the same copied system library to load. Only the task-local launcher's private `/tmp` tmpfs option changed for attempt 02. The container root remained read-only and networking remained disabled. Both launch receipts preserve their original arguments and exits. Earlier import-only guard failures are retained separately under `environment/`; they are excluded from the live denominator.

## Environment and provenance

The [image receipt](provenance/image-inspect.json) identifies Linux arm64 image `sha256:c758cd09229cf4e17bb799063d5213f784b37d74ea8e2e312c9b438ac3fec08b`, built from the official Python 3.13.15 slim Bookworm base. The pinned base index and ARM64 manifest digests are retained in [provisioning](provenance/provisioning.json). The exercised runtime was Python 3.13.15 on aarch64, kokoro-onnx 0.6.1, onnxruntime 1.29.0, and `CPUExecutionProvider`; Torch was absent. Exact [Python packages](provenance/pip-freeze.txt), [Debian packages](provenance/dpkg-packages.txt), runtime/native-extension hashes, build inputs, and task-local validator snapshot are retained.

[The final launch](runs/run-02/launch.json) used the image content ID, no network, a read-only root, no added capabilities, no new privileges, four CPUs, 6 GiB memory, 256 PIDs, and a private 512 MiB temporary mount. Only the task evidence directory was a writable host mount. Source, models, and probe files were read-only mounts; no host credential files or credential environment were supplied. App configuration, cache, data, and blends used private output paths. The model and voice bundle were explicitly supplied existing files; inference could not download replacements.

The source was worktree `tts-macos-burndown`, HEAD `2e3389e694e93592a1c66e5c3416bf29a1057d6c` plus captured changes. All four before/after maps match across 2,253 source files; the [shared source manifest](provenance/source-hashes.json) identifies the actual tested content. Runtime receipts also bind the helper and validator hashes and both supplied asset hashes. HEAD alone does not describe the tested uncommitted changes.

## Release and package verification

[The release receipt](provenance/ownership-release.json) confirms the ASR PID was absent, all task-owned containers had been removed, runtime resources were zero, and the pre-existing `tldw_postgres_test` container remained running with the same ID and image. The exclusive model/audio slot was then released. No shared container or image was changed or removed.

To keep this package compact, only the four repeated source-hash maps are factored into one manifest. Restoring those maps and removing each `packaging` field reproduces both original runtime JSON objects exactly. Other copied receipts are byte-preserving. [Package provenance](package-provenance.json) records original paths/hashes and packaged hashes; [verification](verification.json) records copy equality, semantic round trips, full WAV hash/frame checks, ASR coverage, cancellation ordering, and cleanup checks.

Raw runs, WAVs, logs, and task-local provisioning remain at [/private/tmp/tts-macos-burndown/linux](/private/tmp/tts-macos-burndown/linux). Models, audio binaries, image layers, private profiles, and the build log are excluded from this repository package. Packaging itself ran no model, playback, container, or test server and changed no production code.
