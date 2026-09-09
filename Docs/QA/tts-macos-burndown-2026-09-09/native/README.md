# Native audio.cpp shutdown qualification — 2026-09-09

The isolated transport repair passed real macOS ARM CPU and Metal shutdown/recovery checks. This package preserves the portable patch and compact receipts for TASK-32147. Chatbook's approved runtime baseline remains unchanged; shipping the repair is tracked in [TASK-32163](<../../../../backlog/tasks/task-32163 - Ship-the-audio.cpp-request-drain-repair-in-approved-runtimes.md>).

The [patch](native-shutdown.patch) makes `serve_http` retain accepted requests, stop socket admission/I/O, and drain request lifetimes before returning or unwinding. It includes CMake registration and the new C++ test. Upstream base: `238ab6a9e321c17de8e120559f57efeedaeb1345`. Patch SHA256: `faa692effa31ef5b7836e2b727fda0a9452a19008a8c679686dceda1c1831dac`.

| Evidence | Result |
| --- | --- |
| [Baseline transport cases](transport/baseline-results.json) | Six of nine fail: active handler, streamed callback, accept-loop exception, idle client, partial header, partial body. Three controls pass. |
| [Patched transport cases](transport/fixed-results.json) | Nine of nine pass. |
| [Targeted CTest run](transport/targeted-native-tests.log) | Eleven of eleven pass: the same nine shutdown cases plus live-body and busy-guard regressions. These are overlapping checks, not twenty distinct cases. |
| [CPU01 runtime](runtime/cpu01/evidence.json) | Fault PID 83819 drains to exit 0 in 7.398 s; fresh PID 84059 completes playback and exits 0. |
| [Metal03 runtime](runtime/metal03/evidence.json) | Fault PID 91269 drains to exit 0 in 3.526 s; fresh PID 91327 completes playback and exits 0. |
| [Independent full-content ASR](content/full-content-asr.json) | All four complete source/device clips match the expected text exactly. |

Each native test SIGTERM followed an exact-PID sample containing Supertonic synthesis and ggml computation while the real application's HTTP response remained pending. Sample return to signal was 31.4 ms on CPU and 22.5 ms on Metal. Both runs released all application owners with no forced controller kill. The successor used the production Speech pane, trusted Console store/admission, TTS handler, managed supervisor, HTTP adapter and real output sink. This was not a full Console navigation test or a room-microphone recording; device WAVs contain the actual output callback PCM.

Expected and raw transcribed text for every clip:

> The recovered server speaks this complete reply. It finishes with the words amber sunrise.

The ordered beginning/middle/end anchors were `The recovered server`, `this complete reply`, and `amber sunrise`; lexical differences were empty. Source duration was 6.291 s on CPU and 6.288 s on Metal; both device clips were 6.300 s. The [four-clip input manifest](content/full-content-input.json) fixes the denominator. [ASR results](content/full-content-asr.json) preserve raw text, segment times and model/audio hashes; the [command receipt](content/full-content-asr-command.json) records local Whisper-small, CPU int8, exit 0, 16.004 s and unchanged verifier hash.

Metal01 observed shader compilation only; Metal02's HTTP response completed before the sample returned. Both observation gates refused the test signal, and cleanup exited 0. Metal03 used a longer bounded fault passage with the same strict compute/pending gate and unchanged short successor. These observation misses remain explicit in the [aggregate receipt](provenance/native-validation-summary.json).

## Provenance and retained raw evidence

The application worktree was `tts-macos-burndown` at tested HEAD `2e3389e694e93592a1c66e5c3416bf29a1057d6c`. Both successful runs preserved matching before/after manifests of 98 relevant files. [CPU build](provenance/build-cpu-receipt.json), [Metal build](provenance/build-metal-receipt.json), [patch manifest](provenance/native-patch-manifest.json), and runtime receipts record the binary/model/config hashes, binary UUIDs, PIDs and exact timings. [Package provenance](package-provenance.json) maps every copied artifact to its original path and SHA256; [verification](verification.json) records byte equality and fresh forward/reverse patch checks.

Full raw evidence stays local: [CPU compute sample](/private/tmp/tts-macos-burndown/native/runs/cpu-shutdown-recovery-01/01-owned-sigterm-during-native-compute-sample-4.txt), [Metal compute sample](/private/tmp/tts-macos-burndown/native/runs/metal-shutdown-recovery-03/01-owned-sigterm-during-native-compute-sample-3.txt), [CPU source/device audio directory](/private/tmp/tts-macos-burndown/native/runs/cpu-shutdown-recovery-01/artifacts), [Metal source/device audio directory](/private/tmp/tts-macos-burndown/native/runs/metal-shutdown-recovery-03/artifacts), [CPU build log](/private/tmp/tts-macos-burndown/native/build-cpu.log), [Metal build log](/private/tmp/tts-macos-burndown/native/build-metal.log), and [full native report](/private/tmp/tts-macos-burndown/native/native-validation-report.md). Binaries, models, audio, and large raw logs are excluded from this package.

## Sanitizer qualification and macOS limit

macOS ASan/UBSan remains unqualified. Nine original [sandbox timeouts](sanitizer/macos-sandbox-results.json) are preserved. A single [unsandboxed no-argument check](sanitizer/macos-unsandboxed-startup.json) also timed out before `main`, without starting transport fixtures. Its [raw stack](/private/tmp/tts-macos-burndown/native/transport-tests/fixed-sanitized-unsandboxed/startup-sample.txt) shows ASan shadow-memory initialization allocating through dyld metadata, re-entering ASan via malloc, and waiting in `StaticSpinMutex::LockSlow`. Apple clang 17.0.0 on macOS 26.5.2 ARM64 needs a runtime/toolchain that completes startup before sanitizer behavior can be qualified. This is startup evidence, not nine transport defects. The exact diagnostic child was killed and joined.

The initial [Linux ARM prerequisite check](sanitizer/linux/results.json) used the supplied pinned image `sha256:c758cd09229cf4e17bb799063d5213f784b37d74ea8e2e312c9b438ac3fec08b`. Neither `cc` nor `c++` was present, so zero builds or transport cases ran. No packages were provisioned in that first check. The unique network-none container mounted native source read-only and was removed; [its controller receipt](sanitizer/linux/controller.json) preserves image/command/source provenance and a documented correction to the observer's case-sensitive parsing of Docker's `no such object` response.

After separate authorization to provision an isolated build-tools image, [TASK-32164 Linux sanitizer qualification](sanitizer/linux-qualified-02/README.md) passed all nine unchanged transport cases under ASan/UBSan, with no diagnostics. The new image is pinned, the two-CPU network-none container was removed, and all 507 source inputs and the native patch remained unchanged. macOS sanitizer startup remains a documented platform/toolchain limitation.

All task-owned servers, playback, ASR and diagnostic processes were joined. No preexisting container was touched. Sanitizer qualification is tracked in [TASK-32164](<../../../../backlog/tasks/task-32164 - Qualify-native-shutdown-under-a-working-sanitizer-toolchain.md>), separately from TASK-32163 approved-runtime rollout. Root owns Backlog status and integration. No production source or runtime baseline was modified by this evidence packaging.
