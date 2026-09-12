# Final Qodo fixes: Linux playback qualification

These runs qualify source `d2610bfc6ab3a87de0804842ddd18438f556e382`, based on dev `71313cccd8bf13cbea0d25d341764463629375a4`, including all three Qodo fixes: shutdown admission, shared device validation and guarded optional PyTorch loading. The [wheel identity](provenance/qodo-final-wheel-identity.json) records SHA256 `9359e4a11342f7b36644ecfa99c7ef4c16078945ae6620306e481d9320ea9892`. Complete source, wheel and installed-file sets match: 2,341 application files plus nine profile-core files, checked in both runtime environments before and after execution.

| Runtime | Runtime result | Independent full-text ASR | Observed Logi streams |
| --- | --- | --- | --- |
| [Kokoro CUDA run03](runs/kokoro-cuda-03/evidence.json) | Passed | [Whisper medium 6/6](runs/kokoro-cuda-03/content-medium.json) | 6 |
| [Kokoro ONNX CPU MP3 run03](runs/kokoro-onnx-03/evidence.json) | Passed | [Whisper medium 6/6](runs/kokoro-onnx-03/content-medium.json) | 6 |
| [Chatterbox CUDA run05](runs/chatterbox-cuda-05/evidence.json) | Passed | [Medium 6/7](runs/chatterbox-cuda-05/content-medium.json), [small 7/7](runs/chatterbox-cuda-05/content-small.json) | 7 |

Each runtime passed mounted Lab, trusted Console warmup, Stop during actual inference, successor and three repeats. Chatterbox additionally exercised the supported synthetic-reference Lab picker. The [summary](provenance/qodo-final-summary.json) retains all 19 streams on Logi sink 55, joined cleanup with every resource counter zero, absent workers/children, and empty NVIDIA compute and audio-stream lists. User configuration and the system default sink were unchanged. Pinned model/runtime provenance and observer limits in the [earlier report](../README.md) and [Chatterbox report](../chatterbox/README.md) still apply.

Both Chatterbox recognizers were scheduled before this run. Medium omitted the opening sentence of the reference clip; small recovered all three sentences from the identical recording, SHA256 `75f343ce85ea2543013075cbfcf3e743970a644ad3129fb188db6136ecac6dad`. Both reports bind the same runtime evidence. The [orchestration exit 1](provenance/qodo-final-qualification.exit) deliberately preserves that disagreement even though every runtime command exited 0. This does not establish a synthesis defect or unanimous recognizer acceptance. No clip or expected text was replaced.

The first final-build check stopped before model execution because the source scanner omitted profile-core's nested source directory. Its [failure](provenance/attempt1-qodo-qualification.log.txt), manifest and recipe remain intact. The corrected recipe checks both package roots and requires complete set/hash equality. The historical rebased 2,344-file scan covered only the application; it did not verify the nine profile-core files.

Local verification: 229 targeted TTS tests passed, including isolated-worker encoding and two regressions that reproduced generation starting after shutdown admission closed. Independent review passed 83 focused cases and verified this 22-record package, source identity, cancellation, routing, content denominators and cleanup. Changed runner/tests pass Ruff; all five changed Python files pass formatting, and production Ruff findings do not increase against the baseline. Diagnostic inventory, profile-owned-path census, task-ID guard and diff checks pass. No full-suite run was requested.

The [manifest](package-manifest.json) lists 22 byte-preserving copies. Raw audio, models and environments remain on the Linux host for replay. Human listening is recorded separately; receipt/ASR review does not establish acoustic quality. Existing ADR-023 governs the routine lifecycle fix; no new ADR was required.

## Human listening confirmation

On 2026-09-12 the user requested a replay, then confirmed **“All four were clear and complete”** through the Logi headset. The four retained final recordings were Kokoro CUDA Console repeat01, Kokoro ONNX MP3 Console repeat01, Chatterbox default Console repeat01, and the disputed Chatterbox reference Lab clip. Each ffplay replay exited 0. [The listening record](human-listening.json) binds the files and hashes and retains the exact question/answer. This confirms the reference clip includes the opening, middle and ending; the medium ASR disagreement remains preserved. It is human listening of these four recordings, not a claim that the user audited every historical repeat or a microphone capture. Combined with the mounted runtime/device-drain evidence, TASK-32153/32157/32158 qualification criteria are complete.

Final integration rebase onto dev `ea406f4d41114f806a7909e866a1722d2240d014` adds only the branch-protection baseline document. The complete 2,350-file application/profile-core map and validator hash still equal the tested source above; runtime receipts remain immutable.
