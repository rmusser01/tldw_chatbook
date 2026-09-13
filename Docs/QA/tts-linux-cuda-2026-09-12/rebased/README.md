# Validation after rebasing onto dev

These runs use dev `a766133fc4` plus TASK-32505's unchanged Chatterbox lifecycle fix, including the newly landed audio-format selection/admission changes. [Wheel identity](provenance/rebased-wheel-identity.json) records SHA256 `2f166ef75edcbcfc7af20354b41219f8c3fed84b7b1a26b3b2075261d2b6be50`; the 2,344 application Python files match that worktree, clean source, wheel and both installed environments. This historical scan omitted the nine profile-core files because their sources live under `packages/tldw_profile_core/src/`; the final review rerun checks both complete packages. Earlier runs remain separate historical evidence.

| Runtime | Full-content evidence | Stop to return | Observed Logi streams |
| --- | --- | --- | --- |
| [Kokoro CUDA](runs/kokoro-cuda-02/evidence.json) | [Whisper medium 6/6](runs/kokoro-cuda-02/content-medium.json) | 0.168 s | 6 |
| [Kokoro ONNX CPU, MP3](runs/kokoro-onnx-02/evidence.json) | [Whisper medium 6/6](runs/kokoro-onnx-02/content-medium.json) | 2.629 s | 6 |
| [Chatterbox CUDA](runs/chatterbox-cuda-04/evidence.json) | [Whisper small 7/7](runs/chatterbox-cuda-04/content-small.json); [medium 6/7](runs/chatterbox-cuda-04/content-medium.json) | 0.136 s, child terminated | 8 |

Every runtime passed mounted Lab, trusted Console warmup, inference-overlap Stop, successor and three repeats. Chatterbox additionally used the actual Lab synthetic-reference picker. The [summary](provenance/rebased-summary.json) records all final resource counts zero, workers/children absent, no NVIDIA compute processes or sink inputs, and unchanged user configuration/default sink. All 20 observed streams used Logi sink 55 across 19 successful clips. Chatterbox had one additional, single-sample transient stream near repeat02; the raw routing summary retains it. No human listening or acoustic capture is claimed.

## Retained recognizer disagreement

The initial medium recognizer recovered only “Silver Compass opens this reply.” from the complete Chatterbox reference clip. Consequently the original [qualification orchestration exited 1](provenance/rebased-qualification.log.txt), although all three runtime commands exited 0. The independently applied small recognizer recovered the entire expected text from all seven unchanged recordings. Both reports bind the same evidence hash and per-clip audio hashes; the disputed clip is `73cbc2ff1fbc397387373b7eee12810ce0f80b833102d7926d028b02ab1ceabb`. Neither recording nor expected text was replaced. This is a documented recognizer disagreement, not unanimous ASR acceptance or proof of a synthesis defect. Human listening remains outstanding, especially for this reference clip and the earlier run03 third repeat.

The small-model follow-up used the same `guard-asr.py` and `source-rebased/scripts/verify_live_tts_content.py` command as [the orchestration recipe](recipes/qualify-rebased.py.txt), with model `models/whisper-small` and output `runs/chatterbox-cuda-04/content-small.json`. Its model hashes are in that report and its exit code is retained. Model generation, playback and ASR ran serially in this repeat. Prior runtime/model/reference provenance and observer limitations still apply.

## Local verification

217 targeted tests passed after the rebase: live validation/input admission, Chatterbox initialization/audio delivery, audio-player formats, playback capability, speech-format adaptation and Console autoplay. Ruff check passed for the changed runner/tests; all four changed Python files passed formatting. The production Chatterbox file introduces no Ruff findings relative to its baseline. No diagnostic statement changed in that file. Backlog IDs and the profile-owned-path census passed. This was a targeted run, not the full suite.

[Package manifest](package-manifest.json) contains 18 byte-preserving copies. Raw audio and environments remain on the Linux host for listening/reproduction. TASK-32153/32157/32158 listening criteria remain open. TASK-32505 repairs ownership under existing ADR-023; no new ADR was required.
