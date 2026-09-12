# Linux CUDA TTS qualification plan

> **For agentic workers:** Use subagent-driven-development for the bounded validator change; run hardware qualification serially. Steps use checkbox syntax.

**Goal:** Qualify Kokoro CUDA, Chatterbox CUDA, and physical Linux Kokoro ONNX playback on the available RTX 3090 and Logi USB headset.

**Architecture:** Exercise existing registered backends, mounted Speech Lab, and trusted Console delivery. Observe real native calls and output-device drain; preserve original audio and independently transcribe complete successful clips. Keep profiles, environments, models and evidence under a dedicated remote directory.

**Tech stack:** Debian 13, Python 3.12, PyTorch CUDA, Kokoro, Chatterbox, ONNX Runtime, PipeWire/ALSA and the existing opt-in validation scripts.

**Spec:** TASK-32153, TASK-32157 and TASK-32158 acceptance criteria in backlog/tasks.

ADR required: no
ADR path: N/A; existing ADR-023, speech-settings ADR-039 and speech-Lab ADR-040 apply.
Reason: Qualification and opt-in observation tooling preserve production provider and ownership contracts. Reassess before changing those contracts.

## Constraints

- Use isolated worktree `codex/tts-linux-cuda`, based on `8ab21ecaf3`.
- Remote task directory: `/home/ml-user/tts-linux-cuda-20260912`; no real profile mutation or system dependency replacement.
- CUDA execution must identify actual model placement; CPU fallback cannot count as CUDA evidence.
- Cancellation must overlap actual native execution and record settlement before cleanup/successor; GPU observations must account for asynchronous execution.
- Use the Logi USB headset explicitly, keeping inferred hardware submission separate from human listening confirmation.
- Keep synthetic test text/reference audio, exact versions and hashes, all failed attempts, complete content checks, and process/device cleanup evidence.
- Run only targeted tests. No full suite or unrelated checkout changes.

## Task 1: Extend the existing opt-in Kokoro runner for CUDA

Files: `scripts/validate_live_tts.py`, `Tests/TTS/test_live_validation_harness.py`, `Docs/Development/TTS/Live_Validation.md`.

- [x] Add red targeted tests for CUDA CLI admission, unavailable CUDA rejection, real-device identity, synchronized native completion and allocated/reserved/peak device memory observations.
- [x] Preserve ONNX's CPU restriction and the current CPU/MPS behavior. Add the smallest CUDA observation helpers needed; do not change production inference.
- [x] Run `python -m pytest Tests/TTS/test_live_validation_harness.py Tests/TTS/test_live_tts_input_validation.py -q`, relevant lint, and independent review.

## Task 2: Provision and qualify Kokoro CUDA and ONNX

- [x] Install managed Python 3.12 and task-owned virtual environments; pin CUDA Torch and public model assets. Save install logs and package manifests.
- [x] Build a clean wheel from the worktree and verify source/wheel/installed Python hashes before running it.
- [x] Run the existing validator with `--engine pytorch --device cuda --voice af_heart --format wav --scenarios playback,cancel,repeat --repeats 3 --play-audio`, explicit local model/voice paths, isolated package root and a new output directory.
- [x] Run the analogous `--engine onnx --device cpu --format mp3` tuple using pinned local ONNX model/voices.
- [x] Transcribe every full successful clip with `scripts/verify_live_tts_content.py`; retain begin/middle/end coverage, native overlap, repeated settled memory, device PCM and drained shutdown evidence.
- [ ] Coordinate a complete known utterance through the headset with the user and record the actual listening response separately.

## Task 3: Qualify the registered Chatterbox CUDA runtime

- [x] Reuse the previous Chatterbox qualification recipe where available; adapt only observation/configuration for Linux/CUDA and the current installed wheel.
- [x] Record pinned runtime/model/reference provenance, actual device placement, real Lab/Console clips, inference-overlap Stop, successor and three bounded repeated runs.
- [x] Use synthetic reference speech and independent full-clip ASR; preserve discrepancies instead of weakening expected text.
- [x] Verify model/native work and output devices settle before close; retain cleanup and process-exit evidence.

## Task 4: Close out evidenced work

- [x] Curate credential-free records under `Docs/QA/tts-linux-cuda-2026-09-12/` with immutable failure records and source/runtime/model manifests.
- [ ] Run targeted verification for changed tooling and review the diff. Update backlog criteria only where evidence establishes them; keep any remaining criteria open.
- [x] Release task-owned processes and device handles, retain reproducible assets/evidence, and report remaining validation limits.

## Discovered fix: TASK-32505

ADR required: no. Existing ADR-023 initialization/resource ownership applies; this is a lifecycle correction, preserving asynchronous initialization and the registered subprocess/native fallback boundaries.

- [x] Retain startup, process acquisition and readiness ownership so close cannot return before a late child or native loader settles.
- [x] Preserve nonblocking initialization, join cancellation-safe close, and prove queued startup, gated spawn, late readiness and fallback behavior with targeted regressions.
- [x] Rebuild the installed wheel and repeat Chatterbox CUDA qualification on base `8ab21ecaf3` plus the corrected backend.
- [ ] Complete final review and repeat installed-wheel qualification after rebasing onto the newer dev audio-player changes.

## Execution record

- Task 1: 62 targeted tests pass in independent review and root rerun; Ruff and diff checks pass. No production backend changes. CUDA controls were red before implementation.
- Kokoro CUDA and Linux ONNX: both first hardware attempts passed runtime lifecycle checks and all six successful clips per tuple passed full normalized-text ASR. PipeWire observed all 12 playback streams on the Logi sink. Human listening confirmation remains pending.
- Wheel: 2,352 application/profile-core Python files match source, wheel and installed package. Base source is `8ab21ecaf3`; the separately hashed runner carries the CUDA observation changes.
- Setup adjustment: current dev requires Python 3.12. Task-owned Python 3.11 environments were provisioned initially but never used for application qualification. Kokoro ONNX requires NumPy 2 while Chatterbox requires NumPy below 2; separate environments preserve these constraints. The failed dependency resolution log is retained.
- Chatterbox observation decision: qualify the registered child-process implementation, retaining the original process script and recording the task-owned observation launcher. Cancellation terminates/reaps that process in production; report process termination separately from synchronized natural inference completion.
- CUDA ASR briefly overlapped ONNX CPU validation. Neither run is a throughput benchmark; lifecycle, full-content and device-routing observations remain the intended evidence.
- Chatterbox setup run 01 loaded and executed the real CUDA model but failed admission expectations because a local reference stem was not a catalogued global voice. The UI correctly selected `default`; the supported Lab reference picker must be exercised instead. Retain this attempt as a setup failure.
- Chatterbox baseline run 02 passed default-voice Lab/Console playback, actual `t3.inference` cancellation with child termination/reap and NVIDIA release, successor and three repeats. All six successful clips passed full-text ASR. Repeat after TASK-32505 rather than calling this corrected-source evidence.
- A standalone preflight incorrectly assumed awaiting `initialize()` awaited readiness. Closing immediately afterward exposed a separately reproduced lifecycle defect: unretained startup/readiness work can publish resources after close. TASK-32505 covers the confirmed production race; the precise cause of the preflight's event-loop shutdown hang is not established by the synthetic reproductions.

- Corrected Chatterbox run 03: seven successful complete clips (six default plus one supported synthetic-reference Lab audition), all seven streams on Logi sink 55. Actual inner CUDA inference was observed; Stop terminated/reaped its child, NVIDIA release preceded settlement/successor, and three repeats retained stable allocated/reserved GPU samples. All final process/GPU/audio owners cleared; user configuration/default sink were unchanged.
- Whisper small passed 6/7 on run 03 and preserved “Sylph or Compass” on repeat 03. Whisper medium passed all seven unchanged recordings against the same receipt/audio hashes; this does not erase the recognizer disagreement or replace pending human listening.
- TASK-32505: nine lifecycle regressions plus 30 existing delivery tests passed (39); real installed queued-startup/close passed with no pending tasks and exit 0. Corrected wheel/source/install identity matched all 2,352 Python files. Combined targeted verification later passed 101 tests. Final review and later-head runtime evidence remain pending.
- Chatterbox package: 51 copied records independently match recorded hashes, including failed attempts, child observations, both content reports and reproducible recipes. See `Docs/QA/tts-linux-cuda-2026-09-12/chatterbox/README.md`.

## Rebase verification

Rebased onto `a766133fc4`, preserving both independent lesson entries. A landed task already owned ID 32494, so our lifecycle fix moved to TASK-32505 with provenance after sweeping 57 origin refs and 41 worktrees. Rebuilt the wheel and verified all 2,344 Python files against current source and both installed environments.

All three repeated runtime checks passed. Twenty observed streams across nineteen successful clips reached Logi sink 55, and all cleanup/process/GPU/audio checks passed. Kokoro CUDA and ONNX each passed 6/6 full-text checks with Whisper medium. Chatterbox passed 7/7 with Whisper small; medium recovered only the opening sentence from its reference clip. Both same-audio reports and the original orchestration exit 1 are retained. No synthesis replacement was made, and human listening criteria remain open.

Rebased targeted tests: 217 passed. Ruff/format/diff/backlog/profile census checks passed. See `Docs/QA/tts-linux-cuda-2026-09-12/rebased/README.md`.
