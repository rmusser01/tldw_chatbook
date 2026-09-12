---
id: TASK-32153
title: Qualify Kokoro ONNX on Linux ARM and physical Linux playback
status: Done
assignee:
  - '@codex'
created_date: '2026-09-09 05:50'
updated_date: '2026-09-12 16:51'
labels: []
dependencies: []
documentation:
  - Docs/QA/tts-macos-burndown-2026-09-09/linux-arm/README.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Docker provides a Linux ARM CPU VM for runtime checks, but native physical Linux audio remains unqualified. Record those levels separately.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A fresh isolated Linux ARM environment initializes pinned Kokoro ONNX assets and synthesizes complete known speech through the production backend.
- [x] #2 Real session execution overlaps cancellation and joined cleanup, followed by a successful successor; source, runtime, model and full audio provenance are retained.
- [x] #3 A Linux host with an actual output device plays the complete Lab and Console clips and verifies device drain and shutdown; a headless container does not satisfy this criterion.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A; existing ADR-023 provider/runtime ownership and ADR-039/040 Global/Studio/Lab behavior apply.
Reason: Evidence-only provider qualification; reassess and file a scoped fix before changing provider contracts or runtime support policy.

1. Pin public primary-source runtime and asset requirements, preflight resources, and provision only task-owned environments, caches or containers.
2. Run a bounded real initialization and synthesis attempt while coordinating the exclusive inference/audio slot with other TTS validation.
3. Exercise production application routing, complete playback where the platform exposes a device, cancellation and successor behavior. Preserve exact logs, sources, model hashes and resource joins.
4. Run targeted tests for any discovered application defect and retain explicit prerequisite evidence for any unmet criterion; do not mark Done while required hardware or provider validation is missing.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Qualified headless Linux ARM production Kokoro ONNX inference in a fresh task-owned Python 3.13.15 image with kokoro-onnx 0.6.1, onnxruntime 1.29.0, CPUExecutionProvider and no Torch. Exact image ID, base digests, package lists, source/runtime/model hashes, launch arguments and raw evidence provenance are retained in Docs/QA/tts-macos-burndown-2026-09-09/linux-arm/README.md and its linked receipts.

Warmup and same-backend successor each produced a complete 8.405-second, 24 kHz mono PCM16 WAV. Both successful clips passed separate local Whisper-small ASR against the entire expected text and ordered beginning/middle/end anchors; the only raw difference was Compass capitalization. Stop overlapped one actual unchanged InferenceSession.run call, started no subsequent native call for the cancelled request, waited 9.280 seconds for joined settlement, emitted zero bytes, and left all recorded resources zero. Final backend close joined. ASR exited 0 and its PID was absent; all task-owned containers were removed and the unrelated pre-existing PostgreSQL container remained running unchanged.

The initial live attempt is retained as a setup failure before any native call: phonemizer could not dlopen its copied eSpeak library from Docker's noexec temporary mount. A no-model mount/library control failed with the original option and passed with exec; only the task-local private tmpfs option changed for the successful run. Read-only container root, network none, isolated app paths and read-only source/model mounts remained in force. No production code was changed for this qualification.

Verification checked 27 byte-preserving copied receipts, two exact semantic round trips after shared source-map factoring, all four matching maps across 2,253 source files, complete WAV frame/PCM/encoded hashes, ASR denominator 2/2, cancellation ordering, successor ordering, launch isolation and cleanup. No full test sweep was run for this evidence-only change.

AC1 and AC2 are achieved. AC3 remains open: the headless Docker VM exposed no /dev/snd and did not exercise Linux Speech Lab/Console playback, output-device drain, or shutdown after physical playback. Status is To Do pending a Linux host with an actual output device and the complete Lab/Console/device validation. This evidence must not be reported as physical Linux playback qualification.

ADR required: no. Existing ADR-023 provider/runtime ownership and ADR-039/040 Global/Studio/Lab behavior apply; this evidence-only work changes no provider contract or runtime support policy.

2026-09-12 physical Linux device update: the Debian 13 RTX 3090 host completed real Kokoro ONNX CPU MP3 Speech Lab and trusted Console playback, Stop during InferenceSession.run, successor and three repeats. Six of six successful clips passed full-text independent ASR. PipeWire observed all six ffplay streams on the selected Logi USB headset; complete file playback and zero owners at settlement were verified, worker exited 0. Evidence: Docs/QA/tts-linux-cuda-2026-09-12/README.md. This advances the earlier headless-only result, but AC3 and task status remain open pending human listening confirmation. No acoustic capture or full-shell navigation claim.

Linux ONNX MP3 run02 passed runtime and medium ASR 6/6 on rebased dev a766133fc4. Current-source identity and final device/process cleanup are recorded in Docs/QA/tts-linux-cuda-2026-09-12/rebased/README.md. Human listening criterion remains open.

ONNX CPU MP3 run03: runtime pass and medium ASR 6/6. Final reviewed source d2610bfc matches complete 2,350-file source/wheel/install maps; all playback streams used Logi and resource/process/device cleanup passed. See Docs/QA/tts-linux-cuda-2026-09-12/qodo-final/README.md. User requested a replay; four final recordings were replayed with ffplay to the selected headset, each exit 0. Human confirmation remains pending.

Human playback confirmation completed 2026-09-12: the user heard the four final-run replays through Logi and answered “All four were clear and complete,” including the Chatterbox reference opening/middle/ending. Exact replay files/hashes and response are in Docs/QA/tts-linux-cuda-2026-09-12/qodo-final/human-listening.json. Combined with mounted Lab/Console repeats, full-content ASR, device drain and cleanup evidence, all qualification ACs are complete. Historical recognizer discrepancies remain intact. This is bounded qualification of the documented voices/formats, not every language or long-duration stability. Targeted tests, static checks and independent review passed; existing ADR-023/039/040 apply.
<!-- SECTION:NOTES:END -->
