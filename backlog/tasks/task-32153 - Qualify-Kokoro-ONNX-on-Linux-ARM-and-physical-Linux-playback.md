---
id: TASK-32153
title: Qualify Kokoro ONNX on Linux ARM and physical Linux playback
status: To Do
assignee:
  - '@codex'
created_date: '2026-09-09 05:50'
updated_date: '2026-09-09 07:40'
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
- [ ] #3 A Linux host with an actual output device plays the complete Lab and Console clips and verifies device drain and shutdown; a headless container does not satisfy this criterion.
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
<!-- SECTION:NOTES:END -->
