---
id: TASK-32158
title: Qualify Chatterbox on a real CUDA host
status: Done
assignee:
  - '@codex'
created_date: '2026-09-09 05:51'
updated_date: '2026-09-12 16:51'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Chatterbox CPU and MPS qualification leaves the CUDA-specific loading, inference, cancellation and memory paths untested; no NVIDIA device is available here.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A provisioned NVIDIA host initializes the registered Chatterbox runtime with exact model and CUDA device provenance.
- [x] #2 Real Lab and repeated Console clips play completely, including the supported synthetic-reference path, with independent content evidence.
- [x] #3 Cancellation during CUDA inference joins native work before cleanup, successor requests succeed, and bounded repeated runs record settled device memory and ownership.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A; existing ADR-023 and speech ADR-039/040 apply. Reason: real qualification of existing CUDA execution, without changing provider ownership or supported model contracts.
1. Use task-owned Python/CUDA environment and pin registered Chatterbox model plus synthetic reference provenance.
2. Exercise real mounted Lab and trusted Console generation, complete physical Logi output and independent full-content ASR.
3. Observe actual CUDA inference-overlap Stop, joined ownership/device work, successor and repeated settled memory.
4. Preserve failure attempts and exact source/runtime/assets, verify cleanup, and close only evidenced acceptance criteria.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Qualified the registered Chatterbox subprocess on RTX 3090 with Python 3.12.8, Chatterbox 0.1.7 and Torch 2.6.0+cu124. Run 03 uses base 8ab21ecaf3 plus the Chatterbox initialization lifecycle fix; all 2,352 source/wheel/installed Python files match and every t3/s3gen/ve parameter is float32 on cuda:0. Seven successful WAV clips cover default Lab, trusted Console warmup/successor/three repeats, and the supported synthetic-reference Lab picker. All seven streams routed to Logi sink 55. Stop overlapped real t3.inference; production terminated/reaped the child and NVIDIA PID disappearance was observed before settlement/successor, without claiming natural native return or earlier GPU-release timing. Repeated allocated/reserved memory was stable and final processes/GPU/audio owners cleared.
Whisper medium passed all seven unchanged recordings. The original Whisper-small 6/7 report retains its “Sylph or Compass” discrepancy on repeat 03; no text or audio was rewritten. AC2 remains open for human listening confirmation. Failed setup/preflight and pre-fix run 02 remain preserved; later rebased source needs separate runtime evidence. QA and 51 verified copied records: Docs/QA/tts-linux-cuda-2026-09-12/chatterbox/README.md. User configuration/default sink were unchanged. ADR required: no; existing ADR-023 and speech ADR-039/040 apply. Task remains In Progress.

Chatterbox run04 passed runtime and small ASR 7/7; medium ASR 6/7 reference discrepancy retained on rebased dev a766133fc4. Current-source identity and final device/process cleanup are recorded in Docs/QA/tts-linux-cuda-2026-09-12/rebased/README.md. Human listening criterion remains open.

Chatterbox CUDA run05: runtime pass, medium ASR 6/7 and small 7/7; reference opening disagreement preserved. Final reviewed source d2610bfc matches complete 2,350-file source/wheel/install maps; all playback streams used Logi and resource/process/device cleanup passed. See Docs/QA/tts-linux-cuda-2026-09-12/qodo-final/README.md. User requested a replay; four final recordings were replayed with ffplay to the selected headset, each exit 0. Human confirmation remains pending.

Human playback confirmation completed 2026-09-12: the user heard the four final-run replays through Logi and answered “All four were clear and complete,” including the Chatterbox reference opening/middle/ending. Exact replay files/hashes and response are in Docs/QA/tts-linux-cuda-2026-09-12/qodo-final/human-listening.json. Combined with mounted Lab/Console repeats, full-content ASR, device drain and cleanup evidence, all qualification ACs are complete. Historical recognizer discrepancies remain intact. This is bounded qualification of the documented voices/formats, not every language or long-duration stability. Targeted tests, static checks and independent review passed; existing ADR-023/039/040 apply.
<!-- SECTION:NOTES:END -->
