---
id: TASK-32157
title: Qualify Kokoro on a real CUDA host
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-09 05:51'
updated_date: '2026-09-12 15:25'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
This macOS ARM host has no CUDA-capable NVIDIA device. CPU, MPS and Metal evidence cannot establish CUDA runtime behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A provisioned NVIDIA host records real CUDA device, driver, framework and model provenance for the supported Kokoro execution path.
- [ ] #2 Lab and repeated Console synthesis preserve complete speech through playback or an explicitly paired real client device with independent content evidence.
- [x] #3 Cancellation overlaps actual CUDA inference, ownership and device work finish before close, and successor playback and bounded repeated runs succeed.
- [x] #4 The opt-in validator supports CUDA and records actual device placement, native completion and settled GPU memory without reporting CPU fallback as CUDA qualification.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A; existing ADR-023 and the speech ADR-039/040 apply. Reason: qualify the existing supported CUDA runtime and extend opt-in evidence tooling only; reconsider if production contracts change.
1. Pin source, Python, CUDA framework and public model assets in task-owned environments on the RTX 3090 host.
2. Extend the existing Kokoro live validator to accept CUDA with real device identity and synchronized memory/native-work evidence; verify targeted validator tests.
3. Run mounted Lab and repeated trusted Console generation with full decoded speech, real inference-overlap Stop, joined cleanup and successor.
4. Verify full content independently, physical Logi playback and bounded device memory; preserve failures and mark only evidenced criteria.
5. Retain source/runtime/model hashes and cleanup receipts; update QA and backlog, review any scoped changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-09-12 Linux RTX 3090 update: clean dev source 8ab21ecaf3, Python 3.12.8 and Torch 2.6.0+cu124 qualified Kokoro CUDA. Mounted Lab, Console warmup, native-overlap Stop, successor and three repeats all passed; all six successful clips passed full normalized-text ASR. All observed playback streams used the Logi USB headset. All owners settled, worker exited 0 and NVIDIA compute process list was empty. CUDA tooling passed 62 targeted tests and independent review. Evidence: Docs/QA/tts-linux-cuda-2026-09-12/README.md. AC2 remains open pending human listening confirmation; no acoustic-quality or full-shell claim. No production backend changes. ADR required: no; existing speech ADR-023/039/040 apply.
<!-- SECTION:NOTES:END -->
