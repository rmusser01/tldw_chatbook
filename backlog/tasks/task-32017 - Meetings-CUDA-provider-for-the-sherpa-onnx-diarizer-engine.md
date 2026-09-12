---
id: TASK-32017
title: 'Meetings: CUDA provider for the sherpa-onnx diarizer engine'
status: To Do
assignee: []
created_date: '2026-09-07 23:50'
labels:
  - audio
dependencies:
  - TASK-31827
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The ONNX engine runs CPU-only. sherpa-onnx ships CUDA wheels for Linux and Windows x64; add an opt-in provider setting and the install guidance, measured against the CPU numbers in Docs/STT_Evaluation/task-31827/.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A `[meetings] onnx_provider = cuda` setting selects the GPU when the CUDA wheel is installed and falls back to CPU otherwise
- [ ] #2 The bake-off harness can run with the CUDA provider and records the speed-up
<!-- AC:END -->
