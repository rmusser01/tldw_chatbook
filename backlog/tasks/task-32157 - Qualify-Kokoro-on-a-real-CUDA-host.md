---
id: TASK-32157
title: Qualify Kokoro on a real CUDA host
status: To Do
assignee: []
created_date: '2026-09-09 05:51'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
This macOS ARM host has no CUDA-capable NVIDIA device. CPU, MPS and Metal evidence cannot establish CUDA runtime behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A provisioned NVIDIA host records real CUDA device, driver, framework and model provenance for the supported Kokoro execution path.
- [ ] #2 Lab and repeated Console synthesis preserve complete speech through playback or an explicitly paired real client device with independent content evidence.
- [ ] #3 Cancellation overlaps actual CUDA inference, ownership and device work finish before close, and successor playback and bounded repeated runs succeed.
<!-- AC:END -->
