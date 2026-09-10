---
id: TASK-32158
title: Qualify Chatterbox on a real CUDA host
status: To Do
assignee: []
created_date: '2026-09-09 05:51'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Chatterbox CPU and MPS qualification leaves the CUDA-specific loading, inference, cancellation and memory paths untested; no NVIDIA device is available here.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A provisioned NVIDIA host initializes the registered Chatterbox runtime with exact model and CUDA device provenance.
- [ ] #2 Real Lab and repeated Console clips play completely, including the supported synthetic-reference path, with independent content evidence.
- [ ] #3 Cancellation during CUDA inference joins native work before cleanup, successor requests succeed, and bounded repeated runs record settled device memory and ownership.
<!-- AC:END -->
