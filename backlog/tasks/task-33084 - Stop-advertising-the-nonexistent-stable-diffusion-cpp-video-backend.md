---
id: TASK-33084
title: Stop advertising the nonexistent stable-diffusion-cpp video backend
status: To Do
assignee: []
created_date: '2026-09-27 19:46'
labels: [bug, video]
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Video_Generation/adapter_registry.py registers a stable_diffusion_cpp backend whose adapter module does not exist (the task-3401.7 backend never shipped). Because resolution is lazy, enabling the backend succeeds at registration and then fails at generation time with an import error, which is the worst place for a user to discover it. This is drift from the image/video mirror: the image side ships the adapter, the video side registered the name anyway.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The video registry no longer advertises stable_diffusion_cpp until the backend actually ships.
- [ ] #2 A registration test fails when a registered adapter path cannot be imported.
- [ ] #3 Targeted Video_Generation tests pass.
<!-- AC:END -->
