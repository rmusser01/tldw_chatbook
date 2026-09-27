---
id: TASK-33084
title: Stop advertising the nonexistent stable-diffusion-cpp video backend
status: Done
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

## Implementation Plan

Verify the finding against origin/dev before changing code.

ADR required: no
ADR path: N/A
Reason: no code change resulted; the existing ADR-176 (one-media-generation-core) already governs this registry design.

## Implementation Notes

REFUTED on origin/dev — closed with no production change. The premise was formed against a stale branch (chore/task-32482-done, 1872 commits behind dev). On dev:

- The VideoAdapterRegistry class docstring documents all three DEFAULT_ADAPTERS specs as deliberate skeletons for backend tasks 3401.3/.6/.7, with lazy resolution failing cleanly at generation time by design, under ADR-176.
- Tests/Video_Generation/test_adapter_registry.py::test_default_adapters_point_at_local_package pins the exact key set {minimax, comfyui, stable_diffusion_cpp}, and ::test_lazy_specs_do_not_import_until_get_adapter pins the not-yet-importable contract explicitly.
- AC #1 and #2 would reverse a shipped, tested design decision and are withdrawn. The drift-prevention intent behind AC #2 is already satisfied: any accidental phantom entry changes the pinned key set and fails the existing test.

