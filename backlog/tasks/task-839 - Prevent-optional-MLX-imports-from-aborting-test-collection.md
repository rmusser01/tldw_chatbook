---
id: TASK-839
title: Prevent optional MLX imports from aborting test collection
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-27 02:06'
updated_date: '2026-07-27 20:25'
labels:
  - testing
  - optional-deps
  - stt
dependencies: []
references:
  - Docs/superpowers/specs/2026-07-27-lazy-mlx-import-boundary-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep optional Parakeet and Lightning MLX backends from aborting Python during unrelated test collection when the native MLX runtime is installed but unsafe to initialize. Tests and non-STT application imports must degrade through the existing optional-dependency path instead of requiring temporary module stubs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Unrelated test modules collect without importing or initializing optional MLX backends
- [ ] #2 Unavailable or unsafe MLX backends resolve through a bounded optional-dependency failure instead of aborting Python
- [ ] #3 Focused regression coverage reproduces the current config import path without temporary PYTHONPATH stubs
- [ ] #4 TASK-553.15 verification no longer needs parakeet_mlx or lightning_whisper_mlx stubs
<!-- AC:END -->
