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
  - Docs/superpowers/plans/2026-07-27-lazy-mlx-import-boundary.md
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Detailed plan:
`Docs/superpowers/plans/2026-07-27-lazy-mlx-import-boundary.md`

1. Add failing subprocess and loader-lifecycle regressions for non-importing
   MLX discovery, first-use caching, and bounded import failure.
2. Replace module-level native imports with `find_spec` flags and two explicit
   lazy loaders.
3. Route Lightning file transcription and Parakeet file, buffer, and streaming
   model loads through those loaders.
4. Remove the three ProductionApp `sys.modules` stubs and run only the
   affected import, provider, MLX-loader, and app tests.
5. Run touched-file Ruff and diff checks, review scope, and close the task.

ADR required: no
ADR path: N/A
Reason: This defers existing optional imports to their point of use without
changing provider ownership, dependencies, storage, schema, or service
contracts.
<!-- SECTION:PLAN:END -->
