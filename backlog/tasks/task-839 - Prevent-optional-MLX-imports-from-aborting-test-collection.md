---
id: TASK-839
title: Prevent optional MLX imports from aborting test collection
status: Done
assignee:
  - '@codex'
created_date: '2026-07-27 02:06'
updated_date: '2026-07-27 21:33'
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
- [x] #1 Unrelated test modules collect without importing or initializing optional MLX backends
- [x] #2 Unavailable or unsafe MLX backends resolve through a bounded optional-dependency failure instead of aborting Python
- [x] #3 Focused regression coverage reproduces the current config import path without temporary PYTHONPATH stubs
- [x] #4 TASK-553.15 verification no longer needs parakeet_mlx or lightning_whisper_mlx stubs
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented cheap MLX package discovery with two cached first-use loaders in transcription_service.py. Lightning file transcription and Parakeet file, buffer, and streaming model creation now load native runtimes only when explicitly used; import failures are chained as TranscriptionError and disable same-process retries.

Added subprocess and loader lifecycle/path regression coverage, then removed the three ProductionApp sys.modules stubs. Scoped verification passed: 20 import/config/ProductionApp tests, 6 exact legacy MLX availability/loading/cache tests, Ruff check/format for all five touched code/test files, and git diff --check. The planned broad -k not_available selector was narrowed to exact node IDs because it also selected an unrelated soundfile test that initializes the unsafe native runtime. Repository-wide tests were intentionally not run per user direction.

ADR required: no. No provider ownership, citation behavior, routing/defaults, schema, dependency, security, or license boundary changed.

PR review follow-up: rebased onto the latest dev, routed first-use MLX imports through Utils.optional_deps.require_dependency() while keeping that helper import local to preserve startup laziness, and removed the misleading pre-load callable debug log. The dedicated regressions now fail on direct importlib bypasses and guard against restoring that stale log.
<!-- SECTION:NOTES:END -->
