---
id: TASK-1744
title: 'Fix the sys.modules patch.dict leak in the MLX transcription test fixture'
status: To Do
assignee: []
created_date: '2026-08-01 11:40'
labels: [tests, transcription, hygiene]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/Local_Ingestion/test_transcription_service_lazy_mlx.py`'s `service_module` fixture patches
`sys.modules` via `patch.dict` in a way that leaks between imports: the module under test is only
imported once per run today, so the defect is latent rather than active. The identical bug was found
and fixed in the sibling file `test_transcription_service_parakeet_buffer_wav.py` while addressing
PR #1171 review findings (merged as 74ddf7c6a); that fix was deliberately not extended to this file
to keep the PR scoped. Left alone, it will bite the first time this file needs a second import in
one process -- the failure mode is an import-order-dependent error that looks unrelated to its cause
(numpy's "cannot load module more than once per process" was the shape it took in the sibling).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 The fixture no longer leaks patched entries in `sys.modules` between tests or imports.
- [ ] #2 A test proves the fixture is safe across two imports of the module under test in one process (fails against the current fixture).
- [ ] #3 The file passes in isolation and when run alongside `test_transcription_service_parakeet_buffer_wav.py` in the same process.
<!-- AC:END -->
