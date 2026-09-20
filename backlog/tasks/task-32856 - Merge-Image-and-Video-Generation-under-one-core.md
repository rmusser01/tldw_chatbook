---
id: TASK-32856
title: Merge Image_Generation and Video_Generation under one modality-parameterized core
status: To Do
assignee: []
created_date: '2026-09-19 08:24'
labels:
  - core-review
  - review-cascade
dependencies: []
parent_task_id: TASK-32850
references:
  - qa/cascade-review-2026-09-19/report.md
  - backlog/decisions/044-ephemeral-generated-video-storage-playback-and-streaming.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Video_Generation/` (4,083 LOC) is a structural clone of `Image_Generation/` (6,704 LOC): `adapter_registry.py` differs by 63 normalized lines of ~309; `config.py` duplicates ~250-280 lines of TOML/keyring/secret-precedence machinery (video's docstring admits the mirroring); worker and request-validation skeletons are shared; the ComfyUI adapters share ~50% lifecycle, and video already imports image's transport (`comfyui_video_adapter.py:26-32`). ADR-044 decision 2 mandates the mirror, but its own Consequences pre-sanction this merge: "Any future modality… should evaluate merging Image_Generation/Video_Generation under a Media_Generation umbrella rather than adding a third near-duplicate package."

The merge polarity matters: image carries hardening video lacks (config snapshot context + runtime lock in `adapter_registry.py:34-36,:100,:145`; ComfyUI `_BodyChunkSupervisor`/`_SendSupervisor`/`/object_info` schema validation). Image's stricter behavior wins per site, or the exception is an explicit tested decision — never a silent average. ADR-044 decisions 1/3/7 (ephemeral store, opt-in uploads, `metadata_json`) are orthogonal and preserved; `video_store.py` has no image twin and is not in scope.

ADR required: yes — supersedes ADR-044 decision 2 only; draft it before implementation. Est. net deletion 900–1,400 LOC plus mirrored test scaffolding.

Source: cascade review 2026-09-19 — `qa/cascade-review-2026-09-19/report.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A superseding ADR is recorded and linked from ADR-044 (merge under a modality-parameterized core; decisions 1/3/7 intact)
- [ ] #2 One core owns the adapter registry, config machinery, worker skeleton, and validation skeleton; per-modality backends are data tables plus thin adapters
- [ ] #3 Every image-vs-video behavioral delta (snapshot context, runtime lock, ComfyUI supervisors, `/object_info` validation) resolves to image's stricter behavior or is recorded as an explicit tested exception
- [ ] #4 The 26 image and 15 video test files pass unchanged in behavior as the regression base
- [ ] #5 Net deletion landed; size ratchets re-pinned in the same PR
<!-- AC:END -->
