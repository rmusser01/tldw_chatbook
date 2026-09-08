---
id: TASK-32016
title: 'Library: ingest-time speaker diarization via the sherpa-onnx engine'
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
The Library's after-the-fact speaker labels on imported media still run the torch-based `Local_Ingestion/diarization_service.py`. With the ONNX engine in the base install (TASK-31827), route that pass through sherpa-onnx too so a base install gets post-ingest speaker labels without the `diarization` extra; keep torch as the alternative.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `post_diarize` works on a base install (no torch) using the ONNX segmentation + embedder
- [ ] #2 Output shape and the transcript rewrite are unchanged
- [ ] #3 The torch path remains selectable
<!-- AC:END -->
