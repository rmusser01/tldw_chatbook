---
id: TASK-3309
title: >-
  Surface server-only ingest options and verify forwarded extras against a live server
status: To Do
assignee: []
created_date: '2026-08-07 19:30'
labels:
  - library
  - ingest
  - parity
  - server
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the 2026-08-07 parity audit. Two server-path gaps: (1) the server media service/schema accepts `generate_embeddings`, `force_regenerate_embeddings`, `overwrite_existing`, `keep_original_file`, and `custom_prompt`/`system_prompt`, none surfaced in the Library ingest UI's server mode; (2) the local-option fields the client forwards as `extra="allow"` request fields (`pdf_engine`, `ocr`, `transcription_*`, `timestamps`, `diarization`, `encoding`) have unverified server-side honoring — statically unresolvable, needs a live tldw server run recording which extras take effect.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Live-server verification records, per forwarded extra, whether the server honors it; dishonored extras are removed from the request or annotated in the UI
- [ ] #2 Server-only options worth exposing are surfaced in server mode (or their exclusion recorded)
<!-- AC:END -->
