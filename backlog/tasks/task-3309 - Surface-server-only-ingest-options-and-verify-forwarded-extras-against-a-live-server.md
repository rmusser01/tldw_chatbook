---
id: TASK-3309
title: >-
  Surface server-only ingest options and verify forwarded extras against a live server
status: Done
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
- [x] #1 Live-server verification records, per forwarded extra, whether the server honors it; dishonored extras are removed from the request or annotated in the UI
- [x] #2 Server-only options worth exposing are surfaced in server mode (or their exclusion recorded)
<!-- AC:END -->

## Implementation Plan

1. Read the live server's own contract for `POST /api/v1/media/ingest/jobs`
2. Diff every field the client forwards against what the server declares
3. Translate the names that differ only in spelling; stop sending the rest
4. Guard the contract with a test that fails when a new field is added blind
5. Record the server-only options rather than build new UI in this task

## Implementation Notes

**AC#1 — answered, and the answer was worse than "unverified".** The extras were
not being ignored *sometimes*; nineteen of them never reached the server at all.
The endpoint binds every form field with an explicit `Form(...)` in
`get_add_media_form` and never reads `request.form()`, so FastAPI discards any
undeclared multipart field silently and still answers 200. `extra="allow"` on
`MediaIngestJobSubmitRequest` had been read as "the server takes these"; it only
means the client will serialize them.

Verified two ways against a live server (`127.0.0.1:8000`, `auth_mode =
single_user`): the running instance's own `/openapi.json`, which enumerates
exactly 72 declared fields, and the server's source for the handler. Per-field
effect observation via real submissions was not possible — the instance rejects
the API key in `~/.config/tldw_cli/config.toml` as invalid, and its key is
env-only on the server process. That limits nothing about the finding: a field
the endpoint never binds cannot take effect, whatever a submission would show.

Seven were pure spelling differences and are now translated
(`SERVER_FIELD_ALIASES`): `pdf_engine`->`pdf_parsing_engine`, `ocr`->`enable_ocr`,
`ocr_language`->`ocr_lang`, `diarization`->`diarize`, `timestamps`->
`timestamp_option`, `vad_filter`->`vad_use`, `language`->`transcription_language`.
Eleven have no equivalent and are no longer sent, with
`server_unsupported_options()` reporting the ones the user actually set so the
loss is stated rather than inferred. `cookies_file` is in that set for a
different reason: the server's `cookies` is a cookie *string*, the canvas holds
a *path*, so forwarding it would put a filename where a header belongs.

The nineteenth was found by widening the check past the options loop:
`ServerMediaReadingService.submit_ingest_jobs` named
`force_regenerate_embeddings` in its own signature and sent it on **every**
submission, undeclared and so never once honoured.

**AC#2 — recorded, not built.** 48 declared fields cannot be set from the
canvas. The handful worth exposing (`overwrite_existing`, `keep_original_file`,
`custom_prompt`, `system_prompt`, user-settable `generate_embeddings`) need a
decision this task should not make alone: the canvas serves both backends and
none of these exist locally, so they are either mode-dependent controls or
always-present-and-gated. Filed as **task-15513** with that question stated;
the full inventory is in the design doc.

**Two existing tests were pinning the bug.** `test_per_type_options_are_passed_
through_for_the_detected_group` and `test_document_group_options_travel_for_docx`
asserted that `pdf_engine` and `processing_method` travel verbatim -- i.e. they
had turned the silent drop into a requirement. Updated to the corrected
contract rather than worked around.

**Guard.** `Tests/Library/test_server_ingest_field_contract.py` asserts every
field reaching the wire -- from the options loop and from the service signature
-- is one the server declares, against a fixture captured from the live server
with its provenance. Mutation-checked: reverting the translation turns 5 of
these red.

Added: `Tests/Library/test_server_ingest_field_contract.py`,
`Tests/fixtures/server_ingest_jobs_form_fields.json`,
`Docs/Design/2026-08-11-server-ingest-field-contract.md`.
Modified: `tldw_chatbook/Library/server_ingest_request.py`,
`tldw_chatbook/Media/server_media_reading_service.py`,
`Tests/Library/test_server_ingest_request.py`.
