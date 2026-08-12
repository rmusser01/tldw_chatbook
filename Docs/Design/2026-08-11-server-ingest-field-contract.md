# Server ingest field contract (task-3309)

**Verified against a live tldw server on 2026-08-11** (`http://127.0.0.1:8000`,
`auth_mode = single_user`), using the running instance's own `/openapi.json`
plus the server source for `POST /api/v1/media/ingest/jobs`.

## The failure mode

The endpoint binds every form field with an explicit `Form(...)` in
`get_add_media_form` and never reads `request.form()`. FastAPI therefore
**discards any multipart field the endpoint does not declare** — no error, no
warning, and the submission still answers `200`. From the client this is
indistinguishable from success: the job runs, and the settings the user chose
simply did not happen.

`MediaIngestJobSubmitRequest` sets `extra="allow"`, which was read as "the
server accepts these". It does not mean that. It only means the *client* will
serialize them onto the wire.

## What was being dropped

Nineteen fields. Seven are pure spelling differences and are now translated
(`SERVER_FIELD_ALIASES`):

| client | server |
|---|---|
| `pdf_engine` | `pdf_parsing_engine` |
| `ocr` | `enable_ocr` |
| `ocr_language` | `ocr_lang` |
| `diarization` | `diarize` |
| `timestamps` | `timestamp_option` |
| `vad_filter` | `vad_use` |
| `language` | `transcription_language` |

Eleven have no server equivalent and are no longer sent
(`SERVER_UNSUPPORTED_OPTIONS`): `cookies_file`, `encoding`, `extraction_method`,
`include_toc`, `max_pages`, `processing_method`, `scrape_method`,
`transcription_model_dir`, `transcription_precision`, `transcription_provider`,
`translate_to_english`. `server_unsupported_options()` reports the ones a user
actually set, so the loss can be stated instead of inferred.

`cookies_file` is listed as unsupported for a different reason than the rest:
the server *does* have `cookies`, but that is a cookie **string** while the
canvas collects a **path** to a cookies.txt. Forwarding the path under that
name would put a filename where a cookie header belongs — worse than dropping.

The nineteenth is `force_regenerate_embeddings`, which
`ServerMediaReadingService.submit_ingest_jobs` named in its own signature and
sent on **every** submission. Undeclared, so never once honoured. It is no
longer sent, and a caller who asks for it is warned.

## Guard

`Tests/Library/test_server_ingest_field_contract.py` asserts that every field
reaching the wire — from the options loop *and* from the service's own
signature — is one the server declares. The declared list is captured in
`Tests/fixtures/server_ingest_jobs_form_fields.json` with its provenance;
refresh it by re-fetching `/openapi.json` from a server of the target version.

Mutation-checked: reverting the translation turns 5 of these tests red.

## Server-only options NOT surfaced (AC#2)

48 declared fields cannot be set from the Library's ingest canvas. The ones the
original audit called out as worth exposing are `generate_embeddings` (already
sent), `overwrite_existing`, `keep_original_file`, `custom_prompt`, and
`system_prompt`. The remaining 43 are mostly server-side chunking/embedding
/claims machinery with no local counterpart (`hierarchical_chunking`,
`contextual_llm_model`, `claims_extractor_mode`, `ocr_dpi`, the collection and
idempotency plumbing, and so on).

Surfacing them is deliberately **not** done here: it is new UI in server mode,
not a request-layer fix, and it needs a decision about which belong in a
client whose other backend cannot honour them. Tracked separately.
