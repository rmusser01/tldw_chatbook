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

## Proven live

With a valid `SINGLE_USER_API_KEY`, against the running server:

```
POST /api/v1/media/ingest/jobs  -F pdf_parsing_engine=bogus
  -> 422  {"loc": ["body","pdf_parsing_engine"], "msg": "Input should be
           'pymupdf4llm', 'pymupdf' or 'docling'"}

POST /api/v1/media/ingest/jobs  -F pdf_engine=bogus      # the client's name
  -> 200  {"batch_id": "...", "jobs": [{"id": 285, "status": "queued"}],
           "errors": []}
```

The same 422-naming-the-field response confirms `diarize`, `enable_ocr`,
`vad_use` and `timestamp_option` are read. An invalid value under the server's
name is rejected; the same value under the client's name is accepted and
ignored.

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

Nine are not accepted by this endpoint and are no longer sent. Calling them
"no server equivalent" was **wrong**, and in a way that matters — for several,
the server can do the thing and simply does not take the instruction here.
`server_unsupported_options()` returns a reason with each name:

| option | why not sent |
|---|---|
| `transcription_provider` | the transcription **core** supports it (`transcribe_audio(transcription_provider=...)`); no media endpoint exposes it |
| `translate_to_english` | the core supports it (`task="translate"` in `stt_provider_adapter`); no media endpoint exposes it |
| `transcription_precision` | server-side config (faster-whisper `compute_type`), not a per-request field |
| `transcription_model_dir` | the server resolves its model directory from its own config |
| `extraction_method` | accepted on `/media/process-ebooks`, but not on the ingest-jobs API |
| `cookies_file` | shape mismatch: the server's `cookies` is a cookie **string**, the canvas holds a **path** |
| `encoding`, `include_toc`, `processing_method` | no counterpart in the server's media path |

**`scrape_method` and `max_pages` are deliberately NOT in that set.** They are
accepted — by `/media/ingest-web-content` — and the web group never reaches this
builder anyway: it raises `ServerIngestUnsupported` and routes through
`build_web_clip_kwargs`, which already sends both correctly.

### A server-side gap worth raising

`transcription_provider` and `translate_to_english` are the notable ones: the
server is *capable* and its HTTP API does not surface the switch. Neither
`/media/add` nor `/media/ingest/jobs` nor the `process-*` endpoints accept them.
That belongs in the server's own backlog, not this client's.

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

Endpoint surfaces compared (so the narrow one is not mistaken for the whole
API): `/media/add` is a strict **subset** of `/media/ingest/jobs` (68 vs 72
fields), so the client is already on the fullest general surface. The
`process-*` endpoints add only a handful each — `extraction_method` on
process-ebooks, `vlm_*`/`proposition_*` on process-pdfs, `api_provider`/
`model_name` on the audio/video/document ones.

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
