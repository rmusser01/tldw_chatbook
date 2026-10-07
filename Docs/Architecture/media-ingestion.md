# Media ingestion pipeline

This document describes how source material becomes indexed media: source classification, the parse-without-DB worker model, extraction backends per type, the single DB write seam, chunk stamping, and the handoff into semantic indexing.

## Authoritative files

| File | Role |
| --- | --- |
| `Local_Ingestion/local_file_ingestion.py` | The pipeline: `classify_ingest_source`, `canonicalize_url`, `parse_local_file_for_ingest`, `persist_parsed_media`, `ingest_local_file`, `batch_ingest_files`, `ingest_directory` |
| `Local_Ingestion/ingest_parse_worker.py` | The spawn parse pool — parse payloads are picklable and DB-free |
| `Local_Ingestion/web_article_ingestion.py` | Article extraction (httpx + trafilatura, streamed, size-capped, SSRF-guarded) |
| `Local_Ingestion/video_processing.py` + `audio_processing.py` + `transcription_service.py` | yt-dlp download → audio extract → STT; whisper/parakeet artifacts |
| `PDF_Processing_Lib.py`, `Document_Processing_Lib.py`, `Book_Ingestion_Lib.py`, `Image_Processing_Lib.py` (+`OCR_Backends.py`) | Per-type extraction (lazy-loaded) |
| `Chunking/` (`Chunk_Lib.py`, `engine/`, `token_chunker.py`, `language_chunkers.py`, `auto_selection.py`, `template_runtime.py`) | Chunking methods and template runtime |
| `RAG_Search/ingestion_indexing.py` | Post-commit semantic indexing hook (see [rag.md](./rag.md)) |
| `Library/` (`library_ingest_state.py`, `ingest_capabilities.py`) + `DB/Library_Ingest_Jobs_DB.py` | The real queue and job ownership |
| `Widgets/NewIngest/` | The unified ingest UI (`UnifiedProcessor`, smart drop zone, processing dashboard) |

## The two-phase design

**Parsing never touches the database.** `parse_local_file_for_ingest` runs in a worker process with no DB handle and returns a picklable payload; `persist_parsed_media` is the **only DB writer** in the pipeline. A parse crash therefore cannot corrupt rows, and extraction backends (PDF, document, ebook, image/OCR libraries) are lazy-loaded per type inside the worker only.

## Ingest flow

1. **Admission**: the source is classified — `article`, `audio`, or `video` (video hosts: youtube/youtu.be/vimeo/dailymotion); URLs are canonicalized and egress-guarded (`Utils/egress.guarded_fetch_httpx`); local files are typed by extension.
2. **Extraction** per type: web articles (trafilatura, streamed with a size cap; permanent vs retryable errors distinctly typed), video (yt-dlp download → audio extraction → STT; transcripts keyed by whisper model), audio (STT), PDF, ebooks, images (OCR), plaintext. Empty extraction hard-fails (`_reject_empty_extraction`).
3. **Optional LLM analysis** runs in-band during parse; a literal `"Error: …"` analysis body is forbidden — failures degrade to a recorded `analysis_failed_reason`.
4. **Persist** (single writer thread): `add_media_with_keywords` — Media row (unique url/content-hash dedup, `overwrite` flag, Library-only `restore_trashed`), keyword links, Python-side FTS sync, chunks into `UnvectorizedMediaChunks` with `chunk_engine_version` and `chunking_template`/`chunking_params` stamped on every chunk, optional `DocumentVersions`.
5. **Semantic indexing**: the post-commit hook enqueues the new item for background chunk → embed → vector upsert (see [rag.md](./rag.md)). The `generate_embeddings=False` path suppresses indexing for that write.

## Media type mapping

Source classes normalize into `Media.type` ∈ pdf / document / ebook / plaintext / html / image / audio / video. Chunking methods available per item include semantic, tokens, paragraphs, sentences, words, and ebook chapters; templates live in the `ChunkingTemplates` table (with six server built-ins). The chunking subsystem itself is documented in `Docs/Design/Chunking/`.

## Failure behaviors

| Case | Behavior |
| --- | --- |
| Permanent URL error (4xx, bad content type) | Typed permanent failure — no retry |
| Transport error (timeout, 5xx) | Typed retryable failure |
| Empty extraction | Hard fail with the reason |
| Duplicate (same url/content hash) | Skipped unless `overwrite`; trashed duplicates restored only by the Library writer |
| LLM analysis failure | Warning + `analysis_failed_reason`; item still ingests |
| Parse worker crash | No DB effects (parse is DB-free) |
| SSRF-shaped fetch | Refused by the egress guard |

## Governing decisions

ADR-013 (library ingest ownership and job lifecycle), ADR-014 (ingest service authority and recovery), ADR-030 (derived-index lifecycle and atomic media migrations), ADR-061 (parse progress channel), ADR-065 (active source admission and override), ADR-055 (destructive action reversibility). User guide: `Docs/User_Guide/library/`, `Docs/User_Guide/meetings.md`.

## Verified gotchas

1. Audio/video processors are constructed with `media_db=None` during parse — an earlier shared-handle design caused real double-write/dedup bugs.
2. The UI's `Widgets/NewIngest/BackendIntegration.py` is partially simulated/WIP; the real queue and ownership live in `Library/` + `Library_Ingest_Jobs_DB`.
3. `tldw_api/` is the typed client for the separate tldw **server**, not part of the local pipeline.
4. Chunk stamps (`chunk_engine_version`, template/params) are load-bearing for re-index decisions — never write chunks without them.
5. `Chat/document_generator.py` generates LLM documents **from conversations**; it is not a media export path.

## Related docs

- [rag.md](./rag.md) — what happens after the post-commit hook
- [database-layer.md](./database-layer.md) — the Media DB schema and write seam
