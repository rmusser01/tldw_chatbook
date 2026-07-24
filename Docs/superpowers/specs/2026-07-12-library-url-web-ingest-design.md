# URL / web source local ingest (task 162)

**Status:** Design approved (brainstorm), pending spec review + a tldw_server pipeline review.
**Backlog:** task-162 — "URL / web source local ingest".
**Builds on:** F3 parallel-parse ingest (#594), the heavy-lane cap (#612), persistent job history (#613).

## Problem

Library ingest accepts local file paths only — a URL typed into the ingest form is rejected at `validate_path_simple(require_exists=True)` before it ever reaches the pipeline. We want to ingest a URL: a web page → article extraction, an audio/video URL → transcription, both flowing through the existing parse-pool → persist pipeline.

## Goal / Acceptance

- **AC1** — the user can ingest a URL into the Library.
- **AC2** — web extraction routes through the existing parse/persist pipeline.

## Chosen approach

A URL flows through the SAME submit → parse-pool → persist pipeline as a file. A unified classifier decides the source type at submit (for the heavy-lane cap + routing); the parse worker routes a URL to either the existing audio/video transcription branches (media URLs — they already accept URLs) or a new lightweight `httpx` + `trafilatura` article extractor (web pages, no browser). Both produce the same `result`/payload dict the writer already persists.

## Components

### 1. Unified classifier — `classify_ingest_source(source: str) -> str`

New helper in `tldw_chatbook/Local_Ingestion/local_file_ingestion.py`, wrapping `detect_file_type`:
- If `source` is an http/https URL (`urlparse(source).scheme in ("http", "https")`):
  - a known video host (`youtube.com`/`youtu.be`/`vimeo.com`/…) or a URL path ending in a video extension → `"video"`;
  - a URL path ending in an audio extension → `"audio"`;
  - otherwise → `"article"`.
  (Pattern heuristic — no network at submit; ambiguous URLs default to `"article"`. Mirrors the existing `video_processing.py` host heuristic. **Limitation:** a media URL from an unknown host with no extension is misclassified as an article and will fail extraction — acceptable v1.)
- Else → today's extension-based `detect_file_type(source)` (unchanged).

`submit_library_ingest_job` uses this instead of `detect_file_type`, so `detected_type` = article/audio/video/pdf/… . **The heavy-lane cap (task 160) applies for free:** a media URL → `detected_type ∈ {audio, video}` → heavy lane; an article → `"article"` → light.

### 2. URL-aware `parse_local_file_for_ingest` (the integration seam)

Today the function does `file_path = Path(file_path)` early and the shared payload tail hardcodes `'url': f"file://{file_path.absolute()}"`. Both break for a URL. Change:
- At the top, detect a URL (`urlparse(source).scheme in ("http","https")`) BEFORE any `Path()` conversion; keep the URL as a `str`.
- Compute `source_url` once: the real URL for a URL source, else `f"file://{Path(file_path).absolute()}"`.
- Route by the classifier: URL + `"article"` → `extract_article_for_ingest(url, options)` (§3, returns a `result` dict); URL + `"audio"`/`"video"` → the EXISTING audio/video branches, passing the URL string as the processor input (unchanged processor code — it downloads + transcribes URLs today); a file → today's extension dispatch.
- The shared tail uses `source_url` for the payload `'url'` and does not call `.absolute()` on a URL. `media_metadata['file_path']`/payload `'file_path'` hold the URL string for a URL source (metadata only; not forwarded to the DB).

### 3. New article extractor — `extract_article_for_ingest(url, options) -> dict`

Sync, pool-worker-friendly (no Playwright/browser). New module `tldw_chatbook/Local_Ingestion/web_article_ingestion.py` (heavy imports deferred inside the function — the worker's module scope stays stdlib-only):
- `httpx.Client` GET with: `timeout=30.0`, `follow_redirects=True`, a browser-like `User-Agent`, and a **max-response-size guard** (stream + abort past ~10 MB, so a giant/binary response can't exhaust a worker).
- Reject a non-HTML `Content-Type` (e.g. `application/pdf`, `video/*`) with a clear permanent error (the classifier default-to-article can land here).
- `trafilatura.extract(html, include_comments=False, include_tables=False, ...)` for the article text (markdown) + `trafilatura.extract_metadata(html)` for metadata.
- **Metadata (borrowed from tldw_server, which leaves these on the table):** pull `title`, `author`, `date`, AND the free extras `sitename`, `hostname`, `language` off trafilatura's metadata into the payload `metadata` dict.
- **Boilerplate strip (borrowed from server `_strip_boilerplate_sections`):** a small dependency-free regex line-filter over the extracted text removing "subscribe now / sign up / share on … / follow us / newsletter / read more / thanks for reading" lines, then collapse blank lines.
- **Canonical, post-redirect `url` (improves on the server, which stores the *original* input):** the payload `url` is `httpx`'s final `resp.url` after redirects, canonicalized — lowercase scheme/host, strip default port, drop fragment, strip trailing slash (except root), and strip tracking params (`utm_source/medium/campaign/term/content`, `gclid`, `fbclid`, `igshid`, `mc_cid`, `mc_eid`, `ref`, `ref_src`). Small dependency-free helper `canonicalize_url(url)`.
- **Empty extraction → permanent error** (lightweight take on the server's `_js_required()`): if `trafilatura` returns `None` or content that is empty/whitespace-only after the boilerplate strip, fail with "Couldn't extract article content — the page may require JavaScript (not supported)." (permanent, retryable=False) rather than persisting an empty row.
- Returns a `result`-shaped dict `{content, title, author, keywords: [], analysis: '', chunks: [], metadata: {...}}` so §2's shared tail builds the standard payload (`media_type='article'`; `url=<canonical post-redirect URL>` — §2's `source_url` is set from this for a URL source). Chunking/analysis are handled by the shared pipeline exactly as for a file (the extracted article text is treated like plaintext).
- `import trafilatura` is deferred; ImportError → `FileIngestionError` with an install hint (permanent, retryable=False). `httpx` is a core dep.

### 4. UI form (`library_screen.py` + `library_ingest_canvas.py`)

- `#library-ingest-path` accepts a URL. In `_submit_library_ingest_form`, branch the validation: if `urlparse(raw).scheme in ("http","https")` → `validate_url(raw)` (rejects non-http(s) schemes — see Security); else → `validate_path_simple(require_exists=True)` (today's path). The submitted `source_path` is the URL string for a URL.
- Placeholder/label copy: "Path to a local file or a URL…"; the Browse button still sets a file path.
- `submit_library_ingest_job` no longer rejects a URL (its `detect_file_type` → `classify_ingest_source` swap means a URL yields a real `detected_type` instead of the swallowed `FileIngestionError`).

## Data flow

```
URL in form → validate_url → submit (classify → detected_type) → registry → parse pool
  worker parse_local_file_for_ingest(url):
    url? article → httpx+trafilatura → result{content,title,...}
        audio/video → existing processor(inputs=[url]) download+transcribe → result
    → shared tail: payload{media_type, content, url=<real URL>, chunks, ...}
  → writer persist_parsed_media → media_db.add_media_with_keywords
```

## Error handling

- **Retryable/permanent taxonomy (borrowed from server `http_client._should_retry`):** retryable = a network/timeout error OR HTTP status in `{408, 429, 500, 502, 503, 504}`; permanent = a DNS-resolution failure, any other 4xx, a non-HTML content-type, empty extraction, or missing `trafilatura`. Threaded through the existing `classify_parse_failure` seam (which already maps to the job's `permanent` flag) with clear messages. No backoff (retries are manual). Media-URL download failures use the processors' existing handling.
- The max-response-size + content-type guards prevent a worker from hanging or OOMing on a hostile/huge response.

## Security (desktop threat model)

`validate_url` restricts to `^https?://` with a valid host, so `file://`, `javascript:`, `ftp://`, and scheme-less local paths cannot masquerade as a URL and cause the worker to read local files or execute a non-http fetch — the relevant guard for a single-user desktop app (classic server SSRF — tricking a shared server into reaching internal services — does not apply; the user fetches their own URLs on their own machine).

## Testing

- **Classifier (`Tests/Local_Ingestion/`):** article/audio/video URL patterns classify correctly; a file path still returns `detect_file_type`'s result; a non-http scheme is not treated as a URL.
- **Article extractor:** a recorded-HTML fixture (or a mocked `httpx` response) → asserts the `result` shape (`content`, `title`, `url`, media_type via the tail = `'article'`); a non-HTML content-type → permanent error; a mocked missing `trafilatura` → permanent error with the install hint; the max-size guard aborts an oversized body; **empty/thin extraction → permanent error with the JS hint**; **`canonicalize_url` strips tracking params + trailing slash + fragment and lowercases host** (unit); **boilerplate strip** removes the known phrase lines; a **retryable status (503) vs permanent status (404)** map to the right `permanent` flag.
- **`parse_local_file_for_ingest` URL routing:** a URL → article branch produces a payload whose `url` is the real URL (NOT `file://`) and never calls `.absolute()` on it; a media URL → audio/video branch (processor mocked) yields the standard payload.
- **Submit + UI:** a URL bypasses the file-exists gate, sets the expected `detected_type` (article vs video); a non-http string is rejected.
- **End-to-end-ish:** a URL job's payload is accepted by `persist_parsed_media` (media row with `url`, `media_type`).

## Scope / non-goals

- No Playwright/JS rendering (SPA/JS-heavy pages won't extract — a Playwright fallback is a possible follow-up); no crawling/multi-page; one URL per job.
- Media URLs reuse the existing audio/video processors unchanged.
- The existing `Web_Scraping/Article_Extractor_Lib.py` (Playwright) is left untouched.
- **Deferred to follow-ups (server ideas beyond this AC):** a JSON-LD-first extraction chain (more robust on structured news/blogs) → its own task; using yt-dlp's `extract_info(download=False)` as the definitive classifier for ambiguous media URLs → its own task; the server's two-tier URL→content-hash dedup applies to the existing **task 208** (ingest dedup), not here.
